"""Choose the coverage policy for a request: retrieval over the LCD texts, then a code check.

Retrieval proposes and code decides. ChromaDB ranks the extracted LCD texts in policies/sources/ by how
close their passages are to the order, written in the clinician's words. The closest LCD must be within
MAX_DISTANCE, a policy here must be built on it, and that policy must cover the request's service code.
Otherwise the request escalates and the route says why. The agent never chooses its own policy.

Hybrid retrieval adds a keyword search (BM25) over the same passages. It changes only the orders the
embedding search finds too far: one of those can still reach an LCD whose passages hold enough of the order's
own words, such as the LCD's abbreviations ("MBB"), which a general embedding model may not know.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from uuid import uuid4

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from priorauth.schemas import Contract, Policy

CHUNK_WORDS = 40  # orders are a few words long, so short passages keep the match sharp
OVERLAP_WORDS = 10
# Cosine distance, chosen on the dev cases only (python -m evals.routing, Nov 15): orders that should match
# came within 0.597, unrelated ones (knee replacement, cataract surgery) started at 0.617. A thin margin.
MAX_DISTANCE = 0.61
# The keyword score an order needs to reach an LCD the embeddings find too far, chosen on the dev requests only
# (Nov 19, before the fresh requests were scored): halfway between the highest-scoring dev order that must not
# match (cataract surgery, 0.27) and the dev order the embeddings miss ("Dx MBB L4-5, L5-S1", 0.43).
MIN_KEYWORD = 0.35
BM25_K1, BM25_B = 1.5, 0.75  # the usual BM25 settings, not tuned
STOPWORDS = frozenset("a an and are as at be both by for from in into is it of on or the to with without".split())
LCD_URL = re.compile(r"lcdid=(\d+)&ver=(\d+)")


class Retrieval(StrEnum):
    EMBEDDING = "embedding"  # the closest LCD by meaning, within MAX_DISTANCE
    HYBRID = "hybrid"  # the same; when nothing is that close, the best keyword match, if it reaches MIN_KEYWORD


class Outcome(StrEnum):
    MATCHED = "matched"
    NO_ORDER = "no_order"  # nothing to search with
    NO_MATCH = "no_match"  # no LCD is close enough to the order
    UNSUPPORTED = "unsupported"  # the closest LCD has no policy here yet
    CODE_NOT_COVERED = "code_not_covered"  # the LCD has policies, but none covers the requested code


class Candidate(Contract):
    source: str  # the LCD's URL, as in the header of its file in policies/sources/
    distance: float  # cosine distance from the order to the LCD's closest passage
    keyword: float = 0.0  # the LCD's best passage's BM25 score, as a share of the most this order could score


class Route(Contract):
    outcome: Outcome
    policy_id: str | None = None
    candidates: list[Candidate] = []  # closest LCD first
    reason: str
    by_keyword: bool = False  # the LCD was reached by keyword score, the embeddings having found it too far


def lcd_name(source: str) -> str:
    """'LCD L38773 v32' from an MCD URL, or the URL itself."""
    match = LCD_URL.search(source)
    return f"LCD L{match[1]} v{match[2]}" if match else source


def passages(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Overlapping windows of words, so a phrase cut at one window's edge is whole in the next."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, max(len(words) - overlap, 1), step)]


def terms(text: str) -> list[str]:
    """Lowercase words and codes, without stopwords or single characters ("L4-5" gives "l4")."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


class KeywordIndex:
    """BM25 over the LCD passages. Each LCD scores by its best passage, as a share of an ideal passage's score.

    The ideal passage holds each of the order's terms once at average length, so the share is about how much
    of the order's vocabulary, weighted by rarity, one passage of the LCD contains. Words found in no LCD lower
    the share: an order about a knee scores low everywhere.
    """

    def __init__(self, sources: dict[str, str]) -> None:
        self._passages = [
            (source, Counter(terms(p))) for source, text in sorted(sources.items()) for p in passages(text)
        ]
        if not self._passages:
            raise ValueError("no LCD text to index")
        self._sources = sorted(sources)
        lengths = [sum(counts.values()) for _, counts in self._passages]
        self._average = sum(lengths) / len(lengths) or 1.0
        self._containing = Counter(t for _, counts in self._passages for t in counts)

    def idf(self, term: str) -> float:
        n, total = self._containing[term], len(self._passages)
        return math.log(1 + (total - n + 0.5) / (n + 0.5))  # positive even for a term in every passage

    def scores(self, text: str) -> dict[str, float]:
        """Each LCD's keyword score for the order, from 0 to 1."""
        query = list(dict.fromkeys(terms(text)))
        ideal = sum(self.idf(t) for t in query)
        best = dict.fromkeys(self._sources, 0.0)
        if not ideal:
            return best
        for source, counts in self._passages:
            length = sum(counts.values())
            score = sum(
                self.idf(t)
                * counts[t]
                * (BM25_K1 + 1)
                / (counts[t] + BM25_K1 * (1 - BM25_B + BM25_B * length / self._average))
                for t in query
                if counts[t]
            )
            best[source] = max(best[source], min(1.0, score / ideal))
        return best


class PolicyIndex:
    """The LCD texts, split into passages and embedded in an in-memory ChromaDB collection.

    With no embedding function, ChromaDB's default is used (all-MiniLM-L6-v2, run locally with ONNX). A keyword
    index over the same passages scores each LCD too, for hybrid retrieval.
    """

    def __init__(self, sources: dict[str, str], embedding_function: EmbeddingFunction | None = None) -> None:
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        options = {"embedding_function": embedding_function} if embedding_function else {}
        # in-memory clients in one process share their collections, so each index gets its own name
        self._collection = client.create_collection(
            f"lcd_passages_{uuid4().hex}", metadata={"hnsw:space": "cosine"}, **options
        )
        ids, documents, metadatas = [], [], []
        for n, (source, text) in enumerate(sorted(sources.items())):
            for i, passage in enumerate(passages(text)):
                ids.append(f"{n}-{i}")
                documents.append(passage)
                metadatas.append({"source": source})
        if not documents:
            raise ValueError("no LCD text to index")
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._keywords = KeywordIndex(sources)

    def search(self, text: str) -> list[Candidate]:
        """Every LCD, ranked by its closest passage, with its keyword score."""
        hits = self._collection.query(
            query_texts=[text], n_results=self._collection.count(), include=["metadatas", "distances"]
        )
        best: dict[str, float] = {}
        for metadata, distance in zip(hits["metadatas"][0], hits["distances"][0], strict=True):
            source = str(metadata["source"])
            best[source] = min(distance, best.get(source, distance))
        keyword = self._keywords.scores(text)
        return [
            Candidate(source=s, distance=round(d, 4), keyword=round(keyword[s], 4))
            for s, d in sorted(best.items(), key=lambda item: item[1])
        ]


def route(
    index: PolicyIndex,
    policies: Iterable[Policy],
    ordered_service: str | None,
    service_code: str,
    max_distance: float = MAX_DISTANCE,
    retrieval: Retrieval = Retrieval.EMBEDDING,
    min_keyword: float = MIN_KEYWORD,
) -> Route:
    """The policy for this order and code, or the reason the request must go to a person."""
    if not ordered_service:
        return Route(outcome=Outcome.NO_ORDER, reason="The request has no order text to find a policy with.")
    candidates = index.search(ordered_service)
    best, by_keyword = candidates[0], False
    if best.distance > max_distance:
        reason = (
            f"No coverage document is close enough to the order: the closest, {lcd_name(best.source)}, "
            f"is at distance {best.distance:.2f} and the limit is {max_distance}."
        )
        if retrieval is Retrieval.EMBEDDING:
            return Route(outcome=Outcome.NO_MATCH, candidates=candidates, reason=reason)
        words = max(candidates, key=lambda c: c.keyword)
        if words.keyword < min_keyword:
            reason += (
                f" Its best keyword match, {lcd_name(words.source)}, scores {words.keyword:.2f} "
                f"and the minimum is {min_keyword}."
            )
            return Route(outcome=Outcome.NO_MATCH, candidates=candidates, reason=reason)
        best, by_keyword = words, True
    found = (
        f"The order's words match {lcd_name(best.source)} (keyword score {best.keyword:.2f})"
        if by_keyword
        else f"The order matches {lcd_name(best.source)}"
    )
    built = [p for p in policies if p.source == best.source]
    if not built:
        return Route(
            outcome=Outcome.UNSUPPORTED,
            candidates=candidates,
            reason=f"{found}, but no policy here is built on it yet.",
            by_keyword=by_keyword,
        )
    covering = [p for p in built if service_code in p.service_codes]
    if not covering:
        codes = ", ".join(sorted({code for p in built for code in p.service_codes}))
        return Route(
            outcome=Outcome.CODE_NOT_COVERED,
            candidates=candidates,
            reason=(
                f"{found}, but code {service_code} is not one its policies cover ({codes}): either the order "
                "and the code disagree, or this service is not decided here."
            ),
            by_keyword=by_keyword,
        )
    [policy] = covering  # load_policies refuses two policies on one LCD that cover the same code
    return Route(
        outcome=Outcome.MATCHED,
        policy_id=policy.policy_id,
        candidates=candidates,
        reason=f"{found} and {policy.policy_id} covers code {service_code}.",
        by_keyword=by_keyword,
    )
