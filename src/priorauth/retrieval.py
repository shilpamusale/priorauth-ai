"""Choose the coverage policy for a request: retrieval over the LCD texts, then a code check.

Retrieval proposes and code decides. ChromaDB ranks the extracted LCD texts in policies/sources/ by how
close their passages are to the order, written in the clinician's words. The closest LCD must be within
MAX_DISTANCE, a policy here must be built on it, and that policy must cover the request's service code.
Otherwise the request escalates and the route says why. The agent never chooses its own policy.
"""

import re
from collections.abc import Iterable
from enum import StrEnum
from uuid import uuid4

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings

from priorauth.schemas import Contract, Policy

CHUNK_WORDS = 40  # orders are a few words long, so short passages keep the match sharp
OVERLAP_WORDS = 10
TOP_PASSAGES = 10
# Cosine distance, chosen on the dev cases only (python -m evals.routing, Nov 15): orders that should match
# came within 0.597, unrelated ones (knee replacement, cataract surgery) started at 0.617. A thin margin.
MAX_DISTANCE = 0.61
LCD_URL = re.compile(r"lcdid=(\d+)&ver=(\d+)")


class Outcome(StrEnum):
    MATCHED = "matched"
    NO_ORDER = "no_order"  # nothing to search with
    NO_MATCH = "no_match"  # no LCD is close enough to the order
    UNSUPPORTED = "unsupported"  # the closest LCD has no policy here yet
    CODE_NOT_COVERED = "code_not_covered"  # the LCD has policies, but none covers the requested code


class Candidate(Contract):
    source: str  # the LCD's URL, as in the header of its file in policies/sources/
    distance: float  # cosine distance from the order to the LCD's closest passage


class Route(Contract):
    outcome: Outcome
    policy_id: str | None = None
    candidates: list[Candidate] = []  # closest LCD first
    reason: str


def lcd_name(source: str) -> str:
    """'LCD L38773 v32' from an MCD URL, or the URL itself."""
    match = LCD_URL.search(source)
    return f"LCD L{match[1]} v{match[2]}" if match else source


def passages(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Overlapping windows of words, so a phrase cut at one window's edge is whole in the next."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[start : start + size]) for start in range(0, max(len(words) - overlap, 1), step)]


class PolicyIndex:
    """The LCD texts, split into passages and embedded in an in-memory ChromaDB collection.

    With no embedding function, ChromaDB's default is used (all-MiniLM-L6-v2, run locally with ONNX).
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

    def search(self, text: str) -> list[Candidate]:
        """Each LCD with a passage among the closest, ranked by its closest passage."""
        hits = self._collection.query(
            query_texts=[text],
            n_results=min(TOP_PASSAGES, self._collection.count()),
            include=["metadatas", "distances"],
        )
        best: dict[str, float] = {}
        for metadata, distance in zip(hits["metadatas"][0], hits["distances"][0], strict=True):
            source = str(metadata["source"])
            best[source] = min(distance, best.get(source, distance))
        return [Candidate(source=s, distance=round(d, 4)) for s, d in sorted(best.items(), key=lambda item: item[1])]


def route(
    index: PolicyIndex,
    policies: Iterable[Policy],
    ordered_service: str | None,
    service_code: str,
    max_distance: float = MAX_DISTANCE,
) -> Route:
    """The policy for this order and code, or the reason the request must go to a person."""
    if not ordered_service:
        return Route(outcome=Outcome.NO_ORDER, reason="The request has no order text to find a policy with.")
    candidates = index.search(ordered_service)
    best = candidates[0]
    if best.distance > max_distance:
        return Route(
            outcome=Outcome.NO_MATCH,
            candidates=candidates,
            reason=(
                f"No coverage document is close enough to the order: the closest, {lcd_name(best.source)}, "
                f"is at distance {best.distance:.2f} and the limit is {max_distance}."
            ),
        )
    built = [p for p in policies if p.source == best.source]
    if not built:
        return Route(
            outcome=Outcome.UNSUPPORTED,
            candidates=candidates,
            reason=f"The order matches {lcd_name(best.source)}, but no policy here is built on it yet.",
        )
    covering = [p for p in built if service_code in p.service_codes]
    if not covering:
        codes = ", ".join(sorted({code for p in built for code in p.service_codes}))
        return Route(
            outcome=Outcome.CODE_NOT_COVERED,
            candidates=candidates,
            reason=(
                f"The order matches {lcd_name(best.source)}, but code {service_code} is not one its policies "
                f"cover ({codes}): either the order and the code disagree, or this service is not decided here."
            ),
        )
    [policy] = covering  # load_policies refuses two policies on one LCD that cover the same code
    return Route(
        outcome=Outcome.MATCHED,
        policy_id=policy.policy_id,
        candidates=candidates,
        reason=f"The order matches {lcd_name(best.source)} and {policy.policy_id} covers code {service_code}.",
    )
