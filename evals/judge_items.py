"""What the LLM judge grades, and the sample labelled by hand to calibrate it.

Three kinds of item:

    evidence   a criterion the single agent marked met on its first trial, with the quotes it cited
    planted    a real quote from the same case paired with a different criterion: a mismatch the judge
               should catch. Without these, nearly every item would be "supports" and agreement between
               the judge and a person would say little.
    request    a request the documentation agent wrote for a documentation gap

Item IDs are opaque hashes, so neither the labelling page nor the sample file shows which items are
planted. The sample is written once to evals/judge/sample.yaml and committed, so it never shifts.

    python -m evals.judge_items sample      # writes the sample (already committed; for a new gold set only)
"""

import argparse
import hashlib
import random
from enum import StrEnum
from pathlib import Path

import yaml

from evals.gaps import run_gaps
from evals.gold import GoldCase, load_split
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode, run_split
from priorauth.policies import load_policies
from priorauth.schemas import Contract, CriterionStatus, Evidence, Policy

SPLIT = "dev"
SAMPLE = Path("evals/judge/sample.yaml")
LABELS = Path("evals/judge/labels.yaml")
SEED = 7
SAMPLE_SIZE = {"evidence": 24, "planted": 8}  # plus every documentation request


class ItemKind(StrEnum):
    EVIDENCE = "evidence"
    PLANTED = "planted"
    REQUEST = "request"


GRADES = {  # the labels a person (and later the judge) may give each kind of item
    ItemKind.EVIDENCE: ["supports", "partly", "does_not_support"],
    ItemKind.PLANTED: ["supports", "partly", "does_not_support"],
    ItemKind.REQUEST: ["acceptable", "needs_revision"],
}


class HumanLabel(Contract):
    label: str
    note: str = ""


class JudgeItem(Contract):
    item_id: str
    kind: ItemKind
    case_id: str
    criterion_id: str
    criterion_text: str
    quotes: list[Evidence] = []  # what the agent cited (evidence) or a quote meant for another criterion (planted)
    request: str = ""  # request items: the documentation agent's request
    note_type: str = ""


def item_id(kind: ItemKind, case_id: str, criterion_id: str) -> str:
    return "item_" + hashlib.sha256(f"{kind}|{case_id}|{criterion_id}".encode()).hexdigest()[:10]


def build_items(
    policies: dict[str, Policy], gold: Path = GOLD, recordings: Path = RECORDINGS
) -> tuple[list[JudgeItem], dict[str, GoldCase]]:
    """Every item, from the committed recordings: no model is called."""
    cases = load_split(gold, SPLIT, policies)
    items = []
    for result in run_split(SPLIT, Mode.REPLAY, policies, gold=gold, recordings=recordings, trials=1):
        if result.error:
            raise ValueError(f"{result.case_id}: {result.error}")
        policy = policies[cases[result.case_id].policy_id]
        texts = {c.criterion_id: c.text for c in policy.criteria}
        met = {f.criterion_id: f for f in result.record.determination.criteria if f.status is CriterionStatus.MET}
        for criterion_id, finding in met.items():
            items.append(_item(ItemKind.EVIDENCE, result.case_id, criterion_id, texts, quotes=finding.evidence))
        order = [c.criterion_id for c in policy.criteria]
        for index, source in enumerate(order):  # each met criterion's quotes, offered for the criterion halfway round
            target = order[(index + len(order) // 2) % len(order)]
            if source in met and target != source:
                items.append(_item(ItemKind.PLANTED, result.case_id, target, texts, quotes=met[source].evidence))
    for gap in run_gaps(Mode.REPLAY, policies, gold=gold, recordings=recordings):
        if gap.error:
            raise ValueError(f"{gap.case_id}: {gap.error}")
        texts = {c.criterion_id: c.text for c in policies[cases[gap.case_id].policy_id].criteria}
        for written in gap.run.requests:
            items.append(
                _item(
                    ItemKind.REQUEST,
                    gap.case_id,
                    written.criterion_id,
                    texts,
                    request=written.request,
                    note_type=written.note_type,
                )
            )
    return items, cases


def _item(kind: ItemKind, case_id: str, criterion_id: str, texts: dict[str, str], **fields) -> JudgeItem:
    return JudgeItem(
        item_id=item_id(kind, case_id, criterion_id),
        kind=kind,
        case_id=case_id,
        criterion_id=criterion_id,
        criterion_text=texts[criterion_id],
        **fields,
    )


def choose_sample(items: list[JudgeItem], seed: int = SEED) -> list[str]:
    """The items to label: some evidence, some planted, every request, shuffled together.

    A case and criterion used for a planted item is left out of the evidence sample, so the same criterion
    of the same case never appears twice with different quotes, which would give the planted item away.
    """
    rng = random.Random(seed)
    planted = rng.sample([i for i in items if i.kind is ItemKind.PLANTED], SAMPLE_SIZE["planted"])
    taken = {(i.case_id, i.criterion_id) for i in planted}
    evidence = [i for i in items if i.kind is ItemKind.EVIDENCE and (i.case_id, i.criterion_id) not in taken]
    chosen = planted + rng.sample(evidence, SAMPLE_SIZE["evidence"]) + [i for i in items if i.kind is ItemKind.REQUEST]
    rng.shuffle(chosen)
    return [i.item_id for i in chosen]


def load_sample(path: Path = SAMPLE) -> list[str]:
    return yaml.safe_load(path.read_text())["items"]


def load_labels(path: Path = LABELS) -> dict[str, HumanLabel]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {item: HumanLabel.model_validate(label) for item, label in (raw.get("labels") or {}).items()}


def save_labels(labels: dict[str, HumanLabel], sample: list[str], path: Path = LABELS) -> None:
    """Write the labels in sample order, so the file diffs cleanly as labelling goes on."""
    ordered = {item: labels[item].model_dump() for item in sample if item in labels}
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Hand labels for judge calibration, made blind with ui/label.py before the judge ran.\n"
    path.write_text(header + yaml.safe_dump({"labels": ordered}, sort_keys=False, allow_unicode=True))


def check_label(item: JudgeItem, label: HumanLabel) -> None:
    if label.label not in GRADES[item.kind]:
        raise ValueError(f"{item.item_id}: {label.label!r} is not one of {GRADES[item.kind]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the sample of judge items to label by hand.")
    parser.add_argument("command", choices=["sample"])
    parser.add_argument("--out", type=Path, default=SAMPLE)
    args = parser.parse_args(argv)
    items, _ = build_items(load_policies(POLICIES))
    sample = choose_sample(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Items to label by hand for judge calibration (evals/judge_items.py). Do not edit.\n"
    args.out.write_text(header + yaml.safe_dump({"seed": SEED, "items": sample}, sort_keys=False))
    kinds = {kind: sum(1 for i in items if i.kind is kind) for kind in ItemKind}
    print(f"{len(items)} items ({', '.join(f'{n} {k}' for k, n in kinds.items())}); sampled {len(sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
