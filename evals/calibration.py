"""How far to trust the LLM judge: its agreement with blind hand labels on the 40-item sample.

Agreement is reported as the share of items where the judge and the person gave the same grade, and as
Cohen's kappa, which discounts the agreement two raters would reach by chance. Evidence is compared on
the three grades and on the simpler question "supports or not". Requests are compared on acceptable or
needs revision. The report also lists every disagreement, with both sides' reasons.

    python -m evals.calibration --out evals/results/judge_calibration.json
"""

import argparse
from collections import Counter
from pathlib import Path

from evals.gold import GoldCase
from evals.judge import JUDGE_MODEL, PROMPT_VERSION, Judgement, judged_items, run_judge
from evals.judge_items import SAMPLE, HumanLabel, ItemKind, JudgeItem, build_items, load_labels, load_sample
from evals.runner import GOLD, POLICIES, RECORDINGS, Mode
from priorauth.policies import load_policies
from priorauth.schemas import Contract


class Agreement(Contract):
    items: int
    agreed: int
    kappa: float | None  # None when it is undefined: both raters used a single grade for every item


class Disagreement(Contract):
    item_id: str
    kind: ItemKind
    case_id: str
    criterion_id: str
    person: str
    judge: str
    person_note: str
    judge_reasoning: str


class CalibrationReport(Contract):
    judge_model: str
    prompt_version: str
    evidence: Agreement  # evidence and planted items, three grades
    evidence_supports_or_not: Agreement
    requests: Agreement
    planted: int
    planted_caught_by_judge: int  # graded anything but supports
    planted_caught_by_person: int
    disagreements: list[Disagreement]
    judge_grades_all_evidence: dict[str, int]  # over every evidence item, labelled or not


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement beyond chance between two raters: 1 is perfect, 0 is what chance alone would give."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[c] * second[c] for c in first) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(pairs: list[tuple[str, str]]) -> Agreement:
    kappa = cohen_kappa(pairs)
    return Agreement(
        items=len(pairs), agreed=sum(a == b for a, b in pairs), kappa=None if kappa is None else round(kappa, 3)
    )


def calibrate(
    items: dict[str, JudgeItem],
    sample: list[str],
    labels: dict[str, HumanLabel],
    judgements: dict[str, Judgement],
) -> CalibrationReport:
    """Compare the judge with the person on every labelled item. A missing label or grade is an error."""
    if missing := [i for i in sample if i not in labels or judgements.get(i) is None or judgements[i].error]:
        raise ValueError(f"no label or no judge grade for {missing}")
    graded = [(items[i], labels[i], judgements[i].grade) for i in sample]
    evidence = [(label.label, grade.grade) for item, label, grade in graded if item.kind is not ItemKind.REQUEST]
    planted = [(label.label, grade.grade) for item, label, grade in graded if item.kind is ItemKind.PLANTED]
    return CalibrationReport(
        judge_model=JUDGE_MODEL,
        prompt_version=PROMPT_VERSION,
        evidence=agreement(evidence),
        evidence_supports_or_not=agreement([(a == "supports", b == "supports") for a, b in evidence]),
        requests=agreement([(label.label, g.grade) for item, label, g in graded if item.kind is ItemKind.REQUEST]),
        planted=len(planted),
        planted_caught_by_judge=sum(judge != "supports" for _, judge in planted),
        planted_caught_by_person=sum(person != "supports" for person, _ in planted),
        disagreements=[
            Disagreement(
                item_id=item.item_id,
                kind=item.kind,
                case_id=item.case_id,
                criterion_id=item.criterion_id,
                person=label.label,
                judge=grade.grade,
                person_note=label.note,
                judge_reasoning=grade.reasoning,
            )
            for item, label, grade in graded
            if label.label != grade.grade
        ],
        judge_grades_all_evidence=dict(
            Counter(j.grade.grade for i, j in judgements.items() if j.grade and items[i].kind is ItemKind.EVIDENCE)
        ),
    )


def describe(report: CalibrationReport) -> str:
    def rate(a: Agreement) -> str:
        kappa = "undefined" if a.kappa is None else f"{a.kappa:.2f}"
        return f"agreed on {a.agreed} of {a.items} ({a.agreed / a.items:.0%}), Cohen's kappa {kappa}"

    lines = [
        f"Judge {report.judge_model} ({report.prompt_version}) against blind hand labels",
        f"Evidence, three grades: {rate(report.evidence)}",
        f"Evidence, supports or not: {rate(report.evidence_supports_or_not)}",
        f"Planted mismatches caught: judge {report.planted_caught_by_judge} of {report.planted}, "
        f"person {report.planted_caught_by_person} of {report.planted}",
        f"Requests: {rate(report.requests)}",
        f"Disagreements ({len(report.disagreements)}):",
        *(
            f"  {d.case_id} {d.criterion_id.split('.', 1)[-1]} ({d.kind}): person {d.person}, judge {d.judge}"
            for d in report.disagreements
        ),
        "Judge grades over all evidence items: "
        + ", ".join(f"{n} {grade}" for grade, n in sorted(report.judge_grades_all_evidence.items())),
    ]
    return "\n".join(lines)


def load_all(
    gold: Path = GOLD, recordings: Path = RECORDINGS, policies: Path = POLICIES, sample_path: Path = SAMPLE
) -> tuple[dict[str, JudgeItem], dict[str, GoldCase], list[str], dict[str, Judgement]]:
    items, cases = build_items(load_policies(policies), gold, recordings)
    sample = load_sample(sample_path)
    judgements = run_judge(Mode.REPLAY, judged_items(items, sample), cases, recordings=recordings, workers=8)
    return {i.item_id: i for i in items}, cases, sample, {j.item_id: j for j in judgements}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the LLM judge with the blind hand labels.")
    parser.add_argument("--out", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)
    items, _, sample, judgements = load_all()
    report = calibrate(items, sample, load_labels(), judgements)
    print(describe(report))
    if args.out:
        args.out.write_text(report.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
