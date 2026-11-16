# Error analysis, written before the final held-out run

Written Nov 15, 2026, before the 7 held-out cases and 3 held-out routing requests are run for the first and
only time. It records what went wrong on the dev set, why, and what will and will not change. The system
the held-out run measures is frozen as of this commit:

| Part | Frozen version |
|---|---|
| Model and prompt | claude-sonnet-5-5, `matcher_v1` |
| Facet policy (LCD L38773 v32) | fingerprint `b798094e7762f2da` |
| Blepharoplasty policy (LCD L33944 v28) | fingerprint `bcbb12e42642849e` |
| Retrieval | all-MiniLM-L6-v2, 40-word passages, distance limit 0.61 |
| Judge | claude-opus-5-5, `judge_v1` (evaluation only) |

## 1. The agent's misses on the dev set

Across 51 single-agent runs, 2 of 372 criterion statuses differ from the gold labels, both in a third
trial, and neither changed a decision. The multi-agent pipeline, run once per case, has 2 more.

| Run | Criterion | Gold | Agent | What happened |
|---|---|---|---|---|
| dev_006, trial 3 | functional deficit | met | insufficient evidence | Wanted the limits scored on a disability scale |
| dev_009, trial 3 | no other pathology | met | not met | Counted the untreated radiculopathy here as well as under its own criterion |
| dev_009, multi-agent | no other pathology | met | not met | The same reading |
| dev_011, multi-agent | pain severity | met | insufficient evidence | Would not treat a pain score that covers mostly buttock and thigh pain as a low-back score |

None is a grounding failure: every quote in every run was found in the note it named, and no hard gate
failed. All four are about where one criterion ends and another begins.

## 2. What the human review says about them

The review queue held every dev case the agent escalated or whose trials disagree. Reviewing them from the
chart, without the gold labels on screen:

- **Functional deficit (dev_006, dev_012).** The review marked it insufficient evidence on both: the limits
  are described in narrative but measured on no scale. That is the agent's dev_006 trial-3 reading. The
  criterion text says "measured on a pain or disability scale"; the gold labels read a pain score plus
  described limits as enough. Two careful readers disagree, so the wording is the problem.
- **Pain severity (dev_011).** The review marked it not met. The multi-agent pipeline said insufficient
  evidence and the LLM judge said `partly`. Three independent readings doubt the gold label.
- **No other pathology (dev_009).** The review marked it not met, for a reason neither the gold label nor
  the agent considered: the MRI was taken on 7/30, before the leg symptoms began around 8/7, so it cannot
  rule out a disc cause for them. The agent's trial 3 reached the same status for a different reason.

So three of the four misses sit on labels the review questions. The agent is less wrong than its score
says, and the gold set is less certain than its labels look.

## 3. Decisions made now, before the held-out run

- **Nothing changes before the held-out run.** No gold label, policy wording or prompt is edited. The
  held-out cases were labelled with the same reading of the policies as the dev cases, so the comparison
  stays like for like. Relabelling dev cases now would raise the dev score without changing the system.
- **After the held-out run:**
  1. dev_011 pain severity becomes insufficient evidence, the reading two of the three doubts share.
  2. The functional-deficit criterion says outright whether described limits without a scale count. That
     is a policy change: a new policy fingerprint, re-recorded runs and a new judge sample for that
     criterion.
  3. The "no other pathology" criterion gets a note on imaging that predates the symptoms it is meant to
     explain. This is a question for the policy owner, not a fix to make alone.

## 4. Retrieval

- **The miss.** dev_002's order, "Dx MBB L4-5, L5-S1", lands at distance 0.678 from the facet LCD: the
  general embedding model does not know the shorthand. It escalates, which is safe.
- **The margin.** Matched orders came within 0.597 and the closest unrelated order at 0.617, so the 0.61
  limit has little room on either side.
- **A fix that does not tune to one case.** The facet LCD itself uses "MBB" eight times. A keyword search
  over the same LCD passages, combined with the embedding search, would match dev_002 without a
  hand-written abbreviation list, and would widen the margin for orders that share the LCD's own terms.
  It is a candidate for after the held-out run, measured on the dev requests and then on new ones.
- **Expected weak spots in the held-out run.** Two held-out orders use shorthand, one with "MBB" and one
  with "bleph". Both also contain words the embedding model knows ("lumbar"; "dermatochalasis"), so either
  may route correctly. Each result will be reported whichever way it goes.

## 5. The judge

- More lenient than a person on `partly`, and inconsistent about it across similar cases.
- Not shown the request date, so it wrongly failed pain duration on dev_008 and dev_010. The first change
  for a `judge_v2`, which would need new hand labels to be measured fairly.
- Calibrated on the facet policy only; it does not grade the blepharoplasty cases.

## 6. What the held-out run will report

For the 7 held-out cases, 3 trials each: decisions, criterion statuses, hard-gate failures, trials that
disagree and cost. For the 3 held-out routing requests and the 7 held-out orders: routes right, and any
unsafe route. Every held-out miss will be placed in one of the categories above, or named as a new one.

## Result, appended after the held-out run

Recorded once, on Nov 15, 2026, with the system frozen as above.

- **Agent:** 21 of 21 runs reached the expected decision and all 156 criterion statuses match the gold
  labels. No hard gate failed and no case's trials disagreed. Cost $0.61.
- **Retrieval:** 9 of 10 routed right and none unsafe. Both shorthand orders routed. The miss was the thin
  margin of section 4 on a new order: "Anterior cervical discectomy and fusion, C5-C6" came within the
  distance limit of the facet LCD (0.514), and the code check stopped it, so it escalated as
  `code_not_covered` instead of `no_match`. A new category is not needed: it is the margin problem, caught
  by the code check the design puts after retrieval.
- **Done after the run, as decided in section 3:** dev_011's pain severity label became insufficient
  evidence. The dev criterion score moves from 370 to 367 of 372; no decision changes.
- **Still to do:** the functional-deficit wording (a policy change that needs re-recorded runs), the
  question on "no other pathology" and imaging dates, keyword search alongside the embedding search, and
  `judge_v2`.
