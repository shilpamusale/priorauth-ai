# PriorAuth-AI

[![CI](https://github.com/shilpamusale/priorauth-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/shilpamusale/priorauth-ai/actions/workflows/ci.yml)

**[Live demo](https://priorauth-ai-140535548045.us-central1.run.app)** · recorded runs of the agent on synthetic charts, replayed through the real pipeline

An agentic prior-authorization workflow with an evaluation harness, built from the provider side: given
an ordered service and a patient's clinical notes, decide whether the request is ready to submit to the
payer, which documentation is missing, or whether a human needs to look at it.

> **Status: in progress.** The agent, its verification and decision steps, the evaluation harness, a
> multi-agent comparison, a documentation agent, an LLM judge calibrated against hand labels, a public
> demo and a private API are live and tested in CI. Policy retrieval comes next (see [Roadmap](#roadmap)).
> All patients are synthetic; no PHI is used anywhere.

## Why provider-side

Most prior-authorization demos play the payer and approve or deny. Provider revenue-cycle teams have a
different job: get the request right before it is submitted, because a request missing documentation
comes back as a denial and costs days of rework. That is the workflow this project automates, and the
one I worked on at R1 RCM.

## How it works

```mermaid
flowchart LR
    A[PA request<br/>service code, ICD-10,<br/>clinical notes] --> B[Matcher agent<br/>Claude with tools:<br/>read_note, search_chart,<br/>submit_finding]
    P[Coverage policy<br/>criteria quoted from<br/>the LCD] --> B
    B --> C["verify()<br/>every quote must be<br/>in the note it names"]
    C --> D{"decide()<br/>deterministic code"}
    D -->|all required met| E[ready_to_submit]
    D -->|evidence missing| F[needs_documentation<br/>lists the gaps]
    D -->|not met or unchecked| G[escalate<br/>human reviewer]
    B & C & D --> H[(Audit record)]
```

1. **The matcher agent** (Claude, strict tool use) sees the request, the policy criteria and a list of
   the notes, but not their text. It reads notes and searches the chart with tools, then submits one
   finding per criterion: `met`, `not_met` or `insufficient_evidence`, quoting the chart and naming the
   note. A quote that is not in the note is rejected on submission, and the model can correct it.
2. **`verify()`** checks every quote again on the finished findings. A finding with any bad quote is
   dropped, so the criterion counts as unchecked.
3. **`decide()`** turns the grounded findings into the decision in plain code. The model never makes the
   final call.
4. **The audit record** keeps the determination, the policy and prompt fingerprints, the model, every
   tool call and result, token use, and anything skipped or rejected.
5. **The documentation agent** turns each gap into a request to the clinic: what this chart already
   records, what it lacks, and which note should carry it. It never changes a finding or the decision.

## Design decisions

| Decision | Why |
|---|---|
| The decision is computed in code, not generated | Auditable, reproducible, and an instruction hidden in a note cannot change it |
| Anything not clearly met goes to a human | A wrong "ready to submit" becomes a denial; a wrong escalation costs a reviewer a few minutes |
| A skipped or unverified required criterion escalates | The pipeline never assumes a criterion it did not check |
| Every quote is checked twice: when the model submits it, and again before deciding | Grounding is enforced by string match, not by trusting the model |
| Criterion and note IDs are enums in strict tool schemas | The API itself stops the model from citing another policy's criterion or a note that does not exist |
| Every policy criterion quotes its LCD | Checked when policies load, so a criterion cannot drift from the published policy |
| Policies are data, not code | Service names, criteria and thresholds live in `policies/`; the prompt contains no numbers (a test enforces it) |
| Replay is strict | A recording replays only for the exact request it was made for, so CI never scores a version of the agent that was not run |
| Held-out cases are refused unless the final evaluation asks for them | The rule against tuning on held-out cases is enforced in code |
| Synthetic patients, public policies | No PHI; real coverage documents keep the problem realistic |

## Evaluation

Gold cases are written and committed before the agent runs on them. Each case is run three times,
because the agent is not deterministic. Live runs are recorded once; CI replays the recordings through
the current code with no API key and fails the build on any hard-gate failure.

- **Gold set:** 20 synthetic multi-note charts on LCD L38773 (first lumbar medial branch block): 14 dev
  cases and 6 held out until the final run. They cover each decision, documentation gaps, clinic shorthand
  spread across notes, a copied-forward stale note, an old diagnosis since resolved, and a prompt
  injection inside a referral letter.
- **Checks:**

| Check | Fails when |
|---|---|
| `decision_correct` | The decision differs from the gold case |
| `criteria_correct` | Any criterion's status differs from the gold labels (no finding counts as wrong) |
| `evidence_notes_match` | A met criterion cites none of the notes the gold label names |
| `evidence_grounded` (hard gate) | A quote does not appear in the note it names |
| `ready_requires_all_met` (hard gate) | `ready_to_submit` while a required criterion lacks a met finding |
| `decision_follows_criteria` (hard gate) | The decision is not the one the output's own criteria imply |

**Dev set baseline** (claude-sonnet-5-5, prompt `matcher_v1`, 14 cases × 3 trials, recorded Nov 14, 2026):

| Metric | Value |
|---|---|
| Decision accuracy | 1.00 (42 of 42 trials) |
| Criterion accuracy | 0.99 (334 of 336 criterion labels) |
| Hard-gate failures | 0 |
| Cases whose trials reached different decisions | 0 of 14 |
| Cost | $1.25 for 42 runs, about $0.03 per case |
| Model calls per run | 2: read the notes, then submit every finding |

These are dev cases, written and used while building, so they show the pipeline works rather than how
well it generalizes. The 6 held-out cases run once, in the final evaluation. Both criterion misses were
about where one criterion ends and another begins, not about grounding: in one trial the model asked for
a scored disability scale before calling a functional deficit met (the policy accepts a pain score with
described limits), and in another it counted an untreated radiculopathy against "no other pathology",
although radiculopathy has its own criterion. Neither changed a decision.

CI replays these recordings on every pull request and fails below 0.95 decision accuracy or on any
hard-gate failure.

**Single agent vs. multi-agent** (the same 14 dev cases; multi-agent recorded once per case, Nov 14, 2026):

| | Single agent | Multi-agent: one evidence agent per criterion, in parallel |
|---|---|---|
| Decisions correct | 42 of 42 (14 of 14 on trial 1) | 14 of 14 |
| Cases with every criterion right | 40 of 42 (14 of 14 on trial 1) | 12 of 14 |
| Hard-gate failures | 0 | 0 |
| Model calls per case | 2 | 16 to 19 |
| Tokens per case | about 5.3k in, 1.9k out | about 36k in, 3.8k out |
| Cost per case | $0.03 | $0.11 |

Splitting the work cost 3.7 times as much and was not more accurate, so the single agent is the one
deployed. Each evidence agent uses the single agent's prompt and tools on a one-criterion view of the
policy, so the comparison measures the orchestration alone. The multi-agent pipeline repeated the single
agent's radiculopathy slip on dev_009 (counting it against "no other pathology", which has its own
criterion), which points to how the criteria are worded rather than to one agent handling eight criteria.
Its other miss is a judgment call worth a second look: on dev_011 the only pain score covers pain that is
mostly in the buttock and thigh, and the agent would not treat it as a low-back score. Both recordings
replay in CI, and the demo can show either pipeline.

**Documentation agent** (on the single agent's first trial of each dev case): 7 dev cases had
documentation gaps, and the agent wrote 8 requests, one per gap, for $0.04 in total. Each says what the
chart already records and what it lacks, and names the note that should carry it, for example on dev_004:
"Please document a lower-extremity neurological exam (strength, sensation, reflexes, straight leg raise)."
None invents a result or asks for a particular score. CI replays the requests and fails if a gap is left
without one; the LLM judge below grades their quality.

**LLM judge** (claude-opus-5-5, prompt `judge_v1`): the string match proves a quote is in the chart, not
that it supports the criterion. The judge reads each met finding's quotes, and only the notes they cite,
and grades the evidence `supports`, `partly` or `does_not_support`. For each documentation request it
answers four yes/no checks (specific, asks only for what is needed, invents nothing, says where to put
it), and code turns them into `acceptable` or `needs_revision`. To measure how far to trust it, I
labelled 40 items blind on a local page before the judge ran: 24 of the agent's evidence findings,
8 planted mismatches (a criterion paired with another criterion's evidence) and the 8 requests. The page
never shows which kind an item is.

| Judge vs. my labels | Agreed | Cohen's kappa |
|---|---|---|
| Evidence and planted, three grades | 27 of 32 (84%) | 0.71 |
| Evidence and planted, supports or not | 28 of 32 (88%) | 0.74 |
| Planted mismatches caught | judge 8 of 8, me 8 of 8 | |
| Documentation requests | 8 of 8 | undefined: both of us found every request acceptable |

What the disagreements show:

- **The judge is more lenient on `partly`.** Four of the five disagreements are evidence I graded
  `partly` and the judge graded `supports`: two baseline disability scores with nothing showing they came
  before the block, and two "no untreated radiculopathy" findings that say nothing about neurogenic
  claudication.
- **It applies its rubric unevenly.** On the unlabelled items it graded those same gaps `partly`
  (baseline timing on dev_002 and dev_013, claudication on dev_008).
- **It lacks one piece of context.** The judge is not shown the request date, so on dev_008 and dev_010
  it measured pain duration to the note date and graded the evidence `does_not_support`. That is a
  judge bug, not an agent error, and the first change for a `judge_v2`.
- **It raises the same question on dev_011** that the multi-agent pipeline did: it graded the single
  agent's pain score `partly` because most of that pain is in the buttock and thigh, as I did.
- **The request sample cannot measure it.** With no bad request in the sample, kappa is undefined;
  planted bad requests would be needed.

Over all 99 evidence findings the judge graded 91 `supports`, 6 `partly` and 2 `does_not_support` (the two
duration false alarms), for $1.13 across 115 items. I have not tuned the judge on these 40 labels: a
`judge_v2` would need fresh labels to be measured fairly. CI replays the judge and recomputes these
numbers in a step that is allowed to fail, because the judge informs a review and never blocks a merge.

## Demo and API

- **Public demo** ([link](https://priorauth-ai-140535548045.us-central1.run.app)): replays recorded runs and
  never calls a model. Every recording is a real Claude run, replayed through the same code CI tests, so a
  visitor sees exactly what was evaluated, cannot run up a bill, and the image holds no API key or model SDK.
  On the single agent's first trial it also shows the LLM judge's grade and reasoning for each met criterion
  and each request to the clinic, false alarms included.
- **Private API** (FastAPI on Cloud Run, Google sign-in required): runs the live pipeline.
  `POST /determinations` takes a policy ID and a request and returns the determination with its audit
  record; `GET /policies` lists policies with their fingerprints; interactive docs are at `/docs`. A daily
  run limit caps spending, the Anthropic workspace's monthly limit is the hard backstop, and the API key
  is read from Secret Manager at run time.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test                                      # unit tests, no API key needed
python -m evals.runner replay                  # score the committed recordings, no API key needed
python -m evals.runner replay --track          # ...and log the run to MLflow (local mlflow.db)
python -m evals.gaps replay                    # the documentation agent's requests, replayed
python -m evals.calibration                    # the LLM judge against the hand labels, replayed
python -m streamlit run ui/app.py              # the demo, locally
uvicorn priorauth.api:app --port 8000          # the API, locally (live runs need ANTHROPIC_API_KEY)

# live runs need ANTHROPIC_API_KEY
python -m evals.runner record --cases dev_001 --trials 1

# extract a new LCD version (the PDF stays out of the repo)
python -m priorauth.ingest path/to/LCD.pdf policies/sources/l38773_v32.txt
```

## Repository layout

```
src/priorauth/
  schemas.py      typed contracts for every handoff (Pydantic)
  policies.py     policy loader: ID rules, quotes checked against the LCD, fingerprint
  ingest.py       extracts the coverage section of an LCD PDF
  llm/            model interface, Claude adapter, scripted and record/replay clients
  tools.py        the agent's chart tools and quote checks at submission
  matcher.py      bounded tool loop; prompts/ holds the versioned prompt
  verify.py       final grounding check
  decision.py     deterministic decision rule
  orchestrator.py multi-agent matching: one evidence agent per criterion, in parallel
  documentation.py  documentation agent: a request to the clinic for each gap
  pipeline.py     match, verify, decide, and the audit record
  api.py          FastAPI service with a daily run limit
policies/         LCD L38773 policy and the extracted LCD text its criteria quote
evals/
  gold/           dev and held-out cases
  recordings/     recorded live runs, replayed in CI and in the demo
  checks.py, runner.py, scoring.py   checks, record/replay runner, metrics and the CI gate
  tracking.py     MLflow logging of each evaluation run
  gaps.py         runs the documentation agent on the dev cases with gaps
  judge_items.py, judge.py, calibration.py   items to grade, the LLM judge, and its agreement with hand labels
  judge/          the blind-labelled sample
ui/               Streamlit demo, and the local labelling page for the judge sample
tests/            unit tests for every module, policy and gold case
```

## Roadmap

- [x] Contracts, decision rule, CI
- [x] Policy ingestion from LCD PDFs; first real policy (LCD L38773) with every criterion quoting the LCD
- [x] Multi-note synthetic charts and a gold set of 20 cases, 6 held out, written before the agent
- [x] Criteria-matching agent (Claude API, strict tool use) with quote checks and an audit record
- [x] Record-and-replay evaluation runner, checks with hard gates, metrics and a CI eval gate
- [x] Public demo on Cloud Run (Docker, Streamlit) showing recorded runs
- [x] MLflow tracking of evaluation runs (params, metrics, fingerprints of the policy, prompt and gold set)
- [x] FastAPI service, deployed privately, with a daily run limit and the key in Secret Manager
- [x] Multi-agent orchestration: one evidence agent per criterion, compared with the single agent
- [x] Documentation agent: a request to the clinic for each documentation gap
- [x] LLM-as-judge grading, calibrated against blind hand labels
- [ ] Policy retrieval with ChromaDB, and a second policy added with no code changes
- [ ] Reviewer screen with overrides feeding the gold set
- [ ] Error analysis and the final held-out run

## Out of scope

Payer-side adjudication, live payer or EHR integration, real patient data, and clinical use.
