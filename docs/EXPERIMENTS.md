# Reproducing The Relay experiments

Current status: architecture and retrieval winners are **unselected**. Existing CSVs
are retained. No correctness grades are inferred from successful statuses. The
original assignment PDF is now present as `The_Relay_Assignment_Brief.pdf`.
See `docs/ASSIGNMENT_COMPLIANCE.md` for the verified headings and raw-results schema.

## Environment and checks

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The existing `.env` is loaded automatically. Do not commit it.

## Experiment 1: architecture quality before optimization

The `formal18` set includes 13 answerable paper/comparison questions and all five
unsupported paper-related questions. Every candidate receives the same question
set, model, retrieval configuration, routing policy, revision limit, and scheduler.

```bash
python -m relay.evaluate --subset formal18 --workers 1 --mode end-to-end \
  --max-model-calls 200 --wall-seconds 3600 \
  --output results/architecture_formal18.csv
python -m relay.experiment_review packet --input results/architecture_formal18.csv \
  --output results/architecture_review
```

Independently review `blind_answers.json` against the three original PDFs. Enter
grades in `grades.csv`, recording checked pages, a reviewer identifier, and an
explanation. Accuracy and citation grounding are binary; unsupported questions
also require a correct-refusal grade. Provider failures count as unsuccessful
service outcomes, not evidence refusals. Do not copy semantic-verifier verdicts.
Check each claim and its cited passage, omissions, numerical values, and units.

```bash
python -m relay.experiment_review compare --input results/architecture_formal18.csv \
  --grades results/architecture_review/grades.csv --output results/architecture_selection
```

Selection requires identical 15+ question coverage and complete independent grades.
The predeclared order is accuracy, citation grounding, correct refusals, provider
reliability, input tokens, then end-to-end latency. Treat ties and small-sample
differences cautiously. The selection file stays null while grades are incomplete.

Runs stop at their explicit model-call and wall-time budgets. Completed rows are
checkpointed. Resume only under the identical manifest; changed source/index/model
settings require a new filename. Real Groq has no automatic retry.

## Experiment 2: retrieval, after an architecture is selected

Prepare isolated indexes and local retrieval diagnostics:

```bash
HF_HUB_OFFLINE=1 python -m relay.retrieval_lab --output results/final_retrieval_lab
```

Set ARCH to the measured winner in `architecture_selection/selection.json`.
Do not substitute a preferred candidate before the first experiment is graded.

```bash
ARCH=$(python -c 'import json; d=json.load(open("results/architecture_selection/selection.json")); assert d["winner"], "No measured winner"; print(d["winner"])')
python -m relay.evaluate --subset formal18 --architectures "$ARCH" --workers 1 \
  --index results/final_retrieval_lab/chunks_300.jsonl --top-k 3 --configuration 300_3 \
  --output results/retrieval_300_3.csv
python -m relay.evaluate --subset formal18 --architectures "$ARCH" --workers 1 \
  --index results/final_retrieval_lab/chunks_500.jsonl --top-k 5 --configuration 500_5 \
  --output results/retrieval_500_5.csv
python -m relay.evaluate --subset formal18 --architectures "$ARCH" --workers 1 \
  --index results/final_retrieval_lab/chunks_800.jsonl --top-k 7 --configuration 800_7 \
  --output results/retrieval_800_7.csv
python -m relay.experiment_review packet --input results/retrieval_300_3.csv \
  results/retrieval_500_5.csv results/retrieval_800_7.csv --output results/retrieval_review
# Independently grade against the PDFs, then:
python -m relay.experiment_review compare --input results/retrieval_300_3.csv \
  results/retrieval_500_5.csv results/retrieval_800_7.csv \
  --grades results/retrieval_review/grades.csv --output results/retrieval_selection
python -m relay.finalize \
  --architecture-selection results/architecture_selection/selection.json \
  --retrieval-selection results/retrieval_selection/selection.json \
  --variants results/final_retrieval_lab/variants.json
python -m relay.cli ask --deployment results/deployment.json --question "Explain multi-head attention"
```

Chunk sizes are whitespace-token approximations. Keep embedding, overlap=80,
hybrid fusion, grounding, and verification settings constant. Compare final answer
quality as well as passage coverage. No local retrieval diagnostic proves accuracy.

## Calibration and load experiments (local, automatic)

```bash
python -m relay.load_lab --output results/final_load_lab --seed 42
python -m relay.simulator_server --calibration results/final_load_lab/calibration.json
```

The HTTP server binds localhost:8081. POST chat messages and optional `stream:true`
to `/openai/v1/chat/completions`. Replies are explicitly simulated; no paper-answer
accuracy is claimed. Stop with Ctrl-C. The baseline uses empirical probe service
timing; architecture bundle timing is available through `--architecture` in load_lab
after selecting the real-system winner. It conservatively reserves a whole bundle.

The steady baseline offers 20 requests/minute. Four scenarios use 120 requests;
sustained overload uses 600 over approximately five minutes. All use seed 42 and compare zero and two bounded
admission retries. Detailed rows include queue length, waits, service, tokens,
outcomes and a slow-request breakdown. `quota_3pm.json` contains the exact response.
Prices are absent by default. Supply verified contractual rates explicitly with
`--input-usd-per-million` and `--output-usd-per-million` to produce estimated costs.
Simulated admission retries are not real Groq retries or inferred production rates.

The priced run in `results/final_load_lab_complete/` uses public Groq list prices
of $0.075 input / $0.30 output per million tokens (checked against
https://console.groq.com/docs/models on 2026-11-13). These are estimated paid-tier
equivalents, not actual free-tier charges. Reproduce in a new directory with:

```bash
python -m relay.load_lab --output results/load_reproduction \
  --input-usd-per-million 0.075 --output-usd-per-million 0.30
```

```bash
python -m relay.submission
```

This writes a two-page **progress report**, audit, provisional `results.csv`, and
status to `results/submission`. It does not certify completion. Final architecture
integration, final-system load results, graded quality tables, and exact assignment
format remain blocked on the measured choices and outstanding experiments.
Export the PDF-aligned raw file and required tables with `python -m relay.assignment_export`.
