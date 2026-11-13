# Formal18 execution recovery

These notes describe the earlier scheduler recovery. For the current 52/54
snapshot and corrected RAG behavior, use [FORMAL18_REPAIRS.md](FORMAL18_REPAIRS.md).
The original experiment configuration cannot be silently resumed after prompt changes.

The original `results/architecture_formal18.csv` and its manifest remain unchanged.
The initial audit found 14 attempt rows. A still-running legacy process subsequently
added three more local timeouts; it was stopped with SIGINT. The preserved file has
17 rows: 2 `ok`, 2 `insufficient_evidence`, 4 provider 429s, and 9 local admission failures.
Four evaluations are complete and 50 remain out of
the fixed 54-pair plan. Completion is not a quality grade.

## Diagnosis

- Retry-After values were 160, 1077, 553 and 1082 seconds. The previous provider
  discarded the 429 response body and headers, so these historical limit types
  remain **unknown**. Neither a long wait nor a 429 proves TPM exhaustion.
- Preceding successful calls reported hundreds of daily requests and thousands
  of TPM tokens remaining. These snapshots do not identify the later failing
  quota: daily-token or other account limits were not retained.
- After the long cooldown, the old runner continued launching evaluations. Six
  independent 240-second queue deadlines initially consumed about 24 minutes locally;
  the legacy process accumulated three more before it was stopped.
- Eight calls reported usage: 19,278 input + 2,146 output tokens. Their reservations
  totaled 23,895 versus 21,424 actual tokens. Inputs were underestimated by as much
  as 367 tokens. Unknown usage on other attempts is not recorded as zero.
- Revision and re-verification legitimately add calls; their budgets and evidence
  remain intact. The classifier and verifier JSON schema now contributes to the
  input reservation estimate. Citation and answer-quality checks are unchanged.

## Safe read-only inspection (no model load or Groq calls)

```bash
python -m relay.evaluate --subset formal18 --workers 1 \
  --resume-from results/architecture_formal18.csv \
  --output results/architecture_formal18_recovered.csv --remaining
```

This validates the index hash, model, routing, retrieval, revision and fairness
settings before recognizing earlier results. The legacy whole-package source hash
is retained as provenance; changes to scheduling/exports do not invalidate completed
answers merely because unrelated Python files changed. New checkpoints also hash
the actual RAG and prompt logic, which must match on subsequent resumes.

## Resume (deliberate real API execution)

```bash
source .venv/bin/activate
python -m relay.evaluate --subset formal18 --workers 1 \
  --resume-from results/architecture_formal18.csv \
  --output results/architecture_formal18_recovered.csv \
  --max-cooldown 60 --request-deadline 240 \
  --max-model-calls 200 --wall-seconds 3600
```

Repeat that command after an eligible recovery. A long provider cooldown stops the
run after checkpointing the triggering failure. No automatic API retry occurs.
`architecture_formal18_recovered.stop.json` reports the reason, the earliest UTC
retry bound when supplied, and the recovery condition. Without a provider reset
time, daily exhaustion requires verified account recovery; no reset is invented.

An unexpired Retry-After in imported history is honored. The shared
`data/groq_quota_state.json` persists local reservations, provider counters, reset
times and holds across restarts. Do not delete it to bypass a hold. An account lock
prevents concurrent evaluation processes from using separate quota budgets.

## Data contract

- `*.attempts.jsonl`: append-only, fsynced raw attempt journal, including imported
  historical failures and execution-source provenance. No attempt is removed.
- `*_recovered.csv`: atomic derived view, one row per question ID + architecture +
  evaluation configuration. A retry replaces only the infrastructure-failed view
  row; its earlier raw attempt stays in the journal.
- `*.manifest.json`: immutable evaluation settings. Meaningful model, retrieval,
  router, prompt, citation or verification changes require a separate experiment.
- `--remaining`: lists unfinished and infrastructure-failed evaluations without
  starting the model service. Successful evaluations, application refusals and
  completed verification failures are skipped; selectively rerunning wrong
  answers would bias the comparison.
- Every 429 retains the error body, all rate headers, provider execution time,
  local reservation metadata and limit classification. RPM, TPM, ITPM, OTPM, RPD,
  TPD and unknown provider limits are distinguished from local admission failures.
- Locally rejected admissions count as stage attempts, not dispatched model calls.
- Grading packets exclude infrastructure failures. They are reliability outcomes,
  never automatically incorrect answers.

Stop exit code is 2; complete exit code is 0. Use a new output filename rather than
`--overwrite`. Preserve the journal with the CSV when copying or archiving a run.
