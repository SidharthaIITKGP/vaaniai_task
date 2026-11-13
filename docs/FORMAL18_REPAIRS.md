# Recovered formal18 audit and repair workflow

The repair run is now complete and independently audited. See
[FORMAL18_AUDIT.md](FORMAL18_AUDIT.md) for the 54-run snapshot, authored grades,
quota findings and selection limitations. The following documents the earlier
repair workflow and its original 52-row starting point.

The original `results/architecture_formal18_recovered.csv` contains 52 distinct
question/architecture evaluations of the planned 54. Its SHA-256 is
`0982b706477f4b2cc471995c8ecaf13ebcf755aaeb0c948fa8e70f2f9446a92c`.
It has 29 `ok`, 14 `insufficient_evidence`, six `citation_validation_failed`,
one `verification_failed`, one HTTP 400 `provider_error`, and one `rate_limited`.
These are execution outcomes, not independent correctness grades. All 14 explicit
insufficient-evidence responses are on U questions, but still require review.

## Failure evidence and corrections

- H6 Fast, X1 Adaptive, M1 Verified, T1 Adaptive, M1 Adaptive, and M3 Fast all
  failed with `no citation aliases found`; none had an invalid alias. The old
  CSV retained only the final refusal, so their discarded drafts cannot be
  reconstructed or retroactively scored as correct. Generation now returns
  structured claims and request-local source aliases in its existing one model
  call. The application renders `[S1]` citations and still validates them. Missing
  sources, invented aliases, and unsupported drafts fail closed. This changes
  the common generation contract for all three architectures, without changing
  retrieval, models, routing settings, or their distinct call flows.
- H4 Verified: the verifier flagged many paraphrases as unsupported, but the
  draft also claimed compression loses no detail. The cited H-Net PDF p.5
  discusses encoder/main/decoder design; it does not establish that guarantee.
  The prompt now distinguishes supported paraphrases from overclaims. A refusal
  containing unsupported claims can receive one correction and re-verification,
  just like a `revise` verdict. A second failed check still refuses. Revision
  source aliases are limited to the evidence actually supplied to revision.
- M3 Adaptive: Groq's `json_validate_failed` HTTP 400 says the verifier omitted
  required `unsupported_claims`, while putting a long discussion into `reason`.
  The prompt explicitly requires all three fields and short feedback. There is
  one format-only retry, with the same strict schema and a 1024-token output cap
  instead of the first attempt's 640. A truncated or locally invalid verifier
  object gets the same bounded recovery. No missing field defaults to acceptance.
  All attempts pass through the same scheduler and model-call budget. Their
  timing, queue waits and known usage are counted; unreported HTTP 400 usage
  remains unknown. Authentication, network and rate-limit errors do not trigger
  a format retry.
- T1 Verified: the last 429 explicitly reports **TPD**, limit 200,000, used
  199,716, requested 1,500, Retry-After 526 seconds. The simultaneous headers
  showed 924 daily requests and 5,838 minute tokens remaining; these are separate
  quotas and do not establish daily-token availability. Existing persisted holds,
  admission checks, shared account lock, RPM/TPM limits and early stopping remain
  active. A daily-token 429 stops immediately, even if Retry-After is short.
  An expired Retry-After is a lower bound for retry, not proof a whole daily
  budget has reset. If a request is rejected again, the runner checkpoints and
  stops again; it never loops through doomed evaluations. Do not clear quota state.

New logs retain the rejected generation and revision drafts for diagnosis; these
are never returned as accepted answers. Structured generation is buffered, so
provider TTFT is unavailable and stays blank; user-visible release latency is
still measured. A passing citation check proves source identity, not semantics.

## Independent grading of the existing snapshot (local only)

```bash
python -m relay.experiment_review packet \
  --input results/architecture_formal18_recovered.csv \
  --output results/formal18_review_52
```

The packet contains 50 completed outcomes, including the seven application
failures/refusals, and lists the two infrastructure failures separately. Open
`results/formal18_review_52/blind_answers.csv` or `.json`, inspect the original
PDFs, and fill `grades.csv` following the included README. The packet hides
architecture, pipeline status and verifier judgment from the answer dossier.
All grades start blank. Repeating packet generation preserves existing grades.
The manifest records input and paper hashes. Previously authored progress
reviews remain separate and unchanged.

After filling the snapshot grades, aggregate them locally:

```bash
python -m relay.experiment_review compare \
  --input results/architecture_formal18_recovered.csv \
  --grades results/formal18_review_52/grades.csv \
  --output results/formal18_snapshot_comparison
```

This reports reviewed accuracy and grounding, but cannot select a winner from
the incomplete 52/54 snapshot. Infrastructure failures never receive answer grades.

## Targeted repair resume

```bash
python -m relay.evaluate --subset formal18 --workers 1 \
  --repair-from results/architecture_formal18_recovered.csv \
  --output results/architecture_formal18_repair_v2.csv --remaining
```

This read-only command recognizes 43 accepted outcomes and lists 11 pending:
H4 Verified, H6 Fast, X1 Adaptive, M1 Verified, T1 Adaptive, M1 Adaptive,
M3 Fast, M3 Adaptive, T1 Verified, T5 Verified, and U1 Verified.
The first seven are completed application failures deliberately reopened for
repair, the next two are infrastructure failures, and the last two are missing.

To execute real requests deliberately, after the provider hold permits:

```bash
python -m relay.evaluate --subset formal18 --workers 1 \
  --repair-from results/architecture_formal18_recovered.csv \
  --output results/architecture_formal18_repair_v2.csv \
  --quota-state data/groq_quota_state.json \
  --max-cooldown 60 --request-deadline 240 \
  --max-model-calls 80 --wall-seconds 3600
```

Repeat the same command to resume. Successful request IDs are retained, not
regenerated. The CSV has one latest row per question ID + architecture +
configuration; every imported and new attempt remains in its append-only journal.
Original CSVs, journals and manifests are never rewritten. Each imported row
retains `origin_evaluation_config_id`. Exit code 2 indicates a checkpointed stop;
inspect the output's `.stop.json` for the reason and earliest safe condition.
Use the shared quota file for every run, and do not launch parallel experiment
processes. No extra quota or midnight reset is assumed.

### Switching to a genuinely different Groq account

Use one stable quota profile per account. Rotating a key within the same account
must retain that account's profile. After placing a different account's key in
`.env`, run:

```bash
env -u GROQ_API_KEY python -m relay.evaluate --subset formal18 --workers 1 \
  --repair-from results/architecture_formal18_recovered.csv \
  --output results/architecture_formal18_repair_v2.csv \
  --quota-profile account2 --quota-state data/groq_quota_state.json \
  --max-cooldown 60 --max-model-calls 80 --wall-seconds 3600
```

The `env -u` applies only to this process and ensures an old exported key cannot
override `.env`. This uses `data/groq_quota_state.account2.json` and its own account
lock. The original state and hold remain intact. Historical rows without a profile
belong to `default`; only cooldowns from the selected account are imported.
Completed evaluations are still shared and deduplicated regardless of account.
New rows record `quota_profile`. Keep reusing `account2` for that account: its own
RPM, TPM, daily quotas, Retry-After and stop rules remain enforced. This command
does not validate that the new account has quota until the user explicitly runs it.
Add `--remaining` to inspect pending work and holds without API calls.

## Fair final architecture selection

**The repair campaign cannot select a winner.** It mixes old successful answers
with corrected behavior only on failed cases. Its immutable manifest marks it
`selection_eligible=false`; copying it through ordinary resume cannot remove
that restriction. Normal resume still skips all completed application outcomes
and only retries infrastructure failures or missing evaluations.

For selection under the corrected implementation, a matched full experiment is
required. This is a separate opt-in real run, not part of targeted repair:

```bash
python -m relay.evaluate --subset formal18 --workers 1 \
  --output results/architecture_formal18_v2_full.csv \
  --quota-state data/groq_quota_state.json \
  --max-cooldown 60 --request-deadline 240 \
  --max-model-calls 200 --wall-seconds 3600
```

Repeat that command after eligible recovery; add `--remaining` for a read-only
pending list. The 54 cases share retrieval, model, routing and corrected behavior.
After completion:

```bash
python -m relay.experiment_review packet \
  --input results/architecture_formal18_v2_full.csv \
  --output results/formal18_v2_review
# Independently review the original PDFs and fill results/formal18_v2_review/grades.csv.
python -m relay.experiment_review compare \
  --input results/architecture_formal18_v2_full.csv \
  --grades results/formal18_v2_review/grades.csv \
  --output results/architecture_selection_v2
```

`selection.json` stays incomplete with no winner until full planned coverage,
compatible provenance, non-infrastructure outcomes and independent grades exist.
`status=ok` alone never establishes correctness. Preserve application failures
and grade them; do not selectively retry incorrect answers in this fair campaign.

## Local regression checks

```bash
python -m pytest -q
```

Tests use mock providers or HTTPX MockTransport. Regressions cover all six
citation cases, bounded H4 correction and failure, M3's schema retry, missing
verifier fields, retry accounting, TPD persistence without retry, immutable repair
history, successful resume deduplication, grading preservation, and selection guards.
These checks establish code behavior; they do not demonstrate that unseen real
Groq answers are factually correct. No real repaired evaluation has been run.
