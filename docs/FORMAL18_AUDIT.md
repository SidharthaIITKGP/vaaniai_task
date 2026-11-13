# Completed formal18 repair audit

The completed source is `results/architecture_formal18_repair_v2.csv`, SHA-256
`679178cd245854296e9b51611c676f3792db453ba3ca650ec4bbbb49e7630a6f`.
The CSV, journals, original results, quota state and PDFs are preserved.
The audit uses the original PDF versions 1706.03762v7, 2410.00037v1 and
2507.07955v2. No inference requests are made by this workflow.

The authored reviews are in `audits/formal18_repair_v2_review.json`. All 39 factual
answers were read and checked against both the PDFs and the exact retrieved
citation excerpts. The 15 U-question refusals were also checked against the
papers. This is an independent model-assisted review, not a blinded human study.
Binary accuracy requires a complete correct requested explanation without a
material error or unestablished extension; partial answers receive 0. Citation
support is judged separately. Valid paraphrases are accepted; unsupported claims
are distinguished from direct contradictions in the per-answer notes.

| Architecture | Correct factual answers | Citation support | Both | Correct refusals |
|---|---:|---:|---:|---:|
| Fast | 9/13 | 9/13 | 9/13 | 5/5 |
| Verified | 9/13 | 8/13 | 7/13 | 5/5 |
| Adaptive | 11/13 | 8/13 | 8/13 | 5/5 |

These full-snapshot numbers are descriptive only: 43 rows retain the original
implementation, while 11 failed/missing cases were executed with corrected
generation/verification. Only ten questions have all three answers from the same
original version and account. Of their six factual questions, all three
architectures have three answers passing both correctness and citation support;
all also pass the four refusal questions. There are no questions with all three
architectures represented under the corrected version. The ten-question matched
subset falls below the 15-question formal minimum and is selected by historical
availability. It is not a replacement for the full fair comparison.

Do not finalize a winner. Adaptive leads strict factual accuracy in the mixed
snapshot, while Fast has the most correct-and-grounded answers and lower observed
resource use. Verified's extra calls did not prevent the observed errors or ensure
completeness. These observations nominate hypotheses for a frozen, full matched
experiment, not an unbiased architecture selection. The selection guard keeps
`winner=null` and does not relabel the repair campaign as eligible.

Important findings include Adaptive M6 reversing ASR/TTS; Verified H6 incorrectly
claiming only two-stage H-Net ever beats the tokenized baseline; Fast X1 asserting
fixed input lengths/token granularity and variable per-example hierarchy depth;
all M4 answers adding incorrect or unsupported latency explanations; H2 missing
the actual routing-mechanism source; and Verified X1 citing only H-Net even though
the question requires both papers. H4 Verified's revised answer is grounded but
omits important requested roles and residual detail.

The ten account2 rows contain 20 accepted calls, 47,462 input and 2,517 output
tokens over 415.974 seconds. The recorded admissions fit the implemented
6,400-token capacity/refill bucket, with nonnegative counters and no 429s. But
the reconstructed 60-second gross-token peak is 10,737, above the 8,000 TPM
header: a continuously refilling bucket is not a strict rolling-window limiter.
Cache-exempt tokens and exact server window semantics are not recorded. Groq's
[rate-limit documentation](https://console.groq.com/docs/rate-limits) distinguishes
organization-level limits, TPM and RPD headers, and cache exemptions. The exports
cannot independently certify strict minute or full-day token compliance. Before
claiming that guarantee, retain cache-aware usage and exact dispatch times, confirm
the account-specific limits, and use a conservative rolling reservation ledger.

H4 Verified's four metered calls recover exactly 6,065 input plus 872 output tokens
(6,937 known tokens). Its fifth call was HTTP 400 with no reported usage in the
CSV, journal or trace. The full actual total cannot be recovered offline. Adding
the failed call's 1,744-token reservation gives 8,681 as an explicitly labeled
estimate, not measured usage or a proven upper bound. Raw totals remain blank.

Reproduce this audit offline:

```bash
python -m relay.result_audit \
  --input results/architecture_formal18_repair_v2.csv \
  --review audits/formal18_repair_v2_review.json \
  --logs logs/relay.jsonl --quota-state data/groq_quota_state.account2.json \
  --output results/formal18_repair_v2_audit
python -m pytest -q
```

The source hash prevents applying these authored grades to changed answers.
The output includes `factual_answer_grades.csv` (39 detailed reviews),
`graded_results.csv` (all 54 rows with derived annotations), `grades.csv`,
the reference checklist and rubric, `cross_paper_coverage.csv`,
`account2_row_audit.csv`, `account2_call_audit.csv`, `quota_audit.json`,
`h4_usage_recovery.json`, descriptive/matched comparisons, `recommendation.json`,
and a detailed `REPORT.md`. `integrity.json` verifies preserved input hashes.
The pipeline's answer behavior and historical experiment manifests are unchanged.
