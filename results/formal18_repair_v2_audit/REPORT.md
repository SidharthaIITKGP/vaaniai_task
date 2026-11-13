# Formal18 repaired-run independent audit

All 54 final pipeline runs are present: 39 factual answers and 15 refusals. The raw results are unchanged; no live model calls were made. Grades are an explicitly authored model-assisted PDF review, not blinded human judgments and not pipeline/verifier scores. See `rubric.json`, `grade_details.json` and `grades.csv`.

## Full repaired snapshot: descriptive comparison only

| Architecture | Factual accuracy | Citation support | Correct + grounded | Correct refusals |
|---|---:|---:|---:|---:|
| fast | 9/13 | 9/13 | 9/13 | 5/5 |
| verified | 9/13 | 8/13 | 7/13 | 5/5 |
| adaptive | 11/13 | 8/13 | 8/13 | 5/5 |

Accuracy means a fully correct and complete requested explanation, with no material false or unestablished extension. Partial answers receive 0 under the binary rubric. Citation support is checked against the exact retrieved excerpts; being true elsewhere in the PDF does not repair a weak citation. Refusals are shown separately to avoid inflating factual accuracy.

## Fairness and recommendation

Selective repair mixes 43 original-version successes with 11 corrected runs. Only ten same-version/same-account questions cover all three candidates; corrected-version common coverage is zero.

Matched question IDs: H1, H2, M4, M6, T3, T6, U2, U3, U4, U5.

| Architecture | Factual accuracy | Citation support | Correct + grounded | Correct refusals |
|---|---:|---:|---:|---:|
| fast | 3/6 | 3/6 | 3/6 | 4/4 |
| verified | 4/6 | 3/6 | 3/6 | 4/4 |
| adaptive | 4/6 | 3/6 | 3/6 | 4/4 |

All candidates tie on fully correct-and-grounded answers in this limited matched subset. It was selected by provenance availability after repairs and is not a representative replacement for the planned experiment. No winning architecture is established. Adaptive has the highest descriptive factual accuracy in the mixed snapshot; Fast has the most factual answers passing both correctness and citation support. Neither observation removes the version/repair confound. The existing selection guard remains incomplete with winner=null.

## Errors, citation weaknesses and omissions

- **M4 / verified** (accuracy 0, citations 0): 160/200 ms is right, but the claimed latency cause is a tuned delay between text and audio. PDF pp.14,26 instead distinguish the 1-2-frame acoustic delay between semantic and acoustic tokens; the 160 ms design uses Mimi's 80 ms frame plus one acoustic frame. It also conflates Inner Monologue with multi-stream turn handling. The selected excerpts do not support that causal account. Reference: Moshi PDF pp.1,14,24,26,29.
- **H4 / verified** (accuracy 0, citations 1): Retained claims about components, recursive main network and compression-oriented Mamba-2 are grounded in cited pp.4,19. But the corrected answer omits the main network's role on compressed sequences, encoder residual preservation and decoder coarse/fine combination from p.5. It substitutes general hierarchy description for important requested component roles. A verifier return verdict did not check answer completeness. Reference: H-Net PDF pp.4-5,19.
- **H2 / verified** (accuracy 1, citations 0): Correct inverse relationship between adjacent similarity and boundary likelihood. However S3 p.17 and S4 p.2 mention cosine routing only at a high level; they omit the inverse mapping and downsampler details. Cite the actual p.6 equation 4 passage to ground the mechanism. Reference: H-Net PDF p.6, equation 4, section 2.2.1; contrast cited pp.2,17.
- **M6 / adaptive** (accuracy 0, citations 0): Reverses the task labels and conditioning: audio ahead/text delayed produces ASR, not TTS; text ahead/audio delayed produces TTS, not ASR. Its two cases also both describe text delayed relative to audio. Cited pp.16,31 explicitly give the opposite directions. Reference: Moshi PDF pp.16,31, sections 3.4.4 and 5.7.
- **H2 / adaptive** (accuracy 1, citations 0): Correctly says contextual similarity drops imply higher boundary probability. The simplified raw-vector notation omits learned q/k projections; not material to this conceptual answer. Its cited pp.2,17 do not contain the probability mapping, so p.6 equation 4 is needed for full citation support. Reference: H-Net PDF p.6, equation 4, section 2.2.1; contrast cited pp.2,17.
- **M4 / adaptive** (accuracy 0, citations 0): Latency figures and general speech-to-speech/parallel-stream ideas are right. The additional claim that Inner Monologue reduces cross-modal decoding work is unsupported and conflicts with p.29: it marginally increases inference work, generating 17 rather than 16 tokens per multi-stream step. The architecture bullets have no attached supporting citation; the sole S3 citation is placed on the latency figures. Reference: Moshi PDF pp.1,14,24,26,29.
- **X1 / fast** (accuracy 0, citations 0): The static-tokenizer versus learned-chunking contrast is correct. But Transformer inputs are not fixed-length sequences with the same boundaries/equal token granularity across examples: BPE/word-piece units and sequence lengths vary. H-Net's stage count is fixed for a configured model, not varying per-example hierarchy depth as claimed. The cited pages do not support these extensions. Reference: Transformer PDF pp.3,5,7; H-Net PDF pp.4-6.
- **H1 / fast** (accuracy 0, citations 0): Fixed external segmentation and tokenization drawbacks are correct. The added contrast between predicting then detokenizing and learning a single continuous representation is not established by the paper's discrete byte-model hierarchy. The appendix's online-boundary impossibility does not establish the broader claimed learning impairment. These material additions are not supported by S3/S2. Reference: H-Net PDF pp.1,39.
- **H1 / verified** (accuracy 0, citations 0): Correctly identifies fixed preprocessing and future-byte information in BPE boundary reconstruction. It then asserts that this bias causes the complex-language/modality degradation and that fixed vocabulary prevents better scaling. PDF p.39 discusses online boundary recovery; p.1 lists drawbacks and says learned models would ideally scale better. Neither establishes the added causal chain or absolute scaling claim. Reference: H-Net PDF pp.1,39.
- **M4 / fast** (accuracy 0, citations 0): 160/200 ms is correct, but the PDF does not promise that practical latency on a typical GPU or CPU. It conflates acoustic semantic-to-acoustic delay with ASR/TTS relative text/audio delay, and attributes listening/speaking to Inner Monologue rather than multi-stream modeling. Cited pp.1,26 do not establish these extensions. Reference: Moshi PDF pp.1,14,24,26,29.
- **H6 / adaptive** (accuracy 1, citations 0): Correct two-stage ordering, all H-Net variants eventually crossing, and 30B versus 100B. The claim of several percentage points over one-stage is imprecise: Table 2 p.13 gives mean gaps 1.9 and 2.0 points. Attached S5 p.12 gives 2.2/2.6 points versus Transformer, not one-stage. The core requested conclusion is correct, but that extra quantitative comparison lacks its cited support. Reference: H-Net PDF pp.12-13, Figure 3 and Table 2.
- **H2 / fast** (accuracy 0, citations 0): Describes similarity scores and high boundary probabilities but omits the essential direction linking them: low adjacent projected cosine similarity gives high p_t. This is the core of the question, not just an optional equation. The cited pp.2,17 also do not establish the asserted probability/downsampler mechanics; equation 4 and selection details are on p.6. Reference: H-Net PDF p.6, equation 4, section 2.2.1; contrast cited pp.2,17.
- **X1 / verified** (accuracy 1, citations 0): The factual comparison is broadly correct against both PDFs. However its final citations contain only H-Net pp.4-5, while the Transformer paragraph has no citation at all. Retrieval included Transformer evidence but the answer did not cite it; this fails the explicit two-paper grounding requirement. Reference: Transformer PDF pp.3,5,7; H-Net PDF pp.4-6.
- **H6 / verified** (accuracy 0, citations 0): The 30B/100B/200B values are correct, but it says two-stage is the only byte-level model that ultimately beats the tokenized Transformer. Its own next paragraph and p.12 state that one-stage and space also cross. This material exclusivity claim is false and contradicts the cited source. Reference: H-Net PDF pp.12-13, Figure 3 and Table 2.
- **M3 / adaptive** (accuracy 1, citations 0): Core Inner Monologue definition and linguistic benefit are correct. The ASR/TTS delay statement is true in pp.16,31 but its attached S5 p.28 is unrelated audio-language-model benchmarking. Separate speaker streams are a multi-stream design choice, so the additional attribution to Inner Monologue is imprecise. Reference: Moshi PDF pp.1,5,15-16,26,29.

Verified X1 retrieved Transformer evidence but did not cite it: all final cited passages are H-Net. Fast and Adaptive X1 cite both required papers. Retrieval coverage and answer citation coverage are separate checks.

## Account2 quota audit

Ten rows contain 20 real model calls and 47,462 input + 2,517 output = 49,979 gross tokens over 415.974 seconds. All calls returned 200; the final RPD header is 980. The original account state was not cleared or reused for account2.

Recorded local bucket capacity is 6,400 with 106.667 tokens/second refill. All recorded reservations fit, actual per-call totals fit their reservations, and admission counters are nonnegative. The initial burst plus refill envelope is 50770.6 tokens, consistent with observed aggregate use.

**Strict minute compliance is not certified.** Reconstructed serial-call timing intervals give a peak of 5 calls and 10,737 gross tokens in 60 seconds, above the 8,000 TPM header and the local 6,400-token refill amount. A bucket permits an initial burst plus refill and therefore does not enforce a strict rolling token window. This is an audit concern, not proof of a provider violation: cached-token counts were not retained and precise server window semantics are unknown.

[Groq documents](https://console.groq.com/docs/rate-limits) organization-level limits, TPM and RPD header meanings, and cached-token exemptions. These headers do not expose remaining TPD. Consequently the full-day organization quota cannot be certified from this export. Separate profile names and user confirmation are the available evidence of the account switch; no authenticated organization identity was recorded. Before a strict-quota certification, retain cache-adjusted usage and exact dispatch timestamps, obtain account-specific limits, and add a conservative rolling reservation ledger without changing answer behavior.

## H4 Verified token recovery

Request `4923420f-acc3-4068-afe9-260964bee41b` contains five calls. Four calls have exact usage: 6,065 input and 872 output (6,937 measured tokens). The HTTP 400 formatting attempt has no usage in the CSV, journal or request trace. Exact total recovery is unavailable offline.

Its failed reservation was 1,744 tokens. Adding it gives 8,681, a reservation-based estimate, **not actual usage or a proven upper bound**. Raw totals stay blank. `usage_recovery.csv` exposes known totals, unknown calls and exact totals separately; `h4_usage_recovery.json` records matching log locations. Provider request ID `req_01m7b1305me3pvb5xj4ftb7fs2` is the unresolved call for a future provider-side usage export.

## Speed and resource comparison

| Architecture | Mean total s | Mean queue s | Mean processing s | Model calls | Exact tokens |
|---|---:|---:|---:|---:|---:|
| fast | 24.507 | 23.091 | 1.416 | 18 | 62404 |
| verified | 42.323 | 39.149 | 3.174 | 36 | unknown; >= 88169 |
| adaptive | 47.093 | 44.505 | 2.587 | 39 | 89063 |

Processing is measured total latency minus measured queue wait; it includes retrieval and checks. It is not solely provider execution time. Quantiles and provider latency sums are in the comparison CSVs. Reported tokens/calls here cover final selected attempts only, not all historical failed attempts. Provider TTFT is unavailable for buffered strict-JSON runs and is not invented. Account, version, order and warmup confound end-to-end latency across the mixed run; the matched subset is still descriptive and below the selection minimum.

Reproduce offline with:

```bash
python -m relay.result_audit --input results/architecture_formal18_repair_v2.csv \
  --review audits/formal18_repair_v2_review.json \
  --logs logs/relay.jsonl --quota-state data/groq_quota_state.account2.json \
  --output results/formal18_repair_v2_audit
```
