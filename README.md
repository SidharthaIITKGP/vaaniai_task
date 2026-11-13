# The Relay — three RAG architectures

The experiment workflow, measured diagnostics, and remaining quality gates are documented in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Current architecture and retrieval winners remain
unselected until complete matched evaluations and independent PDF-based grading are available.

This project implements the three Experiment 1 candidate architectures over exactly the required papers. It does not run benchmarks, generate evaluation data, or select a winner.

## Setup

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY only when you want to make deliberate real calls.
```

The indexer accepts the assignment's `research paper/` directory and common capitalization/plural variants, including this repository's existing `Research Papers/`. It matches PDFs by arXiv ID and checks title text/metadata and the required version. A version that is not present in the filename or extracted arXiv text is explicitly marked unverified.

## Commands

```bash
python -m relay.cli index
python -m relay.cli ask --architecture fast --question "What is multi-head attention?"
python -m relay.cli ask --architecture verified --question "How does Moshi handle real-time dialogue?"
python -m relay.cli ask --architecture adaptive --question "Compare the sequence modeling approaches discussed in the Transformer and Dynamic Chunking papers."
python -m pytest -q
```

Intent routing defaults to `hybrid`: obvious greetings, acknowledgements, and capability questions are
answered deterministically with no retrieval or model call; ambiguous/substantive inputs use a small
structured Groq classification call. Select a mode explicitly when needed:

```bash
python -m relay.cli ask --architecture fast --intent-router rules --question "Hey"
python -m relay.cli ask --architecture verified --intent-router hybrid --question "Explain Moshi's full-duplex architecture"
python -m relay.cli ask --architecture adaptive --intent-router llm --question "Compare Transformers and Dynamic Chunking"
```

The router can return `conversation`, `paper_question`, or `out_of_scope`. Only `paper_question` enters
the selected RAG strategy. Invalid classifier output or router-provider failure falls back conservatively
to paper-grounded RAG; it never enables unrestricted general-knowledge answering.

## Fixed architecture evaluation matrix

The supplied 30-question set is encoded in `relay/evaluation_questions.py`. Preview all 90
question/architecture runs without loading the embedding model or calling Groq:

```bash
python -m relay.evaluate --dry-run
```

Run the full matrix yourself. A shared provider scheduler enforces 30 RPM and 8,000 TPM with an
80% safety margin across routing, planning, generation, verification, and revision calls:

```bash
python -m relay.evaluate --intent-router hybrid \
  --output results/architecture_evaluation_fixed.csv
```

The CSV is checkpointed after every run and an existing file is resumed. Provider/rate-limit failures
remain eligible for a later manual rerun; there are no automatic retries. Useful smaller runs:

```bash
python -m relay.evaluate --question-id T1 --question-id M1
python -m relay.evaluate --architectures fast --question-id T1
python -m relay.evaluate --delay 5 --rpm 30 --tpm 8000 --safety-margin 0.8
```

Use `--overwrite` only when you intentionally want to replace the selected CSV. Each row stores the
answer, validated citations, complete cited passage metadata, every retrieval query and ranked passage,
BM25/dense/RRF scores, routing and verification results, per-call usage, latency, and errors.

Runs are interleaved in a deterministic shuffled order (`--seed 42`) to reduce time-order bias. The
provider scheduler reserves estimated prompt plus output capacity, reconciles actual usage, separates
queue wait from provider execution, consumes rate-limit headers, and honors `Retry-After` without
unbounded retries. The original `results/architecture_evaluation.csv` is never modified by default.

The TPM scheduler uses continuously refilling capacity instead of waiting for an entire fixed
60-second window. Stage defaults are 256 completion tokens for routing, 384 for planning, and 640
for generation, revision, and semantic verification. Definitive HTTP 4xx rejections refund their token
reservation; timeouts and unknown-usage failures retain conservative accounting.

Interactive CLI requests default to a 3-second admission deadline and 10-second total deadline:

```bash
python -m relay.cli ask --architecture fast --question "Explain multi-head attention" \
  --queue-deadline 3 --total-deadline 10
```

Evaluation subsets and bounded workers share one scheduler:

```bash
python -m relay.evaluate --subset smoke6 --workers 2 --output results/scheduler_smoke6.csv
python -m relay.evaluate --subset formal15 --workers 2 --output results/formal15.csv
python -m relay.evaluate --subset full30 --workers 2 --output results/architecture_evaluation_optimized.csv
```

`--mode end-to-end` measures the complete workflow. `--mode architecture-only` precomputes shared
routing and initial retrieval and labels those rows accordingly; cached timings are not uncached
end-to-end measurements. Verification defaults to only the deterministically cited passages; this
retains the evidence needed to check each claim while avoiding retransmission of uncited chunks.
Full retrieved evidence remains available for controlled comparison via `--verification-evidence full`.

Run the deterministic scheduler comparison against existing smoke data with:

```bash
python -m relay.scheduler_benchmark --csv results/smoke_fixed.csv
```

Generation uses request-local citation aliases (`[S1]`, `[S2]`, …). Validated aliases resolve back to
the exact stable chunk IDs and full PDF metadata before an answer is released.

Index data is written to `data/chunks.jsonl`; request traces are appended to `logs/relay.jsonl`. Both are ignored by Git. The existing `groq_benchmark.py` and `groq_probe.csv` are not read or modified.

## Architecture behavior

- `fast`: shared retrieval, heuristic pre-gate, one grounded generation call, deterministic citation traceability check.
- `verified`: the fast path plus independent semantic verification. A `revise` decision permits exactly one correction and re-verification; failure remains closed.
- `adaptive`: uses the verified simple path normally. Transparent complexity rules route comparisons/multi-part/long questions through one planning call and bounded retrieval expansion before generation and verification.

All strategies receive the same `HybridRetriever` instance and retrieval configuration. BM25 and local Sentence Transformers rankings are combined with reciprocal-rank fusion. Dense vectors use NumPy cosine similarity; no vector database is involved.
Questions naming two or more library papers receive per-paper retrieval coverage before fusion, so a cross-paper answer cannot be generated from one paper's passages alone.

## Deliberate limitations

- The pre-generation evidence gate only determines whether retrieval found plausibly relevant text; it is not a calibrated confidence score and does not prove answerability.
- Fast RAG checks citation IDs and source metadata, not whether the cited passage semantically entails a claim.
- Answer and revision generation are streamed internally and buffered, so provider TTFT is measured without leaking an unverified draft. Non-streamed planning/verification calls have no TTFT and are recorded as missing (`null`), never zero. User-visible TTFT includes all required checks and is the safe answer-release time.
- PDF page numbers are physical PDF pages (1-based), which are stable and traceable but may differ from a paper's printed page label.
- Section labels are conservative heading inferences and remain `null` when not reliable.
- Real integration calls are never part of the default test suite and the provider performs no automatic retries.

