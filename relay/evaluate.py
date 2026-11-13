from __future__ import annotations

import argparse
import asyncio
import csv
import json
import hashlib
import random
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RelayConfig
from .evaluation_questions import QUESTIONS, EvaluationQuestion
from .ingestion import load_chunks
from .pipeline import RAGPipeline
from .intent_router import IntentRouter
from .providers.groq_provider import GroqProvider
from .providers.base import ProviderError
from .retrieval import HybridRetriever
from .schemas import PipelineResult, RetrievedPassage


ARCHITECTURES = ("fast", "verified", "adaptive")
DEFAULT_OUTPUT = "results/architecture_evaluation_fixed.csv"
SMOKE_6 = {"T1", "M2", "H2", "X1", "U1", "R1"}
FORMAL_15 = {"T1", "T3", "T5", "T6", "M1", "M3", "M4", "M6",
             "H1", "H2", "H4", "H6", "X1", "U1", "R3"}
FORMAL_18 = (FORMAL_15 - {"R3"}) | {"U2", "U3", "U4", "U5"}


class BudgetedGroqProvider(GroqProvider):
    def __init__(self, *args, max_calls: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_calls = max_calls
        self.attempted_calls = 0

    async def _reserve(self, payload):
        if self.attempted_calls >= self.max_calls:
            raise ProviderError('Explicit experiment model-call budget reached')
        self.attempted_calls += 1
        return await super()._reserve(payload)

FIELDS = (
    "configuration",
    "run_started_at", "run_completed_at", "execution_order", "random_seed",
    "evaluation_mode", "worker_count",
    "question_id", "category", "source_papers",
    "difficulty", "question", "expected_behavior", "architecture", "intent_router",
    "request_id", "status", "detected_intent", "routing_method", "routing_reason",
    "routing_complexity", "routing_relevant_papers_json", "routing_error",
    "routing_schema_error", "routing_fallback_used",
    "adaptive_route", "adaptive_route_reason", "answer", "validated_citations_json",
    "cited_sources_json", "retrieved_chunk_ids_json", "retrieval_trace_json",
    "citation_validation_json", "verification_json", "llm_calls_json", "provider_error_json",
    "citation_alias_mapping_json", "citation_error_reason", "verification_executed",
    "verification_supported", "pipeline_failure_stage", "model_calls_by_stage_json",
    "queue_wait_ms", "provider_status_code", "rate_limit_type", "retry_after_seconds",
    "routing_latency_ms", "routing_tokens_in", "routing_tokens_out", "routing_model_calls",
    "architecture_model_calls", "total_model_calls", "query_processing_ms", "retrieval_ms",
    "evidence_decision_ms", "planning_ms", "answer_generation_ms", "verification_ms",
    "revision_ms", "provider_ttft_ms", "user_visible_ttft_ms", "total_ms",
    "total_input_tokens", "total_output_tokens", "error",
)


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RecordingRetriever:
    """Transparent wrapper that captures every retrieval expansion for CSV provenance."""

    def __init__(self, delegate: HybridRetriever,
                 initial_cache: dict[str, list[RetrievedPassage]] | None = None):
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []
        self.initial_cache = initial_cache or {}

    def reset(self) -> None:
        self.calls.clear()

    def retrieve(self, query: str, top_k: int | None = None,
                 paper_ids: set[str] | None = None) -> list[RetrievedPassage]:
        use_cache = query in self.initial_cache and not paper_ids
        passages = (deepcopy(self.initial_cache[query]) if use_cache
                    else self.delegate.retrieve(query, top_k, paper_ids=paper_ids))
        self.calls.append({
            "query": query,
            "paper_filter": sorted(paper_ids) if paper_ids else [],
            "passages": [passage_record(item) for item in passages],
        })
        return passages


def passage_record(item: RetrievedPassage) -> dict[str, Any]:
    chunk = item.chunk
    return {
        "chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
        "paper_title": chunk.paper_title, "arxiv_version": chunk.arxiv_version,
        "filename": chunk.filename, "pdf_page": chunk.page_number,
        "section": chunk.section, "passage": chunk.text, "rank": item.rank,
        "rrf_score": item.rrf_score, "bm25_score": item.bm25_score,
        "dense_score": item.dense_score, "bm25_rank": item.bm25_rank,
        "dense_rank": item.dense_rank,
    }


def build_run_plan(question_ids: set[str] | None,
                   architectures: tuple[str, ...], seed: int = 42) -> list[tuple[EvaluationQuestion, str]]:
    selected = [q for q in QUESTIONS if not question_ids or q.question_id in question_ids]
    plan = [(question, architecture) for question in selected for architecture in architectures]
    random.Random(seed).shuffle(plan)
    return plan


def completed_keys(path: Path, router_mode: str) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("question_id", ""), row.get("architecture", ""), row.get("intent_router", ""))
            for row in csv.DictReader(handle)
            if row.get("status") not in {
                "runner_error", "provider_error", "rate_limited",
                "capacity_unavailable", "capacity_timeout",
            }
        }


def cited_sources(result: PipelineResult, chunk_by_id: dict[str, Any],
                  retrieval_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_by_id: dict[str, dict[str, Any]] = {}
    for call in retrieval_calls:
        for passage in call["passages"]:
            score_by_id.setdefault(passage["chunk_id"], passage)
    sources = []
    for chunk_id in result.citations:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            sources.append({"chunk_id": chunk_id, "metadata_error": "citation missing from index"})
            continue
        record = {
            "chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
            "paper_title": chunk.paper_title, "arxiv_version": chunk.arxiv_version,
            "filename": chunk.filename, "pdf_page": chunk.page_number,
            "section": chunk.section, "passage": chunk.text,
        }
        if chunk_id in score_by_id:
            record["retrieval_scores"] = {
                key: score_by_id[chunk_id][key]
                for key in ("rank", "rrf_score", "bm25_score", "dense_score", "bm25_rank", "dense_rank")
            }
        sources.append(record)
    return sources


def result_row(question: EvaluationQuestion, architecture: str, router_mode: str,
               started_at: str, result: PipelineResult, recorder: RecordingRetriever,
               chunk_by_id: dict[str, Any], execution_order: int = 0,
               random_seed: int = 42) -> dict[str, Any]:
    metrics = result.metrics
    verification = asdict(result.verification) if result.verification else metrics.get("verification_outcome")
    row = {field: "" for field in FIELDS}
    row.update({
        "run_started_at": started_at,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "execution_order": execution_order, "random_seed": random_seed,
        "question_id": question.question_id, "category": question.category,
        "source_papers": question.source_papers, "difficulty": question.difficulty,
        "question": question.question, "expected_behavior": question.expected_behavior,
        "architecture": architecture, "intent_router": router_mode,
        "request_id": result.request_id, "status": result.status,
        "detected_intent": metrics.get("detected_intent"),
        "routing_method": metrics.get("routing_method"),
        "routing_reason": metrics.get("routing_reason"),
        "routing_complexity": metrics.get("routing_complexity"),
        "routing_relevant_papers_json": json_cell(metrics.get("routing_relevant_papers", [])),
        "routing_error": metrics.get("routing_error"),
        "routing_schema_error": metrics.get("routing_schema_error"),
        "routing_fallback_used": metrics.get("routing_fallback_used"),
        "adaptive_route": result.route, "adaptive_route_reason": result.route_reason,
        "answer": result.answer,
        "validated_citations_json": json_cell(result.citations),
        "cited_sources_json": json_cell(cited_sources(result, chunk_by_id, recorder.calls)),
        "retrieved_chunk_ids_json": json_cell(result.retrieved_chunk_ids),
        "retrieval_trace_json": json_cell(recorder.calls),
        "citation_validation_json": json_cell(metrics.get("citation_validation")),
        "verification_json": json_cell(verification),
        "llm_calls_json": json_cell(metrics.get("llm_calls", [])),
        "provider_error_json": json_cell(metrics.get("provider_error")),
        "citation_alias_mapping_json": json_cell(metrics.get("citation_alias_mapping", {})),
        "citation_error_reason": metrics.get("citation_error_reason"),
        "verification_executed": metrics.get("verification_executed"),
        "verification_supported": metrics.get("verification_supported"),
        "pipeline_failure_stage": metrics.get("pipeline_failure_stage"),
        "model_calls_by_stage_json": json_cell(metrics.get("model_calls_by_stage", {})),
        "error": result.error,
    })
    for field in (
        "routing_latency_ms", "routing_tokens_in", "routing_tokens_out", "routing_model_calls",
        "architecture_model_calls", "total_model_calls", "query_processing_ms", "retrieval_ms",
        "evidence_decision_ms", "planning_ms", "answer_generation_ms", "verification_ms",
        "revision_ms", "provider_ttft_ms", "user_visible_ttft_ms", "total_ms",
        "total_input_tokens", "total_output_tokens",
        "queue_wait_ms", "provider_status_code", "rate_limit_type", "retry_after_seconds",
    ):
        row[field] = metrics.get(field)
    return row


class CachedEvaluationRouter:
    def __init__(self, decisions: dict[str, Any]):
        self.decisions = decisions

    async def route(self, question: str, **kwargs: Any):
        decision = deepcopy(self.decisions[question])
        decision.provider_response = None
        decision.model_calls = 0
        decision.latency_ms = 0.0
        decision.reason = f"cached for architecture-only evaluation; {decision.reason}"
        return decision


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


async def run(args: argparse.Namespace) -> int:
    architectures = tuple(args.architectures)
    requested = set(args.question_id) if args.question_id else None
    if requested is None and args.subset == "smoke6":
        requested = SMOKE_6
    elif requested is None and args.subset == "formal15":
        requested = FORMAL_15
    elif requested is None and args.subset == "formal18":
        requested = FORMAL_18
    known_ids = {q.question_id for q in QUESTIONS}
    unknown = (requested or set()) - known_ids
    if unknown:
        raise SystemExit(f"Unknown question ID(s): {', '.join(sorted(unknown))}")
    plan = build_run_plan(requested, architectures, args.seed)
    if args.max_runs is not None:
        plan = plan[:args.max_runs]
    output = Path(args.output).expanduser().resolve()
    if args.dry_run:
        print(f"Dry run only: {len(plan)} runs, {len({q.question_id for q, _ in plan})} questions")
        print(f"Architectures: {', '.join(architectures)}")
        print(f"Intent router: {args.intent_router}")
        print(f"Fixed random seed: {args.seed}")
        print(f"Mode/workers: {args.mode}/{args.workers}")
        print(f"Output: {output}")
        for question, architecture in plan:
            print(f"{question.question_id:>2}  {architecture:<8}  {question.question}")
        return 0

    config = RelayConfig()
    if args.index:
        config.index_override = Path(args.index)
    config.retrieval.top_k = args.top_k
    config.intent_router.mode = args.intent_router
    config.provider.scheduler.rpm_limit = args.rpm
    config.provider.scheduler.tpm_limit = args.tpm
    config.provider.scheduler.safety_margin = args.safety_margin
    config.provider.scheduler.request_deadline_seconds = args.request_deadline
    config.provider.scheduler.max_queue_size = args.max_queue_size
    config.verification_evidence_mode = args.verification_evidence
    manifest = {
        'configuration': args.configuration,
        'architectures': architectures, 'questions': sorted(q.question_id for q, _ in plan),
        'model': config.provider.model, 'top_k': args.top_k,
        'index_sha256': hashlib.sha256(config.index_path.read_bytes()).hexdigest(),
        'routing': args.intent_router, 'verification_evidence': args.verification_evidence,
        'revision': config.enable_revision, 'seed': args.seed, 'mode': args.mode,
        'workers': args.workers, 'rpm': args.rpm, 'tpm': args.tpm,
        'safety_margin': args.safety_margin,
        'source_sha256': hashlib.sha256(b''.join(
            p.read_bytes() for p in sorted(Path(__file__).parent.rglob('*.py')))).hexdigest(),
    }
    manifest_path = output.with_suffix('.manifest.json')
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    if output.exists() and output.stat().st_size and not args.overwrite:
        if not manifest_path.exists() or manifest_path.read_text() != manifest_text:
            raise SystemExit('Resume configuration/code differs or manifest is missing. Use a new output filename.')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text)
    chunks = load_chunks(config.index_path)
    base_retriever = HybridRetriever(chunks, config.retrieval)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    provider = BudgetedGroqProvider(config.provider, env_path=config.root / ".env",
                                    max_calls=args.max_model_calls)
    done = set() if args.overwrite else completed_keys(output, args.intent_router)
    if args.overwrite:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
    pending = [(order, q, a) for order, (q, a) in enumerate(plan, 1)
               if (q.question_id, a, args.intent_router) not in done]
    initial_cache: dict[str, list[RetrievedPassage]] = {}
    cached_router = None
    if args.mode == "architecture-only":
        questions = {question.question: question for _, question, _ in pending}
        initial_cache = {text: base_retriever.retrieve(text) for text in questions}
        router = IntentRouter(provider, config.intent_router)
        decisions = {text: await router.route(text) for text in questions}
        cached_router = CachedEvaluationRouter(decisions)
    print(f"Running {len(pending)} of {len(plan)} planned requests with {args.workers} worker(s). Output: {output}")
    output_lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[int, EvaluationQuestion, str]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)

    async def run_one(execution_order: int, question: EvaluationQuestion, architecture: str):
        recorder = RecordingRetriever(base_retriever, initial_cache)
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"[{execution_order}/{len(plan)}] {question.question_id} / {architecture}", flush=True)
        try:
            pipeline = RAGPipeline(architecture, recorder, provider, config, router=cached_router)
            result = await pipeline.run(question.question)
            row = result_row(question, architecture, args.intent_router, started_at,
                             result, recorder, chunk_by_id, execution_order, args.seed)
        except Exception as exc:  # Preserve progress and make unexpected runner failures inspectable.
            row = {field: "" for field in FIELDS}
            row.update({
                "run_started_at": started_at,
                "run_completed_at": datetime.now(timezone.utc).isoformat(),
                "execution_order": execution_order, "random_seed": args.seed,
                "question_id": question.question_id, "category": question.category,
                "source_papers": question.source_papers, "difficulty": question.difficulty,
                "question": question.question, "expected_behavior": question.expected_behavior,
                "architecture": architecture, "intent_router": args.intent_router,
                "status": "runner_error", "error": f"{type(exc).__name__}: {exc}",
                "retrieval_trace_json": json_cell(recorder.calls),
            })
        row["evaluation_mode"] = args.mode
        row['configuration'] = args.configuration or architecture
        row["worker_count"] = args.workers
        async with output_lock:
            append_row(output, row)
        if args.delay > 0:
            await asyncio.sleep(args.delay)

    async def worker():
        while True:
            if provider.attempted_calls >= provider.max_calls:
                return
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await run_one(*item)
            finally:
                queue.task_done()

    try:
        async with asyncio.timeout(args.wall_seconds):
            await asyncio.gather(*(worker() for _ in range(args.workers)))
    except TimeoutError:
        print('Experiment wall deadline reached; completed rows saved. Resume with the same command.')
    print(f"Finished. CSV: {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the fixed 30-question architecture matrix")
    result.add_argument("--architectures", nargs="+", choices=ARCHITECTURES,
                        default=list(ARCHITECTURES))
    result.add_argument("--intent-router", choices=("rules", "llm", "hybrid"), default="hybrid")
    result.add_argument("--question-id", action="append", help="Run one ID; repeat for multiple IDs")
    result.add_argument("--subset", choices=("smoke6", "formal15", "formal18", "full30"), default="full30")
    result.add_argument("--mode", choices=("end-to-end", "architecture-only"), default="end-to-end")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--max-queue-size", type=int, default=8)
    result.add_argument("--output", default=DEFAULT_OUTPUT)
    result.add_argument("--index", help="Explicit variant index; never rebuilds the original")
    result.add_argument("--top-k", type=int, default=5)
    result.add_argument('--configuration', help='Retrieval candidate label for matched comparisons')
    result.add_argument("--delay", type=float, default=0.0,
                        help="Seconds between requests; no automatic retries are performed")
    result.add_argument("--seed", type=int, default=42, help="Fixed execution-order seed")
    result.add_argument("--max-runs", type=int, help="Maximum pipeline runs for an opt-in smoke test")
    result.add_argument('--max-model-calls', type=int, default=200)
    result.add_argument('--wall-seconds', type=float, default=3600)
    result.add_argument("--rpm", type=int, default=30)
    result.add_argument("--tpm", type=int, default=8000)
    result.add_argument("--safety-margin", type=float, default=0.80)
    result.add_argument("--request-deadline", type=float, default=240.0)
    result.add_argument("--verification-evidence", choices=("full", "cited"), default="cited")
    result.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace the selected output instead of resuming it")
    result.add_argument("--dry-run", action="store_true", help="Print the plan; make no API calls")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")
    if args.rpm <= 0 or args.tpm <= 0 or not 0 < args.safety_margin <= 1:
        raise SystemExit("RPM/TPM must be positive and safety margin must be in (0, 1]")
    if args.max_runs is not None and args.max_runs <= 0:
        raise SystemExit("--max-runs must be positive")
    if args.max_model_calls <= 0 or args.wall_seconds <= 0 or args.top_k <= 0:
        raise SystemExit('Model-call budget, wall deadline and top-K must be positive')
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_queue_size <= 0 or args.workers > args.max_queue_size:
        raise SystemExit("--max-queue-size must be positive and at least --workers")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
