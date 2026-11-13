from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from .config import RelayConfig
from .ingestion import ingest, load_chunks
from .pipeline import RAGPipeline
from .providers.groq_provider import GroqProvider
from .retrieval import HybridRetriever


def index_command(config: RelayConfig) -> int:
    report = ingest(config)
    for message in report.messages:
        print(message)
    for paper_id, verified in report.verified.items():
        print(f"{paper_id}: {'identity/version verified' if verified else 'not fully verified'}")
    return 0 if report.chunks else 1


async def ask_command(args: argparse.Namespace, config: RelayConfig) -> int:
    if args.deployment:
        deployment = json.loads(Path(args.deployment).read_text())
        args.architecture = deployment['architecture']
        config.index_override = Path(deployment['index'])
        config.retrieval.top_k = deployment['top_k']
    config.intent_router.mode = args.intent_router
    config.provider.scheduler.request_deadline_seconds = args.queue_deadline
    config.total_request_deadline_seconds = args.total_deadline
    provider = GroqProvider(config.provider, env_path=config.root / ".env")
    def build_retriever():
        return HybridRetriever(load_chunks(config.index_path), config.retrieval)
    result = await RAGPipeline(args.architecture, build_retriever, provider, config).run(args.question)
    metrics = result.metrics
    print(f"Status: {result.status}")
    print(f"Architecture: {result.architecture}")
    print(f"Detected intent: {metrics.get('detected_intent')}")
    print(f"Routing: {metrics.get('routing_method')} ({metrics.get('routing_mode')})")
    print(f"Router model calls: {metrics.get('routing_model_calls')}")
    print(f"Architecture model calls: {metrics.get('architecture_model_calls')}")
    if result.route:
        print(f"Adaptive route: {result.route} ({result.route_reason})")
    print(f"Total model calls: {metrics.get('total_model_calls')}")
    print(f"Routing latency: {metrics.get('routing_latency_ms')} ms")
    print(f"Total latency: {metrics.get('total_latency_ms')} ms")
    print(f"Provider TTFT: {metrics.get('provider_ttft_ms')} ms")
    print(f"User-visible TTFT: {metrics.get('user_visible_ttft_ms')} ms")
    print(f"Retrieval: {metrics.get('retrieval_ms')} ms")
    print(f"Tokens (input/output): {metrics.get('total_input_tokens')}/{metrics.get('total_output_tokens')}")
    if result.verification:
        print(f"Verification: {result.verification.decision} — {result.verification.reason}")
    if result.error:
        print(f"Error: {result.error}")
    print("\nAnswer:\n" + (result.answer or "(no answer released)"))
    print("\nValidated citations: " + (", ".join(result.citations) or "none"))
    return 0 if result.status in {"ok", "insufficient_evidence", "conversation", "out_of_scope"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The Relay: three traceable RAG architectures")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="extract and index the three research PDFs")
    ask = sub.add_parser("ask", help="ask a question")
    selection = ask.add_mutually_exclusive_group(required=True)
    selection.add_argument("--architecture", choices=("fast", "verified", "adaptive"))
    selection.add_argument('--deployment', help='Measured configuration from python -m relay.finalize')
    ask.add_argument("--intent-router", choices=("rules", "llm", "hybrid"), default="hybrid")
    ask.add_argument("--question", required=True)
    ask.add_argument("--queue-deadline", type=float, default=3.0)
    ask.add_argument("--total-deadline", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RelayConfig()
    code = index_command(config) if args.command == "index" else asyncio.run(ask_command(args, config))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

