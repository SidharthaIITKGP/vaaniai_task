from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(slots=True)
class ProviderConfig:
    model: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "low"
    max_output_tokens: int = 1200
    timeout_seconds: float = 60.0
    scheduler: "SchedulerConfig" = field(default_factory=lambda: SchedulerConfig())


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool = True
    rpm_limit: int = 30
    tpm_limit: int = 8000
    safety_margin: float = 0.80
    max_queue_size: int = 8
    request_deadline_seconds: float = 240.0


@dataclass(slots=True)
class IntentRouterConfig:
    mode: str = "hybrid"
    max_output_tokens: int = 384
    reasoning_effort: str = "low"


@dataclass(slots=True)
class RelayConfig:
    root: Path = ROOT
    chunk_tokens: int = 500
    chunk_overlap: int = 80
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    intent_router: IntentRouterConfig = field(default_factory=IntentRouterConfig)
    enable_revision: bool = False
    max_revisions: int = 1
    max_retrieval_expansions: int = 4

    @property
    def index_path(self) -> Path:
        return self.root / "data" / "chunks.jsonl"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "relay.jsonl"

