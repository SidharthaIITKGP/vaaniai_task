from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .chunking import PageText, chunk_pages
from .config import RelayConfig
from .schemas import Chunk


@dataclass(frozen=True, slots=True)
class PaperSpec:
    title: str
    arxiv_id: str
    version: str
    title_markers: tuple[str, ...]


MANIFEST = (
    PaperSpec("Attention Is All You Need", "1706.03762", "v7", ("attention is all you need",)),
    PaperSpec("Moshi: a speech-text foundation model for real-time dialogue", "2410.00037", "v1", ("moshi", "real-time dialogue")),
    PaperSpec("Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "2507.07955", "v2", ("dynamic chunking", "hierarchical sequence modeling")),
)


@dataclass(slots=True)
class IngestionReport:
    chunks: list[Chunk]
    messages: list[str]
    verified: dict[str, bool]


def find_paper_dir(root: Path) -> Path | None:
    for name in ("research paper", "Research Papers", "research papers", "Research Paper"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _remove_repeated_marginal_lines(raw_pages: list[str]) -> list[str]:
    candidates: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for text in raw_pages:
        lines = [line.strip() for line in text.splitlines()]
        page_lines.append(lines)
        edge = {x for x in lines[:3] + lines[-3:] if 2 < len(x) < 160}
        candidates.update(edge)
    threshold = max(3, int(len(raw_pages) * 0.35))
    repeated = {line for line, count in candidates.items() if count >= threshold}
    cleaned = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repeated and not re.fullmatch(r"\s*\d+\s*", line)]
        cleaned.append("\n".join(kept))
    return cleaned


def _match_pdf(files: list[Path], spec: PaperSpec) -> Path | None:
    id_matches = [p for p in files if spec.arxiv_id in p.name]
    return id_matches[0] if len(id_matches) == 1 else None


def ingest(config: RelayConfig) -> IngestionReport:
    directory = find_paper_dir(config.root)
    if directory is None:
        return IngestionReport([], [
            "No paper directory found. Create 'research paper/' at the project root and add the three required PDFs."
        ], {})
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for indexing. Run: pip install -r requirements.txt") from exc
    files = sorted(directory.glob("*.pdf"))
    messages: list[str] = []
    verified: dict[str, bool] = {}
    all_chunks: list[Chunk] = []
    for spec in MANIFEST:
        path = _match_pdf(files, spec)
        if path is None:
            messages.append(f"Missing unambiguous PDF for arXiv {spec.arxiv_id}{spec.version} ({spec.title}).")
            verified[spec.arxiv_id] = False
            continue
        with pymupdf.open(path) as doc:
            raw_pages = [page.get_text("text", sort=True) for page in doc]
            metadata = doc.metadata or {}
        sample = " ".join(raw_pages[:2]).lower()
        filename_l = path.name.lower()
        title_ok = any(marker in sample or marker in str(metadata.get("title", "")).lower()
                       for marker in spec.title_markers)
        version_ok = f"{spec.arxiv_id}{spec.version}" in filename_l or bool(
            re.search(rf"arxiv:\s*{re.escape(spec.arxiv_id)}{re.escape(spec.version)}\b", sample)
        )
        verified[spec.arxiv_id] = title_ok and version_ok
        if not title_ok:
            messages.append(f"Identity warning for {path.name}: expected title was not verified in metadata/text.")
        if not version_ok:
            messages.append(f"Version warning for {path.name}: required {spec.version} could not be verified reliably.")
        cleaned = _remove_repeated_marginal_lines(raw_pages)
        pages = [PageText(i + 1, text) for i, text in enumerate(cleaned)]
        all_chunks.extend(chunk_pages(
            pages, paper_id=spec.arxiv_id, paper_title=spec.title,
            arxiv_version=spec.version if version_ok else "unverified",
            filename=path.name, chunk_tokens=config.chunk_tokens,
            overlap=config.chunk_overlap,
        ))
    if all_chunks:
        config.index_path.parent.mkdir(parents=True, exist_ok=True)
        with config.index_path.open("w", encoding="utf-8") as handle:
            for chunk in all_chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        messages.append(f"Wrote {len(all_chunks)} chunks to {config.index_path}.")
    return IngestionReport(all_chunks, messages, verified)


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run: python -m relay.cli index")
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]
