from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .schemas import Chunk


TOKEN_RE = re.compile(r"\S+")
SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|introduction|background|related work|"
    r"method(?:ology)?|model|experiments?|results?|discussion|conclusion|appendix|references)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad\n", "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _section_at(lines: list[str], position: int, prior: str | None) -> str | None:
    for line in lines[:position]:
        candidate = line.strip()
        if len(candidate) <= 100 and SECTION_RE.match(candidate):
            prior = candidate
    return prior


def chunk_pages(
    pages: list[PageText], *, paper_id: str, paper_title: str,
    arxiv_version: str, filename: str, chunk_tokens: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    if chunk_tokens <= 0 or overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("chunk_tokens must be positive and 0 <= overlap < chunk_tokens")
    chunks: list[Chunk] = []
    current_section: str | None = None
    step = chunk_tokens - overlap
    for page in pages:
        text = normalize_text(page.text)
        lines = text.splitlines()
        tokens = list(TOKEN_RE.finditer(text))
        for start in range(0, len(tokens), step):
            window = tokens[start:start + chunk_tokens]
            if not window:
                break
            begin, end = window[0].start(), window[-1].end()
            passage = text[begin:end].strip()
            line_pos = text[:begin].count("\n") + 1
            current_section = _section_at(lines, line_pos, current_section)
            identity = f"{paper_id}|{arxiv_version}|{page.page_number}|{start}|{passage}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=f"{paper_id}-p{page.page_number:03d}-{digest}",
                paper_id=paper_id, paper_title=paper_title,
                arxiv_version=arxiv_version, filename=filename,
                page_number=page.page_number, section=current_section,
                text=passage, token_start=start, token_end=start + len(window),
            ))
            if start + chunk_tokens >= len(tokens):
                break
    return chunks

