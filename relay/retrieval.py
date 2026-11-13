from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .config import RetrievalConfig
from .schemas import Chunk, RetrievedPassage


WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [x.lower() for x in WORD_RE.findall(text)]


class Embedder(Protocol):
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval. Run: pip install -r requirements.txt") from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class BM25Index:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.lengths = np.array([len(d) for d in documents], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(documents) else 0.0
        self.term_frequencies = [Counter(doc) for doc in documents]
        df = Counter(term for doc in documents for term in set(doc))
        n = len(documents)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        result = np.zeros(len(self.documents), dtype=float)
        if not query or not self.avgdl:
            return result
        for i, frequencies in enumerate(self.term_frequencies):
            for term in set(query):
                tf = frequencies.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                    result[i] += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
        return result


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=float).reshape(-1)
    matrix = np.asarray(matrix, dtype=float)
    qn = np.linalg.norm(query)
    dn = np.linalg.norm(matrix, axis=1)
    return (matrix @ query) / np.maximum(dn * qn, 1e-12)


class HybridRetriever:
    """The single shared retriever used by all three strategies."""

    def __init__(self, chunks: list[Chunk], config: RetrievalConfig | None = None,
                 embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("Cannot build a retriever without chunks")
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.dense_model)
        self.bm25 = BM25Index([tokenize(c.text) for c in chunks])
        self.embeddings = np.asarray(self.embedder.encode([c.text for c in chunks]), dtype=float)

    def retrieve(self, query: str, top_k: int | None = None,
                 paper_ids: set[str] | None = None) -> list[RetrievedPassage]:
        top_k = top_k or self.config.top_k
        allowed = [index for index, chunk in enumerate(self.chunks)
                   if not paper_ids or chunk.paper_id in paper_ids]
        if not allowed:
            return []
        candidate_k = min(max(top_k, self.config.candidate_k), len(allowed))
        sparse = self.bm25.scores(tokenize(query))
        query_vector = np.asarray(self.embedder.encode([query]))[0]
        dense = _cosine(query_vector, self.embeddings)
        allowed_array = np.asarray(allowed, dtype=int)
        sparse_order = allowed_array[np.argsort(-sparse[allowed_array], kind="stable")[:candidate_k]]
        dense_order = allowed_array[np.argsort(-dense[allowed_array], kind="stable")[:candidate_k]]
        sparse_rank = {int(idx): rank + 1 for rank, idx in enumerate(sparse_order)}
        dense_rank = {int(idx): rank + 1 for rank, idx in enumerate(dense_order)}
        candidates = set(sparse_rank) | set(dense_rank)
        fused = {
            idx: (1 / (self.config.rrf_k + sparse_rank[idx]) if idx in sparse_rank else 0)
            + (1 / (self.config.rrf_k + dense_rank[idx]) if idx in dense_rank else 0)
            for idx in candidates
        }
        ordered = sorted(candidates, key=lambda i: (-fused[i], self.chunks[i].chunk_id))
        selected: list[int] = []
        for idx in ordered:
            chunk = self.chunks[idx]
            duplicate = any(
                chunk.paper_id == self.chunks[prior].paper_id
                and _token_overlap(chunk.text, self.chunks[prior].text) >= 0.82
                for prior in selected
            )
            if not duplicate:
                selected.append(idx)
            if len(selected) == top_k:
                break
        return [RetrievedPassage(
            chunk=self.chunks[idx], rank=rank, rrf_score=fused[idx],
            bm25_score=float(sparse[idx]), dense_score=float(dense[idx]),
            bm25_rank=sparse_rank.get(idx), dense_rank=dense_rank.get(idx),
        ) for rank, idx in enumerate(selected, 1)]


def _token_overlap(a: str, b: str) -> float:
    left, right = set(tokenize(a)), set(tokenize(b))
    return len(left & right) / max(1, min(len(left), len(right)))


def merge_passages(groups: list[list[RetrievedPassage]], top_k: int) -> list[RetrievedPassage]:
    by_id: dict[str, RetrievedPassage] = {}
    for group in groups:
        for item in group:
            old = by_id.get(item.chunk.chunk_id)
            if old is None or item.rrf_score > old.rrf_score:
                by_id[item.chunk.chunk_id] = item
    ordered = sorted(by_id.values(), key=lambda x: (-x.rrf_score, x.chunk.chunk_id))[:top_k]
    for rank, item in enumerate(ordered, 1):
        item.rank = rank
    return ordered


PAPER_NAME_PATTERNS = {
    "1706.03762": re.compile(r"\b(?:original\s+)?transformer\b|attention is all you need", re.I),
    "2410.00037": re.compile(r"\bmoshi\b|\bmimi\b|\bhelium\b", re.I),
    "2507.07955": re.compile(r"\bh[- ]?net\b|dynamic chunking", re.I),
}


def comparison_paper_ids(question: str) -> list[str]:
    named = [paper_id for paper_id, pattern in PAPER_NAME_PATTERNS.items() if pattern.search(question)]
    # Mentioning two library papers is sufficient to require coverage from both, even when
    # the comparison is phrased indirectly (for example, "How does Moshi build on Transformers?").
    return named if len(named) >= 2 else []


def merge_passages_with_coverage(groups: list[list[RetrievedPassage]], top_k: int,
                                 required_papers: list[str]) -> list[RetrievedPassage]:
    candidates = merge_passages(groups, max(top_k, sum(len(group) for group in groups)))
    selected: list[RetrievedPassage] = []
    for paper_id in required_papers:
        match = next((item for item in candidates if item.chunk.paper_id == paper_id), None)
        if match and match.chunk.chunk_id not in {item.chunk.chunk_id for item in selected}:
            selected.append(match)
    for item in candidates:
        if len(selected) >= top_k:
            break
        if item.chunk.chunk_id not in {chosen.chunk.chunk_id for chosen in selected}:
            selected.append(item)
    for rank, item in enumerate(selected, 1):
        item.rank = rank
    return selected

