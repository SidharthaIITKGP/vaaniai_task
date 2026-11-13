from __future__ import annotations

import csv

from relay.evaluate import (FIELDS, FORMAL_15, SMOKE_6, RecordingRetriever,
                            build_run_plan, completed_keys, result_row)
from relay.evaluation_questions import QUESTIONS
from relay.schemas import PipelineResult


def test_fixed_question_matrix_has_30_unique_questions_and_90_runs():
    assert len(QUESTIONS) == 30
    assert len({question.question_id for question in QUESTIONS}) == 30
    plan = build_run_plan(None, ("fast", "verified", "adaptive"))
    assert len(plan) == 90
    assert len(build_run_plan(SMOKE_6, ("fast", "verified", "adaptive"))) == 18
    assert len(build_run_plan(FORMAL_15, ("fast", "verified", "adaptive"))) == 45


def test_recording_retriever_captures_passage_metadata(retriever):
    recorder = RecordingRetriever(retriever)
    results = recorder.retrieve("multi head attention")
    assert results and len(recorder.calls) == 1
    passage = recorder.calls[0]["passages"][0]
    assert passage["chunk_id"] and passage["paper_title"]
    assert passage["rrf_score"] is not None and "passage" in passage


def test_result_row_contains_answer_sources_and_metrics(retriever):
    recorder = RecordingRetriever(retriever)
    passages = recorder.retrieve("multi head attention")
    cited = passages[0].chunk
    result = PipelineResult(
        "request", "fast", "ok", "answer", citations=[cited.chunk_id],
        retrieved_chunk_ids=[cited.chunk_id], metrics={
            "detected_intent": "paper_question", "routing_method": "llm",
            "routing_model_calls": 1, "architecture_model_calls": 1,
            "total_model_calls": 2, "llm_calls": [], "total_ms": 12.0,
        },
    )
    row = result_row(QUESTIONS[0], "fast", "hybrid", "start", result, recorder,
                     {chunk.chunk_id: chunk for chunk in retriever.chunks})
    assert tuple(row) == FIELDS
    assert cited.chunk_id in row["cited_sources_json"]
    assert "rrf_score" in row["retrieval_trace_json"]


def test_resume_retries_only_failed_provider_rows(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({"question_id": "T1", "architecture": "fast",
                         "intent_router": "hybrid", "status": "ok"})
        writer.writerow({"question_id": "T2", "architecture": "fast",
                         "intent_router": "hybrid", "status": "rate_limited"})
        writer.writerow({"question_id": "T3", "architecture": "fast",
                         "intent_router": "hybrid", "status": "capacity_unavailable"})
    done = completed_keys(path, "hybrid")
    assert ("T1", "fast", "hybrid") in done
    assert ("T2", "fast", "hybrid") not in done
    assert ("T3", "fast", "hybrid") not in done
