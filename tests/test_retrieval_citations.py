from relay.citation_validator import build_alias_mapping, validate_citations
from relay.retrieval import merge_passages


def test_hybrid_retrieval_and_dedup(retriever):
    results = retriever.retrieve("multi head attention", top_k=3)
    assert results[0].chunk.paper_id == "1706.03762"
    assert len({x.chunk.chunk_id for x in results}) == len(results)
    assert all(x.bm25_score is not None and x.dense_score is not None for x in results)


def test_citation_validation(retriever):
    evidence = retriever.retrieve("attention")
    aliases = build_alias_mapping(evidence)
    valid_id = aliases["S1"]
    result = validate_citations("A claim [S1]", evidence, aliases)
    assert result.valid and result.cited_ids == [valid_id]
    invalid = validate_citations("A claim [S99]", evidence, aliases)
    assert not invalid.valid and invalid.invalid_ids
    mixed = validate_citations("A [S1] but also [invented-source]", evidence, aliases)
    assert not mixed.valid and mixed.invalid_ids == ["invented-source"]


def test_multiple_aliases_and_grouped_alias_rejection(retriever):
    evidence = retriever.retrieve("attention")
    aliases = build_alias_mapping(evidence)
    assert validate_citations("Claims [S1][S2]", evidence, aliases).valid
    assert not validate_citations("Claims [S1, S2]", evidence, aliases).valid


def test_explicit_refusal_needs_no_citation(retriever):
    evidence = retriever.retrieve("attention")
    refusal = "The provided papers do not establish an answer to this question."
    assert validate_citations(refusal, evidence, refusal_text=refusal).valid


def test_merge_deduplicates(retriever):
    group = retriever.retrieve("attention")
    merged = merge_passages([group, group], 5)
    assert len(merged) == len(group)
