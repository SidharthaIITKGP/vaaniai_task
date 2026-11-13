from relay.citation_validator import build_alias_mapping, validate_citations
from relay.retrieval import (comparison_paper_ids, merge_passages,
                             merge_passages_with_coverage)


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
    mixed = validate_citations("A [S1] with ordinary [dimension] notation", evidence, aliases)
    assert mixed.valid and not mixed.invalid_ids


def test_multiple_aliases_and_grouped_alias_rejection(retriever):
    evidence = retriever.retrieve("attention")
    aliases = build_alias_mapping(evidence)
    assert validate_citations("Claims [S1][S2]", evidence, aliases).valid
    assert not validate_citations("Claims [S1, S2]", evidence, aliases).valid


def test_latex_and_numeric_brackets_are_not_citations(retriever):
    evidence = retriever.retrieve("attention")
    aliases = build_alias_mapping(evidence)
    answer = r"""\[
    Q \in \mathbb{R}^{[\,\text{queries}\times d_k\,]},\quad
    K \in \mathbb{R}^{[\,\text{keys}\times d_k\,]}
    \]
    The scaling is required [S1]. Prior work is numbered [3][4].
    """
    result = validate_citations(answer, evidence, aliases)
    assert result.valid
    assert result.cited_aliases == ["S1"]
    assert result.invalid_ids == []


def test_numeric_brackets_without_aliases_report_missing_citations(retriever):
    evidence = retriever.retrieve("attention")
    result = validate_citations("A claim with references [3][4].", evidence)
    assert not result.valid
    assert result.reason == "no citation aliases found"


def test_explicit_refusal_needs_no_citation(retriever):
    evidence = retriever.retrieve("attention")
    refusal = "The provided papers do not establish an answer to this question."
    assert validate_citations(refusal, evidence, refusal_text=refusal).valid


def test_merge_deduplicates(retriever):
    group = retriever.retrieve("attention")
    merged = merge_passages([group, group], 5)
    assert len(merged) == len(group)


def test_cross_paper_retrieval_guarantees_named_paper_coverage(retriever):
    question = "Compare the original Transformer and H-Net input approaches."
    required = comparison_paper_ids(question)
    assert required == ["1706.03762", "2507.07955"]
    groups = [retriever.retrieve(question)]
    groups.extend(retriever.retrieve(question, top_k=2, paper_ids={paper}) for paper in required)
    merged = merge_passages_with_coverage(groups, 3, required)
    assert set(required).issubset({item.chunk.paper_id for item in merged})


def test_paper_filter_never_leaks_other_papers(retriever):
    results = retriever.retrieve("input sequence", top_k=3, paper_ids={"1706.03762"})
    assert results
    assert {item.chunk.paper_id for item in results} == {"1706.03762"}
