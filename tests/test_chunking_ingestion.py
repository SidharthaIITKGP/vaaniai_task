import sys
from types import SimpleNamespace

from relay.chunking import PageText, chunk_pages
from relay.ingestion import ingest


def test_chunking_preserves_location_and_is_stable():
    pages = [PageText(7, "2 Method\n" + "attention mechanism token " * 30)]
    kwargs = dict(paper_id="1706.03762", paper_title="Attention Is All You Need",
                  arxiv_version="v7", filename="paper.pdf", chunk_tokens=20, overlap=5)
    first = chunk_pages(pages, **kwargs)
    second = chunk_pages(pages, **kwargs)
    assert [x.chunk_id for x in first] == [x.chunk_id for x in second]
    assert all(x.page_number == 7 and x.filename == "paper.pdf" for x in first)
    assert first[0].section == "2 Method"


def test_chunk_validation():
    try:
        chunk_pages([], paper_id="x", paper_title="x", arxiv_version="v1",
                    filename="x", chunk_tokens=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap accepted")


def test_ingestion_metadata_and_version(monkeypatch, config):
    paper_dir = config.root / "research paper"
    paper_dir.mkdir()
    for name in ("1706.03762v7.pdf", "2410.00037v1.pdf", "2507.07955v2.pdf"):
        (paper_dir / name).touch()

    texts = {
        "1706.03762": "Attention Is All You Need\narXiv:1706.03762v7\n1 Introduction\nattention text " * 20,
        "2410.00037": "Moshi real-time dialogue\narXiv:2410.00037v1\n1 Introduction\nspeech text " * 20,
        "2507.07955": "Dynamic Chunking hierarchical sequence modeling\narXiv:2507.07955v2\n1 Method\nchunk text " * 20,
    }

    class Page:
        def __init__(self, text): self.text = text
        def get_text(self, *args, **kwargs): return self.text

    class Doc(list):
        metadata = {"title": ""}
        def __enter__(self): return self
        def __exit__(self, *args): return None

    def open_pdf(path):
        paper_id = next(key for key in texts if key in str(path))
        return Doc([Page(texts[paper_id])])

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=open_pdf))
    report = ingest(config)
    assert all(report.verified.values())
    assert {chunk.paper_id for chunk in report.chunks} == set(texts)
    assert all(chunk.arxiv_version in {"v1", "v2", "v7"} for chunk in report.chunks)
