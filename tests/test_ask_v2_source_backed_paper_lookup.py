from __future__ import annotations

from knowledge_hub.ai.ask_v2_source_backed_paper_lookup import source_backed_paper_lookup_results


class _VectorDatabase:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_documents(self, **kwargs):  # noqa: ANN003
        self.calls.append(dict(kwargs))
        return {
            "ids": ["paper_1706.03762_0"],
            "documents": [
                "Title: Attention Is All You Need\n\n"
                "The Transformer relies on self-attention, removes recurrence and "
                "convolutions, and enables parallelizable sequence transduction."
            ],
            "metadatas": [
                {
                    "title": "Attention Is All You Need",
                    "source_type": "paper",
                    "arxiv_id": "1706.03762",
                }
            ],
        }


class _Searcher:
    def __init__(self) -> None:
        self.database = _VectorDatabase()


def test_source_backed_paper_lookup_reads_only_resolved_paper_vector_doc() -> None:
    # Given: a single-paper lookup has been resolved to the Transformer paper.
    searcher = _Searcher()

    # When: active vector evidence is collected for that scoped lookup.
    results = source_backed_paper_lookup_results(
        searcher=searcher,
        paper_family="paper_lookup",
        query_frame={"resolved_source_ids": ["1706.03762"]},
        metadata_filter={"source_type": "paper", "arxiv_id": "1706.03762"},
        limit=2,
    )

    # Then: the helper returns a source-backed active vector result without broad search.
    assert searcher.database.calls == [
        {
            "filter_dict": {"source_type": "paper", "arxiv_id": "1706.03762"},
            "limit": 2,
            "include_ids": True,
            "include_documents": True,
            "include_metadatas": True,
        }
    ]
    assert len(results) == 1
    assert results[0].document_id == "paper_1706.03762_0"
    assert results[0].metadata["arxiv_id"] == "1706.03762"
    assert results[0].retrieval_mode == "active-vector-paper"
    assert results[0].lexical_extras is not None
    assert results[0].lexical_extras["quality_flag"] == "ok"


def test_source_backed_paper_lookup_stays_off_for_unscoped_lookup() -> None:
    # Given: a paper query has not resolved to exactly one paper id.
    searcher = _Searcher()

    # When: active vector evidence is requested.
    results = source_backed_paper_lookup_results(
        searcher=searcher,
        paper_family="paper_lookup",
        query_frame={"resolved_source_ids": ["1706.03762", "2010.11929"]},
        metadata_filter={"source_type": "paper"},
        limit=2,
    )

    # Then: no vector document is read.
    assert results == []
    assert searcher.database.calls == []
