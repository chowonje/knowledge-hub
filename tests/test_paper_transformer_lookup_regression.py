from __future__ import annotations

from typing import Any

from knowledge_hub.domain.ai_papers.families import PAPER_FAMILY_LOOKUP
from knowledge_hub.domain.ai_papers.query_plan import build_rule_based_query_frame


class TransformerLookupSQLite:
    def search_paper_cards_v2(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        _ = query
        _ = limit
        return []

    def search_papers(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        _ = limit
        if query == "Attention Is All You Need":
            return [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                }
            ]
        return []


def test_rule_based_query_frame_keeps_transformer_rescue_for_noisy_english_paper_lookup() -> None:
    # Given: a noisy English lookup asks about the Transformer paper, while the
    # representative paper is only reachable through the local papers title index.
    sqlite_db = TransformerLookupSQLite()

    # When: the paper query frame is built for the public paper surface.
    frame = build_rule_based_query_frame(
        "Explain the core idea of the Transformer paper.",
        source_type="paper",
        sqlite_db=sqlite_db,
    ).to_dict()

    # Then: strict noisy-title parsing must not discard the resolved Transformer paper.
    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert "Attention Is All You Need" in frame["expanded_terms"]
    assert frame["resolved_source_ids"] == ["1706.03762"]
