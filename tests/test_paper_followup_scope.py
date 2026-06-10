from __future__ import annotations

from knowledge_hub.domain.ai_papers.families import PAPER_FAMILY_LOOKUP
from knowledge_hub.domain.ai_papers.followup_scope import canonical_followup_title_candidate
from knowledge_hub.domain.ai_papers.query_plan import build_rule_based_query_frame


def test_transformer_recurrence_followup_pins_attention_is_all_you_need_scope():
    # Given: a paper follow-up that names the Transformer and asks for evidence
    query = "What evidence supports removing recurrence in the Transformer?"

    # When: the paper query frame is built
    frame = build_rule_based_query_frame(query, source_type="paper").to_dict()

    # Then: the frame stays scoped to the canonical Transformer paper
    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["1706.03762"]
    assert frame["expanded_terms"][0] == "Attention Is All You Need"


def test_transformer_followup_scope_does_not_capture_vision_transformer_questions():
    # Given: a follow-up that explicitly names the Vision Transformer family
    query = "What evidence supports patch embeddings in the Vision Transformer?"

    # When: canonical Transformer follow-up scope is checked
    title = canonical_followup_title_candidate(query)

    # Then: the original Transformer paper is not silently selected
    assert title == ""
