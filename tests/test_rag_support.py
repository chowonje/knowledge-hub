from __future__ import annotations

from knowledge_hub.ai.rag_support import (
    apply_paper_answer_readiness_p1_prompt_overlay,
    build_answer_prompt,
    build_paper_answer_readiness_p1_conservative_answer,
    build_paper_definition_context,
)
from knowledge_hub.core.models import SearchResult


def test_build_answer_prompt_enforces_concept_first_order_for_paper_definition_queries():
    prompt = build_answer_prompt(
        query="Transformer의 핵심 아이디어를 설명해줘",
        answer_signals={
            "quality_counts": {"ok": 1, "needs_review": 0, "reject": 0, "unscored": 1},
            "caution_required": False,
            "paper_definition_mode": True,
            "answer_mode": "representative_paper_explainer_beginner",
            "concept_core_evidence": {
                "paperId": "1706.03762",
                "title": "Transformer",
                "summary": "Use self-attention to model token relations in parallel.",
            },
            "representative_paper": {
                "paperId": "1706.03762",
                "title": "Attention Is All You Need",
                "citationLabel": "S1",
                "sourceCount": 2,
            },
            "supporting_paper_count": 1,
        },
    )

    assert "답변 모드: paper_definition_anchor" in prompt
    assert "대표 논문: Attention Is All You Need (1706.03762)" in prompt
    assert "보조 논문 수: 1" in prompt
    assert "한줄 정의 -> 작동 원리 -> 왜 중요한지 -> 대표 사례" in prompt
    assert "대표 논문은 개념의 본문 주인공이 아니라 대표 사례 또는 전환점으로만 소개한다." in prompt
    assert "쉬운 설명 요청이므로 전문 용어를 바로 풀어쓰고" in prompt
    assert "audience=beginner" in prompt
    assert prompt.index("한줄 정의") < prompt.index("작동 원리") < prompt.index("왜 중요한지") < prompt.index("대표 사례")


def test_build_paper_definition_context_places_concept_core_before_representative_example():
    context = build_paper_definition_context(
        query="Transformer의 핵심 아이디어를 설명해줘",
        filtered=[
            SearchResult(
                document=(
                    "Title: Attention Is All You Need\n"
                    "## 한줄 요약\n\n"
                    "The Transformer uses only attention mechanisms, removing recurrence.\n\n"
                    "## 핵심 아이디어\n\n"
                    "Use self-attention to model token relations in parallel.\n\n"
                    "## 방법\n\n"
                    "An encoder-decoder stack built entirely from attention and feed-forward layers.\n"
                ),
                metadata={"title": "Attention Is All You Need", "arxiv_id": "1706.03762"},
                distance=0.1,
                score=0.9,
                semantic_score=0.9,
                lexical_score=0.1,
                retrieval_mode="hybrid",
                document_id="paper:1706.03762",
            ),
            SearchResult(
                document=(
                    "Title: An Image is Worth 16x16 Words\n"
                    "## 한줄 요약\n\n"
                    "Apply the transformer encoder to image patches.\n"
                ),
                metadata={"title": "An Image is Worth 16x16 Words", "arxiv_id": "2010.11929"},
                distance=0.12,
                score=0.8,
                semantic_score=0.8,
                lexical_score=0.08,
                retrieval_mode="hybrid",
                document_id="paper:2010.11929",
            ),
        ],
        evidence=[
            {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "citation_label": "S1", "excerpt": "fallback one"},
            {"title": "An Image is Worth 16x16 Words", "arxiv_id": "2010.11929", "citation_label": "S2", "excerpt": "fallback two"},
        ],
        answer_signals={
            "paper_definition_mode": True,
            "answer_mode": "representative_paper_explainer_beginner",
            "concept_core_evidence": {
                "paperId": "1706.03762",
                "title": "Transformer",
                "summary": "Use self-attention to model token relations in parallel.",
            },
            "representative_paper": {"paperId": "1706.03762", "title": "Attention Is All You Need"},
            "supporting_paper_count": 1,
        },
        claim_context="",
    )

    assert "=== Concept Core Evidence ===" in context
    assert "=== Why It Matters ===" in context
    assert "=== Representative Paper Example 1 ===" in context
    assert "Audience: beginner" in context
    assert "intuition_hint=Use one simple intuition or everyday metaphor and avoid jargon-heavy wording." in context
    assert context.index("=== Concept Core Evidence ===") < context.index("=== Representative Paper Example 1 ===")
    assert "one_line_summary=The Transformer uses only attention mechanisms, removing recurrence." in context
    assert "one_line_definition=The Transformer uses only attention mechanisms, removing recurrence." in context
    assert "mechanism_summary=Use self-attention to model token relations in parallel." in context
    assert "support_note=Apply the transformer encoder to image patches." in context


def test_paper_answer_readiness_p1_prompt_overlay_is_idempotent():
    prompt = apply_paper_answer_readiness_p1_prompt_overlay("base prompt")
    again = apply_paper_answer_readiness_p1_prompt_overlay(prompt)

    assert prompt == again
    assert "최대 4개의 짧은 bullet" in prompt
    assert "- [근거: <논문 또는 출처 제목>]" in prompt


def test_paper_answer_readiness_p1_v2_prompt_overlay_uses_two_bullet_contract():
    prompt = apply_paper_answer_readiness_p1_prompt_overlay(
        "base prompt",
        budget_v2=True,
        max_bullets=2,
    )
    again = apply_paper_answer_readiness_p1_prompt_overlay(
        prompt,
        budget_v2=True,
        max_bullets=2,
    )

    assert prompt == again
    assert "답변 준비도 P1 short/citation-first 규칙 v2" in prompt
    assert "최대 2개의 bullet" in prompt
    assert "- [근거: <제목>] <근거에서 직접 확인되는 한 문장>" in prompt
    assert "근거 title/excerpt에 없는 숫자, 연도, 성능 비교, 배경지식" in prompt
    assert "근거에서 확인되는 범위" in prompt
    for banned in ["synthetic_fallback", "verification", "unsupported", "claim card"]:
        assert banned not in prompt

    capped = apply_paper_answer_readiness_p1_prompt_overlay("base prompt", budget_v2=True, max_bullets=4)
    assert "최대 2개의 bullet" in capped


def test_paper_answer_readiness_p1_fallback_uses_title_and_excerpt_only():
    answer = build_paper_answer_readiness_p1_conservative_answer(
        evidence=[
            {
                "title": "Attention Is All You Need",
                "excerpt": "The Transformer is based solely on attention mechanisms.",
            }
        ]
    )

    assert "근거에서 확인되는 범위:" in answer
    assert "- [근거: Attention Is All You Need] The Transformer is based solely on attention mechanisms." in answer
    for banned in ["synthetic_fallback", "verification", "unsupported", "claim card"]:
        assert banned not in answer
