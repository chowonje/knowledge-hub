from __future__ import annotations

from knowledge_hub.ai.answer_rewrite import apply_conservative_fallback_if_needed


class _ConfigStub:
    def get_nested(self, *keys, default=None):
        if keys == ("labs", "answer_readiness", "paper_short_citation_first", "enabled"):
            return True
        return default


class _SearcherStub:
    config = _ConfigStub()

    def _build_conservative_answer(self, **_kwargs):  # pragma: no cover - asserted by raising
        raise AssertionError("P1 fallback must use evidence-only answer builder")


def test_paper_answer_readiness_p1_fallback_uses_evidence_only_answer(monkeypatch):
    monkeypatch.setattr(
        "knowledge_hub.ai.answer_rewrite.verify_answer",
        lambda *_args, **_kwargs: {
            "status": "caution",
            "supportedClaimCount": 0,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 1,
            "conflictMentioned": True,
            "needsCaution": True,
            "warnings": ["heuristic limitation"],
            "claims": [],
        },
    )

    answer, rewrite_meta, verification = apply_conservative_fallback_if_needed(
        _SearcherStub(),
        query="attention mechanism",
        answer="generated answer",
        rewrite_meta={
            "applied": False,
            "requiresConservativeFallback": True,
            "warnings": ["answer rewrite skipped"],
        },
        verification={
            "status": "caution",
            "needsCaution": True,
            "unsupportedClaimCount": 1,
            "supportedClaimCount": 0,
            "claims": [{"claim": "The generated uncertain claim is copied from the model.", "verdict": "uncertain"}],
        },
        evidence=[
            {
                "title": "Attention Is All You Need",
                "excerpt": "The Transformer is based solely on attention mechanisms.",
                "source_type": "paper",
            }
        ],
        answer_signals={"paper_definition_mode": True},
        contradicting_beliefs=[],
        allow_external=False,
        routing_meta={"route": "local"},
    )

    assert "The generated uncertain claim" not in answer
    assert "- [근거: Attention Is All You Need] The Transformer is based solely on attention mechanisms." in answer
    assert rewrite_meta["finalAnswerSource"] == "conservative_fallback"
    assert verification["needsCaution"] is True
    for banned in ["synthetic_fallback", "verification", "unsupported", "claim card"]:
        assert banned not in answer
