from __future__ import annotations

from knowledge_hub.ai.answer_rewrite import rewrite_answer
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.config import Config
from tests.test_rag_search import DummyEmbedder, DummyVectorDB, StaticLLM, _build_records


def _local_searcher() -> RAGSearcher:
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    return RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=StaticLLM("unused"), config=config)


def test_verify_answer_fails_when_it_contradicts_rejected_belief_without_caveat():
    searcher = _local_searcher()

    verification = searcher._verify_answer(
        query="attention mechanism",
        answer="Attention은 seq2seq training에 필요합니다.",
        evidence=[{"title": "Paper A", "excerpt": "attention is needed for seq2seq training.", "source_type": "paper"}],
        answer_signals={"contradictory_source_count": 0},
        contradicting_beliefs=[{"belief_id": "belief_1", "statement": "Attention은 필요하지 않다.", "status": "rejected"}],
        allow_external=False,
    )

    assert verification["status"] == "failed"
    assert verification["needsCaution"] is True
    assert verification["contradictsRejectedBelief"] is True
    assert verification["rejectedBeliefConflictCount"] == 1
    assert verification["reasonCode"] == "contradicts_rejected_belief"
    assert any("contradicts rejected belief" in warning for warning in verification["warnings"])


def test_verify_answer_keeps_rejected_belief_conflict_as_caution_when_answer_mentions_it():
    searcher = _local_searcher()

    verification = searcher._verify_answer(
        query="attention mechanism",
        answer="Attention이 필요하다는 근거가 있지만, 기존 판단과 상충할 수 있어 단정하기 어렵습니다.",
        evidence=[{"title": "Paper A", "excerpt": "attention is needed for seq2seq training.", "source_type": "paper"}],
        answer_signals={"contradictory_source_count": 0},
        contradicting_beliefs=[{"belief_id": "belief_1", "statement": "Attention은 필요하지 않다.", "status": "rejected"}],
        allow_external=False,
    )

    assert verification["status"] == "caution"
    assert verification["needsCaution"] is True
    assert verification["conflictMentioned"] is True
    assert verification["contradictsRejectedBelief"] is True
    assert verification["rejectedBeliefConflictCount"] == 1


def test_verify_answer_fails_when_only_retrieval_signals_support_grounding():
    searcher = _local_searcher()

    verification = searcher._verify_answer(
        query="attention mechanism",
        answer="Attention은 최신 prerequisite입니다.",
        evidence=[
            {
                "title": "Learning edge",
                "excerpt": "attention prerequisite edge",
                "source_id": "learning_edge:rag:prereq",
                "source_type": "learning_edge",
            }
        ],
        answer_signals={"contradictory_source_count": 0},
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert verification["status"] == "failed"
    assert verification["needsCaution"] is True
    assert verification["retrievalSignalCount"] == 1
    assert verification["groundingEvidenceCount"] == 0
    assert verification["reasonCode"] == "signal_only_grounding"
    assert any("retrieval signals are not citation-grade evidence" in warning for warning in verification["warnings"])


def test_rewrite_answer_skips_when_verification_reports_signal_only_grounding():
    searcher = _local_searcher()
    answer = "Attention은 최신 prerequisite입니다."

    rewritten, rewrite_meta = rewrite_answer(
        searcher,
        query="attention mechanism",
        answer=answer,
        evidence=[],
        answer_signals={},
        verification={
            "status": "failed",
            "needsCaution": True,
            "supportedClaimCount": 0,
            "unsupportedClaimCount": 0,
            "uncertainClaimCount": 0,
            "retrievalSignalCount": 1,
            "groundingEvidenceCount": 0,
            "conflictMentioned": True,
        },
        contradicting_beliefs=[],
        allow_external=False,
    )

    assert rewritten == answer
    assert rewrite_meta["applied"] is False
    assert rewrite_meta["requiresConservativeFallback"] is True
    assert any(
        "retrieval signals without citation-grade evidence require conservative fallback" in warning
        for warning in rewrite_meta["warnings"]
    )
