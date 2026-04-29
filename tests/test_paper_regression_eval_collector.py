from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/collect_paper_regression_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_regression_eval_collector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostic_fieldnames_are_additive_tail():
    module = _load_script()

    assert module.LIVE_GENERATION_DIAGNOSTIC_FIELDNAMES == [
        "answerProvider",
        "answerModel",
        "answerRoute",
        "finalAnswerSource",
        "answerGenerationFallbackUsed",
        "answerGenerationErrorType",
        "answerGenerationWarning",
        "generationFallbackUsed",
        "generationFallbackReason",
        "conservativeFallbackApplied",
        "answerVerificationStatus",
        "modelCallMs",
        "promptChars",
        "contextChars",
        "unsupportedClaimCount",
        "uncertainClaimCount",
        "needsCaution",
        "citationTracePresent",
        "citationCount",
        "sourceTitleTracePresent",
        "latencyMs",
        "timeoutFlag",
    ]
    assert module.PAPER_REGRESSION_EVAL_FIELDNAMES[-len(module.ASK_V2_DIAGNOSTIC_FIELDNAMES):] == module.ASK_V2_DIAGNOSTIC_FIELDNAMES
    for field in module.LIVE_GENERATION_DIAGNOSTIC_FIELDNAMES:
        assert field in module.PAPER_REGRESSION_EVAL_FIELDNAMES


def test_serialize_row_adds_live_generation_diagnostics_without_changing_regression_judgment():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "ask_v2", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Deep_Residual_Learning_efbb7871", "title": "Deep Residual Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "explicit_metadata_filter"},
        "answerVerification": {
            "status": "weak",
            "needsCaution": True,
            "unsupportedClaimCount": 2,
            "uncertainClaimCount": 1,
        },
        "answerRewrite": {"applied": True, "finalAnswerSource": "conservative_fallback"},
        "answerGeneration": {
            "fallbackUsed": True,
            "stage": "initial_answer",
            "errorType": "ReadTimeout",
            "warnings": ["answer generation fallback applied: initial_answer:ReadTimeout"],
            "modelCallMs": 60000,
            "promptChars": 321,
            "contextChars": 654,
        },
        "answerProvenance": {"mode": "weak_claim_fallback"},
        "router": {"selected": {"route": "local", "provider": "ollama", "model": "gemma4:e4b"}},
        "sources": [{"title": "Deep Residual Learning", "paper_id": "Deep_Residual_Learning_efbb7871"}],
        "answer": "conservative answer",
    }

    serialized = module._serialize_row(row, payload, top_k=5, retrieval_mode="hybrid", latency_ms=56.7, gate_mode="live_smoke")

    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "unsupported_claims"
    assert serialized["unsupported_claim_count"] == "2"
    assert serialized["answerProvider"] == "ollama"
    assert serialized["answerModel"] == "gemma4:e4b"
    assert serialized["answerRoute"] == "local"
    assert serialized["finalAnswerSource"] == "conservative_fallback"
    assert serialized["answerGenerationFallbackUsed"] == "1"
    assert serialized["answerGenerationErrorType"] == "ReadTimeout"
    assert serialized["answerGenerationWarning"] == "answer generation fallback applied: initial_answer:ReadTimeout"
    assert serialized["generationFallbackUsed"] == "1"
    assert serialized["generationFallbackReason"] == "initial_answer:ReadTimeout"
    assert serialized["conservativeFallbackApplied"] == "1"
    assert serialized["answerVerificationStatus"] == "weak"
    assert serialized["modelCallMs"] == "60000"
    assert serialized["promptChars"] == "321"
    assert serialized["contextChars"] == "654"
    assert serialized["unsupportedClaimCount"] == "2"
    assert serialized["uncertainClaimCount"] == "1"
    assert serialized["needsCaution"] == "1"
    assert serialized["citationTracePresent"] == "1"
    assert serialized["citationCount"] == "0"
    assert serialized["sourceTitleTracePresent"] == "1"
    assert serialized["latencyMs"] == "56.7"
    assert serialized["timeoutFlag"] == "0"


def test_serialize_row_marks_scoped_lookup_match_as_good():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "ask_v2", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Deep_Residual_Learning_efbb7871", "title": "Deep Residual Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "resolved_source_id"},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "paper_scoped_answer"},
        "sources": [{"title": "Deep Residual Learning", "paper_id": "Deep_Residual_Learning_efbb7871"}],
        "answer": "resnet paper summary",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.3)

    assert serialized["family_match"] == "1"
    assert serialized["answer_mode_match"] == "1"
    assert serialized["source_match"] == "1"
    assert serialized["actual_paper_scope_applied"] == "1"
    assert serialized["pred_label"] == "good"
    assert serialized["pred_reason"] == "family_scope_and_source_match"
    assert serialized["v2_verification_status"] == ""
    assert serialized["ask_v2_hard_gate"] == "0"
    assert serialized["selected_card_count"] == "0"
    assert serialized["filtered_evidence_count"] == "1"


def test_serialize_row_marks_compare_undercoverage_as_partial():
    module = _load_script()
    row = {
        "query": "RAG와 FiD를 비교해줘",
        "source": "paper",
        "eval_bucket": "compare",
        "expected_family": "paper_compare",
        "expected_top1_or_set": "2005.11401|Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks|2007.01282|Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        "expected_answer_mode": "paper_comparison",
        "expected_match_count": "2",
        "expected_scope_applied": "0",
        "allowed_fallback": "need_multiple_papers",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_compare",
        "queryFrame": {"family": "paper_compare", "answer_mode": "paper_comparison"},
        "familyRouteDiagnostics": {"answerMode": "paper_comparison", "runtimeUsed": "legacy"},
        "evidencePolicy": {"policyKey": "paper_compare_policy"},
        "representativePaper": {},
        "paperAnswerScope": {"paperScoped": False, "applied": False, "fallbackUsed": False, "reason": ""},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "raw_evidence"},
        "v2": {
            "routing": {"selected_card_ids": ["paper-card:self-rag"]},
            "cardSelection": {
                "resolvedPaperIds": ["2005.11401", "2007.01282"],
                "selectionStage": "compare_ranked_diversity",
                "selectionReason": "filled_from_ranked_candidate_diversity",
                "candidateCountBeforeRerank": 7,
                "candidateCountAfterRerank": 6,
                "resolvedPairPreserved": False,
                "selected": [
                    {
                        "cardId": "paper-card:self-rag",
                        "sourceId": "2310.11511",
                        "paperId": "2310.11511",
                        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
                        "selectionStage": "compare_ranked_diversity",
                        "selectionReason": "filled_from_ranked_candidate_diversity",
                    }
                ],
            },
        },
        "sources": [{"title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "paper_id": "2310.11511"}],
        "answer": "compare answer",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=15.0)

    assert serialized["family_match"] == "1"
    assert serialized["source_match"] == "0"
    assert serialized["matched_expected_count"] == "0"
    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "source_undercoverage"
    assert serialized["resolved_paper_ids"] == "2005.11401 | 2007.01282"
    assert serialized["selected_card_ids"] == "paper-card:self-rag"
    assert serialized["selected_card_titles"] == "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
    assert serialized["selected_card_paper_ids"] == "2310.11511"
    assert serialized["selected_card_stage"] == "compare_ranked_diversity"
    assert serialized["selection_reason"] == "filled_from_ranked_candidate_diversity"
    assert serialized["candidate_count_before_rerank"] == "7"
    assert serialized["candidate_count_after_rerank"] == "6"
    assert serialized["resolved_pair_preserved"] == "0"


def test_serialize_row_records_ask_v2_hard_gate_diagnostics_without_changing_judgment():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "no_result",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "ask_v2", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Deep_Residual_Learning_efbb7871", "title": "Deep Residual Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "explicit_metadata_filter"},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "weak_claim_fallback"},
        "v2": {
            "routing": {"selected_card_ids": ["paper-card:resnet"]},
            "cardSelection": {"selected": [{"cardId": "paper-card:resnet", "sourceId": "Deep_Residual_Learning_efbb7871"}]},
            "runtimeExecution": {"used": "ask_v2", "fallbackReason": "section_blocked_to_claim_cards"},
            "evidenceVerification": {
                "verificationStatus": "missing",
                "unsupportedFields": [],
                "anchorIdsUsed": ["anchor-resnet"],
            },
            "consensus": {"weakClaimCount": 3, "unsupportedClaimCount": 0},
        },
        "evidencePacket": {
            "askV2HardGate": True,
            "answerableDecisionReason": "missing_verification_evidence",
            "uniquePaperCount": 1,
        },
        "sources": [{"title": "Deep Residual Learning", "paper_id": "Deep_Residual_Learning_efbb7871"}],
        "answer": "제공된 근거만으로는 검증 가능한 답변을 생성하기 어렵습니다.",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.3)

    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "allowed_no_result_fallback"
    assert serialized["source_match"] == "1"
    assert serialized["actual_paper_scope_applied"] == "1"
    assert serialized["unsupported_claim_count"] == "0"
    assert serialized["v2_verification_status"] == "missing"
    assert serialized["ask_v2_hard_gate"] == "1"
    assert serialized["answerable_decision_reason"] == "missing_verification_evidence"
    assert serialized["weak_claim_count"] == "3"
    assert serialized["selected_card_count"] == "1"
    assert serialized["filtered_evidence_count"] == "1"
    assert serialized["compare_unique_paper_count"] == "1"
    assert serialized["compare_anchor_coverage"] == "unique_papers=1;anchors=1;selected_cards=1"
    assert serialized["hard_gate_reason"] == "missing_verification_evidence"


def test_serialize_row_keeps_answer_verification_count_separate_from_v2_consensus():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "no_result",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "ask_v2", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Deep_Residual_Learning_efbb7871", "title": "Deep Residual Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "explicit_metadata_filter"},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "weak_claim_fallback"},
        "v2": {
            "routing": {"selected_card_ids": ["paper-card:resnet"]},
            "cardSelection": {"selected": [{"cardId": "paper-card:resnet", "sourceId": "Deep_Residual_Learning_efbb7871"}]},
            "runtimeExecution": {"used": "ask_v2", "fallbackReason": ""},
            "evidenceVerification": {
                "verificationStatus": "weak",
                "unsupportedFields": [],
                "anchorIdsUsed": ["anchor-resnet"],
            },
            "claimVerification": [
                {
                    "status": "unsupported",
                    "verdict": "unsupported",
                    "reasons": ["no_anchor_backed_evidence"],
                }
            ],
            "consensus": {"supportCount": 1, "weakClaimCount": 1, "unsupportedClaimCount": 2},
            "preHardGateAnswerable": True,
            "preHardGateReason": "substantive_evidence_found",
        },
        "evidencePacket": {
            "askV2HardGate": True,
            "answerableDecisionReason": "ask_v2_unsupported_claim_cards",
            "uniquePaperCount": 1,
        },
        "sources": [{"title": "Deep Residual Learning", "paper_id": "Deep_Residual_Learning_efbb7871"}],
        "answer": "제공된 근거만으로는 검증 가능한 답변을 생성하기 어렵습니다.",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.3)

    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "allowed_no_result_fallback"
    assert serialized["unsupported_claim_count"] == "0"
    assert serialized["v2ConsensusUnsupportedClaimCount"] == "2"
    assert serialized["v2ConsensusWeakClaimCount"] == "1"
    assert serialized["v2ConsensusSupportedClaimCount"] == "1"
    assert serialized["v2ClaimReasonSummary"] == "unsupported:no_anchor_backed_evidence=1"
    assert serialized["hard_gate_reason"] == "ask_v2_unsupported_claim_cards"
    assert serialized["preHardGateAnswerable"] == "1"
    assert serialized["preHardGateReason"] == "substantive_evidence_found"


def test_serialize_row_preserves_relaxed_claim_card_gate_diagnostics_without_changing_judgment():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "ask_v2", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Deep_Residual_Learning_efbb7871", "title": "Deep Residual Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "explicit_metadata_filter"},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "weak_claim_fallback"},
        "v2": {
            "routing": {"selected_card_ids": ["paper-card:resnet"]},
            "cardSelection": {"selected": [{"cardId": "paper-card:resnet", "sourceId": "Deep_Residual_Learning_efbb7871"}]},
            "runtimeExecution": {"used": "ask_v2", "fallbackReason": ""},
            "evidenceVerification": {
                "verificationStatus": "weak",
                "unsupportedFields": [],
                "anchorIdsUsed": ["anchor-resnet"],
            },
            "claimVerification": [
                {
                    "status": "unsupported",
                    "verdict": "unsupported",
                    "reasons": ["no_anchor_backed_evidence"],
                }
            ],
            "consensus": {"supportCount": 1, "weakClaimCount": 1, "unsupportedClaimCount": 2},
            "preHardGateAnswerable": True,
            "preHardGateReason": "substantive_evidence_found",
            "askV2OriginalHardGateReason": "ask_v2_unsupported_claim_cards",
            "claimCardGateRelaxed": True,
            "claimCardGateRelaxationReason": "claim_card_unsupported_relaxed_for_paper_family",
        },
        "evidencePacket": {
            "answerable": True,
            "askV2HardGate": False,
            "askV2OriginalHardGateReason": "ask_v2_unsupported_claim_cards",
            "claimCardGateRelaxed": True,
            "claimCardGateRelaxationReason": "claim_card_unsupported_relaxed_for_paper_family",
            "uniquePaperCount": 1,
        },
        "sources": [{"title": "Deep Residual Learning", "paper_id": "Deep_Residual_Learning_efbb7871"}],
        "answer": "resnet paper summary",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.3)

    assert serialized["pred_label"] == "good"
    assert serialized["pred_reason"] == "family_scope_and_source_match"
    assert serialized["ask_v2_hard_gate"] == "0"
    assert serialized["hard_gate_reason"] == ""
    assert serialized["askV2OriginalHardGateReason"] == "ask_v2_unsupported_claim_cards"
    assert serialized["claimCardGateRelaxed"] == "1"
    assert serialized["claimCardGateRelaxationReason"] == "claim_card_unsupported_relaxed_for_paper_family"
    assert serialized["v2ConsensusUnsupportedClaimCount"] == "2"
    assert serialized["v2ClaimReasonSummary"] == "unsupported:no_anchor_backed_evidence=1"


def test_serialize_row_leaves_diagnostic_columns_empty_when_v2_is_missing():
    module = _load_script()
    row = {
        "query": "Sequence to Sequence Learning with Neural Networks 논문의 방법을 설명해줘",
        "source": "paper",
        "eval_bucket": "method",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Sequence_to_Sequence_Learning_68e9c2a0|Sequence to Sequence Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_scoped_answer"},
        "familyRouteDiagnostics": {"answerMode": "paper_scoped_answer", "runtimeUsed": "legacy", "resolvedSourceScopeApplied": True},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "representativePaper": {"paperId": "Sequence_to_Sequence_Learning_68e9c2a0", "title": "Sequence to Sequence Learning"},
        "paperAnswerScope": {"paperScoped": True, "applied": True, "fallbackUsed": False, "reason": "explicit_metadata_filter"},
        "answerVerification": {"needsCaution": False, "unsupportedClaimCount": 0},
        "answerRewrite": {"applied": False},
        "answerProvenance": {"mode": "raw_evidence"},
        "sources": [{"title": "Sequence to Sequence Learning", "paper_id": "Sequence_to_Sequence_Learning_68e9c2a0"}],
        "answer": "method answer",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.3)

    assert serialized["pred_label"] == "good"
    assert serialized["v2_verification_status"] == ""
    assert serialized["ask_v2_hard_gate"] == "0"
    assert serialized["selected_card_count"] == "0"
    assert serialized["filtered_evidence_count"] == "1"
    assert serialized["v2_diagnostics_keys"] == ""
    assert serialized["selected_card_ids"] == ""
    assert serialized["resolved_pair_preserved"] == ""
    assert serialized["answerProvider"] == ""
    assert serialized["answerModel"] == ""
    assert serialized["answerRoute"] == ""
    assert serialized["finalAnswerSource"] == ""
    assert serialized["answerGenerationFallbackUsed"] == "0"
    assert serialized["answerGenerationErrorType"] == ""
    assert serialized["answerGenerationWarning"] == ""
    assert serialized["generationFallbackUsed"] == "0"
    assert serialized["generationFallbackReason"] == ""
    assert serialized["conservativeFallbackApplied"] == "0"
    assert serialized["answerVerificationStatus"] == ""
    assert serialized["modelCallMs"] == ""
    assert serialized["promptChars"] == ""
    assert serialized["contextChars"] == ""
    assert serialized["unsupportedClaimCount"] == "0"
    assert serialized["uncertainClaimCount"] == "0"
    assert serialized["needsCaution"] == "0"
    assert serialized["citationTracePresent"] == "1"
    assert serialized["citationCount"] == "0"
    assert serialized["sourceTitleTracePresent"] == "1"
    assert serialized["latencyMs"] == "12.3"
    assert serialized["timeoutFlag"] == "0"


def test_error_row_adds_latency_and_live_generation_diagnostics_without_changing_judgment():
    module = _load_script()
    row = {
        "query": "Deep Residual Learning 논문 설명해줘",
        "source": "paper",
        "eval_bucket": "lookup",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "Deep_Residual_Learning_efbb7871|1512.03385|Deep Residual Learning",
        "expected_answer_mode": "paper_scoped_answer",
        "expected_match_count": "1",
        "expected_scope_applied": "1",
        "allowed_fallback": "paper_scoped_no_result",
    }

    serialized = module._error_row(
        row,
        top_k=5,
        retrieval_mode="hybrid",
        latency_ms=123.456,
        error=TimeoutError("boom"),
        gate_mode="live_smoke",
    )

    assert serialized["pred_label"] == "bad"
    assert serialized["pred_reason"] == "family_mismatch"
    assert serialized["no_result"] == "0"
    assert serialized["notes"] == "collector_error=TimeoutError: boom"
    assert serialized["latency_ms"] == "123.456"
    assert serialized["latencyMs"] == "123.456"
    assert serialized["timeout_flag"] == "1"
    assert serialized["timeoutFlag"] == "1"
    assert serialized["gate_mode"] == "live_smoke"
    assert serialized["answerProvider"] == ""
    assert serialized["answerModel"] == ""
    assert serialized["answerRoute"] == ""
    assert serialized["finalAnswerSource"] == ""
    assert serialized["answerGenerationFallbackUsed"] == "0"
    assert serialized["answerGenerationErrorType"] == ""
    assert serialized["answerGenerationWarning"] == ""
    assert serialized["generationFallbackUsed"] == "0"
    assert serialized["generationFallbackReason"] == ""
    assert serialized["conservativeFallbackApplied"] == "0"
    assert serialized["answerVerificationStatus"] == ""
    assert serialized["modelCallMs"] == ""
    assert serialized["promptChars"] == ""
    assert serialized["contextChars"] == ""
    assert serialized["unsupportedClaimCount"] == "0"
    assert serialized["uncertainClaimCount"] == "0"
    assert serialized["needsCaution"] == "0"
    assert serialized["citationTracePresent"] == "0"
    assert serialized["citationCount"] == "0"
    assert serialized["sourceTitleTracePresent"] == "0"


def test_live_smoke_selection_keeps_one_query_per_bucket_subset():
    module = _load_script()
    rows = [
        {"query": "Deep Residual Learning 논문 설명해줘"},
        {"query": "Deep Residual Learning 논문의 방법을 설명해줘"},
        {"query": "Batch Normalization 논문의 핵심 결과를 설명해줘"},
        {"query": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering을 비교해줘"},
        {"query": "Attention Is All You Need 논문 요약해줘"},
    ]

    selected = module._selected_queries(rows, gate_mode="live_smoke")

    assert [item["query"] for item in selected] == [
        "Deep Residual Learning 논문 설명해줘",
        "Deep Residual Learning 논문의 방법을 설명해줘",
        "Batch Normalization 논문의 핵심 결과를 설명해줘",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering을 비교해줘",
    ]







def test_live_smoke_summary_marks_route_not_ready_when_pair_not_preserved():
    module = _load_script()

    summary = module._build_live_smoke_summary(
        [
            {
                "pred_label": "good",
                "no_result": "0",
                "ask_v2_hard_gate": "0",
                "resolved_pair_preserved": "0",
                "answerGenerationFallbackUsed": "0",
                "generationFallbackUsed": "0",
                "conservativeFallbackApplied": "0",
                "answerGenerationErrorType": "",
                "answerGenerationWarning": "",
                "generationFallbackReason": "",
                "timeoutFlag": "0",
                "timeout_flag": "0",
                "latencyMs": "12000",
                "citationTracePresent": "1",
            }
        ]
    )

    assert summary["rowCount"] == 1
    assert summary["generationFallbackCount"] == 0
    assert summary["readTimeoutCount"] == 0
    assert summary["timeoutCount"] == 0
    assert summary["p95LatencyMs"] == 12000
    assert summary["unsupportedClaimsJudgmentCount"] == 0
    assert summary["structuralRouteReady"] is False
    assert summary["routeAcceptanceReady"] is False
    assert summary["answerAcceptanceReady"] is False
    assert summary["answerAcceptanceBlockers"] == ["structural_route_not_ready"]


def test_live_smoke_summary_separates_structural_route_from_unsupported_answer_quality():
    module = _load_script()

    summary = module._build_live_smoke_summary(
        [
            {
                "pred_label": "partial",
                "pred_reason": "unsupported_claims",
                "no_result": "0",
                "ask_v2_hard_gate": "0",
                "resolved_pair_preserved": "1",
                "answerGenerationFallbackUsed": "0",
                "generationFallbackUsed": "0",
                "conservativeFallbackApplied": "1",
                "answerGenerationErrorType": "",
                "answerGenerationWarning": "",
                "generationFallbackReason": "",
                "timeoutFlag": "0",
                "timeout_flag": "0",
                "latencyMs": "12000",
                "citationTracePresent": "1",
            }
        ]
    )

    assert summary["structuralRouteReady"] is True
    assert summary["routeAcceptanceReady"] is False
    assert summary["unsupportedClaimsJudgmentCount"] == 1
    assert summary["answerAcceptanceReady"] is False
    assert summary["answerAcceptanceBlockers"] == ["unsupported_claims"]
