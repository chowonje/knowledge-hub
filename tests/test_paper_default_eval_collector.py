from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "eval/knowledgeos/scripts/collect_paper_default_eval.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_default_eval_collector_test", SCRIPT)
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
    assert module.PAPER_DEFAULT_EVAL_FIELDNAMES[-len(module.ASK_V2_DIAGNOSTIC_FIELDNAMES):] == module.ASK_V2_DIAGNOSTIC_FIELDNAMES
    for field in module.LIVE_GENERATION_DIAGNOSTIC_FIELDNAMES:
        assert field in module.PAPER_DEFAULT_EVAL_FIELDNAMES


def test_serialize_row_adds_live_generation_diagnostics_without_changing_default_judgment():
    module = _load_script()
    row = {
        "query": "AlexNet 논문 요약해줘",
        "source": "paper",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "alexnet-2012",
        "expected_answer_mode": "paper_summary",
        "allowed_fallback": "",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryPlan": {"family": "paper_lookup"},
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_summary"},
        "representativePaper": {"paperId": "alexnet-2012", "title": "ImageNet Classification with Deep Convolutional Neural Networks"},
        "answerSignals": {},
        "answerVerification": {
            "status": "weak",
            "needsCaution": True,
            "unsupportedClaimCount": 2,
            "uncertainClaimCount": 1,
        },
        "answerRewrite": {"applied": True, "finalAnswerSource": "conservative_fallback"},
        "answerGeneration": {
            "status": "fallback",
            "fallbackUsed": True,
            "stage": "initial_answer",
            "errorType": "ReadTimeout",
            "errorMessage": "timed out",
            "warnings": ["answer generation fallback applied: initial_answer:ReadTimeout"],
            "modelCallMs": 60001,
            "promptChars": 123,
            "contextChars": 456,
        },
        "router": {"selected": {"route": "local", "provider": "ollama", "model": "gemma4:e4b"}},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "paper_summary", "runtimeUsed": "ask_v2"},
        "sources": [{"title": "ImageNet Classification with Deep Convolutional Neural Networks", "citation_target": "alexnet-2012"}],
        "citations": [{"target": "alexnet-2012"}],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=9.5, gate_mode="live_smoke")

    assert serialized["pred_label"] == "good"
    assert serialized["pred_reason"] == "family_and_mode_match"
    assert serialized["answerProvider"] == "ollama"
    assert serialized["answerModel"] == "gemma4:e4b"
    assert serialized["answerRoute"] == "local"
    assert serialized["finalAnswerSource"] == "conservative_fallback"
    assert serialized["answerGenerationFallbackUsed"] == "1"
    assert serialized["answerGenerationErrorType"] == "ReadTimeout"
    assert serialized["answerGenerationWarning"] == "answer generation fallback applied: initial_answer:ReadTimeout"
    assert serialized["generationFallbackUsed"] == "1"
    assert serialized["generationFallbackReason"] == "initial_answer:ReadTimeout:timed out"
    assert serialized["conservativeFallbackApplied"] == "1"
    assert serialized["answerVerificationStatus"] == "weak"
    assert serialized["modelCallMs"] == "60001"
    assert serialized["promptChars"] == "123"
    assert serialized["contextChars"] == "456"
    assert serialized["unsupportedClaimCount"] == "2"
    assert serialized["uncertainClaimCount"] == "1"
    assert serialized["needsCaution"] == "1"
    assert serialized["citationTracePresent"] == "1"
    assert serialized["citationCount"] == "1"
    assert serialized["sourceTitleTracePresent"] == "1"
    assert serialized["latencyMs"] == "9.5"
    assert serialized["timeoutFlag"] == "0"


def test_serialize_row_marks_beginner_concept_match_as_good():
    module = _load_script()
    row = {
        "query": "CNN을 쉽게 설명해줘",
        "source": "paper",
        "expected_family": "concept_explainer",
        "expected_top1_or_set": "alexnet-2012|ImageNet Classification with Deep Convolutional Neural Networks",
        "expected_answer_mode": "representative_paper_explainer_beginner",
        "allowed_fallback": "planner_retry",
    }
    payload = {
        "status": "ok",
        "paperFamily": "concept_explainer",
        "queryPlan": {"family": "concept_explainer"},
        "queryFrame": {"family": "concept_explainer", "answer_mode": "representative_paper_explainer_beginner"},
        "representativePaper": {
            "paperId": "alexnet-2012",
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        },
        "answerSignals": {
            "representative_selection": {
                "score": 7.25,
                "titleHits": 2,
                "reason": "resolved_source_and_title_match",
            }
        },
        "v2": {"runtimeExecution": {"used": "ask_v2", "fallbackReason": ""}},
        "evidencePolicy": {"policyKey": "concept_explainer_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "representative_paper_explainer_beginner", "runtimeUsed": "legacy"},
        "sources": [{"title": "ImageNet Classification with Deep Convolutional Neural Networks"}],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=12.5)

    assert serialized["family_match"] == "1"
    assert serialized["answer_mode_match"] == "1"
    assert serialized["representative_match"] == "1"
    assert serialized["actual_representative_selection_reason"] == "resolved_source_and_title_match"
    assert serialized["pred_label"] == "good"
    assert serialized["pred_reason"] == "family_and_mode_match"
    assert serialized["actual_runtime_used"] == "ask_v2"
    assert serialized["actual_fallback_reason"] == ""
    assert serialized["gate_mode"] == "standard"
    assert serialized["timeout_flag"] == "0"
    assert serialized["v2_verification_status"] == ""
    assert serialized["ask_v2_hard_gate"] == "0"
    assert serialized["selected_card_count"] == "0"
    assert serialized["filtered_evidence_count"] == "1"


def test_serialize_row_marks_compare_no_result_partial_when_need_multiple_papers_allowed():
    """paper_default_eval compare queries allow need_multiple_papers → no_result must not be 'bad'."""
    module = _load_script()
    row = {
        "query": "RAG와 FiD를 비교해줘",
        "source": "paper",
        "expected_family": "paper_compare",
        "expected_top1_or_set": "2005.11401|2007.01282",
        "expected_answer_mode": "paper_comparison",
        "allowed_fallback": "need_multiple_papers",
    }
    payload = {
        "status": "no_result",
        "paperFamily": "paper_compare",
        "queryPlan": {"family": "paper_compare"},
        "queryFrame": {"family": "paper_compare", "answer_mode": "paper_comparison"},
        "representativePaper": {},
        "answerSignals": {},
        "v2": {"runtimeExecution": {"used": "ask_v2", "fallbackReason": "insufficient_compare_anchor_coverage"}},
        "evidencePolicy": {"policyKey": "paper_compare_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "paper_comparison", "runtimeUsed": "ask_v2"},
        "sources": [],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=1.0)

    assert serialized["no_result"] == "1"
    assert serialized["actual_runtime_used"] == "ask_v2"
    assert serialized["actual_fallback_reason"] == "insufficient_compare_anchor_coverage"
    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "allowed_no_result_fallback"


def test_serialize_row_records_ask_v2_hard_gate_diagnostics_without_changing_judgment():
    module = _load_script()
    row = {
        "query": "RAG와 FiD를 비교해줘",
        "source": "paper",
        "expected_family": "paper_compare",
        "expected_top1_or_set": "2005.11401|2007.01282",
        "expected_answer_mode": "paper_comparison",
        "allowed_fallback": "need_multiple_papers",
    }
    payload = {
        "status": "no_result",
        "paperFamily": "paper_compare",
        "queryFrame": {"family": "paper_compare", "answer_mode": "paper_comparison"},
        "representativePaper": {},
        "answerSignals": {},
        "v2": {
            "routing": {"selected_card_ids": ["paper-card:rag", "paper-card:fid"]},
            "cardSelection": {
                "resolvedPaperIds": ["2005.11401", "2007.01282"],
                "selectionStage": "compare_focus_form | compare_resolved_paper_id",
                "selectionReason": "matched_focus_form:RAG | matched_resolved_paper_id:2007.01282",
                "candidateCountBeforeRerank": 5,
                "candidateCountAfterRerank": 4,
                "resolvedPairPreserved": True,
                "selected": [
                    {
                        "cardId": "paper-card:rag",
                        "sourceId": "2005.11401",
                        "paperId": "2005.11401",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "selectionStage": "compare_focus_form",
                        "selectionReason": "matched_focus_form:RAG",
                    },
                    {
                        "cardId": "paper-card:fid",
                        "sourceId": "2007.01282",
                        "paperId": "2007.01282",
                        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                        "selectionStage": "compare_resolved_paper_id",
                        "selectionReason": "matched_resolved_paper_id:2007.01282",
                    },
                ]
            },
            "runtimeExecution": {"used": "ask_v2", "fallbackReason": "section_blocked_to_claim_cards"},
            "evidenceVerification": {
                "verificationStatus": "weak",
                "unsupportedFields": ["compare_axis_gap"],
                "anchorIdsUsed": ["anchor-rag", "anchor-fid"],
            },
            "claimVerification": [
                {
                    "status": "unsupported",
                    "verdict": "unsupported",
                    "reasons": ["no_anchor_backed_evidence"],
                }
            ],
            "consensus": {"supportCount": 1, "weakClaimCount": 2, "unsupportedClaimCount": 1},
            "preHardGateAnswerable": True,
            "preHardGateReason": "substantive_evidence_found",
            "v2ConsensusUnsupportedClaimCount": 1,
            "v2ConsensusWeakClaimCount": 2,
            "v2ConsensusSupportedClaimCount": 1,
            "v2ClaimReasonSummary": "unsupported:no_anchor_backed_evidence=1",
            "fallback": {"used": True, "reason": "unsupported_v2_fields:compare_axis_gap"},
        },
        "evidencePacket": {
            "askV2HardGate": True,
            "answerableDecisionReason": "unsupported_v2_fields:compare_axis_gap",
            "uniquePaperCount": 2,
        },
        "evidencePolicy": {"policyKey": "paper_compare_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "paper_comparison", "runtimeUsed": "ask_v2"},
        "sources": [
            {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "paper_id": "2005.11401"},
            {"title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "paper_id": "2007.01282"},
        ],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=1.0)

    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "allowed_no_result_fallback"
    assert serialized["v2_verification_status"] == "weak"
    assert serialized["ask_v2_hard_gate"] == "1"
    assert serialized["answerable_decision_reason"] == "unsupported_v2_fields:compare_axis_gap"
    assert serialized["preHardGateAnswerable"] == "1"
    assert serialized["preHardGateReason"] == "substantive_evidence_found"
    assert serialized["v2ConsensusUnsupportedClaimCount"] == "1"
    assert serialized["v2ConsensusWeakClaimCount"] == "2"
    assert serialized["v2ConsensusSupportedClaimCount"] == "1"
    assert serialized["v2ClaimReasonSummary"] == "unsupported:no_anchor_backed_evidence=1"
    assert serialized["unsupported_fields"] == "compare_axis_gap"
    assert serialized["unsupported_claim_count"] == "1"
    assert serialized["weak_claim_count"] == "2"
    assert serialized["selected_card_count"] == "2"
    assert serialized["filtered_evidence_count"] == "2"
    assert serialized["compare_unique_paper_count"] == "2"
    assert serialized["compare_anchor_coverage"] == "unique_papers=2;anchors=2;selected_cards=2"
    assert serialized["hard_gate_reason"] == "unsupported_v2_fields:compare_axis_gap"
    assert serialized["resolved_paper_ids"] == "2005.11401 | 2007.01282"
    assert serialized["selected_card_ids"] == "paper-card:rag | paper-card:fid"
    assert serialized["selected_card_titles"] == (
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | "
        "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
    )
    assert serialized["selected_card_paper_ids"] == "2005.11401 | 2007.01282"
    assert serialized["selected_card_stage"] == "compare_focus_form | compare_resolved_paper_id"
    assert serialized["selection_reason"] == "matched_focus_form:RAG | matched_resolved_paper_id:2007.01282"
    assert serialized["candidate_count_before_rerank"] == "5"
    assert serialized["candidate_count_after_rerank"] == "4"
    assert serialized["resolved_pair_preserved"] == "1"
    assert "evidenceVerification" in serialized["v2_diagnostics_keys"]


def test_serialize_row_leaves_diagnostic_columns_empty_when_v2_is_missing():
    module = _load_script()
    row = {
        "query": "RAG 관련 논문 찾아줘",
        "source": "paper",
        "expected_family": "paper_discover",
        "expected_top1_or_set": "",
        "expected_answer_mode": "paper_shortlist",
        "allowed_fallback": "no_result",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_discover",
        "queryFrame": {"family": "paper_discover", "answer_mode": "paper_shortlist"},
        "representativePaper": {},
        "answerSignals": {},
        "evidencePolicy": {"policyKey": "paper_discover_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "paper_shortlist", "runtimeUsed": "legacy"},
        "sources": [],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=1.0)

    assert serialized["pred_label"] == "good"
    assert serialized["v2_verification_status"] == ""
    assert serialized["ask_v2_hard_gate"] == "0"
    assert serialized["selected_card_count"] == "0"
    assert serialized["filtered_evidence_count"] == "0"
    assert serialized["compare_unique_paper_count"] == ""
    assert serialized["resolved_paper_ids"] == ""
    assert serialized["selected_card_ids"] == ""
    assert serialized["selected_card_titles"] == ""
    assert serialized["selected_card_paper_ids"] == ""
    assert serialized["candidate_count_before_rerank"] == ""
    assert serialized["candidate_count_after_rerank"] == ""
    assert serialized["resolved_pair_preserved"] == ""
    assert serialized["v2_diagnostics_keys"] == ""
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
    assert serialized["latencyMs"] == "1.0"
    assert serialized["timeoutFlag"] == "0"


def test_error_row_adds_latency_and_live_generation_diagnostics_without_changing_judgment():
    module = _load_script()
    row = {
        "query": "AlexNet 논문 요약해줘",
        "source": "paper",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "alexnet-2012",
        "expected_answer_mode": "paper_summary",
        "allowed_fallback": "",
    }

    serialized = module._error_row(
        row,
        top_k=6,
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


def test_serialize_row_marks_representative_drift_as_partial():
    module = _load_script()
    row = {
        "query": "Transformer의 핵심 아이디어를 설명해줘",
        "source": "paper",
        "expected_family": "concept_explainer",
        "expected_top1_or_set": "1706.03762|Attention Is All You Need",
        "expected_answer_mode": "representative_paper_explainer",
        "allowed_fallback": "planner_retry",
    }
    payload = {
        "status": "ok",
        "paperFamily": "concept_explainer",
        "queryPlan": {"family": "concept_explainer"},
        "queryFrame": {"family": "concept_explainer", "answer_mode": "representative_paper_explainer"},
        "representativePaper": {
            "paperId": "2010.11929",
            "title": "An Image is Worth 16x16 Words",
        },
        "answerSignals": {
            "representative_selection": {
                "score": 4.1,
                "titleHits": 0,
                "reason": "retrieval_score_lead",
            }
        },
        "v2": {"runtimeExecution": {"used": "legacy", "fallbackReason": "ask_v2_not_used"}},
        "evidencePolicy": {"policyKey": "concept_explainer_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "representative_paper_explainer", "runtimeUsed": "legacy"},
        "sources": [{"title": "An Image is Worth 16x16 Words"}],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=15.0)

    assert serialized["family_match"] == "1"
    assert serialized["answer_mode_match"] == "1"
    assert serialized["representative_match"] == "0"
    assert serialized["actual_runtime_used"] == "legacy"
    assert serialized["actual_fallback_reason"] == "ask_v2_not_used"
    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "representative_mismatch"


def test_select_queries_for_live_smoke_keeps_only_two_blocking_queries():
    module = _load_script()
    rows = [
        {"query": "CNN을 쉽게 설명해줘", "expected_family": "concept_explainer"},
        {"query": "AlexNet 논문 요약해줘", "expected_family": "paper_lookup"},
        {"query": "RAG 관련 논문 찾아줘", "expected_family": "paper_discover"},
    ]

    selected = module._select_queries_for_gate(rows, gate_mode="live_smoke")

    assert [item["query"] for item in selected] == [
        "CNN을 쉽게 설명해줘",
        "AlexNet 논문 요약해줘",
    ]


def test_select_queries_for_gate_applies_family_filter_before_gate_mode():
    module = _load_script()
    rows = [
        {"query": "CNN을 쉽게 설명해줘", "expected_family": "concept_explainer"},
        {"query": "AlexNet 논문 요약해줘", "expected_family": "paper_lookup"},
        {"query": "RAG 관련 논문 찾아줘", "expected_family": "paper_discover"},
    ]

    selected = module._select_queries_for_gate(rows, gate_mode="standard", family_filter="concept_explainer,paper_discover")

    assert [item["query"] for item in selected] == [
        "CNN을 쉽게 설명해줘",
        "RAG 관련 논문 찾아줘",
    ]


def test_gate_mode_defaults_force_stub_hard_and_live_smoke_defaults():
    module = _load_script()

    assert module._gate_mode_defaults(gate_mode="stub_hard", stub_llm=False, timeout_seconds=0) == (True, 20)
    assert module._gate_mode_defaults(gate_mode="live_smoke", stub_llm=True, timeout_seconds=0) == (False, 60)


def test_serialize_row_marks_citation_support_match_when_targets_align_to_sources():
    module = _load_script()
    row = {
        "query": "AlexNet 논문 요약해줘",
        "source": "paper",
        "expected_family": "paper_lookup",
        "expected_top1_or_set": "alexnet-2012",
        "expected_answer_mode": "paper_summary",
        "allowed_fallback": "",
    }
    payload = {
        "status": "ok",
        "paperFamily": "paper_lookup",
        "queryPlan": {"family": "paper_lookup"},
        "queryFrame": {"family": "paper_lookup", "answer_mode": "paper_summary"},
        "representativePaper": {"paperId": "alexnet-2012", "title": "ImageNet Classification with Deep Convolutional Neural Networks"},
        "answerSignals": {},
        "v2": {"runtimeExecution": {"fallbackReason": ""}},
        "evidencePolicy": {"policyKey": "paper_lookup_policy"},
        "plannerFallback": {"attempted": False, "used": False, "reason": ""},
        "familyRouteDiagnostics": {"answerMode": "paper_summary", "runtimeUsed": "ask_v2"},
        "sources": [{"title": "ImageNet Classification with Deep Convolutional Neural Networks", "citation_target": "alexnet-2012"}],
        "citations": [{"target": "alexnet-2012"}],
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=9.5)

    assert serialized["citation_count"] == "1"
    assert serialized["citation_support_match"] == "1"







def test_live_smoke_summary_splits_route_and_answer_acceptance():
    module = _load_script()

    summary = module._build_live_smoke_summary(
        [
            {
                "pred_label": "good",
                "no_result": "0",
                "ask_v2_hard_gate": "0",
                "resolved_pair_preserved": "",
                "answerGenerationFallbackUsed": "1",
                "generationFallbackUsed": "1",
                "conservativeFallbackApplied": "1",
                "answerGenerationErrorType": "ReadTimeout",
                "answerGenerationWarning": "answer generation fallback applied: initial_answer:ReadTimeout",
                "generationFallbackReason": "initial_answer:ReadTimeout",
                "timeoutFlag": "0",
                "timeout_flag": "0",
                "latencyMs": "60000",
                "citationTracePresent": "1",
            },
            {
                "pred_label": "good",
                "no_result": "0",
                "ask_v2_hard_gate": "0",
                "resolved_pair_preserved": "1",
                "answerGenerationFallbackUsed": "0",
                "generationFallbackUsed": "0",
                "conservativeFallbackApplied": "0",
                "answerGenerationErrorType": "",
                "answerGenerationWarning": "",
                "generationFallbackReason": "",
                "timeoutFlag": "0",
                "timeout_flag": "0",
                "latencyMs": "30000",
                "citationTracePresent": "1",
            },
        ]
    )

    assert summary["rowCount"] == 2
    assert summary["generationFallbackCount"] == 1
    assert summary["generationFallbackRate"] == 0.5
    assert summary["conservativeFallbackCount"] == 1
    assert summary["readTimeoutCount"] == 1
    assert summary["timeoutCount"] == 0
    assert summary["p95LatencyMs"] == 60000
    assert summary["maxLatencyMs"] == 60000
    assert summary["citationTracePresentCount"] == 2
    assert summary["unsupportedClaimsJudgmentCount"] == 0
    assert summary["structuralRouteReady"] is True
    assert summary["routeAcceptanceReady"] is True
    assert summary["answerAcceptanceReady"] is False
    assert summary["answerAcceptanceBlockers"] == [
        "generation_fallback",
        "read_timeout",
        "p95_latency_gt_45000",
    ]
