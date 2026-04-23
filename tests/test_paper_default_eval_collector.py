from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path("/Users/won/Desktop/allinone/knowledge-hub/eval/knowledgeos/scripts/collect_paper_default_eval.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_default_eval_collector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
