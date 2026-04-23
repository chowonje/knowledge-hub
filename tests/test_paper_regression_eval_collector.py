from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path("/Users/won/Desktop/allinone/knowledge-hub/eval/knowledgeos/scripts/collect_paper_regression_eval.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_regression_eval_collector_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        "sources": [{"title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "paper_id": "2310.11511"}],
        "answer": "compare answer",
    }

    serialized = module._serialize_row(row, payload, top_k=6, retrieval_mode="hybrid", latency_ms=15.0)

    assert serialized["family_match"] == "1"
    assert serialized["source_match"] == "0"
    assert serialized["matched_expected_count"] == "0"
    assert serialized["pred_label"] == "partial"
    assert serialized["pred_reason"] == "source_undercoverage"


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
