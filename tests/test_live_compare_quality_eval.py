from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("eval/knowledgeos/scripts/check_live_compare_quality_eval.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("check_live_compare_quality_eval", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(*, status: str = "ok", compare_packet: dict | None = None, citations: list[dict] | None = None):
    return {
        "schema": "knowledge-hub.compare.result.v1",
        "status": status,
        "query": "compare source A and source B",
        "answer": "Compared.",
        "comparePacket": compare_packet or {},
        "trace": {
            "citations": citations if citations is not None else [{"label": "S1", "source_id": "paper:a#0"}],
            "evidenceSpans": [{"span_id": "span_a", "source_id": "paper:a#0"}],
        },
        "citations": citations if citations is not None else [{"label": "S1", "source_id": "paper:a#0"}],
        "sources": [{"source_id": "paper:a#0"}, {"source_id": "paper:b#0"}],
        "warnings": [],
    }


def _compare_packet(*, source_type: str = "paper", status: str = "conflict"):
    return {
        "schema": "knowledge-hub.compare-packet.v1",
        "packet_id": "cmp_1",
        "query": "compare source A and source B",
        "dimensions": [
            {
                "dimensionId": "dim_metric",
                "label": "accuracy",
                "leftClaim": "A reports higher accuracy.",
                "rightClaim": "B reports lower accuracy.",
                "comparisonStatus": status,
                "supportingSpans": [
                    {"spanRef": "span_a", "sourceId": "paper:a#0", "sourceType": source_type, "quote": "A accuracy"},
                    {"spanRef": "span_b", "sourceId": "paper:b#0", "sourceType": source_type, "quote": "B accuracy"},
                ],
            }
        ],
        "coverage": {
            "dimensionCount": 1,
            "supportingSpanCount": 2,
            "excludedNonEvidenceSpanCount": 0,
            "answerable": True,
        },
    }


def test_live_compare_quality_eval_passes_expected_compare_packet():
    module = _load_module()
    case = {
        "case_id": "compare_accuracy",
        "query": "compare source A and source B",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
    }

    result = module.evaluate_case(case, _payload(compare_packet=_compare_packet()))

    assert result["status"] == "pass"
    assert result["comparePacketPresent"] is True
    assert result["expectedSourceCoverage"] == 1.0
    assert result["dimensionCoverage"] == 1.0
    assert result["traceCitationCoverage"] == 1.0


def test_live_compare_quality_eval_fails_missing_compare_packet():
    module = _load_module()
    case = {
        "case_id": "compare_missing_packet",
        "query": "compare",
        "expected_min_supporting_span_count": 1,
    }

    result = module.evaluate_case(case, _payload(compare_packet={}))

    assert result["status"] == "fail"
    assert "compare_packet_missing" in result["errors"]
    assert "supporting_span_count_below_min:0<1" in result["errors"]


def test_live_compare_quality_eval_blocks_non_evidence_supporting_spans():
    module = _load_module()
    case = {
        "case_id": "compare_signal_leak",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
        "forbidden_non_evidence_support": True,
    }

    result = module.evaluate_case(case, _payload(compare_packet=_compare_packet(source_type="learning_edge")))

    assert result["status"] == "fail"
    assert result["nonEvidenceLeakCount"] == 2
    assert "non_evidence_supporting_span_leak" in result["errors"]


def test_live_compare_quality_eval_requires_expected_sources_in_supporting_spans():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"] = [
        {"spanRef": "span_a", "sourceId": "paper:a#0", "sourceType": "paper", "quote": "A accuracy"}
    ]
    case = {
        "case_id": "compare_one_sided_packet",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 1,
    }

    result = module.evaluate_case(case, _payload(compare_packet=packet))

    assert result["status"] == "fail"
    assert result["expectedSourceCoverage"] == 0.5
    assert result["payloadSourceIds"] == ["paper:a#0", "paper:b#0"]
    assert "expected_source_coverage_incomplete" in result["errors"]


def test_live_compare_quality_eval_summary_enforces_thresholds():
    module = _load_module()
    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "pass",
                "query": "pass",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
            },
            {
                "case_id": "fail",
                "query": "fail",
                "expected_source_ids": ["paper:a#0"],
                "expected_min_supporting_span_count": 1,
            },
        ],
        compare_runner=lambda case: _payload(compare_packet=_compare_packet()) if case["case_id"] == "pass" else _payload(compare_packet={}),
        cases_path=Path("cases.json"),
        min_cases=2,
        min_compare_packet_present_rate=1.0,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "failed"
    assert payload["comparePacketPresentRate"] == 0.5
    assert payload["failedCaseCount"] == 1
    assert "compare_packet_present_rate_below_threshold:0.5<1.0" in payload["errors"]


def test_live_compare_quality_eval_skips_empty_local_cases_without_fail_flag():
    module = _load_module()
    payload = module.run_live_compare_quality_eval(
        cases=[],
        compare_runner=lambda _case: {},
        cases_path=Path("missing.json"),
        min_cases=1,
        min_compare_packet_present_rate=1.0,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=False,
    )

    assert payload["status"] == "skipped"
    assert payload["errors"] == ["insufficient_evaluable_cases:0/1"]


def test_live_compare_quality_eval_ignores_template_cases(tmp_path):
    module = _load_module()
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
        {
          "cases": [
            {"case_id": "template_case", "template": true, "query": "replace me"},
            {"case_id": "real_case", "query": "compare real sources"}
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = module._read_cases(cases_path)

    assert [case["case_id"] for case in cases] == ["real_case"]
