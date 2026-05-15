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
                    {
                        "spanRef": "span_a",
                        "sourceId": "paper:a#0",
                        "sourceType": source_type,
                        "contentHash": "sha256:a",
                        "spanLocator": "chars:1-20",
                        "strictSpanBacked": True,
                        "fallbackSpan": False,
                        "quote": "A accuracy",
                    },
                    {
                        "spanRef": "span_b",
                        "sourceId": "paper:b#0",
                        "sourceType": source_type,
                        "contentHash": "sha256:b",
                        "spanLocator": "chars:30-50",
                        "strictSpanBacked": True,
                        "fallbackSpan": False,
                        "quote": "B accuracy",
                    },
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


def _fallback_only_packet():
    packet = _compare_packet()
    for span in packet["dimensions"][0]["supportingSpans"]:
        span["strictSpanBacked"] = False
        span["fallbackSpan"] = True
    packet["coverage"] = {**packet["coverage"], "answerable": False}
    return packet


def _locator_only_packet():
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"] = [
        {
            "spanRef": "span_a",
            "sourceId": "paper:a#0",
            "sourceType": "paper",
            "contentHash": "sha256:a",
            "spanLocator": "unit:abstract",
            "strictSpanBacked": False,
            "fallbackSpan": False,
            "quote": "A accuracy",
        }
    ]
    packet["coverage"] = {**packet["coverage"], "answerable": False}
    return packet


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
    assert result["strictSpanBackedCount"] == 2
    assert result["strictSpanCoverage"] == 1.0
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


def test_live_compare_quality_eval_accepts_expected_no_answer_status():
    module = _load_module()
    case = {
        "case_id": "compare_expected_no_answer",
        "query": "compare under-evidenced sources",
        "expected_answerable": False,
        "expected_min_supporting_span_count": 0,
        "require_trace_citations": False,
    }

    result = module.evaluate_case(case, _payload(status="insufficient_evidence", compare_packet={}))

    assert "compare_status_not_ok:insufficient_evidence" not in result["errors"]


def test_live_compare_quality_eval_rejects_answerable_no_answer_case():
    module = _load_module()
    case = {
        "case_id": "compare_expected_no_answer",
        "query": "compare under-evidenced sources",
        "expected_answerable": False,
        "expected_min_supporting_span_count": 0,
        "require_trace_citations": False,
    }

    result = module.evaluate_case(case, _payload(compare_packet=_compare_packet()))

    assert result["status"] == "fail"
    assert "compare_packet_unexpectedly_answerable" in result["errors"]


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
        {
            "spanRef": "span_a",
            "sourceId": "paper:a#0",
            "sourceType": "paper",
            "contentHash": "sha256:a",
            "spanLocator": "chars:1-20",
            "strictSpanBacked": True,
            "fallbackSpan": False,
            "quote": "A accuracy",
        }
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


def test_live_compare_quality_eval_requires_strict_source_coverage_for_answerable_cases():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"][1] = {
        "spanRef": "span_b",
        "sourceId": "paper:b#0",
        "sourceType": "paper",
        "contentHash": "sha256:b",
        "spanLocator": "chars:30-50",
        "strictSpanBacked": False,
        "fallbackSpan": True,
        "quote": "B accuracy",
    }
    case = {
        "case_id": "compare_partial_strict_sources",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
        "expected_min_strict_span_count": 1,
        "expected_answerable": True,
    }

    result = module.evaluate_case(case, _payload(compare_packet=packet))

    assert result["status"] == "fail"
    assert result["expectedSourceCoverage"] == 1.0
    assert result["expectedStrictSourceCoverage"] == 0.5
    assert result["strictCoveredExpectedSourceIds"] == ["paper:a#0"]
    assert result["fallbackCoveredExpectedSourceIds"] == ["paper:b#0"]
    assert "expected_strict_source_coverage_below_min" in result["errors"]
    assert "strict_source_coverage_gap" in result["failureCategories"]


def test_live_compare_quality_eval_reports_locator_only_span_diagnostics():
    module = _load_module()
    case = {
        "case_id": "compare_locator_only",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
        "expected_min_strict_span_count": 2,
        "expected_answerable": True,
    }

    result = module.evaluate_case(case, _payload(status="insufficient_evidence", compare_packet=_locator_only_packet()))

    assert result["status"] == "fail"
    assert result["locatorOnlySpanCount"] == 1
    assert result["offsetBackedSpanCount"] == 0
    assert "locator_only_spans_present" in result["provenanceDiagnostics"]
    assert "locator_only_anchor" in result["failureCategories"]


def test_live_compare_quality_eval_fallback_only_answerable_case_has_strict_gap_taxonomy():
    module = _load_module()
    case = {
        "case_id": "compare_fallback_only",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
        "expected_min_strict_span_count": 2,
        "expected_answerable": True,
    }

    result = module.evaluate_case(case, _payload(status="insufficient_evidence", compare_packet=_fallback_only_packet()))

    assert result["status"] == "fail"
    assert result["supportingSpanCoverage"] == 1.0
    assert result["strictSpanBackedCount"] == 0
    assert result["strictSpanCoverage"] == 0.0
    assert result["fallbackSpanCount"] == 2
    assert result["fallbackSpanShare"] == 1.0
    assert result["expectedSourceCoverage"] == 1.0
    assert result["expectedStrictSourceCoverage"] == 0.0
    assert "strict_span_count_below_min:0<2" in result["errors"]
    assert "expected_strict_source_coverage_below_min" in result["errors"]
    assert "compare_packet_not_answerable" in result["errors"]
    assert "compare_status_not_ok:insufficient_evidence" in result["errors"]
    assert "fallback_only_support" in result["provenanceDiagnostics"]
    assert "fallback_only" in result["failureCategories"]


def test_live_compare_quality_eval_respects_expected_min_strict_source_coverage_override():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"][1] = {
        "spanRef": "span_b",
        "sourceId": "paper:b#0",
        "sourceType": "paper",
        "contentHash": "sha256:b",
        "spanLocator": "chars:30-50",
        "strictSpanBacked": False,
        "fallbackSpan": True,
        "quote": "B accuracy",
    }
    packet["coverage"] = {**packet["coverage"], "answerable": False}
    case = {
        "case_id": "compare_strict_source_override",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_answerable": False,
        "expected_min_strict_source_coverage": 0.5,
    }

    result = module.evaluate_case(case, _payload(status="insufficient_evidence", compare_packet=packet))

    assert result["status"] == "pass"
    assert result["expectedStrictSourceCoverage"] == 0.5
    assert result["expectedMinStrictSourceCoverage"] == 0.5
    assert result["strictCoveredExpectedSourceIds"] == ["paper:a#0"]
    assert result["fallbackCoveredExpectedSourceIds"] == ["paper:b#0"]
    assert "expected_strict_source_coverage_below_min" not in result["errors"]


def test_live_compare_quality_eval_reports_unexpected_dimension_status():
    module = _load_module()
    case = {
        "case_id": "compare_unexpected_status",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
    }

    result = module.evaluate_case(case, _payload(compare_packet=_compare_packet(status="supported")))

    assert result["status"] == "fail"
    assert result["dimensionStatuses"] == ["supported"]
    assert "unexpected_dimension_status" in result["errors"]
    assert "unexpected_dimension_status" in result["failureCategories"]


def test_live_compare_quality_eval_reports_trace_citation_gap_with_strict_spans():
    module = _load_module()
    case = {
        "case_id": "compare_trace_gap",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
    }

    result = module.evaluate_case(case, _payload(compare_packet=_compare_packet(), citations=[]))

    assert result["status"] == "fail"
    assert result["traceCitationCount"] == 0
    assert result["traceCitationCoverage"] == 0.0
    assert "trace_citations_missing" in result["errors"]
    assert "citation_gap" in result["failureCategories"]


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
        min_expected_answerable_strict_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "failed"
    assert payload["comparePacketPresentRate"] == 0.5
    assert payload["strictSpanCoverageRate"] == 0.5
    assert payload["fallbackSpanCaseRate"] == 0.0
    assert payload["expectedAnswerableStrictSourceCoverageRate"] == 0.5
    assert payload["failureCategoryCounts"]["compare_packet_missing"] == 1
    assert payload["failureCategoryCounts"]["not_answerable"] == 1
    assert payload["expectedAnswerableCaseCount"] == 2
    assert payload["expectedNoAnswerCaseCount"] == 0
    assert payload["failedCaseCount"] == 1
    assert "compare_packet_present_rate_below_threshold:0.5<1.0" in payload["errors"]


def test_live_compare_quality_eval_summary_separates_expected_no_answer_cases():
    module = _load_module()
    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "answerable",
                "query": "answerable",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
            },
            {
                "case_id": "no_answer",
                "query": "no answer",
                "expected_answerable": False,
                "expected_min_supporting_span_count": 0,
                "require_trace_citations": False,
            },
        ],
        compare_runner=lambda case: _payload(compare_packet=_compare_packet())
        if case["case_id"] == "answerable"
        else _payload(
            status="insufficient_evidence",
            compare_packet={
                **_compare_packet(status="insufficient"),
                "coverage": {
                    **_compare_packet(status="insufficient")["coverage"],
                    "answerable": False,
                },
            },
        ),
        cases_path=Path("cases.json"),
        min_cases=2,
        min_compare_packet_present_rate=0.5,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_expected_answerable_strict_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=0.5,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "ok"
    assert payload["answerableRate"] == 0.5
    assert payload["expectedAnswerablePassRate"] == 1.0
    assert payload["expectedNoAnswerPassRate"] == 1.0


def test_live_compare_quality_eval_summary_reports_provenance_rates():
    module = _load_module()

    def _runner(case):
        if case["case_id"] == "strict":
            return _payload(compare_packet=_compare_packet())
        if case["case_id"] == "fallback":
            return _payload(status="insufficient_evidence", compare_packet=_fallback_only_packet())
        return _payload(status="insufficient_evidence", compare_packet=_locator_only_packet())

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "strict",
                "query": "strict",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
            },
            {
                "case_id": "fallback",
                "query": "fallback",
                "expected_answerable": False,
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
                "expected_min_strict_source_coverage": 0.0,
            },
            {
                "case_id": "locator",
                "query": "locator",
                "expected_answerable": False,
                "expected_source_ids": ["paper:a#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 1,
                "expected_min_strict_source_coverage": 0.0,
            },
        ],
        compare_runner=_runner,
        cases_path=Path("cases.json"),
        min_cases=3,
        min_compare_packet_present_rate=1.0,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_expected_answerable_strict_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "ok"
    assert payload["fallbackSpanCaseRate"] == 0.333333
    assert payload["fallbackOnlyCaseRate"] == 0.333333
    assert payload["locatorOnlyCaseRate"] == 0.333333
    assert payload["provenanceDiagnosticCounts"]["fallback_only_support"] == 1
    assert payload["provenanceDiagnosticCounts"]["locator_only_spans_present"] == 1
    assert payload["provenanceDiagnosticCounts"]["strict_source_coverage_gap"] == 2


def test_live_compare_quality_eval_summary_classifies_case_execution_failures():
    module = _load_module()
    payload = module.run_live_compare_quality_eval(
        cases=[{"case_id": "boom", "query": "compare"}],
        compare_runner=lambda _case: (_ for _ in ()).throw(RuntimeError("boom")),
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=1.0,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_expected_answerable_strict_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=True,
    )

    assert payload["status"] == "failed"
    assert payload["failedCaseCount"] == 1
    assert payload["cases"][0]["errors"] == ["case_execution_failed:boom"]
    assert payload["failureCategoryCounts"]["compare_runtime_failed"] == 1
    assert "failed_cases:1" in payload["errors"]


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
        min_expected_answerable_strict_source_coverage_rate=1.0,
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


def test_live_compare_quality_eval_case_alias_fields_support_wider_live_cases(tmp_path):
    module = _load_module()
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        """
        {
          "cases": [
            {
              "source_type": "paper",
              "query": "compare real sources",
              "expected_source_id": "paper:a#0|paper:b#0",
              "expected_terms": "accuracy, benchmark"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = module._read_cases(cases_path)
    result = module.evaluate_case(cases[0], _payload(compare_packet=_compare_packet()))

    assert cases[0]["case_id"] == "case_1"
    assert result["expectedSourceIds"] == ["paper:a#0", "paper:b#0"]
    assert result["expectedDimensionTerms"] == ["accuracy", "benchmark"]
    assert result["source"] == "paper"
