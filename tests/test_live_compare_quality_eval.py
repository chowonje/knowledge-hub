from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path("eval/knowledgeos/scripts/check_live_compare_quality_eval.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("check_live_compare_quality_eval", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ConfigWithPapersDir:
    def __init__(self, papers_dir: Path):
        self.papers_dir = str(papers_dir)

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        if tuple(args) == ("storage", "papers_dir"):
            return self.papers_dir
        return default


def _write_manifest(path: Path, artifacts: list[dict]) -> Path:
    path.write_text(
        json.dumps({"schema": "knowledge-hub.corpus-manifest.v1", "artifacts": artifacts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _payload(
    *,
    status: str = "ok",
    compare_packet: dict | None = None,
    citations: list[dict] | None = None,
    sources: list[dict] | None = None,
):
    source_items = sources if sources is not None else [{"source_id": "paper:a#0"}, {"source_id": "paper:b#0"}]
    return {
        "schema": "knowledge-hub.compare.result.v1",
        "status": status,
        "query": "compare source A and source B",
        "answer": "Compared.",
        "comparePacket": compare_packet or {},
        "trace": {
            "citations": citations if citations is not None else [{"label": "S1", "source_id": "paper:a#0"}],
            "evidenceSpans": [{"span_id": "span_a", "source_id": "paper:a#0"}],
            "sources": source_items,
        },
        "citations": citations if citations is not None else [{"label": "S1", "source_id": "paper:a#0"}],
        "sources": source_items,
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
                        "sourceContentHash": "sha256:a-source",
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
                        "sourceContentHash": "sha256:b-source",
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
            "sourceContentHash": "sha256:a-source",
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


def test_live_compare_quality_eval_recomputes_strict_spans_from_source_provenance():
    module = _load_module()
    packet = _compare_packet()
    for span in packet["dimensions"][0]["supportingSpans"]:
        span.pop("sourceContentHash", None)
    case = {
        "case_id": "compare_legacy_strict_flags",
        "query": "compare",
        "expected_source_ids": ["paper:a#0", "paper:b#0"],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
    }

    result = module.evaluate_case(case, _payload(compare_packet=packet))

    assert result["status"] == "fail"
    assert result["expectedSourceCoverage"] == 1.0
    assert result["expectedStrictSourceCoverage"] == 0.0
    assert result["strictSpanBackedCount"] == 0
    assert "expected_strict_source_coverage_below_min" in result["errors"]


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
            "sourceContentHash": "sha256:a-source",
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


def test_live_compare_quality_eval_resolves_expected_source_aliases_from_payload_metadata():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"] = [
        {
            "spanRef": "span_resnet",
            "sourceId": "1512.03385",
            "sourceType": "paper",
            "contentHash": "sha256:a",
            "sourceContentHash": "sha256:a-source",
            "spanLocator": "chars:1-20",
            "strictSpanBacked": True,
            "fallbackSpan": False,
            "quote": "ResNet accuracy",
        },
        {
            "spanRef": "span_vit",
            "sourceId": "2010.11929",
            "sourceType": "paper",
            "contentHash": "sha256:b",
            "sourceContentHash": "sha256:b-source",
            "spanLocator": "chars:30-50",
            "strictSpanBacked": True,
            "fallbackSpan": False,
            "quote": "ViT accuracy",
        },
    ]
    case = {
        "case_id": "compare_title_aliases",
        "query": "compare",
        "expected_source_ids": [
            "Deep_Residual_Learning_for_Image_Recognition",
            "An Image is Worth 16x16 Words",
        ],
        "expected_dimension_terms": ["accuracy"],
        "expected_statuses": ["conflict"],
        "expected_min_supporting_span_count": 2,
        "expected_min_strict_span_count": 2,
    }

    result = module.evaluate_case(
        case,
        _payload(
            compare_packet=packet,
            sources=[
                {"source_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition"},
                {"source_id": "2010.11929", "title": "An Image is Worth 16x16 Words"},
            ],
        ),
    )

    assert result["status"] == "pass"
    assert result["expectedSourceCoverage"] == 1.0
    assert result["expectedStrictSourceCoverage"] == 1.0
    assert result["aliasResolvedExpectedSourceIds"] == [
        "Deep_Residual_Learning_for_Image_Recognition",
        "An Image is Worth 16x16 Words",
    ]
    assert "alias_resolved_expected_source" in result["provenanceDiagnostics"]


def test_live_compare_quality_eval_does_not_resolve_sources_from_snippet_hashes():
    module = _load_module()
    source_hash = "sha256:" + ("a" * 64)
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"] = [
        {
            "spanRef": "span_hash",
            "sourceId": "opaque-source",
            "sourceType": "paper",
            "contentHash": source_hash,
            "spanLocator": "chars:1-20",
            "strictSpanBacked": True,
            "fallbackSpan": False,
            "quote": "A accuracy",
        }
    ]
    case = {
        "case_id": "compare_snippet_hash_alias",
        "query": "compare",
        "expected_source_ids": [source_hash],
        "expected_dimension_terms": ["accuracy"],
        "expected_min_supporting_span_count": 1,
        "expected_min_strict_span_count": 1,
    }

    result = module.evaluate_case(case, _payload(compare_packet=packet, sources=[]))

    assert result["status"] == "fail"
    assert result["expectedSourceCoverage"] == 0.0
    assert result["expectedStrictSourceCoverage"] == 0.0
    assert result["strictSpanBackedCount"] == 0
    assert "expected_source_coverage_incomplete" in result["errors"]


def test_live_compare_quality_eval_does_not_overmatch_similar_source_aliases():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"] = [
        {
            "spanRef": "span_other",
            "sourceId": "2501.00001",
            "sourceType": "paper",
            "contentHash": "sha256:a",
            "sourceContentHash": "sha256:a-source",
            "spanLocator": "chars:1-20",
            "strictSpanBacked": True,
            "fallbackSpan": False,
            "quote": "A different deep learning paper",
        }
    ]
    case = {
        "case_id": "compare_similar_title_aliases",
        "query": "compare",
        "expected_source_ids": ["Deep Residual Learning for Image Recognition"],
        "expected_dimension_terms": ["accuracy"],
        "expected_min_supporting_span_count": 1,
        "expected_min_strict_span_count": 1,
    }

    result = module.evaluate_case(
        case,
        _payload(
            compare_packet=packet,
            sources=[{"source_id": "2501.00001", "title": "Deep Learning for Recognition"}],
        ),
    )

    assert result["status"] == "fail"
    assert result["expectedSourceCoverage"] == 0.0
    assert result["unresolvedExpectedSourceAliases"] == ["Deep Residual Learning for Image Recognition"]
    assert "unresolved_expected_source_alias" in result["provenanceDiagnostics"]


def test_live_compare_quality_eval_requires_strict_source_coverage_for_answerable_cases():
    module = _load_module()
    packet = _compare_packet()
    packet["dimensions"][0]["supportingSpans"][1] = {
        "spanRef": "span_b",
        "sourceId": "paper:b#0",
        "sourceType": "paper",
        "contentHash": "sha256:b",
        "sourceContentHash": "sha256:b-source",
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
        "sourceContentHash": "sha256:b-source",
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


def test_live_compare_quality_eval_skips_case_for_missing_corpus_requirement(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "missing_artifact",
                "sourceIds": ["paper:a#0"],
                "expectedFilename": "missing.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "local_corpus",
            }
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "missing",
                "query": "missing corpus",
                "corpusRequirements": [{"artifactId": "missing_artifact"}],
            }
        ],
        compare_runner=lambda _case: (_ for _ in ()).throw(AssertionError("compare should not run")),
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=1.0,
        min_answerable_rate=1.0,
        min_expected_source_coverage_rate=1.0,
        min_expected_answerable_strict_source_coverage_rate=1.0,
        min_dimension_coverage_rate=1.0,
        min_supporting_span_coverage_rate=1.0,
        min_trace_citation_coverage_rate=1.0,
        fail_on_insufficient=False,
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
    )

    case = payload["cases"][0]
    assert case["status"] == "skipped"
    assert case["skipReason"] == "skipped_missing_corpus"
    assert case["errors"] == []
    assert case["corpusRequirements"][0]["status"] == "missing_artifact"
    assert case["corpusRequirements"][0]["searchedPaths"] == ["papers_dir/missing.pdf"]
    assert payload["evaluatedCaseCount"] == 0
    assert payload["skippedForMissingCorpus"] == 1


def test_live_compare_quality_eval_accepts_repo_fixture_requirement(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    fixture = tmp_path / "fixtures" / "fixture.txt"
    fixture.parent.mkdir()
    fixture.write_text("strict fixture evidence", encoding="utf-8")
    fixture_hash = "sha256:" + hashlib.sha256(fixture.read_bytes()).hexdigest()
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "fixture_artifact",
                "sourceIds": ["paper:a#0"],
                "fixturePath": "fixtures/fixture.txt",
                "expectedSourceContentHash": fixture_hash,
                "corpusTier": "repo_fixture",
            }
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "fixture_present",
                "query": "fixture present",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
                "corpusRequirements": [{"artifactId": "fixture_artifact", "minOffsetsRequired": True}],
            }
        ],
        compare_runner=lambda _case: _payload(compare_packet=_compare_packet()),
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
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
    )

    case = payload["cases"][0]
    assert payload["status"] == "ok"
    assert payload["evaluatedCaseCount"] == 1
    assert case["corpusRequirements"][0]["status"] == "ok"
    assert case["corpusRequirements"][0]["path"] == "repo_fixture/fixtures/fixture.txt"
    assert "resolvedPath" not in json.dumps(case["corpusRequirements"])


def test_live_compare_quality_eval_fails_missing_repo_fixture_requirement(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "fixture_artifact",
                "sourceIds": ["paper:a#0"],
                "expectedFilename": "fixture.txt",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "repo_fixture",
            }
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "fixture_missing",
                "query": "fixture missing",
                "corpusRequirements": [{"artifactId": "fixture_artifact"}],
            }
        ],
        compare_runner=lambda _case: (_ for _ in ()).throw(AssertionError("compare should not run")),
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=0.0,
        min_answerable_rate=0.0,
        min_expected_source_coverage_rate=0.0,
        min_expected_answerable_strict_source_coverage_rate=0.0,
        min_dimension_coverage_rate=0.0,
        min_supporting_span_coverage_rate=0.0,
        min_trace_citation_coverage_rate=0.0,
        fail_on_insufficient=True,
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
    )

    case = payload["cases"][0]
    assert case["status"] == "fail"
    assert case["skipReason"] == ""
    assert case["errors"] == ["corpus_requirement_failed:skipped_missing_corpus"]
    assert payload["failedCaseCount"] == 1
    assert payload["skippedCaseCount"] == 0


def test_live_compare_quality_eval_runs_with_missing_optional_local_corpus(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "optional_artifact",
                "sourceIds": ["paper:a#0"],
                "expectedFilename": "optional.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "optional_local_corpus",
            }
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "optional_missing",
                "query": "optional corpus",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
                "corpusRequirements": [{"artifactId": "optional_artifact"}],
            }
        ],
        compare_runner=lambda _case: _payload(compare_packet=_compare_packet()),
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
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
    )

    assert payload["status"] == "ok"
    assert payload["evaluatedCaseCount"] == 1
    assert payload["skippedCaseCount"] == 0
    assert payload["coveragePct"] == 1.0
    assert payload["cases"][0]["corpusRequirements"][0]["status"] == "missing_artifact"


def test_live_compare_quality_eval_summary_reports_corpus_coverage(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    mismatch_file = papers_dir / "mismatch.pdf"
    mismatch_file.write_bytes(b"different")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "missing_artifact",
                "sourceIds": ["paper:missing#0"],
                "expectedFilename": "missing.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "local_corpus",
            },
            {
                "artifactId": "mismatch_artifact",
                "sourceIds": ["paper:mismatch#0"],
                "expectedFilename": "mismatch.pdf",
                "expectedSourceContentHash": "sha256:" + "1" * 64,
                "corpusTier": "local_corpus",
            },
        ],
    )

    def _runner(case):
        if case["case_id"] == "pass":
            return _payload(compare_packet=_compare_packet())
        return _payload(compare_packet={})

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
                "expected_min_supporting_span_count": 1,
            },
            {
                "case_id": "missing",
                "query": "missing corpus",
                "corpusRequirements": [{"artifactId": "missing_artifact"}],
            },
            {
                "case_id": "mismatch",
                "query": "hash mismatch",
                "corpusRequirements": [{"artifactId": "mismatch_artifact"}],
            },
        ],
        compare_runner=_runner,
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=0.0,
        min_answerable_rate=0.0,
        min_expected_source_coverage_rate=0.0,
        min_expected_answerable_strict_source_coverage_rate=0.0,
        min_dimension_coverage_rate=0.0,
        min_supporting_span_coverage_rate=0.0,
        min_trace_citation_coverage_rate=0.0,
        fail_on_insufficient=True,
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
    )

    assert payload["declaredCaseCount"] == 4
    assert payload["status"] == "failed"
    assert "corpus_coverage_rate_below_threshold:0.5<1.0" in payload["errors"]
    assert payload["evaluatedCaseCount"] == 2
    assert payload["evaluableCaseCount"] == 2
    assert payload["passedCaseCount"] == 1
    assert payload["failedCaseCount"] == 1
    assert payload["coveragePct"] == 0.5
    assert payload["coverageReport"] == {
        "passed": 1,
        "evaluable": 2,
        "declared": 4,
        "coverage_pct": 0.5,
        "skipped_for_missing_corpus": 1,
        "skipped_for_hash_mismatch": 1,
        "missing_corpus_requirements": 0,
        "derived_corpus_requirements": 0,
    }
    assert payload["skippedForMissingCorpus"] == 1
    assert payload["skippedForHashMismatch"] == 1
    assert payload["cases"][2]["skipReason"] == "skipped_missing_corpus"
    assert payload["cases"][3]["skipReason"] == "skipped_hash_mismatch"


def test_live_compare_quality_eval_derives_requirements_from_expected_sources(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    paper_a = papers_dir / "a.pdf"
    paper_b = papers_dir / "b.pdf"
    paper_a.write_bytes(b"paper a strict source")
    paper_b.write_bytes(b"paper b strict source")
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "artifact_a",
                "sourceIds": ["paper:a#0"],
                "expectedFilename": "a.pdf",
                "expectedSourceContentHash": "sha256:" + hashlib.sha256(paper_a.read_bytes()).hexdigest(),
                "corpusTier": "local_corpus",
            },
            {
                "artifactId": "artifact_b",
                "sourceIds": ["paper:b#0"],
                "expectedFilename": "b.pdf",
                "expectedSourceContentHash": "sha256:" + hashlib.sha256(paper_b.read_bytes()).hexdigest(),
                "corpusTier": "local_corpus",
            },
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "derived_present",
                "query": "derived corpus present",
                "expected_source_ids": ["paper:a#0", "paper:b#0"],
                "expected_dimension_terms": ["accuracy"],
                "expected_statuses": ["conflict"],
                "expected_min_supporting_span_count": 2,
            }
        ],
        compare_runner=lambda _case: _payload(compare_packet=_compare_packet()),
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
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
        derive_corpus_requirements=True,
    )

    case = payload["cases"][0]
    assert payload["status"] == "ok"
    assert payload["coveragePct"] == 1.0
    assert payload["derivedCorpusRequirementCount"] == 2
    assert payload["missingCorpusRequirementCount"] == 0
    assert [item["artifactId"] for item in case["corpusRequirements"]] == ["artifact_a", "artifact_b"]
    assert all(item["status"] == "ok" for item in case["corpusRequirements"])
    assert case["derivedCorpusRequirementCount"] == 2


def test_live_compare_quality_eval_derived_missing_artifact_fails_default_corpus_gate(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "missing_artifact",
                "sourceIds": ["paper:a#0"],
                "expectedFilename": "missing.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "local_corpus",
            }
        ],
    )

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "derived_missing",
                "query": "derived corpus missing",
                "expected_source_ids": ["paper:a#0"],
            }
        ],
        compare_runner=lambda _case: (_ for _ in ()).throw(AssertionError("compare should not run")),
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=0.0,
        min_answerable_rate=0.0,
        min_expected_source_coverage_rate=0.0,
        min_expected_answerable_strict_source_coverage_rate=0.0,
        min_dimension_coverage_rate=0.0,
        min_supporting_span_coverage_rate=0.0,
        min_trace_citation_coverage_rate=0.0,
        fail_on_insufficient=True,
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
        derive_corpus_requirements=True,
    )

    case = payload["cases"][0]
    assert payload["status"] == "failed"
    assert payload["coveragePct"] == 0.0
    assert payload["skippedForMissingCorpus"] == 1
    assert payload["derivedCorpusRequirementCount"] == 1
    assert "corpus_coverage_rate_below_threshold:0.0<1.0" in payload["errors"]
    assert case["status"] == "skipped"
    assert case["skipReason"] == "skipped_missing_corpus"
    assert case["corpusRequirements"][0]["artifactId"] == "missing_artifact"


def test_live_compare_quality_eval_missing_manifest_mapping_is_not_green(tmp_path):
    module = _load_module()
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    manifest_path = _write_manifest(tmp_path / "manifest.json", [])

    payload = module.run_live_compare_quality_eval(
        cases=[
            {
                "case_id": "missing_requirement",
                "query": "missing requirement mapping",
                "expected_source_ids": ["paper:unknown#0"],
            }
        ],
        compare_runner=lambda _case: (_ for _ in ()).throw(AssertionError("compare should not run")),
        cases_path=Path("cases.json"),
        min_cases=1,
        min_compare_packet_present_rate=0.0,
        min_answerable_rate=0.0,
        min_expected_source_coverage_rate=0.0,
        min_expected_answerable_strict_source_coverage_rate=0.0,
        min_dimension_coverage_rate=0.0,
        min_supporting_span_coverage_rate=0.0,
        min_trace_citation_coverage_rate=0.0,
        fail_on_insufficient=True,
        corpus_manifest_path=manifest_path,
        config=_ConfigWithPapersDir(papers_dir),
        derive_corpus_requirements=True,
    )

    case = payload["cases"][0]
    assert payload["status"] == "failed"
    assert payload["missingCorpusRequirementCount"] == 1
    assert "missing_corpus_requirements:1" in payload["errors"]
    assert "failed_cases:1" in payload["errors"]
    assert case["status"] == "fail"
    assert case["errors"] == ["corpus_requirement_failed:missing_corpus_requirement"]
    assert case["missingCorpusRequirements"] == [
        {
            "sourceId": "paper:unknown#0",
            "status": "missing_manifest_entry",
            "reason": "expected source id has no corpus manifest entry",
        }
    ]


def test_live_compare_quality_eval_cli_fails_required_corpus_coverage_gap(tmp_path):
    module = _load_module()
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "missing",
                    "query": "missing corpus",
                    "corpusRequirements": [{"artifactId": "missing_artifact"}],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "artifactId": "missing_artifact",
                "sourceIds": ["paper:missing#0"],
                "expectedFilename": "missing.pdf",
                "expectedSourceContentHash": "sha256:" + "0" * 64,
                "corpusTier": "local_corpus",
            }
        ],
    )

    exit_code = module.main(
        [
            "--cases",
            str(cases_path),
            "--corpus-manifest",
            str(manifest_path),
            "--fail-on-insufficient",
            "--min-cases",
            "1",
            "--json",
        ]
    )

    assert exit_code == 1


def test_live_compare_quality_eval_rejects_unsupported_corpus_manifest_schema(tmp_path):
    module = _load_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": "wrong.schema", "artifacts": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported corpus manifest schema"):
        module.run_live_compare_quality_eval(
            cases=[],
            compare_runner=lambda _case: {},
            cases_path=Path("cases.json"),
            min_cases=1,
            min_compare_packet_present_rate=1.0,
            min_answerable_rate=1.0,
            min_expected_source_coverage_rate=1.0,
            min_expected_answerable_strict_source_coverage_rate=1.0,
            min_dimension_coverage_rate=1.0,
            min_supporting_span_coverage_rate=1.0,
            min_trace_citation_coverage_rate=1.0,
            fail_on_insufficient=False,
            corpus_manifest_path=manifest_path,
            config=_ConfigWithPapersDir(tmp_path / "papers"),
        )


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
