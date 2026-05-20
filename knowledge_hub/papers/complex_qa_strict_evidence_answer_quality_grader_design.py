"""Report-only strict-evidence answer-quality grader design.

The design consumes the strict-evidence answer-quality dry-run report and
defines deterministic rubric contracts for a later grader.  It does not
generate answers, compute quality scores, call LLMs or judge models, invoke or
change the answer path, search indexes, scan the vault, mutate storage,
reindex, reembed, or create evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_dry_run import (
    COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID,
)


COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-strict-evidence-answer-quality-grader-design.v1"
)
DEFAULT_DRY_RUN_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-strict-evidence-answer-quality-dry-run/"
    "complex-qa-strict-evidence-answer-quality-dry-run.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-strict-evidence-answer-quality-grader-design"
)

NEXT_RECOMMENDED_TRANCHE = "complex QA supplied strict-evidence synthetic grading fixture runner"

COMMON_REQUIRED_GRADER_INPUTS = (
    "question_id",
    "paper_ids",
    "question_category",
    "expected_evidence_type",
    "answerability_expectation",
    "required_evidence_contract",
    "strict_structured_evidence_refs",
    "future_candidate_answer_text",
    "future_answer_citation_map",
)

COMMON_HARD_FAIL_RULES = (
    "expected_no_answer_question_must_not_be_scored",
    "missing_strict_structured_evidence_refs",
    "missing_required_evidence_contract",
    "unsupported_claim_present",
    "citation_missing_or_not_tied_to_strict_evidence",
    "answerability_regression_answered_without_evidence",
    "llm_or_judge_model_used_for_report_only_design",
)

CATEGORY_RUBRICS: dict[str, dict[str, tuple[str, ...]]] = {
    "table_numeric_qa": {
        "required_inputs": (
            "table_id_or_label",
            "row_label",
            "column_label",
            "exact_cell_value",
            "unit_or_denominator",
            "cell_source_content_hash",
            "cell_locator_or_region",
        ),
        "checks": (
            "exact_cell_value_check",
            "unit_or_denominator_check",
            "row_column_provenance_check",
            "no_rounding_drift_check",
        ),
        "hard_fails": (
            "numeric_value_mismatch",
            "missing_unit_or_denominator_when_required",
            "row_column_provenance_mismatch",
            "rounding_without_contract",
        ),
    },
    "equation_citation_qa": {
        "required_inputs": (
            "equation_source_span",
            "equation_label_or_number",
            "equation_citation_anchor",
            "equation_source_content_hash",
            "equation_locator_or_region",
        ),
        "checks": (
            "equation_source_span_check",
            "equation_label_or_number_check",
            "citation_anchor_check",
            "unsupported_interpretation_guard",
        ),
        "hard_fails": (
            "missing_equation_source_span",
            "wrong_equation_label_or_number",
            "missing_citation_anchor",
            "unsupported_equation_interpretation",
        ),
    },
    "figure_caption_qa": {
        "required_inputs": (
            "figure_id_or_label",
            "figure_caption_source_span",
            "figure_page",
            "caption_source_content_hash",
            "caption_locator_or_region",
        ),
        "checks": (
            "figure_caption_source_span_check",
            "figure_id_or_page_check",
            "caption_only_grounding_check",
            "no_visual_claim_beyond_caption_evidence_check",
        ),
        "hard_fails": (
            "missing_caption_source_span",
            "figure_id_or_page_mismatch",
            "visual_claim_not_supported_by_caption",
            "caption_citation_missing",
        ),
    },
    "method_comparison_qa": {
        "required_inputs": (
            "strict_method_span_per_paper",
            "comparison_dimension_contract",
            "paper_id_to_evidence_ref_map",
            "source_content_hash_per_method_span",
            "locator_or_region_per_method_span",
        ),
        "checks": (
            "all_papers_have_strict_method_span_check",
            "symmetric_comparison_dimension_check",
            "cross_paper_citation_coverage_check",
            "unsupported_comparison_guard",
        ),
        "hard_fails": (
            "missing_method_span_for_any_paper",
            "asymmetric_or_missing_comparison_dimension",
            "unsupported_cross_paper_comparison_claim",
            "citation_missing_for_compared_paper",
        ),
    },
    "limitation_qa": {
        "required_inputs": (
            "limitation_section_span",
            "limitation_scope_contract",
            "limitation_source_content_hash",
            "limitation_locator_or_region",
        ),
        "checks": (
            "limitation_section_span_check",
            "limitation_scope_guard",
            "no_overgeneralization_check",
            "citation_coverage_check",
        ),
        "hard_fails": (
            "missing_limitation_section_span",
            "limitation_claim_outside_source_scope",
            "overgeneralized_limitation",
            "citation_missing_for_limitation_claim",
        ),
    },
    "appendix_table_lookup_qa": {
        "required_inputs": (
            "appendix_id_or_label",
            "appendix_table_label",
            "appendix_page",
            "appendix_row_or_cell_provenance",
            "appendix_source_content_hash",
            "appendix_locator_or_region",
        ),
        "checks": (
            "appendix_or_table_label_check",
            "appendix_page_check",
            "cell_or_row_provenance_check",
            "lookup_value_exactness_check",
        ),
        "hard_fails": (
            "missing_appendix_or_table_label",
            "appendix_page_mismatch",
            "missing_cell_or_row_provenance",
            "lookup_value_mismatch",
        ),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item or "").strip()]


def _grader_execution_expectation(row: dict[str, Any]) -> str:
    answerability = str(row.get("answerabilityExpectation") or "")
    if answerability == "expected_no_answer":
        return "not_run_expected_no_answer"
    if row.get("readyForAnswerQualityComparison") is True and _text_list(row.get("structuredEvidenceRefs")):
        return "eligible_for_future_grading"
    return "blocked_missing_strict_evidence"


def _category_rubric(category: str) -> dict[str, tuple[str, ...]]:
    return CATEGORY_RUBRICS.get(
        category,
        {
            "required_inputs": (),
            "checks": ("unknown_question_category_blocker",),
            "hard_fails": ("unknown_question_category",),
        },
    )


def _required_grader_inputs(row: dict[str, Any]) -> list[str]:
    category = str(row.get("questionCategory") or "")
    rubric = _category_rubric(category)
    return list(dict.fromkeys([*COMMON_REQUIRED_GRADER_INPUTS, *rubric["required_inputs"]]))


def _hard_fail_rules(row: dict[str, Any]) -> list[str]:
    category = str(row.get("questionCategory") or "")
    rubric = _category_rubric(category)
    return list(dict.fromkeys([*COMMON_HARD_FAIL_RULES, *rubric["hard_fails"]]))


def _row(row: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("questionCategory") or "")
    rubric = _category_rubric(category)
    execution = _grader_execution_expectation(row)
    return {
        "questionId": str(row.get("questionId") or ""),
        "paperIds": _text_list(row.get("paperIds")),
        "questionCategory": category,
        "expectedEvidenceType": str(row.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": str(row.get("answerabilityExpectation") or ""),
        "readyForAnswerQualityComparison": execution == "eligible_for_future_grading",
        "graderExecutionExpectation": execution,
        "requiredGraderInputs": _required_grader_inputs(row),
        "categorySpecificChecks": list(rubric["checks"]),
        "hardFailRules": _hard_fail_rules(row),
        "structuredEvidenceRefs": _text_list(row.get("structuredEvidenceRefs")),
        "dryRunBlockers": _text_list(row.get("dryRunBlockers")),
        "plannedScoringContract": {
            "scoreComputedNow": False,
            "futureScoreScale": ["pass", "fail"],
            "requiresAllCategorySpecificChecks": True,
            "requiresNoHardFailRulesTriggered": True,
            "requiresCitationCoverageForAllSupportedClaims": True,
            "requiresAbstainForExpectedNoAnswer": True,
        },
        "answerGenerated": False,
        "scoreComputed": False,
        "llmCall": False,
        "judgeModelCall": False,
        "answerPathInvoked": False,
        "unsafeAnswered": False,
        "riskNotes": _as_list(row.get("riskNotes")),
    }


def _semantic_violations(dry_run: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    if dry_run.get("schema") != COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID:
        violations.append("dry_run_report_schema_mismatch")
    question_ids = [str(row.get("questionId") or "") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    for row in rows:
        question_id = str(row.get("questionId") or "")
        if not question_id:
            violations.append("missing_question_id")
        if not row.get("paperIds"):
            violations.append(f"missing_paper_ids:{question_id}")
        category = str(row.get("questionCategory") or "")
        if category not in CATEGORY_RUBRICS:
            violations.append(f"unknown_question_category:{question_id}")
        if (
            row.get("graderExecutionExpectation") == "eligible_for_future_grading"
            and not row.get("structuredEvidenceRefs")
        ):
            violations.append(f"eligible_without_structured_evidence_refs:{question_id}")
        if (
            row.get("answerabilityExpectation") == "expected_no_answer"
            and row.get("graderExecutionExpectation") == "eligible_for_future_grading"
        ):
            violations.append(f"expected_no_answer_marked_eligible:{question_id}")
    return sorted(set(violations))


def build_complex_qa_strict_evidence_answer_quality_grader_design(
    *,
    dry_run_report: str | Path = DEFAULT_DRY_RUN_REPORT,
) -> dict[str, Any]:
    """Build a report-only grader design from a strict-evidence dry-run report."""

    dry_run_path = Path(str(dry_run_report)).expanduser()
    dry_run = _read_json(dry_run_path)
    dry_run_rows = [item for item in _as_list(dry_run.get("rows")) if isinstance(item, dict)]
    rows = [_row(row) for row in dry_run_rows]
    semantic_violations = _semantic_violations(dry_run, rows)
    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_execution = Counter(str(row.get("graderExecutionExpectation") or "") for row in rows)
    by_evidence_type = Counter(str(row.get("expectedEvidenceType") or "") for row in rows)
    ready_rows = by_execution.get("eligible_for_future_grading", 0)
    unsafe_answered_rows = sum(1 for row in rows if row.get("unsafeAnswered") is True)
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and unsafe_answered_rows == 0 else "blocked"
    counts = {
        "questionRows": len(rows),
        "dryRunRows": len(dry_run_rows),
        "rubricDesignRows": len(rows),
        "readyForFutureGradingRows": ready_rows,
        "expectedNoAnswerRows": by_answerability.get("expected_no_answer", 0),
        "blockedMissingStrictEvidenceRows": by_execution.get("blocked_missing_strict_evidence", 0),
        "notRunExpectedNoAnswerRows": by_execution.get("not_run_expected_no_answer", 0),
        "answerGeneratedRows": 0,
        "scoreComputedRows": 0,
        "llmCallRows": 0,
        "judgeModelCallRows": 0,
        "answerPathInvokedRows": 0,
        "unsafeAnsweredRows": unsafe_answered_rows,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "vaultScanRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "citationEvidenceCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": schema_violation_count,
        "readyForFutureGradingRate": _rate(ready_rows, len(rows)),
        "unsafeAnswerRate": _rate(unsafe_answered_rows, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byGraderExecutionExpectation": dict(sorted(by_execution.items())),
        "byExpectedEvidenceType": dict(sorted(by_evidence_type.items())),
    }
    return {
        "schema": COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-strict-evidence-answer-quality-grader-design",
            "version": "2026-05-20",
            "mode": "report_only_grader_design",
            "purpose": (
                "Define deterministic answer-quality grader contracts for strict-evidence-ready "
                "complex-paper QA rows without generating answers or computing scores."
            ),
            "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        },
        "inputs": {
            "dryRunReport": str(dry_run_path),
            "dryRunReportSchema": str(dry_run.get("schema") or ""),
            "dryRunStatus": str(dry_run.get("status") or ""),
            "dryRunQuestionRows": _safe_int((dry_run.get("counts") or {}).get("questionRows")),
            "dryRunReadyRows": _safe_int((dry_run.get("counts") or {}).get("readyForAnswerQualityComparisonRows")),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "designOnly": True,
            "graderDesignOnly": True,
            "strictEvidenceReadOnly": True,
            "questionExecutionRun": False,
            "answerGenerationRun": False,
            "answerQualityScoringRun": False,
            "scoreComputed": False,
            "llmCalls": False,
            "judgeModelCalls": False,
            "answerPathChanged": False,
            "answerPathInvoked": False,
            "searchIndexQueried": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "vaultScan": False,
            "runtimeEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": [
            "grader_design_does_not_execute_answer_generation_or_scoring",
            "eligible_rows_only_define_future_grader_inputs_and_checks",
            "future_grading_requires_supplied_answers_and_strict_evidence_fixture_contracts",
        ],
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_strict_evidence_answer_quality_grader_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Strict-Evidence Answer-Quality Grader Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Question rows: `{counts.get('questionRows', 0)}`",
        f"- Rubric design rows: `{counts.get('rubricDesignRows', 0)}`",
        f"- Ready for future grading rows: `{counts.get('readyForFutureGradingRows', 0)}`",
        f"- Expected no-answer rows: `{counts.get('expectedNoAnswerRows', 0)}`",
        f"- Blocked missing strict evidence rows: `{counts.get('blockedMissingStrictEvidenceRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only grader design. It does not generate answers, compute scores, call LLMs or judge models, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By grader execution: `{json.dumps(counts.get('byGraderExecutionExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By category: `{json.dumps(counts.get('byQuestionCategory') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in _as_list(report.get("rows")):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"### `{row.get('questionId', '')}`",
                "",
                f"- Category: `{row.get('questionCategory', '')}`",
                f"- Papers: `{', '.join([str(item) for item in _as_list(row.get('paperIds'))])}`",
                f"- Grader execution expectation: `{row.get('graderExecutionExpectation', '')}`",
                f"- Checks: `{json.dumps(row.get('categorySpecificChecks') or [], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_strict_evidence_answer_quality_grader_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-strict-evidence-answer-quality-grader-design.json"
    summary_path = root / "complex-qa-strict-evidence-answer-quality-grader-design-summary.json"
    markdown_path = root / "complex-qa-strict-evidence-answer-quality-grader-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_strict_evidence_answer_quality_grader_design_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only strict-evidence answer-quality grader design.")
    parser.add_argument(
        "--dry-run-report",
        default=DEFAULT_DRY_RUN_REPORT,
        help="Path to complex-qa-strict-evidence-answer-quality-dry-run.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local grader design reports.")
    parser.add_argument("--json", action="store_true", help="Print grader design payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_strict_evidence_answer_quality_grader_design(
        dry_run_report=args.dry_run_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_strict_evidence_answer_quality_grader_design_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID",
    "DEFAULT_DRY_RUN_REPORT",
    "DEFAULT_REPORT_DIR",
    "NEXT_RECOMMENDED_TRANCHE",
    "build_complex_qa_strict_evidence_answer_quality_grader_design",
    "render_complex_qa_strict_evidence_answer_quality_grader_design_markdown",
    "write_complex_qa_strict_evidence_answer_quality_grader_design_reports",
]
