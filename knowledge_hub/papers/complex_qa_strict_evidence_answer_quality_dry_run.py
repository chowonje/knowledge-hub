"""Report-only strict-evidence supplied answer-quality dry-run.

The dry-run consumes the structured-evidence gated complex QA comparison report
and plans which strict-evidence-ready questions could enter a later
answer-quality measurement.  It does not generate answers, score answer
quality, call LLMs or judge models, invoke the answer path, search indexes,
scan the vault, mutate storage, reindex, reembed, or create evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
)


COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-strict-evidence-answer-quality-dry-run.v1"
)
DEFAULT_COMPARISON_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-structured-evidence-comparison-runner/"
    "complex-qa-structured-evidence-comparison-runner.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-strict-evidence-answer-quality-dry-run"
)

PLANNED_ANSWER_QUALITY_CHECKS = (
    "strict_evidence_refs_present",
    "required_evidence_contract_recheck",
    "citation_coverage_check",
    "unsupported_claim_guard",
    "answerability_regression_guard",
)


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


def _dry_run_blockers(row: dict[str, Any]) -> list[str]:
    if row.get("readyForAnswerQualityComparison") is True:
        return []
    verdict = str(row.get("comparisonVerdict") or "")
    if verdict == "stable_expected_no_answer":
        return ["expected_no_answer_seed_question"]
    if verdict == "still_blocked_missing_strict_structured_evidence":
        blockers = [str(item) for item in _as_list(row.get("strictMissingBlockers")) if str(item or "").strip()]
        return blockers or ["missing_strict_structured_evidence"]
    if verdict == "answerable_seed_blocked_missing_strict_evidence":
        return ["answerable_seed_missing_strict_evidence"]
    return ["not_ready_for_answer_quality_dry_run"]


def _planned_contract(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "requiresReadyComparisonRow": True,
        "requiresStrictEvidenceRefs": True,
        "requiresRequiredEvidenceContract": True,
        "requiresSourceContentHashOrEquivalentAuthority": True,
        "requiresLocatorOrRegionAuthority": True,
        "forbidsUnsupportedClaims": True,
        "forbidsLLMJudge": True,
        "forbidsAnswerGeneration": True,
        "inputEvidenceRefs": [str(item) for item in _as_list(row.get("structuredEvidenceRefs")) if str(item or "").strip()],
        "requiredEvidenceContract": row.get("requiredEvidenceContract") if isinstance(row.get("requiredEvidenceContract"), dict) else {},
    }


def _dry_run_row(row: dict[str, Any]) -> dict[str, Any]:
    ready = row.get("readyForAnswerQualityComparison") is True
    verdict = str(row.get("comparisonVerdict") or "")
    if ready:
        dry_run_verdict = "planned_strict_evidence_answer_quality_dry_run"
    elif verdict == "stable_expected_no_answer":
        dry_run_verdict = "stable_expected_no_answer_no_quality_run"
    elif verdict in {"still_blocked_missing_strict_structured_evidence", "answerable_seed_blocked_missing_strict_evidence"}:
        dry_run_verdict = "blocked_missing_strict_structured_evidence"
    else:
        dry_run_verdict = "schema_violation_unknown_comparison_verdict"
    return {
        "questionId": str(row.get("questionId") or ""),
        "paperIds": [str(item) for item in _as_list(row.get("paperIds")) if str(item or "").strip()],
        "questionCategory": str(row.get("questionCategory") or ""),
        "expectedEvidenceType": str(row.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": str(row.get("answerabilityExpectation") or ""),
        "comparisonVerdict": verdict,
        "strictEvidenceAvailable": row.get("strictEvidenceAvailable") is True,
        "structuredEvidenceRefs": [str(item) for item in _as_list(row.get("structuredEvidenceRefs")) if str(item or "").strip()],
        "readyForAnswerQualityComparison": ready,
        "answerQualityDryRunVerdict": dry_run_verdict,
        "plannedAnswerQualityDryRun": ready,
        "plannedAnswerQualityChecks": list(PLANNED_ANSWER_QUALITY_CHECKS) if ready else [],
        "plannedAnswerQualityContract": _planned_contract(row) if ready else {},
        "dryRunBlockers": _dry_run_blockers(row),
        "answerGenerated": False,
        "answerQualityScoreComputed": False,
        "llmCall": False,
        "judgeModelCall": False,
        "answerPathInvoked": False,
        "unsafeAnswered": False,
        "riskNotes": _as_list(row.get("riskNotes")),
    }


def _semantic_violations(comparison_report: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    if comparison_report.get("schema") != COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID:
        violations.append("comparison_report_schema_mismatch")
    question_ids = [str(row.get("questionId") or "") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    for row in rows:
        question_id = str(row.get("questionId") or "")
        if not question_id:
            violations.append("missing_question_id")
        if not row.get("paperIds"):
            violations.append(f"missing_paper_ids:{question_id}")
        if str(row.get("answerQualityDryRunVerdict") or "").startswith("schema_violation"):
            violations.append(f"unknown_comparison_verdict:{question_id}")
        if row.get("plannedAnswerQualityDryRun") is True and not row.get("structuredEvidenceRefs"):
            violations.append(f"planned_without_structured_evidence_refs:{question_id}")
    return sorted(set(violations))


def build_complex_qa_strict_evidence_answer_quality_dry_run(
    *,
    comparison_report: str | Path = DEFAULT_COMPARISON_REPORT,
) -> dict[str, Any]:
    """Build a report-only answer-quality dry-run plan from a comparison report."""

    comparison_path = Path(str(comparison_report)).expanduser()
    comparison = _read_json(comparison_path)
    comparison_rows = [item for item in _as_list(comparison.get("rows")) if isinstance(item, dict)]
    rows = [_dry_run_row(row) for row in comparison_rows]
    semantic_violations = _semantic_violations(comparison, rows)
    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_verdict = Counter(str(row.get("answerQualityDryRunVerdict") or "") for row in rows)
    by_evidence_type = Counter(str(row.get("expectedEvidenceType") or "") for row in rows)
    planned_rows = sum(1 for row in rows if row.get("plannedAnswerQualityDryRun") is True)
    ready_rows = sum(1 for row in rows if row.get("readyForAnswerQualityComparison") is True)
    strict_available_rows = sum(1 for row in rows if row.get("strictEvidenceAvailable") is True)
    structured_ref_rows = sum(1 for row in rows if row.get("structuredEvidenceRefs"))
    unsafe_answered_rows = sum(1 for row in rows if row.get("unsafeAnswered") is True)
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and unsafe_answered_rows == 0 else "blocked"
    counts = {
        "questionRows": len(rows),
        "comparisonRows": len(comparison_rows),
        "comparisonReadyRows": _safe_int((comparison.get("counts") or {}).get("readyForAnswerQualityComparisonRows")),
        "strictEvidenceAvailableRows": strict_available_rows,
        "structuredEvidenceRefRows": structured_ref_rows,
        "plannedAnswerQualityDryRunRows": planned_rows,
        "readyForAnswerQualityComparisonRows": ready_rows,
        "stableExpectedNoAnswerRows": by_verdict.get("stable_expected_no_answer_no_quality_run", 0),
        "blockedMissingStrictEvidenceRows": by_verdict.get("blocked_missing_strict_structured_evidence", 0),
        "answerGeneratedRows": 0,
        "answerQualityScoreComputedRows": 0,
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
        "plannedAnswerQualityDryRunRate": _rate(planned_rows, len(rows)),
        "unsafeAnswerRate": _rate(unsafe_answered_rows, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byAnswerQualityDryRunVerdict": dict(sorted(by_verdict.items())),
        "byExpectedEvidenceType": dict(sorted(by_evidence_type.items())),
    }
    return {
        "schema": COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-strict-evidence-answer-quality-dry-run",
            "version": "2026-05-20",
            "mode": "report_only_answer_quality_planning_dry_run",
            "purpose": "Plan which strict-evidence-ready complex-paper QA questions can enter a later answer-quality measurement without generating answers or scoring quality.",
            "nextRecommendedTranche": "complex QA strict-evidence supplied answer-quality grader design",
        },
        "inputs": {
            "comparisonReport": str(comparison_path),
            "comparisonReportSchema": str(comparison.get("schema") or ""),
            "comparisonStatus": str(comparison.get("status") or ""),
            "comparisonQuestionRows": _safe_int((comparison.get("counts") or {}).get("questionRows")),
            "comparisonReadyRows": _safe_int((comparison.get("counts") or {}).get("readyForAnswerQualityComparisonRows")),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "answerQualityPlanningOnly": True,
            "strictEvidenceReadOnly": True,
            "questionExecutionRun": False,
            "answerGenerationRun": False,
            "answerQualityScoringRun": False,
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
            "dry_run_does_not_execute_answer_generation_or_answer_quality_scoring",
            "planned_rows_only_mean_the_question_has_supplied_strict_evidence_refs",
            "future_quality_measurement_requires_a_separate_grader_design_and_answer_source_contract",
        ],
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_strict_evidence_answer_quality_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Strict-Evidence Answer-Quality Dry-Run",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Question rows: `{counts.get('questionRows', 0)}`",
        f"- Strict evidence available rows: `{counts.get('strictEvidenceAvailableRows', 0)}`",
        f"- Planned answer-quality dry-run rows: `{counts.get('plannedAnswerQualityDryRunRows', 0)}`",
        f"- Stable expected no-answer rows: `{counts.get('stableExpectedNoAnswerRows', 0)}`",
        f"- Blocked missing strict evidence rows: `{counts.get('blockedMissingStrictEvidenceRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only planning dry-run. It does not generate answers, score answer quality, call LLMs or judge models, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By verdict: `{json.dumps(counts.get('byAnswerQualityDryRunVerdict') or {}, ensure_ascii=False, sort_keys=True)}`",
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
                f"- Strict evidence available: `{row.get('strictEvidenceAvailable', False)}`",
                f"- Dry-run verdict: `{row.get('answerQualityDryRunVerdict', '')}`",
                f"- Planned answer-quality dry-run: `{row.get('plannedAnswerQualityDryRun', False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_strict_evidence_answer_quality_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-strict-evidence-answer-quality-dry-run.json"
    summary_path = root / "complex-qa-strict-evidence-answer-quality-dry-run-summary.json"
    markdown_path = root / "complex-qa-strict-evidence-answer-quality-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_strict_evidence_answer_quality_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only strict-evidence answer-quality dry-run.")
    parser.add_argument(
        "--comparison-report",
        default=DEFAULT_COMPARISON_REPORT,
        help="Path to complex-qa-structured-evidence-comparison-runner.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local dry-run reports.")
    parser.add_argument("--json", action="store_true", help="Print dry-run payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_strict_evidence_answer_quality_dry_run(
        comparison_report=args.comparison_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_strict_evidence_answer_quality_dry_run_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_DRY_RUN_SCHEMA_ID",
    "DEFAULT_COMPARISON_REPORT",
    "DEFAULT_REPORT_DIR",
    "build_complex_qa_strict_evidence_answer_quality_dry_run",
    "render_complex_qa_strict_evidence_answer_quality_dry_run_markdown",
    "write_complex_qa_strict_evidence_answer_quality_dry_run_reports",
]
