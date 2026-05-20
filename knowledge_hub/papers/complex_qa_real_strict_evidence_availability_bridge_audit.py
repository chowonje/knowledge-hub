"""Report-only real strict-evidence availability bridge audit.

The audit consumes the complex QA structured-evidence comparison report and the
supplied strict-evidence grader baseline report, then classifies the gap between
synthetic supplied fixtures and real strict-evidence availability.  It does not
scan stores or the vault, generate answers, compute real answer-quality scores,
call LLMs or judge models, invoke or change the answer path, mutate storage,
reindex, reembed, or create runtime, citation, or strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_structured_evidence_comparison_runner import (
    COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
)
from knowledge_hub.papers.complex_qa_supplied_strict_evidence_grader_baseline_runner import (
    COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
)


COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-real-strict-evidence-availability-bridge-audit.v1"
)
DEFAULT_COMPARISON_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-structured-evidence-comparison-runner/"
    "complex-qa-structured-evidence-comparison-runner.json"
)
DEFAULT_GRADER_BASELINE_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-supplied-strict-evidence-grader-baseline-runner/"
    "complex-qa-supplied-strict-evidence-grader-baseline-runner.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-real-strict-evidence-availability-bridge-audit"
)

NEXT_RECOMMENDED_TRANCHE = "complex QA real strict-evidence candidate mapping design"

COMMON_NO_MUTATION_FLAGS = {
    "answerGenerated": False,
    "answerQualityScoreComputed": False,
    "scoreComputed": False,
    "llmCall": False,
    "judgeModelCall": False,
    "answerPathInvoked": False,
    "searchIndexQueried": False,
    "databaseMutation": False,
    "indexMutation": False,
    "reindexOrReembed": False,
    "vaultScan": False,
    "runtimeEvidenceCreated": False,
    "citationEvidenceCreated": False,
    "strictEvidenceCreated": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item or "").strip()]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _baseline_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_id = str(row.get("questionId") or "").strip()
        if question_id:
            mapped[question_id].append(row)
    return dict(mapped)


def _bridge_row(comparison_row: dict[str, Any], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerability = str(comparison_row.get("answerabilityExpectation") or "")
    real_refs = _text_list(comparison_row.get("structuredEvidenceRefs"))
    real_available = comparison_row.get("strictEvidenceAvailable") is True and bool(real_refs)
    synthetic_refs = [
        ref
        for baseline_row in baseline_rows
        for ref in _text_list(baseline_row.get("syntheticStrictEvidenceRefs"))
    ]
    synthetic_baseline_verdicts = _text_list([row.get("baselineGraderVerdict") for row in baseline_rows])
    if answerability == "expected_no_answer":
        bridge_status = "not_applicable_expected_no_answer"
        blockers = ["expected_no_answer_policy_guard"]
        ready = False
    elif real_available:
        bridge_status = "ready_for_future_real_supplied_answer_grading"
        blockers = []
        ready = True
    else:
        bridge_status = "blocked_missing_real_strict_evidence"
        blockers = _text_list(comparison_row.get("strictMissingBlockers")) or [
            f"missing_real_strict_{comparison_row.get('expectedEvidenceType') or 'evidence'}"
        ]
        ready = False
    risk_notes = [
        *_text_list(comparison_row.get("riskNotes")),
        "synthetic_baseline_refs_do_not_count_as_real_strict_evidence",
        "bridge_audit_does_not_scan_stores_or_create_evidence",
    ]
    return {
        "questionId": str(comparison_row.get("questionId") or ""),
        "paperIds": _text_list(comparison_row.get("paperIds")),
        "questionCategory": str(comparison_row.get("questionCategory") or ""),
        "expectedEvidenceType": str(comparison_row.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": answerability,
        "comparisonVerdict": str(comparison_row.get("comparisonVerdict") or ""),
        "comparisonReadyForAnswerQuality": comparison_row.get("readyForAnswerQualityComparison") is True,
        "realStrictEvidenceAvailable": real_available,
        "realStrictEvidenceRefs": real_refs,
        "realStrictEvidenceRefCount": len(real_refs),
        "syntheticBaselineFixtureRows": len(baseline_rows),
        "syntheticBaselineVerdicts": synthetic_baseline_verdicts,
        "syntheticStrictEvidenceRefsObserved": synthetic_refs,
        "syntheticStrictEvidenceRefCount": len(synthetic_refs),
        "syntheticRefsCountAsRealEvidence": False,
        "availabilityBridgeStatus": bridge_status,
        "readyForFutureRealGrading": ready,
        "bridgeBlockers": blockers,
        "bridgeContract": {
            "requiresRealStrictEvidenceRefs": True,
            "requiresQuestionNotExpectedNoAnswer": True,
            "requiresFutureSuppliedCandidateAnswer": True,
            "syntheticFixtureRefsAcceptedAsRealEvidence": False,
            "storeScanPerformed": False,
            "realAnswerQualityScoreComputed": False,
        },
        "riskNotes": risk_notes,
        **COMMON_NO_MUTATION_FLAGS,
    }


def _semantic_violations(
    comparison: dict[str, Any],
    baseline: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if comparison.get("schema") != COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID:
        violations.append("comparison_report_schema_mismatch")
    if baseline.get("schema") != COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID:
        violations.append("grader_baseline_report_schema_mismatch")
    question_ids = [str(row.get("questionId") or "") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    if any(not question_id for question_id in question_ids):
        violations.append("missing_question_id")
    for row in rows:
        question_id = str(row.get("questionId") or "")
        if not row.get("paperIds"):
            violations.append(f"missing_paper_ids:{question_id}")
        if row.get("syntheticRefsCountAsRealEvidence") is not False:
            violations.append(f"synthetic_refs_counted_as_real_evidence:{question_id}")
        if row.get("answerabilityExpectation") == "expected_no_answer" and row.get("readyForFutureRealGrading") is True:
            violations.append(f"expected_no_answer_marked_real_grading_ready:{question_id}")
        if row.get("readyForFutureRealGrading") is True and row.get("realStrictEvidenceAvailable") is not True:
            violations.append(f"ready_without_real_strict_evidence:{question_id}")
        if row.get("realStrictEvidenceAvailable") is True and _safe_int(row.get("realStrictEvidenceRefCount")) <= 0:
            violations.append(f"real_availability_without_refs:{question_id}")
        for flag_name, expected in COMMON_NO_MUTATION_FLAGS.items():
            if row.get(flag_name) is not expected:
                violations.append(f"policy_flag_drift:{question_id}:{flag_name}")
    return sorted(set(violations))


def build_complex_qa_real_strict_evidence_availability_bridge_audit(
    *,
    comparison_report: str | Path = DEFAULT_COMPARISON_REPORT,
    grader_baseline_report: str | Path = DEFAULT_GRADER_BASELINE_REPORT,
) -> dict[str, Any]:
    """Build a report-only bridge audit from comparison and baseline reports."""

    comparison_path = Path(str(comparison_report)).expanduser()
    baseline_path = Path(str(grader_baseline_report)).expanduser()
    comparison = _read_json(comparison_path)
    baseline = _read_json(baseline_path)
    comparison_rows = [item for item in _as_list(comparison.get("rows")) if isinstance(item, dict)]
    baseline_rows = [item for item in _as_list(baseline.get("rows")) if isinstance(item, dict)]
    baseline_by_question = _baseline_by_question(baseline_rows)
    rows = [
        _bridge_row(
            comparison_row,
            baseline_by_question.get(str(comparison_row.get("questionId") or ""), []),
        )
        for comparison_row in comparison_rows
    ]
    semantic_violations = _semantic_violations(comparison, baseline, rows)
    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_status = Counter(str(row.get("availabilityBridgeStatus") or "") for row in rows)
    by_evidence_type = Counter(str(row.get("expectedEvidenceType") or "") for row in rows)
    real_available_rows = sum(1 for row in rows if row.get("realStrictEvidenceAvailable") is True)
    ready_rows = sum(1 for row in rows if row.get("readyForFutureRealGrading") is True)
    synthetic_covered_rows = sum(1 for row in rows if _safe_int(row.get("syntheticBaselineFixtureRows")) > 0)
    synthetic_refs_observed_rows = sum(1 for row in rows if _safe_int(row.get("syntheticStrictEvidenceRefCount")) > 0)
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 else "blocked"
    counts = {
        "comparisonQuestionRows": len(comparison_rows),
        "baselineFixtureRows": len(baseline_rows),
        "bridgeAuditRows": len(rows),
        "syntheticBaselineCoveredQuestionRows": synthetic_covered_rows,
        "syntheticStrictEvidenceObservedRows": synthetic_refs_observed_rows,
        "syntheticStrictEvidencePromotedToRealRows": 0,
        "realStrictEvidenceAvailableRows": real_available_rows,
        "realStrictEvidenceRefRows": sum(1 for row in rows if _safe_int(row.get("realStrictEvidenceRefCount")) > 0),
        "readyForFutureRealGradingRows": ready_rows,
        "expectedNoAnswerRows": by_answerability.get("expected_no_answer", 0),
        "blockedMissingRealStrictEvidenceRows": by_status.get("blocked_missing_real_strict_evidence", 0),
        "notApplicableExpectedNoAnswerRows": by_status.get("not_applicable_expected_no_answer", 0),
        "readyRealStrictEvidenceRows": by_status.get("ready_for_future_real_supplied_answer_grading", 0),
        "answerGeneratedRows": 0,
        "answerQualityScoreComputedRows": 0,
        "scoreComputedRows": 0,
        "llmCallRows": 0,
        "judgeModelCallRows": 0,
        "answerPathInvokedRows": 0,
        "searchIndexQueriedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "vaultScanRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "citationEvidenceCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": schema_violation_count,
        "realStrictEvidenceAvailabilityRate": _rate(real_available_rows, len(rows)),
        "futureRealGradingReadinessRate": _rate(ready_rows, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byAvailabilityBridgeStatus": dict(sorted(by_status.items())),
        "byExpectedEvidenceType": dict(sorted(by_evidence_type.items())),
    }
    return {
        "schema": COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-real-strict-evidence-availability-bridge-audit",
            "version": "2026-05-20",
            "mode": "report_only_real_strict_evidence_bridge_audit",
            "purpose": (
                "Audit which complex QA rows have real strict-evidence availability "
                "after the supplied synthetic grader baseline, without treating synthetic "
                "fixture references as real evidence."
            ),
            "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        },
        "inputs": {
            "comparisonReport": str(comparison_path),
            "comparisonReportSchema": str(comparison.get("schema") or ""),
            "comparisonStatus": str(comparison.get("status") or ""),
            "comparisonQuestionRows": len(comparison_rows),
            "graderBaselineReport": str(baseline_path),
            "graderBaselineReportSchema": str(baseline.get("schema") or ""),
            "graderBaselineStatus": str(baseline.get("status") or ""),
            "graderBaselineRows": len(baseline_rows),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "bridgeAuditOnly": True,
            "realStrictEvidenceReadOnly": True,
            "syntheticFixtureRefsReadOnly": True,
            "syntheticRefsPromotedToRealEvidence": False,
            "storeScanPerformed": False,
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
            "bridge_audit_does_not_scan_real_stores_or_vault",
            "synthetic_baseline_refs_are_not_real_strict_evidence",
            "future_real_grading_requires_explicit_real_strict_evidence_refs_and_supplied_candidate_answers",
        ],
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_real_strict_evidence_availability_bridge_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Real Strict-Evidence Availability Bridge Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Comparison question rows: `{counts.get('comparisonQuestionRows', 0)}`",
        f"- Baseline fixture rows: `{counts.get('baselineFixtureRows', 0)}`",
        f"- Real strict evidence available rows: `{counts.get('realStrictEvidenceAvailableRows', 0)}`",
        f"- Ready for future real grading rows: `{counts.get('readyForFutureRealGradingRows', 0)}`",
        f"- Blocked missing real strict evidence rows: `{counts.get('blockedMissingRealStrictEvidenceRows', 0)}`",
        f"- Synthetic strict evidence promoted to real rows: `{counts.get('syntheticStrictEvidencePromotedToRealRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only bridge audit. It does not scan stores or the vault, generate answers, compute real answer-quality scores, call LLMs or judge models, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, or create strict/runtime/citation evidence. Synthetic fixture refs are never counted as real strict evidence.",
        "",
        "## Counts",
        "",
        f"- By bridge status: `{json.dumps(counts.get('byAvailabilityBridgeStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
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
                f"- Answerability: `{row.get('answerabilityExpectation', '')}`",
                f"- Bridge status: `{row.get('availabilityBridgeStatus', '')}`",
                f"- Real strict evidence refs: `{row.get('realStrictEvidenceRefCount', 0)}`",
                f"- Synthetic baseline fixtures: `{row.get('syntheticBaselineFixtureRows', 0)}`",
                f"- Blockers: `{json.dumps(row.get('bridgeBlockers') or [], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_real_strict_evidence_availability_bridge_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-real-strict-evidence-availability-bridge-audit.json"
    summary_path = root / "complex-qa-real-strict-evidence-availability-bridge-audit-summary.json"
    markdown_path = root / "complex-qa-real-strict-evidence-availability-bridge-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_real_strict_evidence_availability_bridge_audit_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only complex QA real strict-evidence bridge audit.")
    parser.add_argument(
        "--comparison-report",
        default=DEFAULT_COMPARISON_REPORT,
        help="Path to complex-qa-structured-evidence-comparison-runner.json.",
    )
    parser.add_argument(
        "--grader-baseline-report",
        default=DEFAULT_GRADER_BASELINE_REPORT,
        help="Path to complex-qa-supplied-strict-evidence-grader-baseline-runner.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local bridge audit reports.")
    parser.add_argument("--json", action="store_true", help="Print bridge audit payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_real_strict_evidence_availability_bridge_audit(
        comparison_report=args.comparison_report,
        grader_baseline_report=args.grader_baseline_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_real_strict_evidence_availability_bridge_audit_reports(
            report,
            args.output_dir,
        )
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_REAL_STRICT_EVIDENCE_AVAILABILITY_BRIDGE_AUDIT_SCHEMA_ID",
    "DEFAULT_COMPARISON_REPORT",
    "DEFAULT_GRADER_BASELINE_REPORT",
    "DEFAULT_REPORT_DIR",
    "NEXT_RECOMMENDED_TRANCHE",
    "build_complex_qa_real_strict_evidence_availability_bridge_audit",
    "render_complex_qa_real_strict_evidence_availability_bridge_audit_markdown",
    "write_complex_qa_real_strict_evidence_availability_bridge_audit_reports",
]
