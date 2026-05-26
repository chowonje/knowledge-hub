"""Report-only supplied strict-evidence grader baseline runner.

The runner consumes the synthetic strict-evidence grading fixture report and
recomputes deterministic baseline verdicts for the supplied fixture inputs.  It
does not generate answers, compute real answer-quality scores, call LLMs or
judge models, invoke or change the answer path, search indexes, scan the vault,
mutate storage, reindex, reembed, or create runtime, citation, or strict
evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner import (
    COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID,
)


COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-supplied-strict-evidence-grader-baseline-runner.v1"
)
DEFAULT_SYNTHETIC_FIXTURE_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner/"
    "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-supplied-strict-evidence-grader-baseline-runner"
)

NEXT_RECOMMENDED_TRANCHE = "complex QA real strict-evidence availability bridge audit"

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

BASELINE_CHECKS = (
    "synthetic_strict_evidence_refs_present_or_blocked",
    "supplied_answer_presence_check",
    "citation_coverage_check",
    "expected_no_answer_guard",
    "triggered_hard_fail_rules_guard",
    "category_specific_checks_declared",
)


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


def _baseline_verdict(fixture: dict[str, Any]) -> tuple[str, list[str]]:
    strict_refs = _text_list(fixture.get("syntheticStrictEvidenceRefs"))
    supplied_answer = str(fixture.get("suppliedAnswer") or "").strip()
    supplied_citations = [item for item in _as_list(fixture.get("suppliedCitations")) if isinstance(item, dict)]
    triggered_hard_fails = _text_list(fixture.get("triggeredHardFailRules"))
    answerability = str(fixture.get("answerabilityExpectation") or "")

    if not strict_refs:
        return "blocked", ["missing_strict_structured_evidence"]

    failure_reasons: list[str] = []
    if answerability == "expected_no_answer" and supplied_answer:
        failure_reasons.append("expected_no_answer_question_answered")
    if answerability != "expected_no_answer" and not supplied_answer:
        return "blocked", ["missing_supplied_answer"]
    if supplied_answer and not supplied_citations:
        failure_reasons.append("missing_citation_coverage")
    failure_reasons.extend(triggered_hard_fails)
    if failure_reasons:
        return "fail", sorted(set(failure_reasons))
    return "pass", []


def _row(fixture: dict[str, Any]) -> dict[str, Any]:
    baseline_verdict, failure_reasons = _baseline_verdict(fixture)
    expected_verdict = str(fixture.get("expectedFixtureVerdict") or "")
    source_verdict = str(fixture.get("actualFixtureVerdict") or "")
    strict_refs = _text_list(fixture.get("syntheticStrictEvidenceRefs"))
    supplied_answer = str(fixture.get("suppliedAnswer") or "")
    supplied_citations = [item for item in _as_list(fixture.get("suppliedCitations")) if isinstance(item, dict)]
    risk_notes = [
        *_text_list(fixture.get("riskNotes")),
        "baseline_verdict_is_deterministic_fixture_contract_not_real_answer_quality_score",
        "synthetic_fixture_inputs_are_not_runtime_or_citation_evidence",
    ]
    return {
        "fixtureId": str(fixture.get("fixtureId") or ""),
        "questionId": str(fixture.get("questionId") or ""),
        "paperIds": _text_list(fixture.get("paperIds")),
        "questionCategory": str(fixture.get("questionCategory") or ""),
        "expectedEvidenceType": str(fixture.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": str(fixture.get("answerabilityExpectation") or ""),
        "fixtureKind": str(fixture.get("fixtureKind") or ""),
        "sourceFixtureVerdict": source_verdict,
        "expectedFixtureVerdict": expected_verdict,
        "baselineGraderVerdict": baseline_verdict,
        "baselineVerdictMatchedExpected": baseline_verdict == expected_verdict,
        "baselineVerdictMatchedSourceFixture": baseline_verdict == source_verdict,
        "baselineChecksApplied": list(BASELINE_CHECKS),
        "categorySpecificChecksObserved": _text_list(fixture.get("categorySpecificChecks")),
        "requiredGraderInputsObserved": _text_list(fixture.get("requiredGraderInputs")),
        "syntheticStrictEvidenceRefs": strict_refs,
        "syntheticStrictEvidenceRefCount": len(strict_refs),
        "suppliedAnswerPresent": bool(supplied_answer.strip()),
        "suppliedCitationCount": len(supplied_citations),
        "triggeredHardFailRules": _text_list(fixture.get("triggeredHardFailRules")),
        "baselineFailureReasons": failure_reasons,
        "baselineGradeContract": {
            "baselineVerdictComputed": True,
            "realAnswerQualityScoreComputed": False,
            "requiresStrictEvidenceRefs": True,
            "requiresCitationCoverageForSuppliedClaims": True,
            "requiresExpectedNoAnswerGuard": True,
            "requiresNoTriggeredHardFailRulesForPass": True,
        },
        "riskNotes": risk_notes,
        **COMMON_NO_MUTATION_FLAGS,
    }


def _semantic_violations(
    fixture_report: dict[str, Any],
    fixture_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if fixture_report.get("schema") != COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID:
        violations.append("synthetic_fixture_report_schema_mismatch")
    fixture_ids = [str(row.get("fixtureId") or "") for row in baseline_rows]
    if len(fixture_ids) != len(set(fixture_ids)):
        violations.append("duplicate_fixture_id")
    if len(fixture_rows) != len(baseline_rows):
        violations.append("fixture_row_count_mismatch")
    for row in baseline_rows:
        fixture_id = str(row.get("fixtureId") or "")
        if not fixture_id:
            violations.append("missing_fixture_id")
        if row.get("baselineVerdictMatchedExpected") is not True:
            violations.append(f"baseline_expected_verdict_mismatch:{fixture_id}")
        if row.get("baselineVerdictMatchedSourceFixture") is not True:
            violations.append(f"baseline_source_fixture_verdict_mismatch:{fixture_id}")
        if row.get("answerabilityExpectation") == "expected_no_answer" and row.get("baselineGraderVerdict") == "pass":
            violations.append(f"expected_no_answer_baseline_passed:{fixture_id}")
        if row.get("baselineGraderVerdict") == "pass" and _safe_int(row.get("syntheticStrictEvidenceRefCount")) <= 0:
            violations.append(f"passing_baseline_missing_strict_evidence_refs:{fixture_id}")
        if row.get("baselineGraderVerdict") == "pass" and _safe_int(row.get("suppliedCitationCount")) <= 0:
            violations.append(f"passing_baseline_missing_citations:{fixture_id}")
        for flag_name, expected in COMMON_NO_MUTATION_FLAGS.items():
            if row.get(flag_name) is not expected:
                violations.append(f"policy_flag_drift:{fixture_id}:{flag_name}")
    return sorted(set(violations))


def build_complex_qa_supplied_strict_evidence_grader_baseline(
    *,
    synthetic_fixture_report: str | Path = DEFAULT_SYNTHETIC_FIXTURE_REPORT,
) -> dict[str, Any]:
    """Build a report-only deterministic grader baseline from supplied fixtures."""

    fixture_path = Path(str(synthetic_fixture_report)).expanduser()
    fixture_report = _read_json(fixture_path)
    fixture_rows = [item for item in _as_list(fixture_report.get("rows")) if isinstance(item, dict)]
    rows = [_row(fixture) for fixture in fixture_rows]
    semantic_violations = _semantic_violations(fixture_report, fixture_rows, rows)

    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_kind = Counter(str(row.get("fixtureKind") or "") for row in rows)
    by_expected_verdict = Counter(str(row.get("expectedFixtureVerdict") or "") for row in rows)
    by_baseline_verdict = Counter(str(row.get("baselineGraderVerdict") or "") for row in rows)
    matched_expected = sum(1 for row in rows if row.get("baselineVerdictMatchedExpected") is True)
    matched_source = sum(1 for row in rows if row.get("baselineVerdictMatchedSourceFixture") is True)
    expected_mismatch = len(rows) - matched_expected
    source_mismatch = len(rows) - matched_source
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and expected_mismatch == 0 and source_mismatch == 0 else "blocked"
    counts = {
        "inputFixtureRows": len(fixture_rows),
        "baselineGradeRows": len(rows),
        "baselineVerdictComputedRows": len(rows),
        "baselinePassRows": by_baseline_verdict.get("pass", 0),
        "baselineFailRows": by_baseline_verdict.get("fail", 0),
        "baselineBlockedRows": by_baseline_verdict.get("blocked", 0),
        "expectedPassRows": by_expected_verdict.get("pass", 0),
        "expectedFailRows": by_expected_verdict.get("fail", 0),
        "expectedBlockedRows": by_expected_verdict.get("blocked", 0),
        "baselineExpectedMatchRows": matched_expected,
        "baselineExpectedMismatchRows": expected_mismatch,
        "baselineSourceFixtureMatchRows": matched_source,
        "baselineSourceFixtureMismatchRows": source_mismatch,
        "suppliedAnswerRows": sum(1 for row in rows if row.get("suppliedAnswerPresent") is True),
        "suppliedCitationRows": sum(1 for row in rows if _safe_int(row.get("suppliedCitationCount")) > 0),
        "syntheticStrictEvidenceRefRows": sum(
            1 for row in rows if _safe_int(row.get("syntheticStrictEvidenceRefCount")) > 0
        ),
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
        "baselineExpectedMatchRate": _rate(matched_expected, len(rows)),
        "baselineSourceFixtureMatchRate": _rate(matched_source, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byFixtureKind": dict(sorted(by_kind.items())),
        "byExpectedFixtureVerdict": {
            "pass": by_expected_verdict.get("pass", 0),
            "fail": by_expected_verdict.get("fail", 0),
            "blocked": by_expected_verdict.get("blocked", 0),
        },
        "byBaselineGraderVerdict": {
            "pass": by_baseline_verdict.get("pass", 0),
            "fail": by_baseline_verdict.get("fail", 0),
            "blocked": by_baseline_verdict.get("blocked", 0),
        },
    }
    return {
        "schema": COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-supplied-strict-evidence-grader-baseline-runner",
            "version": "2026-05-20",
            "mode": "report_only_deterministic_grader_baseline",
            "purpose": (
                "Fix a deterministic baseline verdict contract for supplied synthetic "
                "strict-evidence complex QA grading fixtures."
            ),
            "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        },
        "inputs": {
            "syntheticFixtureReport": str(fixture_path),
            "syntheticFixtureReportSchema": str(fixture_report.get("schema") or ""),
            "syntheticFixtureStatus": str(fixture_report.get("status") or ""),
            "syntheticFixtureRows": len(fixture_rows),
            "syntheticFixtureExpectationMismatchRows": _safe_int(
                (fixture_report.get("counts") or {}).get("expectationMismatchRows")
            ),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "baselineOnly": True,
            "deterministicGraderBaselineOnly": True,
            "suppliedFixtureOnly": True,
            "syntheticStrictEvidenceRefsOnly": True,
            "strictEvidenceReadOnly": True,
            "deterministicBaselineVerdictRun": True,
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
            "baseline_runner_recomputes_fixture_verdicts_only_not_real_answer_quality_scores",
            "supplied_synthetic_inputs_are_not_generated_answers_or_created_evidence",
            "future_real_grading_requires_real_strict_evidence_and_supplied_candidate_answers",
        ],
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_supplied_strict_evidence_grader_baseline_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Supplied Strict-Evidence Grader Baseline Runner",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input fixture rows: `{counts.get('inputFixtureRows', 0)}`",
        f"- Baseline grade rows: `{counts.get('baselineGradeRows', 0)}`",
        f"- Baseline pass/fail/blocked: `{counts.get('baselinePassRows', 0)}` / `{counts.get('baselineFailRows', 0)}` / `{counts.get('baselineBlockedRows', 0)}`",
        f"- Expected mismatches: `{counts.get('baselineExpectedMismatchRows', 0)}`",
        f"- Source fixture mismatches: `{counts.get('baselineSourceFixtureMismatchRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only deterministic grader baseline. It recomputes fixture verdicts over supplied synthetic answers, citations, and strict-evidence reference strings only. It does not generate answers, compute real answer-quality scores, call LLMs or judge models, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By fixture kind: `{json.dumps(counts.get('byFixtureKind') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By baseline verdict: `{json.dumps(counts.get('byBaselineGraderVerdict') or {}, ensure_ascii=False, sort_keys=True)}`",
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
                f"### `{row.get('fixtureId', '')}`",
                "",
                f"- Question: `{row.get('questionId', '')}`",
                f"- Category: `{row.get('questionCategory', '')}`",
                f"- Fixture kind: `{row.get('fixtureKind', '')}`",
                f"- Expected verdict: `{row.get('expectedFixtureVerdict', '')}`",
                f"- Baseline verdict: `{row.get('baselineGraderVerdict', '')}`",
                f"- Failure reasons: `{json.dumps(row.get('baselineFailureReasons') or [], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_supplied_strict_evidence_grader_baseline_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-supplied-strict-evidence-grader-baseline-runner.json"
    summary_path = root / "complex-qa-supplied-strict-evidence-grader-baseline-runner-summary.json"
    markdown_path = root / "complex-qa-supplied-strict-evidence-grader-baseline-runner.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_supplied_strict_evidence_grader_baseline_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only supplied strict-evidence grader baseline report.")
    parser.add_argument(
        "--synthetic-fixture-report",
        default=DEFAULT_SYNTHETIC_FIXTURE_REPORT,
        help="Path to complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local baseline reports.")
    parser.add_argument("--json", action="store_true", help="Print baseline payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_supplied_strict_evidence_grader_baseline(
        synthetic_fixture_report=args.synthetic_fixture_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_supplied_strict_evidence_grader_baseline_reports(
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
    "COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_GRADER_BASELINE_RUNNER_SCHEMA_ID",
    "DEFAULT_SYNTHETIC_FIXTURE_REPORT",
    "DEFAULT_REPORT_DIR",
    "NEXT_RECOMMENDED_TRANCHE",
    "build_complex_qa_supplied_strict_evidence_grader_baseline",
    "render_complex_qa_supplied_strict_evidence_grader_baseline_markdown",
    "write_complex_qa_supplied_strict_evidence_grader_baseline_reports",
]
