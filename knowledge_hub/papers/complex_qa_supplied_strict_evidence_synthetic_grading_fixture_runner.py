"""Report-only synthetic strict-evidence grading fixture runner.

The runner consumes the strict-evidence answer-quality grader design report and
builds deterministic synthetic supplied-answer fixtures.  It does not generate
answers, compute real answer-quality scores, call LLMs or judge models, invoke
or change the answer path, search indexes, scan the vault, mutate storage,
reindex, reembed, or create runtime, citation, or strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_strict_evidence_answer_quality_grader_design import (
    COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID,
)


COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.v1"
)
DEFAULT_GRADER_DESIGN_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-strict-evidence-answer-quality-grader-design/"
    "complex-qa-strict-evidence-answer-quality-grader-design.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner"
)

NEXT_RECOMMENDED_TRANCHE = "complex QA supplied strict-evidence grader baseline runner"

CATEGORY_ORDER = (
    "table_numeric_qa",
    "equation_citation_qa",
    "figure_caption_qa",
    "method_comparison_qa",
    "limitation_qa",
    "appendix_table_lookup_qa",
)

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


def _safe_id(value: Any) -> str:
    return str(value or "").replace("/", "-").replace(" ", "-")


def _first_non_expected_no_answer(rows: list[dict[str, Any]], *, offset: int = 0) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("answerabilityExpectation") != "expected_no_answer"]
    if not candidates:
        return None
    return candidates[offset % len(candidates)]


def _first_expected_no_answer(
    rows: list[dict[str, Any]],
    *,
    preferred_categories: list[str] | None = None,
) -> dict[str, Any] | None:
    for category in preferred_categories or []:
        row = next(
            (
                item
                for item in rows
                if item.get("answerabilityExpectation") == "expected_no_answer"
                and item.get("questionCategory") == category
            ),
            None,
        )
        if row is not None:
            return row
    return next((row for row in rows if row.get("answerabilityExpectation") == "expected_no_answer"), None)


def _good_fixture_rows(design_rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for category in CATEGORY_ORDER:
        row = next(
            (
                item
                for item in design_rows
                if item.get("questionCategory") == category
                and item.get("answerabilityExpectation") != "expected_no_answer"
            ),
            None,
        )
        if row is not None:
            selected.append(("synthetic_good_strict_evidence_answer", row))
    return selected


def _synthetic_strict_evidence_refs(row: dict[str, Any], fixture_id: str) -> list[str]:
    question_id = _safe_id(row.get("questionId"))
    return [
        f"synthetic-strict-evidence://complex-qa-fixture/{fixture_id}/{question_id}/source-span-001"
    ]


def _supplied_citations(refs: list[str]) -> list[dict[str, str]]:
    return [
        {
            "claimId": "synthetic-supported-claim-001",
            "evidenceRef": refs[0],
            "citationRole": "strict_fixture_support",
        }
    ] if refs else []


def _supplied_answer(row: dict[str, Any], fixture_kind: str) -> str:
    question_id = str(row.get("questionId") or "")
    category = str(row.get("questionCategory") or "")
    return (
        f"Synthetic supplied answer for {question_id} in {category}. "
        f"Fixture kind: {fixture_kind}."
    )


def _actual_fixture_verdict(
    *,
    row: dict[str, Any],
    synthetic_strict_evidence_refs: list[str],
    supplied_answer: str,
    supplied_citations: list[dict[str, str]],
    triggered_hard_fail_rules: list[str],
) -> tuple[str, list[str]]:
    failure_reasons: list[str] = []
    if not synthetic_strict_evidence_refs:
        return "blocked", ["missing_strict_structured_evidence"]
    if row.get("answerabilityExpectation") == "expected_no_answer" and supplied_answer.strip():
        failure_reasons.append("expected_no_answer_question_answered")
    if supplied_answer.strip() and not supplied_citations:
        failure_reasons.append("missing_citation_coverage")
    failure_reasons.extend(triggered_hard_fail_rules)
    if failure_reasons:
        return "fail", sorted(set(failure_reasons))
    return "pass", []


def _fixture_row(
    row: dict[str, Any],
    *,
    fixture_kind: str,
    ordinal: int,
    expected_fixture_verdict: str,
    with_strict_evidence: bool = True,
    with_answer: bool = True,
    with_citations: bool = True,
    triggered_hard_fail_rules: list[str] | None = None,
) -> dict[str, Any]:
    question_id = str(row.get("questionId") or "")
    fixture_id = f"{ordinal:02d}-{fixture_kind}-{_safe_id(question_id)}"
    refs = _synthetic_strict_evidence_refs(row, fixture_id) if with_strict_evidence else []
    answer = _supplied_answer(row, fixture_kind) if with_answer else ""
    citations = _supplied_citations(refs) if with_citations else []
    hard_fails = triggered_hard_fail_rules or []
    actual_verdict, failure_reasons = _actual_fixture_verdict(
        row=row,
        synthetic_strict_evidence_refs=refs,
        supplied_answer=answer,
        supplied_citations=citations,
        triggered_hard_fail_rules=hard_fails,
    )
    risk_notes = [
        *_text_list(row.get("riskNotes")),
        "synthetic_fixture_only_not_real_answer_quality_measurement",
        "synthetic_strict_evidence_refs_are_not_runtime_or_citation_evidence",
    ]
    return {
        "fixtureId": fixture_id,
        "questionId": question_id,
        "paperIds": _text_list(row.get("paperIds")),
        "questionCategory": str(row.get("questionCategory") or ""),
        "expectedEvidenceType": str(row.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": str(row.get("answerabilityExpectation") or ""),
        "sourceDesignExpectation": str(row.get("graderExecutionExpectation") or ""),
        "readyForAnswerQualityComparison": row.get("readyForAnswerQualityComparison") is True,
        "readyForSyntheticFixtureComparison": bool(refs and answer and citations),
        "fixtureKind": fixture_kind,
        "fixturePurpose": "exercise_deterministic_grader_contract_with_supplied_synthetic_inputs",
        "requiredGraderInputs": _text_list(row.get("requiredGraderInputs")),
        "categorySpecificChecks": _text_list(row.get("categorySpecificChecks")),
        "hardFailRules": _text_list(row.get("hardFailRules")),
        "syntheticStrictEvidenceRefs": refs,
        "suppliedAnswer": answer,
        "suppliedCitations": citations,
        "triggeredHardFailRules": hard_fails,
        "expectedFixtureVerdict": expected_fixture_verdict,
        "actualFixtureVerdict": actual_verdict,
        "verdictMatched": actual_verdict == expected_fixture_verdict,
        "graderChecksApplied": _text_list(row.get("categorySpecificChecks")),
        "failureReasons": failure_reasons,
        "riskNotes": risk_notes,
        **COMMON_NO_MUTATION_FLAGS,
    }


def _fixture_rows(design_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    ordinal = 1
    for fixture_kind, row in _good_fixture_rows(design_rows):
        fixtures.append(
            _fixture_row(
                row,
                fixture_kind=fixture_kind,
                ordinal=ordinal,
                expected_fixture_verdict="pass",
            )
        )
        ordinal += 1

    unsupported_row = _first_non_expected_no_answer(design_rows, offset=0)
    if unsupported_row is not None:
        fixtures.append(
            _fixture_row(
                unsupported_row,
                fixture_kind="synthetic_unsupported_claim",
                ordinal=ordinal,
                expected_fixture_verdict="fail",
                triggered_hard_fail_rules=["unsupported_claim_present"],
            )
        )
        ordinal += 1

    missing_citation_row = _first_non_expected_no_answer(design_rows, offset=1)
    if missing_citation_row is not None:
        fixtures.append(
            _fixture_row(
                missing_citation_row,
                fixture_kind="synthetic_missing_citation",
                ordinal=ordinal,
                expected_fixture_verdict="fail",
                with_citations=False,
            )
        )
        ordinal += 1

    missing_evidence_row = _first_non_expected_no_answer(design_rows, offset=2)
    if missing_evidence_row is not None:
        fixtures.append(
            _fixture_row(
                missing_evidence_row,
                fixture_kind="synthetic_missing_strict_evidence",
                ordinal=ordinal,
                expected_fixture_verdict="blocked",
                with_strict_evidence=False,
                with_answer=False,
                with_citations=False,
            )
        )
        ordinal += 1

    covered_categories = {str(row.get("questionCategory") or "") for row in fixtures}
    missing_categories = [category for category in CATEGORY_ORDER if category not in covered_categories]
    expected_no_answer_row = _first_expected_no_answer(
        design_rows,
        preferred_categories=missing_categories,
    )
    if expected_no_answer_row is not None:
        fixtures.append(
            _fixture_row(
                expected_no_answer_row,
                fixture_kind="synthetic_expected_no_answer_answered",
                ordinal=ordinal,
                expected_fixture_verdict="fail",
            )
        )
    return fixtures


def _semantic_violations(
    grader_design: dict[str, Any],
    design_rows: list[dict[str, Any]],
    fixture_rows: list[dict[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if grader_design.get("schema") != COMPLEX_QA_STRICT_EVIDENCE_ANSWER_QUALITY_GRADER_DESIGN_SCHEMA_ID:
        violations.append("grader_design_report_schema_mismatch")
    fixture_ids = [str(row.get("fixtureId") or "") for row in fixture_rows]
    if len(fixture_ids) != len(set(fixture_ids)):
        violations.append("duplicate_fixture_id")
    design_question_ids = {str(row.get("questionId") or "") for row in design_rows}
    for row in fixture_rows:
        fixture_id = str(row.get("fixtureId") or "")
        question_id = str(row.get("questionId") or "")
        if not fixture_id:
            violations.append("missing_fixture_id")
        if question_id not in design_question_ids:
            violations.append(f"fixture_question_missing_from_design:{fixture_id}")
        if row.get("actualFixtureVerdict") != row.get("expectedFixtureVerdict"):
            violations.append(f"fixture_expectation_mismatch:{fixture_id}")
        if row.get("answerabilityExpectation") == "expected_no_answer" and row.get("actualFixtureVerdict") == "pass":
            violations.append(f"expected_no_answer_fixture_passed:{fixture_id}")
        if row.get("actualFixtureVerdict") == "pass" and not row.get("syntheticStrictEvidenceRefs"):
            violations.append(f"passing_fixture_missing_strict_evidence_refs:{fixture_id}")
        if row.get("actualFixtureVerdict") == "pass" and not row.get("suppliedCitations"):
            violations.append(f"passing_fixture_missing_citations:{fixture_id}")
        for flag_name, expected in COMMON_NO_MUTATION_FLAGS.items():
            if row.get(flag_name) is not expected:
                violations.append(f"policy_flag_drift:{fixture_id}:{flag_name}")
    return sorted(set(violations))


def build_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner(
    *,
    grader_design_report: str | Path = DEFAULT_GRADER_DESIGN_REPORT,
) -> dict[str, Any]:
    """Build a report-only synthetic supplied-answer grading fixture report."""

    grader_design_path = Path(str(grader_design_report)).expanduser()
    grader_design = _read_json(grader_design_path)
    design_rows = [item for item in _as_list(grader_design.get("rows")) if isinstance(item, dict)]
    fixtures = _fixture_rows(design_rows)
    semantic_violations = _semantic_violations(grader_design, design_rows, fixtures)

    by_category = Counter(str(row.get("questionCategory") or "") for row in fixtures)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in fixtures)
    by_kind = Counter(str(row.get("fixtureKind") or "") for row in fixtures)
    by_expected_verdict = Counter(str(row.get("expectedFixtureVerdict") or "") for row in fixtures)
    by_actual_verdict = Counter(str(row.get("actualFixtureVerdict") or "") for row in fixtures)
    matched_rows = sum(1 for row in fixtures if row.get("verdictMatched") is True)
    mismatch_rows = len(fixtures) - matched_rows
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and mismatch_rows == 0 else "blocked"
    ready_for_real_grading_rows = sum(
        1 for row in design_rows if row.get("graderExecutionExpectation") == "eligible_for_future_grading"
    )
    counts = {
        "designRows": len(design_rows),
        "rubricDesignRows": len(design_rows),
        "fixtureRows": len(fixtures),
        "syntheticGoodFixtureRows": by_kind.get("synthetic_good_strict_evidence_answer", 0),
        "negativeFixtureRows": len(fixtures) - by_kind.get("synthetic_good_strict_evidence_answer", 0),
        "expectedPassRows": by_expected_verdict.get("pass", 0),
        "expectedFailRows": by_expected_verdict.get("fail", 0),
        "expectedBlockedRows": by_expected_verdict.get("blocked", 0),
        "actualPassRows": by_actual_verdict.get("pass", 0),
        "actualFailRows": by_actual_verdict.get("fail", 0),
        "actualBlockedRows": by_actual_verdict.get("blocked", 0),
        "expectationMatchedRows": matched_rows,
        "expectationMismatchRows": mismatch_rows,
        "suppliedAnswerRows": sum(1 for row in fixtures if str(row.get("suppliedAnswer") or "").strip()),
        "suppliedCitationRows": sum(1 for row in fixtures if row.get("suppliedCitations")),
        "syntheticStrictEvidenceRefRows": sum(1 for row in fixtures if row.get("syntheticStrictEvidenceRefs")),
        "readyForRealAnswerQualityComparisonRows": ready_for_real_grading_rows,
        "deterministicFixtureVerdictRows": len(fixtures),
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
        "expectationMatchRate": _rate(matched_rows, len(fixtures)),
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
        "byActualFixtureVerdict": {
            "pass": by_actual_verdict.get("pass", 0),
            "fail": by_actual_verdict.get("fail", 0),
            "blocked": by_actual_verdict.get("blocked", 0),
        },
    }
    return {
        "schema": COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner",
            "version": "2026-05-20",
            "mode": "report_only_synthetic_fixture_runner",
            "purpose": (
                "Exercise deterministic strict-evidence answer-quality grader contracts "
                "with supplied synthetic answers, citations, and strict-evidence references."
            ),
            "nextRecommendedTranche": NEXT_RECOMMENDED_TRANCHE,
        },
        "inputs": {
            "graderDesignReport": str(grader_design_path),
            "graderDesignReportSchema": str(grader_design.get("schema") or ""),
            "graderDesignStatus": str(grader_design.get("status") or ""),
            "graderDesignRows": len(design_rows),
            "graderDesignReadyForFutureGradingRows": _safe_int(
                (grader_design.get("counts") or {}).get("readyForFutureGradingRows")
            ),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "syntheticFixtureOnly": True,
            "suppliedAnswerOnly": True,
            "syntheticStrictEvidenceRefsOnly": True,
            "strictEvidenceReadOnly": True,
            "deterministicFixtureVerdictRun": True,
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
            "synthetic_fixture_runner_does_not_generate_answers_or_call_answer_path",
            "synthetic_strict_evidence_refs_are_fixture_inputs_not_created_evidence_records",
            "fixture_verdicts_are_contract_checks_not_real_answer_quality_scores",
        ],
        "semanticViolations": semantic_violations,
        "rows": fixtures,
    }


def render_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Supplied Strict-Evidence Synthetic Grading Fixture Runner",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Design rows: `{counts.get('designRows', 0)}`",
        f"- Fixture rows: `{counts.get('fixtureRows', 0)}`",
        f"- Expected pass/fail/blocked: `{counts.get('expectedPassRows', 0)}` / `{counts.get('expectedFailRows', 0)}` / `{counts.get('expectedBlockedRows', 0)}`",
        f"- Actual pass/fail/blocked: `{counts.get('actualPassRows', 0)}` / `{counts.get('actualFailRows', 0)}` / `{counts.get('actualBlockedRows', 0)}`",
        f"- Expectation mismatches: `{counts.get('expectationMismatchRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only synthetic fixture runner. It uses supplied synthetic answers, citations, and strict-evidence reference strings only. It does not generate answers, compute real answer-quality scores, call LLMs or judge models, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By fixture kind: `{json.dumps(counts.get('byFixtureKind') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By actual verdict: `{json.dumps(counts.get('byActualFixtureVerdict') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By category: `{json.dumps(counts.get('byQuestionCategory') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Fixtures",
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
                f"- Kind: `{row.get('fixtureKind', '')}`",
                f"- Expected verdict: `{row.get('expectedFixtureVerdict', '')}`",
                f"- Actual verdict: `{row.get('actualFixtureVerdict', '')}`",
                f"- Failure reasons: `{json.dumps(row.get('failureReasons') or [], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.json"
    summary_path = root / "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner-summary.json"
    markdown_path = root / "complex-qa-supplied-strict-evidence-synthetic-grading-fixture-runner.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Generate a report-only supplied strict-evidence synthetic grading fixture report."
    )
    parser.add_argument(
        "--grader-design-report",
        default=DEFAULT_GRADER_DESIGN_REPORT,
        help="Path to complex-qa-strict-evidence-answer-quality-grader-design.json.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local fixture reports.")
    parser.add_argument("--json", action="store_true", help="Print fixture runner payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner(
        grader_design_report=args.grader_design_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_reports(
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
    "COMPLEX_QA_SUPPLIED_STRICT_EVIDENCE_SYNTHETIC_GRADING_FIXTURE_RUNNER_SCHEMA_ID",
    "DEFAULT_GRADER_DESIGN_REPORT",
    "DEFAULT_REPORT_DIR",
    "NEXT_RECOMMENDED_TRANCHE",
    "build_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner",
    "render_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_markdown",
    "write_complex_qa_supplied_strict_evidence_synthetic_grading_fixture_runner_reports",
]
