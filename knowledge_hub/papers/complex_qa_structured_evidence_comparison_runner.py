"""Report-only structured-evidence gated complex QA comparison runner.

The runner consumes the complex-paper QA seed pack and abstain baseline report,
then computes which questions may become future answer-quality comparison
candidates if strict structured evidence is explicitly available.  It does not
call LLMs, execute or change the answer path, search indexes, scan the vault,
mutate storage, reindex, reembed, or create evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_abstain_baseline_runner import (
    COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID,
)
from knowledge_hub.papers.complex_qa_seed_pack import COMPLEX_QA_SEED_PACK_SCHEMA_ID


COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID = (
    "knowledge-hub.paper.complex-qa-structured-evidence-comparison-runner.v1"
)
DEFAULT_SEED_PACK_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-paper-qa-seed-pack/complex-paper-qa-seed-pack.json"
)
DEFAULT_ABSTAIN_BASELINE_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-abstain-baseline-runner/complex-qa-abstain-baseline-runner.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-structured-evidence-comparison-runner"
)

STRICT_EVIDENCE_TRUE_KEYS = (
    "strictEvidenceAvailable",
    "evidenceContractSatisfied",
    "requiredEvidenceContractSatisfied",
    "strictEvidenceContractSatisfied",
)
STRICT_EVIDENCE_REF_KEYS = (
    "structuredEvidenceRefs",
    "strictEvidenceRefs",
    "evidenceRefs",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
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


def _paper_ids(question: dict[str, Any]) -> list[str]:
    values = [str(item) for item in _as_list(question.get("paperIds")) if str(item or "").strip()]
    if values:
        return values
    paper_id = str(question.get("paperId") or "").strip()
    return [paper_id] if paper_id else []


def _question_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("questionId") or "").strip()
        if question_id:
            mapped[question_id] = row
    return mapped


def _evidence_refs(row: dict[str, Any]) -> list[str]:
    for key in STRICT_EVIDENCE_REF_KEYS:
        refs = [str(item) for item in _as_list(row.get(key)) if str(item or "").strip()]
        if refs:
            return refs
    return []


def _availability_by_question(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    availability: dict[str, dict[str, Any]] = {}
    for row in [item for item in _as_list(report.get("rows")) if isinstance(item, dict)]:
        question_id = str(row.get("questionId") or "").strip()
        if not question_id:
            continue
        contract_satisfied = any(bool(row.get(key)) for key in STRICT_EVIDENCE_TRUE_KEYS)
        refs = _evidence_refs(row)
        strict_available = contract_satisfied and bool(refs)
        availability[question_id] = {
            "strictEvidenceAvailable": strict_available,
            "evidenceContractSatisfied": contract_satisfied,
            "structuredEvidenceRefs": refs,
            "sourceReportRowStatus": str(row.get("status") or row.get("comparisonStatus") or ""),
        }
    return availability


def _strict_missing_blockers(question: dict[str, Any], availability: dict[str, Any]) -> list[str]:
    if availability.get("strictEvidenceAvailable") is True:
        return []
    contract = question.get("requiredEvidenceContract")
    contract_obj = contract if isinstance(contract, dict) else {}
    blockers = [str(item) for item in _as_list(contract_obj.get("blockedIfMissing")) if str(item or "").strip()]
    if blockers:
        return blockers
    evidence_type = str(question.get("expectedEvidenceType") or "")
    return [f"missing_strict_{evidence_type or 'evidence'}"]


def _comparison_row(
    question: dict[str, Any],
    *,
    baseline_row: dict[str, Any] | None,
    availability: dict[str, Any] | None,
) -> dict[str, Any]:
    answerability = str(question.get("answerabilityExpectation") or "")
    availability = dict(availability or {})
    strict_available = availability.get("strictEvidenceAvailable") is True
    contract_satisfied = availability.get("evidenceContractSatisfied") is True
    refs = [str(item) for item in _as_list(availability.get("structuredEvidenceRefs")) if str(item or "").strip()]

    if answerability == "expected_no_answer":
        verdict = "stable_expected_no_answer"
        expected_behavior = "no_answer_expected_even_with_structured_evidence"
        ready = False
    elif strict_available:
        verdict = "candidate_answerable_with_strict_structured_evidence"
        expected_behavior = "structured_evidence_gated_answer_candidate"
        ready = True
    elif answerability == "blocked_until_structured_evidence":
        verdict = "still_blocked_missing_strict_structured_evidence"
        expected_behavior = "abstain_until_strict_structured_evidence"
        ready = False
    elif answerability == "answerable":
        verdict = "answerable_seed_blocked_missing_strict_evidence"
        expected_behavior = "blocked_missing_strict_structured_evidence"
        ready = False
    else:
        verdict = "schema_violation_unknown_answerability"
        expected_behavior = "blocked_schema_violation"
        ready = False

    contract = question.get("requiredEvidenceContract")
    return {
        "questionId": str(question.get("questionId") or ""),
        "paperIds": _paper_ids(question),
        "questionCategory": str(question.get("questionCategory") or ""),
        "expectedEvidenceType": str(question.get("expectedEvidenceType") or ""),
        "answerabilityExpectation": answerability,
        "baselineVerdict": str((baseline_row or {}).get("baselineVerdict") or ""),
        "baselineExpectedBehavior": str((baseline_row or {}).get("expectedBaselineBehavior") or ""),
        "strictEvidenceAvailable": strict_available,
        "evidenceContractSatisfied": contract_satisfied,
        "structuredEvidenceRefs": refs,
        "comparisonVerdict": verdict,
        "expectedGatedBehavior": expected_behavior,
        "readyForAnswerQualityComparison": ready,
        "answerGenerated": False,
        "unsafeAnswered": False,
        "llmCall": False,
        "answerPathInvoked": False,
        "strictMissingBlockers": _strict_missing_blockers(question, availability),
        "requiredEvidenceContract": contract if isinstance(contract, dict) else {},
        "riskNotes": _as_list(question.get("riskNotes")),
    }


def _semantic_violations(
    *,
    seed_pack: dict[str, Any],
    abstain_baseline: dict[str, Any],
    evidence_availability: dict[str, Any],
    rows: list[dict[str, Any]],
    evidence_report_supplied: bool,
) -> list[str]:
    violations: list[str] = []
    if seed_pack.get("schema") != COMPLEX_QA_SEED_PACK_SCHEMA_ID:
        violations.append("seed_pack_schema_mismatch")
    if abstain_baseline.get("schema") != COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID:
        violations.append("abstain_baseline_schema_mismatch")
    question_ids = [str(row.get("questionId") or "") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    if any(not item for item in question_ids):
        violations.append("missing_question_id")
    seed_ids = set(question_ids)
    baseline_ids = {
        str(row.get("questionId") or "")
        for row in [item for item in _as_list(abstain_baseline.get("rows")) if isinstance(item, dict)]
        if str(row.get("questionId") or "")
    }
    missing_baseline_ids = sorted(seed_ids - baseline_ids)
    extra_baseline_ids = sorted(baseline_ids - seed_ids)
    if missing_baseline_ids:
        violations.append("missing_abstain_baseline_question_rows")
    if extra_baseline_ids:
        violations.append("extra_abstain_baseline_question_rows")
    evidence_ids = {
        str(row.get("questionId") or "")
        for row in [item for item in _as_list(evidence_availability.get("rows")) if isinstance(item, dict)]
        if str(row.get("questionId") or "")
    }
    if evidence_report_supplied and evidence_ids - seed_ids:
        violations.append("extra_structured_evidence_question_rows")
    for row in rows:
        if not row.get("paperIds"):
            violations.append(f"missing_paper_ids:{row.get('questionId')}")
        if str(row.get("comparisonVerdict") or "").startswith("schema_violation"):
            violations.append(f"unknown_answerability:{row.get('questionId')}")
    return sorted(set(violations))


def build_complex_qa_structured_evidence_comparison(
    *,
    seed_pack_report: str | Path = DEFAULT_SEED_PACK_REPORT,
    abstain_baseline_report: str | Path = DEFAULT_ABSTAIN_BASELINE_REPORT,
    structured_evidence_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only structured-evidence gated comparison report."""

    seed_path = Path(str(seed_pack_report)).expanduser()
    baseline_path = Path(str(abstain_baseline_report)).expanduser()
    evidence_path = Path(str(structured_evidence_report)).expanduser() if structured_evidence_report else None
    seed_pack = _read_json(seed_path)
    abstain_baseline = _read_json(baseline_path)
    evidence_availability = _read_json(evidence_path) if evidence_path else {}
    questions = [item for item in _as_list(seed_pack.get("questions")) if isinstance(item, dict)]
    papers = [item for item in _as_list(seed_pack.get("papers")) if isinstance(item, dict)]
    baseline_by_question = _question_map(
        [item for item in _as_list(abstain_baseline.get("rows")) if isinstance(item, dict)]
    )
    availability_by_question = _availability_by_question(evidence_availability)
    rows = [
        _comparison_row(
            question,
            baseline_row=baseline_by_question.get(str(question.get("questionId") or "")),
            availability=availability_by_question.get(str(question.get("questionId") or "")),
        )
        for question in questions
    ]
    semantic_violations = _semantic_violations(
        seed_pack=seed_pack,
        abstain_baseline=abstain_baseline,
        evidence_availability=evidence_availability,
        rows=rows,
        evidence_report_supplied=evidence_path is not None,
    )
    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_verdict = Counter(str(row.get("comparisonVerdict") or "") for row in rows)
    by_evidence_type = Counter(str(row.get("expectedEvidenceType") or "") for row in rows)
    strict_available_rows = sum(1 for row in rows if row.get("strictEvidenceAvailable") is True)
    contract_satisfied_rows = sum(1 for row in rows if row.get("evidenceContractSatisfied") is True)
    ready_rows = sum(1 for row in rows if row.get("readyForAnswerQualityComparison") is True)
    unsafe_answered_rows = sum(1 for row in rows if row.get("unsafeAnswered") is True)
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and unsafe_answered_rows == 0 else "blocked"
    evidence_rows = [item for item in _as_list(evidence_availability.get("rows")) if isinstance(item, dict)]
    counts = {
        "paperRows": len(papers),
        "questionRows": len(rows),
        "abstainBaselineRows": len(baseline_by_question),
        "structuredEvidenceAvailabilityRows": len(evidence_rows),
        "expectedNoAnswerRows": by_answerability.get("expected_no_answer", 0),
        "blockedUntilStructuredEvidenceRows": by_answerability.get("blocked_until_structured_evidence", 0),
        "answerableRows": by_answerability.get("answerable", 0),
        "strictEvidenceAvailableRows": strict_available_rows,
        "evidenceContractSatisfiedRows": contract_satisfied_rows,
        "stableExpectedNoAnswerRows": by_verdict.get("stable_expected_no_answer", 0),
        "stillBlockedMissingStrictEvidenceRows": by_verdict.get("still_blocked_missing_strict_structured_evidence", 0),
        "candidateAnswerableWithStructuredEvidenceRows": by_verdict.get(
            "candidate_answerable_with_strict_structured_evidence", 0
        ),
        "answerableSeedBlockedMissingStrictEvidenceRows": by_verdict.get(
            "answerable_seed_blocked_missing_strict_evidence", 0
        ),
        "readyForAnswerQualityComparisonRows": ready_rows,
        "baselineAbstainPassRows": _safe_int((abstain_baseline.get("counts") or {}).get("abstainBaselinePassRows")),
        "answerGeneratedRows": 0,
        "llmCallRows": 0,
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
        "readyForAnswerQualityComparisonRate": _rate(ready_rows, len(rows)),
        "unsafeAnswerRate": _rate(unsafe_answered_rows, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byComparisonVerdict": dict(sorted(by_verdict.items())),
        "byExpectedEvidenceType": dict(sorted(by_evidence_type.items())),
    }
    warnings = [
        "comparison_runner_does_not_execute_answer_generation",
        "strict_structured_evidence_must_be_supplied_explicitly_before_questions_become_answer_quality_candidates",
        "future_answer_quality_measurement_requires_a_separate_answer_generation_or_grading_runner",
    ]
    if evidence_path is None:
        warnings.append("no_structured_evidence_report_supplied_all_questions_remain_no_answer_or_blocked")
    return {
        "schema": COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-structured-evidence-comparison-runner",
            "version": "2026-05-20",
            "mode": "report_only_structured_evidence_gate",
            "purpose": "Identify which complex-paper QA seed questions may become future answer-quality comparison candidates when strict structured evidence is explicitly available.",
            "nextRecommendedTranche": "complex QA strict-evidence supplied answer-quality dry-run",
        },
        "inputs": {
            "seedPackReport": str(seed_path),
            "seedPackSchema": str(seed_pack.get("schema") or ""),
            "abstainBaselineReport": str(baseline_path),
            "abstainBaselineSchema": str(abstain_baseline.get("schema") or ""),
            "structuredEvidenceReport": str(evidence_path) if evidence_path else "",
            "structuredEvidenceReportSchema": str(evidence_availability.get("schema") or ""),
            "structuredEvidenceInputMode": (
                "report_supplied_contract_probe"
                if evidence_path
                else "absent_all_strict_evidence_unavailable"
            ),
            "seedPackQuestionRows": _safe_int((seed_pack.get("counts") or {}).get("questionRows")),
            "seedPackPaperRows": _safe_int((seed_pack.get("counts") or {}).get("paperRows")),
            "abstainBaselineQuestionRows": _safe_int((abstain_baseline.get("counts") or {}).get("questionRows")),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "comparisonOnly": True,
            "strictEvidenceReadOnly": True,
            "questionExecutionRun": False,
            "answerGenerationRun": False,
            "llmCalls": False,
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
        "warnings": warnings,
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_structured_evidence_comparison_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Structured Evidence Comparison Runner",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Question rows: `{counts.get('questionRows', 0)}`",
        f"- Strict evidence available rows: `{counts.get('strictEvidenceAvailableRows', 0)}`",
        f"- Ready for answer-quality comparison rows: `{counts.get('readyForAnswerQualityComparisonRows', 0)}`",
        f"- Candidate answerable rows: `{counts.get('candidateAnswerableWithStructuredEvidenceRows', 0)}`",
        f"- Still blocked rows: `{counts.get('stillBlockedMissingStrictEvidenceRows', 0)}`",
        f"- Stable expected no-answer rows: `{counts.get('stableExpectedNoAnswerRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only structured-evidence gate. It does not generate answers, call LLMs, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By verdict: `{json.dumps(counts.get('byComparisonVerdict') or {}, ensure_ascii=False, sort_keys=True)}`",
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
                f"- Answerability: `{row.get('answerabilityExpectation', '')}`",
                f"- Strict evidence available: `{row.get('strictEvidenceAvailable', False)}`",
                f"- Verdict: `{row.get('comparisonVerdict', '')}`",
                f"- Ready for answer-quality comparison: `{row.get('readyForAnswerQualityComparison', False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_structured_evidence_comparison_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-structured-evidence-comparison-runner.json"
    summary_path = root / "complex-qa-structured-evidence-comparison-runner-summary.json"
    markdown_path = root / "complex-qa-structured-evidence-comparison-runner.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_complex_qa_structured_evidence_comparison_markdown(report),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only structured-evidence gated complex QA comparison.")
    parser.add_argument("--seed-pack-report", default=DEFAULT_SEED_PACK_REPORT, help="Path to complex-paper-qa-seed-pack.json.")
    parser.add_argument(
        "--abstain-baseline-report",
        default=DEFAULT_ABSTAIN_BASELINE_REPORT,
        help="Path to complex-qa-abstain-baseline-runner.json.",
    )
    parser.add_argument(
        "--structured-evidence-report",
        default=None,
        help="Optional structured evidence availability report keyed by questionId.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local comparison reports.")
    parser.add_argument("--json", action="store_true", help="Print comparison payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_structured_evidence_comparison(
        seed_pack_report=args.seed_pack_report,
        abstain_baseline_report=args.abstain_baseline_report,
        structured_evidence_report=args.structured_evidence_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_structured_evidence_comparison_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_STRUCTURED_EVIDENCE_COMPARISON_SCHEMA_ID",
    "DEFAULT_ABSTAIN_BASELINE_REPORT",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_SEED_PACK_REPORT",
    "build_complex_qa_structured_evidence_comparison",
    "render_complex_qa_structured_evidence_comparison_markdown",
    "write_complex_qa_structured_evidence_comparison_reports",
]
