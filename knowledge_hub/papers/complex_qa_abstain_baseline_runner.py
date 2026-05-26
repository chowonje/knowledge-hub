"""Report-only complex QA abstain baseline runner.

The runner consumes the complex-paper QA seed pack and computes the current
contract-only baseline: questions that require strict structured evidence
should remain abstain/no-answer until that evidence exists.  It does not call
LLMs, execute the answer path, search indexes, scan the vault, mutate storage,
reindex, reembed, or create evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.complex_qa_seed_pack import COMPLEX_QA_SEED_PACK_SCHEMA_ID


COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID = "knowledge-hub.paper.complex-qa-abstain-baseline-runner.v1"
DEFAULT_SEED_PACK_REPORT = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-paper-qa-seed-pack/complex-paper-qa-seed-pack.json"
)
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/"
    "complex-qa-abstain-baseline-runner"
)

NO_ANSWER_EXPECTATIONS = {"expected_no_answer", "blocked_until_structured_evidence"}


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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _paper_ids(question: dict[str, Any]) -> list[str]:
    values = [str(item) for item in list(question.get("paperIds") or []) if str(item or "").strip()]
    if values:
        return values
    paper_id = str(question.get("paperId") or "").strip()
    return [paper_id] if paper_id else []


def _strict_missing_blockers(question: dict[str, Any]) -> list[str]:
    contract = dict(question.get("requiredEvidenceContract") or {})
    blockers = [str(item) for item in list(contract.get("blockedIfMissing") or []) if str(item or "").strip()]
    if blockers:
        return blockers
    evidence_type = str(question.get("expectedEvidenceType") or "")
    return [f"missing_strict_{evidence_type or 'evidence'}"]


def _baseline_row(question: dict[str, Any]) -> dict[str, Any]:
    answerability = str(question.get("answerabilityExpectation") or "")
    question_id = str(question.get("questionId") or "")
    category = str(question.get("questionCategory") or "")
    evidence_type = str(question.get("expectedEvidenceType") or "")
    paper_ids = _paper_ids(question)
    strict_missing = _strict_missing_blockers(question)
    if answerability == "expected_no_answer":
        verdict = "pass_expected_no_answer"
        expected_behavior = "no_answer"
    elif answerability == "blocked_until_structured_evidence":
        verdict = "pass_blocked_until_structured_evidence"
        expected_behavior = "abstain_until_strict_structured_evidence"
    elif answerability == "answerable":
        verdict = "blocked_answerable_without_strict_evidence"
        expected_behavior = "requires_future_strict_evidence_check"
    else:
        verdict = "schema_violation_unknown_answerability"
        expected_behavior = "blocked_schema_violation"
    unsafe_answered = False
    abstain_pass = answerability in NO_ANSWER_EXPECTATIONS
    return {
        "questionId": question_id,
        "paperIds": paper_ids,
        "questionCategory": category,
        "expectedEvidenceType": evidence_type,
        "answerabilityExpectation": answerability,
        "expectedBaselineBehavior": expected_behavior,
        "baselineVerdict": verdict,
        "abstainBaselinePass": abstain_pass,
        "strictEvidenceAvailable": False,
        "answerGenerated": False,
        "unsafeAnswered": unsafe_answered,
        "llmCall": False,
        "answerPathInvoked": False,
        "strictMissingBlockers": strict_missing,
        "requiredEvidenceContract": dict(question.get("requiredEvidenceContract") or {}),
        "riskNotes": list(question.get("riskNotes") or []),
    }


def _semantic_violations(seed_pack: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    if seed_pack.get("schema") != COMPLEX_QA_SEED_PACK_SCHEMA_ID:
        violations.append("seed_pack_schema_mismatch")
    question_ids = [str(row.get("questionId") or "") for row in rows]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    for row in rows:
        if not row.get("questionId"):
            violations.append("missing_question_id")
        if not row.get("paperIds"):
            violations.append(f"missing_paper_ids:{row.get('questionId')}")
        if str(row.get("baselineVerdict") or "").startswith("schema_violation"):
            violations.append(f"unknown_answerability:{row.get('questionId')}")
    return sorted(set(violations))


def build_complex_qa_abstain_baseline(
    *,
    seed_pack_report: str | Path = DEFAULT_SEED_PACK_REPORT,
) -> dict[str, Any]:
    """Build a report-only abstain baseline from a complex QA seed pack."""

    seed_path = Path(str(seed_pack_report)).expanduser()
    seed_pack = _read_json(seed_path)
    questions = [item for item in list(seed_pack.get("questions") or []) if isinstance(item, dict)]
    papers = [item for item in list(seed_pack.get("papers") or []) if isinstance(item, dict)]
    rows = [_baseline_row(question) for question in questions]
    semantic_violations = _semantic_violations(seed_pack, rows)
    by_category = Counter(str(row.get("questionCategory") or "") for row in rows)
    by_answerability = Counter(str(row.get("answerabilityExpectation") or "") for row in rows)
    by_verdict = Counter(str(row.get("baselineVerdict") or "") for row in rows)
    by_evidence_type = Counter(str(row.get("expectedEvidenceType") or "") for row in rows)
    blocker_counts = Counter(blocker for row in rows for blocker in list(row.get("strictMissingBlockers") or []))
    no_answer_rows = sum(1 for row in rows if row.get("answerabilityExpectation") in NO_ANSWER_EXPECTATIONS)
    abstain_pass_rows = sum(1 for row in rows if row.get("abstainBaselinePass") is True)
    answerable_rows = by_answerability.get("answerable", 0)
    strict_evidence_available_rows = sum(1 for row in rows if row.get("strictEvidenceAvailable") is True)
    unsafe_answered_rows = sum(1 for row in rows if row.get("unsafeAnswered") is True)
    schema_violation_count = len(semantic_violations)
    status = "ok" if schema_violation_count == 0 and unsafe_answered_rows == 0 else "blocked"
    counts = {
        "paperRows": len(papers),
        "questionRows": len(rows),
        "expectedNoAnswerRows": by_answerability.get("expected_no_answer", 0),
        "blockedUntilStructuredEvidenceRows": by_answerability.get("blocked_until_structured_evidence", 0),
        "expectedNoAnswerOrBlockedRows": no_answer_rows,
        "answerableRows": answerable_rows,
        "strictEvidenceAvailableRows": strict_evidence_available_rows,
        "abstainExpectedRows": no_answer_rows,
        "abstainBaselinePassRows": abstain_pass_rows,
        "abstainBaselineFailRows": no_answer_rows - abstain_pass_rows,
        "unsafeAnsweredRows": unsafe_answered_rows,
        "answerGeneratedRows": 0,
        "llmCallRows": 0,
        "answerPathInvokedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "vaultScanRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "citationEvidenceCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "reindexOrReembedRows": 0,
        "schemaViolationCount": schema_violation_count,
        "abstainNoAnswerPassRate": _rate(abstain_pass_rows, no_answer_rows),
        "unsafeAnswerRate": _rate(unsafe_answered_rows, len(rows)),
        "byQuestionCategory": dict(sorted(by_category.items())),
        "byAnswerabilityExpectation": {
            "answerable": by_answerability.get("answerable", 0),
            "expected_no_answer": by_answerability.get("expected_no_answer", 0),
            "blocked_until_structured_evidence": by_answerability.get("blocked_until_structured_evidence", 0),
        },
        "byBaselineVerdict": dict(sorted(by_verdict.items())),
        "byExpectedEvidenceType": dict(sorted(by_evidence_type.items())),
        "byStrictMissingBlocker": dict(sorted(blocker_counts.items())),
    }
    return {
        "schema": COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "runner": {
            "name": "complex-qa-abstain-baseline-runner",
            "version": "2026-05-20",
            "mode": "contract_only_static_baseline",
            "purpose": "Measure the current no-answer/abstain baseline before structured evidence is allowed to improve complex paper QA answerability.",
            "nextRecommendedTranche": "structured evidence gated complex QA comparison runner",
        },
        "inputs": {
            "seedPackReport": str(seed_path),
            "seedPackSchema": str(seed_pack.get("schema") or ""),
            "seedPackQuestionRows": _safe_int((seed_pack.get("counts") or {}).get("questionRows")),
            "seedPackPaperRows": _safe_int((seed_pack.get("counts") or {}).get("paperRows")),
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "contractOnlyStaticBaseline": True,
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
        "warnings": [
            "baseline_runner_does_not_execute_answer_generation",
            "pass_rows_only_mean_no_answer_policy_is_expected_for_current_seed_pack",
            "future_answerability_improvements_require_strict_structured_evidence_and_a_separate_runner",
        ],
        "semanticViolations": semantic_violations,
        "rows": rows,
    }


def render_complex_qa_abstain_baseline_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex QA Abstain Baseline Runner",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Question rows: `{counts.get('questionRows', 0)}`",
        f"- Abstain expected rows: `{counts.get('abstainExpectedRows', 0)}`",
        f"- Abstain/no-answer pass rows: `{counts.get('abstainBaselinePassRows', 0)}`",
        f"- Unsafe answered rows: `{counts.get('unsafeAnsweredRows', 0)}`",
        f"- Abstain/no-answer pass rate: `{counts.get('abstainNoAnswerPassRate', 0)}`",
        f"- Next recommended tranche: `{(report.get('runner') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only static baseline. It does not generate answers, call LLMs, invoke the answer path, query indexes, mutate DB/index state, reindex/reembed, scan the vault, or create strict/runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By verdict: `{json.dumps(counts.get('byBaselineVerdict') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By category: `{json.dumps(counts.get('byQuestionCategory') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By strict missing blocker: `{json.dumps(counts.get('byStrictMissingBlocker') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.extend(
            [
                f"### `{row.get('questionId', '')}`",
                "",
                f"- Category: `{row.get('questionCategory', '')}`",
                f"- Papers: `{', '.join(list(row.get('paperIds') or []))}`",
                f"- Answerability: `{row.get('answerabilityExpectation', '')}`",
                f"- Verdict: `{row.get('baselineVerdict', '')}`",
                f"- Strict missing blockers: `{', '.join(list(row.get('strictMissingBlockers') or []))}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_abstain_baseline_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-qa-abstain-baseline-runner.json"
    summary_path = root / "complex-qa-abstain-baseline-runner-summary.json"
    markdown_path = root / "complex-qa-abstain-baseline-runner.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("runner") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_complex_qa_abstain_baseline_markdown(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only complex QA abstain baseline.")
    parser.add_argument("--seed-pack-report", default=DEFAULT_SEED_PACK_REPORT, help="Path to complex-paper-qa-seed-pack.json.")
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local baseline reports.")
    parser.add_argument("--json", action="store_true", help="Print baseline payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_abstain_baseline(seed_pack_report=args.seed_pack_report)
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_abstain_baseline_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_ABSTAIN_BASELINE_SCHEMA_ID",
    "DEFAULT_REPORT_DIR",
    "DEFAULT_SEED_PACK_REPORT",
    "build_complex_qa_abstain_baseline",
    "render_complex_qa_abstain_baseline_markdown",
    "write_complex_qa_abstain_baseline_reports",
]
