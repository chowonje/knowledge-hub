"""Report-only review gate for structured paper candidate layers.

The gate consumes the structured candidate summary and complex-paper QA eval
design.  It determines whether the candidate layer is ready for human/operator
review while keeping strict evidence, parser routing, and answer integration
explicitly blocked.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-review-gate.v1"
STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID = "knowledge-hub.paper.structured-candidate-summary.v1"
COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID = "knowledge-hub.paper.complex-qa-eval-design.v1"


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


def _violations(summary: dict[str, Any], eval_design: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    summary_counts = dict(summary.get("counts") or {})
    summary_policy = dict(summary.get("policy") or {})
    eval_counts = dict(eval_design.get("counts") or {})
    eval_policy = dict(eval_design.get("policy") or {})
    if summary.get("schema") != STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID:
        violations.append("structured_summary_schema_mismatch")
    if eval_design.get("schema") != COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID:
        violations.append("complex_qa_eval_design_schema_mismatch")
    if _safe_int(summary_counts.get("totalCandidates")) <= 0:
        violations.append("candidate_summary_empty")
    if _safe_int(eval_counts.get("questionCount")) <= 0:
        violations.append("eval_design_empty")
    if _safe_int(summary_counts.get("strictEligibleCandidates")) != 0:
        violations.append("summary_strict_eligible_candidates_nonzero")
    if _safe_int(summary_counts.get("citationGradeCandidates")) != 0:
        violations.append("summary_citation_grade_candidates_nonzero")
    if _safe_int(summary_counts.get("runtimeEvidenceCandidates")) != 0:
        violations.append("summary_runtime_evidence_candidates_nonzero")
    if _safe_int(eval_counts.get("currentRuntimeAnswerableQuestions")) != 0:
        violations.append("eval_design_runtime_answerable_questions_nonzero")
    if _safe_int(eval_counts.get("executedQuestions")) != 0:
        violations.append("eval_design_questions_executed")
    if _safe_int(eval_counts.get("strictEvidenceCreated")) != 0:
        violations.append("eval_design_strict_evidence_created")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(summary_policy.get(key)) is True:
            violations.append(f"summary_policy_{key}_true")
        if bool(eval_policy.get(key)) is True:
            violations.append(f"eval_policy_{key}_true")
    if bool(eval_policy.get("questionsExecuted")) is True:
        violations.append("eval_policy_questionsExecuted_true")
    if bool(eval_policy.get("answerGenerationRun")) is True:
        violations.append("eval_policy_answerGenerationRun_true")
    return list(dict.fromkeys(violations))


def build_candidate_layer_review_gate(
    *,
    structured_summary_report: str | Path,
    complex_qa_eval_design_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only candidate-layer review gate payload."""

    summary_path = Path(str(structured_summary_report)).expanduser()
    eval_path = Path(str(complex_qa_eval_design_report)).expanduser()
    summary = _read_json(summary_path)
    eval_design = _read_json(eval_path)
    summary_counts = dict(summary.get("counts") or {})
    eval_counts = dict(eval_design.get("counts") or {})
    assessment = dict(summary.get("releaseCandidateAssessment") or {})
    violations = _violations(summary, eval_design)
    review_ready = not violations and bool(assessment.get("candidateLayerReviewReady", False))
    return {
        "schema": CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID,
        "status": "ready_for_candidate_layer_review" if review_ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "structuredSummaryReport": str(summary_path),
            "complexQaEvalDesignReport": str(eval_path),
            "structuredSummarySchema": str(summary.get("schema") or ""),
            "complexQaEvalDesignSchema": str(eval_design.get("schema") or ""),
            "expectedStructuredSummarySchema": STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID,
            "expectedComplexQaEvalDesignSchema": COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID,
        },
        "counts": {
            "totalCandidates": _safe_int(summary_counts.get("totalCandidates")),
            "questionCount": _safe_int(eval_counts.get("questionCount")),
            "strictEligibleCandidates": _safe_int(summary_counts.get("strictEligibleCandidates")),
            "citationGradeCandidates": _safe_int(summary_counts.get("citationGradeCandidates")),
            "runtimeEvidenceCandidates": _safe_int(summary_counts.get("runtimeEvidenceCandidates")),
            "currentRuntimeAnswerableQuestions": _safe_int(eval_counts.get("currentRuntimeAnswerableQuestions")),
            "executedQuestions": _safe_int(eval_counts.get("executedQuestions")),
            "strictEvidenceCreated": _safe_int(eval_counts.get("strictEvidenceCreated")),
        },
        "gate": {
            "candidateLayerReviewReady": review_ready,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "ready_for_candidate_layer_review" if review_ready else "blocked",
            "strictPromotionDecision": "blocked",
            "parserRoutingDecision": "blocked",
            "answerIntegrationDecision": "blocked",
            "violations": violations,
            "blockers": list(
                dict.fromkeys(
                    [
                        *list(assessment.get("mainBlockers") or []),
                        "candidate_layers_are_report_only",
                        "runtime_promotion_disabled_for_tranche",
                    ]
                )
            ),
            "recommendedNextTranche": "PR_review_then_candidate_layer_blocker_backlog",
        },
        "policy": {
            "reviewGateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "ready_for_candidate_layer_review_does_not_mean_strict_evidence_ready",
            "ready_for_candidate_layer_review_does_not_mean_parser_routing_ready",
            "candidate_layers_must_not_be_used_for_runtime_answers_without_a_later_explicit_promotion_tranche",
        ],
    }


def render_candidate_layer_review_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Review Gate",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Candidate-layer review ready: `{bool(gate.get('candidateLayerReviewReady'))}`",
        f"- Strict evidence ready: `{bool(gate.get('strictEvidenceReady'))}`",
        f"- Parser routing ready: `{bool(gate.get('parserRoutingReady'))}`",
        f"- Answer integration ready: `{bool(gate.get('answerIntegrationReady'))}`",
        f"- Total candidates: `{int(counts.get('totalCandidates') or 0)}`",
        f"- Eval design questions: `{int(counts.get('questionCount') or 0)}`",
        f"- Strict eligible candidates: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence candidates: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        f"- Current runtime answerable questions: `{int(counts.get('currentRuntimeAnswerableQuestions') or 0)}`",
        "",
        "## Safety",
        "",
        "This gate is report-only. It does not create strict evidence, parser routing, or answer integration.",
        "",
        "## Blockers",
        "",
    ]
    for blocker in list(gate.get("blockers") or []):
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Violations", ""])
    violations = list(gate.get("violations") or [])
    if violations:
        for violation in violations:
            lines.append(f"- `{violation}`")
    else:
        lines.append("- none")
    lines.extend(["", f"Recommended next tranche: `{gate.get('recommendedNextTranche', '')}`", ""])
    return "\n".join(lines)


def write_candidate_layer_review_gate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "candidate-layer-review-gate.json"
    markdown_path = root / "candidate-layer-review-gate.md"
    gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_review_gate_markdown(report), encoding="utf-8")
    return {
        "gate": str(gate_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer review gate.")
    parser.add_argument("--structured-summary-report", required=True, help="Path to structured-candidate-summary.json.")
    parser.add_argument("--complex-qa-eval-design-report", required=True, help="Path to complex-paper-qa-eval-design.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print gate payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_review_gate(
        structured_summary_report=args.structured_summary_report,
        complex_qa_eval_design_report=args.complex_qa_eval_design_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_review_gate_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID",
    "build_candidate_layer_review_gate",
    "render_candidate_layer_review_gate_markdown",
    "write_candidate_layer_review_gate_reports",
]
