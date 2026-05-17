"""Report-only blocker backlog for structured paper candidate layers.

The backlog consumes the candidate-layer review gate plus its upstream summary
and eval-design reports. It turns remaining blockers into concrete next-tranche
items without changing parser routing, runtime answers, canonical artifacts, or
strict evidence policy.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-blocker-backlog.v1"
CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID = "knowledge-hub.paper.candidate-layer-review-gate.v1"
STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID = "knowledge-hub.paper.structured-candidate-summary.v1"
COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID = "knowledge-hub.paper.complex-qa-eval-design.v1"

_BLOCKER_RULES = {
    "equation_quote_alignment_missing": {
        "priority": "P0",
        "layers": ["equation_quote"],
        "category": "canonical_alignment",
        "recommendedNextTranche": "equation_quote_alignment_feasibility_audit",
        "evidenceNeededBeforePromotion": [
            "exact equation text alignment to canonical parsed text",
            "chars:start-end against canonical text",
            "page recovered from canonical parsed artifact",
            "sourceContentHash from source/parsed manifest",
            "quote-only policy with no equation interpretation",
        ],
        "stopRule": "stop_if_equation_text_can_only_be_matched_fuzzily_or_from_generated_markdown_offsets",
    },
    "table_cell_row_column_bbox_provenance_missing": {
        "priority": "P0",
        "layers": ["table_region"],
        "category": "table_cell_provenance",
        "recommendedNextTranche": "table_cell_provenance_feasibility_audit",
        "evidenceNeededBeforePromotion": [
            "row and column labels",
            "cell values",
            "cell-level bbox or source span provenance",
            "table caption source span",
            "sourceContentHash from source/parsed manifest",
        ],
        "stopRule": "stop_if_only_table_caption_or_region_bbox_is_available",
    },
    "figure_region_link_unverified": {
        "priority": "P1",
        "layers": ["figure_caption"],
        "category": "layout_region_link",
        "recommendedNextTranche": "figure_region_link_feasibility_audit",
        "evidenceNeededBeforePromotion": [
            "caption source span",
            "linked figure/image-like layout region",
            "page recovered from canonical parsed artifact",
            "sourceContentHash from source/parsed manifest",
            "confidence that caption and region refer to the same figure",
        ],
        "stopRule": "stop_if_caption_text_and_figure_region_cannot_be_linked_without_layout_guessing",
    },
    "generated_markdown_offsets_are_not_original_pdf_offsets": {
        "priority": "P0",
        "layers": ["sectionspan", "figure_caption", "equation_quote", "table_region"],
        "category": "source_span_provenance",
        "recommendedNextTranche": "source_span_offset_authority_audit",
        "evidenceNeededBeforePromotion": [
            "canonical chars:start-end from existing parsed text",
            "clear distinction between generated Markdown offsets and canonical source offsets",
            "sourceContentHash linkage",
            "page recovery",
        ],
        "stopRule": "stop_if_only_generated_mineru_markdown_offsets_are_available",
    },
    "sectionspan_pdf_offsets_require_human_review_before_strict_promotion": {
        "priority": "P0",
        "layers": ["sectionspan"],
        "category": "source_span_promotion_review",
        "recommendedNextTranche": "sectionspan_pdf_offset_human_review_gate",
        "evidenceNeededBeforePromotion": [
            "human review of recovered original-PDF SectionSpan offsets",
            "explicit strict-promotion gate for SectionSpan only",
            "tests proving reviewed SectionSpan rows do not authorize other candidate types",
            "runtime tests preserving fail-closed no-answer behavior",
        ],
        "stopRule": "stop_if_human_review_or_explicit_promotion_approval_is_missing",
    },
    "non_sectionspan_layers_lack_original_pdf_offsets": {
        "priority": "P0",
        "layers": ["figure_caption", "equation_quote", "table_region"],
        "category": "source_span_provenance",
        "recommendedNextTranche": "non_sectionspan_original_pdf_offset_feasibility_audit",
        "evidenceNeededBeforePromotion": [
            "figure caption original-PDF source span recovery",
            "equation quote original-PDF source span recovery",
            "table caption or table region original-PDF source span recovery",
            "explicit distinction from SectionSpan-only source alignment",
        ],
        "stopRule": "stop_if_non_sectionspan_candidates_only_have_generated_markdown_offsets",
    },
    "figure_caption_pdf_offsets_require_region_link_review": {
        "priority": "P1",
        "layers": ["figure_caption"],
        "category": "layout_region_link",
        "recommendedNextTranche": "figure_caption_region_link_review_pack",
        "evidenceNeededBeforePromotion": [
            "human/operator review of recovered original-PDF caption spans",
            "verified link from caption source span to the correct figure/image-like region",
            "page and sourceContentHash agreement between caption span and layout region",
            "tests proving recovered caption offsets alone do not authorize figure evidence",
        ],
        "stopRule": "stop_if_caption_source_span_is_available_but_figure_region_link_is_unverified",
    },
    "table_caption_pdf_offsets_require_cell_provenance_review": {
        "priority": "P0",
        "layers": ["table_region"],
        "category": "table_cell_provenance",
        "recommendedNextTranche": "table_cell_provenance_review_pack",
        "evidenceNeededBeforePromotion": [
            "operator review of recovered original-PDF table caption spans",
            "row and column labels linked to table cells",
            "cell values linked to bbox or source spans",
            "sourceContentHash agreement between caption span and table region",
            "tests proving table captions alone do not authorize table-cell evidence",
        ],
        "stopRule": "stop_if_table_caption_span_exists_but_cell_row_column_bbox_provenance_is_missing",
    },
    "equation_quote_pdf_offsets_require_quote_review": {
        "priority": "P0",
        "layers": ["equation_quote"],
        "category": "equation_quote_provenance",
        "recommendedNextTranche": "equation_quote_offset_review_pack",
        "evidenceNeededBeforePromotion": [
            "operator review of recovered original-PDF equation quote spans",
            "explicit quote-only policy with no equation interpretation",
            "page and sourceContentHash agreement",
            "tests proving recovered equation offsets alone do not authorize equation semantics",
        ],
        "stopRule": "stop_if_equation_quote_span_exists_but_semantics_or_context_are_unreviewed",
    },
    "candidate_layers_are_report_only": {
        "priority": "P2",
        "layers": ["sectionspan", "figure_caption", "equation_quote", "table_region"],
        "category": "runtime_policy",
        "recommendedNextTranche": "candidate_layer_contract_review",
        "evidenceNeededBeforePromotion": [
            "explicit later promotion tranche",
            "runtime contract that preserves fail-closed strict evidence rules",
            "tests proving candidate-only rows do not answer questions",
        ],
        "stopRule": "stop_if_runtime_integration_is_required_before_promotion_approval",
    },
    "runtime_promotion_disabled_for_tranche": {
        "priority": "P2",
        "layers": ["sectionspan", "figure_caption", "equation_quote", "table_region"],
        "category": "runtime_policy",
        "recommendedNextTranche": "candidate_layer_promotion_policy_review",
        "evidenceNeededBeforePromotion": [
            "explicit user approval for a separate promotion tranche",
            "strict evidence eligibility rule for each candidate type",
            "regression tests for no fallback span, bbox-only, Markdown-offset-only, or paraphrase promotion",
        ],
        "stopRule": "stop_if_any_runtime_evidence_or_answer_generation_change_is_needed",
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


def _schema_violations(gate: dict[str, Any], summary: dict[str, Any], eval_design: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if gate.get("schema") != CANDIDATE_LAYER_REVIEW_GATE_SCHEMA_ID:
        violations.append("candidate_layer_review_gate_schema_mismatch")
    if summary.get("schema") != STRUCTURED_CANDIDATE_SUMMARY_SCHEMA_ID:
        violations.append("structured_candidate_summary_schema_mismatch")
    if eval_design.get("schema") != COMPLEX_QA_EVAL_DESIGN_SCHEMA_ID:
        violations.append("complex_qa_eval_design_schema_mismatch")
    return violations


def _candidate_counts_by_layer(summary: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): _safe_int(value)
        for key, value in dict((summary.get("counts") or {}).get("byLayer") or {}).items()
    }


def _affected_candidate_count(blocker: str, layers: list[str], summary: dict[str, Any]) -> int:
    by_layer = _candidate_counts_by_layer(summary)
    if blocker == "equation_quote_alignment_missing":
        return _safe_int(by_layer.get("equation_quote"))
    if blocker == "table_cell_row_column_bbox_provenance_missing":
        return _safe_int(by_layer.get("table_region"))
    if blocker == "figure_region_link_unverified":
        return _safe_int(by_layer.get("figure_caption"))
    if blocker == "sectionspan_pdf_offsets_require_human_review_before_strict_promotion":
        return _safe_int((summary.get("counts") or {}).get("sectionspanOriginalPdfOffsetReadyForReviewRows")) or _safe_int(
            by_layer.get("sectionspan")
        )
    if blocker == "non_sectionspan_layers_lack_original_pdf_offsets":
        counts = dict(summary.get("counts") or {})
        if _safe_int(counts.get("figureCaptionOriginalPdfOffsetFeasibilityRows")) > 0:
            figure_blocked = _safe_int(counts.get("figureCaptionOriginalPdfOffsetBlockedRows"))
        else:
            figure_blocked = _safe_int(by_layer.get("figure_caption"))
        if _safe_int(counts.get("tableRegionOriginalPdfOffsetFeasibilityRows")) > 0:
            table_blocked = _safe_int(counts.get("tableRegionOriginalPdfOffsetBlockedRows"))
        else:
            table_blocked = _safe_int(by_layer.get("table_region"))
        if _safe_int(counts.get("equationQuoteOriginalPdfOffsetFeasibilityRows")) > 0:
            equation_blocked = _safe_int(counts.get("equationQuoteOriginalPdfOffsetBlockedRows"))
        else:
            equation_blocked = _safe_int(by_layer.get("equation_quote"))
        return figure_blocked + table_blocked + equation_blocked
    if blocker == "figure_caption_pdf_offsets_require_region_link_review":
        return _safe_int((summary.get("counts") or {}).get("figureCaptionOriginalPdfOffsetRecoveredRows")) or _safe_int(
            by_layer.get("figure_caption")
        )
    if blocker == "table_caption_pdf_offsets_require_cell_provenance_review":
        return _safe_int((summary.get("counts") or {}).get("tableRegionOriginalPdfOffsetRecoveredRows")) or _safe_int(
            by_layer.get("table_region")
        )
    if blocker == "equation_quote_pdf_offsets_require_quote_review":
        return _safe_int((summary.get("counts") or {}).get("equationQuoteOriginalPdfOffsetRecoveredRows")) or _safe_int(
            by_layer.get("equation_quote")
        )
    return sum(_safe_int(by_layer.get(layer)) for layer in layers)


def _affected_question_count(blocker: str, layers: list[str], eval_design: dict[str, Any]) -> int:
    total = 0
    for item in list(eval_design.get("questions") or []):
        question_blockers = set(str(value) for value in list(item.get("blocked_by_current_candidates") or []))
        target_layers = set(str(value) for value in list(item.get("target_candidate_layers") or []))
        if blocker in question_blockers or target_layers.intersection(layers):
            total += 1
    return total


def _backlog_item(index: int, blocker: str, summary: dict[str, Any], eval_design: dict[str, Any]) -> dict[str, Any]:
    rule = dict(_BLOCKER_RULES.get(blocker) or {})
    layers = list(rule.get("layers") or [])
    return {
        "backlog_id": f"candidate-layer-blocker-v1-{index:03d}",
        "blocker": blocker,
        "status": "open",
        "priority": str(rule.get("priority") or "P2"),
        "category": str(rule.get("category") or "unknown"),
        "affected_layers": layers,
        "affected_candidate_count": _affected_candidate_count(blocker, layers, summary),
        "affected_eval_question_count": _affected_question_count(blocker, layers, eval_design),
        "evidenceNeededBeforePromotion": list(rule.get("evidenceNeededBeforePromotion") or []),
        "recommendedNextTranche": str(rule.get("recommendedNextTranche") or "manual_review"),
        "allowedActions": [
            "report_only_audit",
            "schema_backed_local_report",
            "focused_tests",
            "human_operator_review",
        ],
        "disallowedActions": [
            "strict_evidence_promotion",
            "parser_routing",
            "answer_integration",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "stopRule": str(rule.get("stopRule") or "stop_if_promotion_or_runtime_changes_are_required"),
        "evidence_tier": "candidate_backlog_only",
        "strict_eligible": False,
        "strict_blockers": list(dict.fromkeys([
            blocker,
            "candidate_backlog_is_report_only",
            "runtime_promotion_disabled_for_tranche",
        ])),
        "non_strict_reason": [
            "backlog_items_are_not_evidence",
            "backlog_items_do_not_authorize_runtime_use",
        ],
    }


def _collect_blockers(gate: dict[str, Any], summary: dict[str, Any], eval_design: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    blockers.extend(str(item) for item in list((gate.get("gate") or {}).get("blockers") or []))
    blockers.extend(str(item) for item in list((summary.get("releaseCandidateAssessment") or {}).get("mainBlockers") or []))
    for question in list(eval_design.get("questions") or []):
        blockers.extend(str(item) for item in list(question.get("blocked_by_current_candidates") or []))
    return [item for item in dict.fromkeys(blockers) if item]


def build_candidate_layer_blocker_backlog(
    *,
    candidate_layer_review_gate_report: str | Path,
    structured_summary_report: str | Path,
    complex_qa_eval_design_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only backlog over current candidate-layer blockers."""

    gate_path = Path(str(candidate_layer_review_gate_report)).expanduser()
    summary_path = Path(str(structured_summary_report)).expanduser()
    eval_path = Path(str(complex_qa_eval_design_report)).expanduser()
    gate = _read_json(gate_path)
    summary = _read_json(summary_path)
    eval_design = _read_json(eval_path)
    schema_violations = _schema_violations(gate, summary, eval_design)
    blockers = _collect_blockers(gate, summary, eval_design)
    items = [_backlog_item(index, blocker, summary, eval_design) for index, blocker in enumerate(blockers, start=1)]
    by_priority = Counter(str(item.get("priority") or "") for item in items)
    by_layer = Counter(layer for item in items for layer in list(item.get("affected_layers") or []))
    gate_payload = dict(gate.get("gate") or {})
    summary_counts = dict(summary.get("counts") or {})
    eval_counts = dict(eval_design.get("counts") or {})
    return {
        "schema": CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID,
        "status": "ok" if items and not schema_violations else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "candidateLayerReviewGateReport": str(gate_path),
            "structuredSummaryReport": str(summary_path),
            "complexQaEvalDesignReport": str(eval_path),
            "candidateLayerReviewGateSchema": str(gate.get("schema") or ""),
            "structuredSummarySchema": str(summary.get("schema") or ""),
            "complexQaEvalDesignSchema": str(eval_design.get("schema") or ""),
        },
        "counts": {
            "backlogItemCount": len(items),
            "openBacklogItemCount": len([item for item in items if item.get("status") == "open"]),
            "byPriority": dict(by_priority),
            "byLayer": dict(by_layer),
            "totalCandidates": _safe_int(summary_counts.get("totalCandidates")),
            "evalQuestionCount": _safe_int(eval_counts.get("questionCount")),
            "strictEligibleCandidates": _safe_int(summary_counts.get("strictEligibleCandidates")),
            "citationGradeCandidates": _safe_int(summary_counts.get("citationGradeCandidates")),
            "runtimeEvidenceCandidates": _safe_int(summary_counts.get("runtimeEvidenceCandidates")),
            "currentRuntimeAnswerableQuestions": _safe_int(eval_counts.get("currentRuntimeAnswerableQuestions")),
        },
        "gate": {
            "candidateLayerReviewReady": bool(gate_payload.get("candidateLayerReviewReady")),
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "blocker_backlog_ready" if items and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "candidate_layer_blocker_backlog_review",
        },
        "policy": {
            "backlogOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "backlog_items_are_not_evidence",
            "backlog_items_do_not_authorize_runtime_candidate_use",
            "parser_routing_and_answer_integration_remain_out_of_scope",
        ],
        "backlog": items,
    }


def render_candidate_layer_blocker_backlog_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Candidate Layer Blocker Backlog",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Backlog items: `{int(counts.get('backlogItemCount') or 0)}`",
        f"- Open items: `{int(counts.get('openBacklogItemCount') or 0)}`",
        f"- Total candidates: `{int(counts.get('totalCandidates') or 0)}`",
        f"- Eval questions: `{int(counts.get('evalQuestionCount') or 0)}`",
        f"- Strict eligible candidates: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence candidates: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        "",
        "## Policy",
        "",
        "This backlog is report-only. It does not create evidence, parser routing, answer integration, canonical parsed artifacts, DB mutations, or index/embedding changes.",
        "",
        "## Items",
        "",
    ]
    for item in list(report.get("backlog") or []):
        lines.extend(
            [
                f"### `{item.get('backlog_id', '')}`",
                "",
                f"- Blocker: `{item.get('blocker', '')}`",
                f"- Priority: `{item.get('priority', '')}`",
                f"- Category: `{item.get('category', '')}`",
                f"- Affected layers: `{', '.join(list(item.get('affected_layers') or []))}`",
                f"- Affected candidates: `{int(item.get('affected_candidate_count') or 0)}`",
                f"- Affected eval questions: `{int(item.get('affected_eval_question_count') or 0)}`",
                f"- Recommended next tranche: `{item.get('recommendedNextTranche', '')}`",
                f"- Stop rule: `{item.get('stopRule', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_candidate_layer_blocker_backlog_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    backlog_path = root / "candidate-layer-blocker-backlog.json"
    markdown_path = root / "candidate-layer-blocker-backlog.md"
    backlog_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_candidate_layer_blocker_backlog_markdown(report), encoding="utf-8")
    return {
        "backlog": str(backlog_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only candidate-layer blocker backlog.")
    parser.add_argument("--candidate-layer-review-gate-report", required=True, help="Path to candidate-layer-review-gate.json.")
    parser.add_argument("--structured-summary-report", required=True, help="Path to structured-candidate-summary.json.")
    parser.add_argument("--complex-qa-eval-design-report", required=True, help="Path to complex-paper-qa-eval-design.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print backlog payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=args.candidate_layer_review_gate_report,
        structured_summary_report=args.structured_summary_report,
        complex_qa_eval_design_report=args.complex_qa_eval_design_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_candidate_layer_blocker_backlog_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_LAYER_BLOCKER_BACKLOG_SCHEMA_ID",
    "build_candidate_layer_blocker_backlog",
    "render_candidate_layer_blocker_backlog_markdown",
    "write_candidate_layer_blocker_backlog_reports",
]
