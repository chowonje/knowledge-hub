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
TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-isolated-extractor-pilot-result.v1"
)
SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-proposal.v1"
)
NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.non-sectionspan-pdf-offset-feasibility-audit.v1"
)
EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.equation-alignment-feasibility-audit.v1"
)
TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-feasibility-audit.v1"
)
FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.figure-region-link-feasibility-audit.v1"
)
CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.candidate-layer-blocker-decision-record.v1"
)

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
    "sectionspan_pdf_offset_human_review_pending": {
        "priority": "P0",
        "layers": ["sectionspan"],
        "category": "source_span_promotion_review",
        "recommendedNextTranche": "sectionspan_pdf_offset_human_review_execution",
        "evidenceNeededBeforePromotion": [
            "human review decision for each recovered SectionSpan original-PDF offset",
            "explicit approval or rejection recorded outside runtime answer generation",
            "tests proving approvals remain later-promotion-design-only",
            "separate explicit apply tranche before any strict/runtime evidence use",
        ],
        "stopRule": "stop_if_any_sectionspan_pdf_offset_rows_are_still_pending_human_review",
    },
    "sectionspan_selected_review_decision_file_required": {
        "priority": "P0",
        "layers": ["sectionspan"],
        "category": "source_span_promotion_review",
        "recommendedNextTranche": "manual_record_selected_sectionspan_review_decisions",
        "evidenceNeededBeforePromotion": [
            "explicit human decision file for the selected SectionSpan review rows",
            "approval or rejection recorded with reviewer/notes",
            "decision record proving proposals were not consumed automatically",
            "separate explicit apply tranche before any strict/runtime evidence use",
        ],
        "stopRule": "stop_if_selected_review_decision_proposals_have_not_been_accepted_by_a_human_decision_file",
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
    "table_cell_isolated_extractor_approval_required": {
        "priority": "P0",
        "layers": ["table_region"],
        "category": "table_cell_extractor_pilot",
        "recommendedNextTranche": "table_cell_isolated_extractor_pilot_requires_explicit_approval",
        "evidenceNeededBeforePromotion": [
            "explicit operator approval before isolated dependency install or extractor run",
            "isolated venv path and package set reviewed before execution",
            "bounded target table allowlist",
            "report proving no canonical parsed artifacts, DB, index, or embeddings were modified",
        ],
        "stopRule": "stop_if_operator_approval_for_isolated_extractor_run_is_missing",
    },
    "table_cell_isolated_extractor_unavailable_or_blocked": {
        "priority": "P0",
        "layers": ["table_region"],
        "category": "table_cell_extractor_pilot",
        "recommendedNextTranche": "table_cell_isolated_extractor_dependency_repair_or_alternative_review",
        "evidenceNeededBeforePromotion": [
            "importable optional extractor in an isolated environment",
            "bounded page/table probe output",
            "row/column/cell bbox candidates from the extractor",
            "explicit report for blocked dependency or runtime failures",
        ],
        "stopRule": "stop_if_optional_extractor_is_missing_or_fails_before_producing_cell_candidates",
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
    "candidate_layer_blocker_decision_record_pending": {
        "priority": "P0",
        "layers": ["sectionspan", "figure_caption", "equation_quote", "table_region"],
        "category": "blocker_decision_review",
        "recommendedNextTranche": "manual_record_candidate_layer_blocker_decisions",
        "evidenceNeededBeforePromotion": [
            "explicit decision file for each candidate-layer blocker decision row",
            "separate human/operator decision record with reviewer notes",
            "proof that operator approval records do not execute parser, dependency, or artifact writes",
            "separate explicit tranche before any blocker resolution can change runtime behavior",
        ],
        "stopRule": "stop_if_candidate_layer_blocker_decision_record_rows_are_still_needs_review",
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


def _affected_candidate_count(
    blocker: str,
    layers: list[str],
    summary: dict[str, Any],
    table_cell_result: dict[str, Any] | None = None,
    sectionspan_human_review_gate: dict[str, Any] | None = None,
    sectionspan_selected_decision_proposal: dict[str, Any] | None = None,
    non_sectionspan_pdf_offset_audit: dict[str, Any] | None = None,
    equation_alignment_audit: dict[str, Any] | None = None,
    table_cell_provenance_audit: dict[str, Any] | None = None,
    figure_region_link_audit: dict[str, Any] | None = None,
    candidate_layer_blocker_decision_record: dict[str, Any] | None = None,
) -> int:
    by_layer = _candidate_counts_by_layer(summary)
    table_cell_counts = dict((table_cell_result or {}).get("counts") or {})
    sectionspan_gate_counts = dict((sectionspan_human_review_gate or {}).get("counts") or {})
    non_sectionspan_counts = dict((non_sectionspan_pdf_offset_audit or {}).get("counts") or {})
    equation_counts = dict((equation_alignment_audit or {}).get("counts") or {})
    table_cell_counts_audit = dict((table_cell_provenance_audit or {}).get("counts") or {})
    figure_counts = dict((figure_region_link_audit or {}).get("counts") or {})
    blocker_decision_counts = dict((candidate_layer_blocker_decision_record or {}).get("counts") or {})
    if blocker == "candidate_layer_blocker_decision_record_pending":
        return _safe_int(blocker_decision_counts.get("needsReviewRows")) or _safe_int(
            blocker_decision_counts.get("recordRows")
        )
    if blocker == "sectionspan_pdf_offset_human_review_pending":
        return _safe_int(sectionspan_gate_counts.get("pendingHumanReviewRows")) or _safe_int(
            sectionspan_gate_counts.get("gateRows")
        )
    if blocker == "sectionspan_selected_review_decision_file_required":
        return _safe_int(((sectionspan_selected_decision_proposal or {}).get("counts") or {}).get("proposalRows"))
    if blocker == "table_cell_isolated_extractor_approval_required":
        return _safe_int(table_cell_counts.get("approvalRequiredRows")) or _safe_int(table_cell_counts.get("targetRows"))
    if blocker == "table_cell_isolated_extractor_unavailable_or_blocked":
        return _safe_int(table_cell_counts.get("blockedRows")) or _safe_int(table_cell_counts.get("targetRows"))
    if blocker == "equation_quote_alignment_missing":
        if _safe_int(equation_counts.get("auditedEquationQuoteCandidates")):
            return _safe_int(equation_counts.get("auditedEquationQuoteCandidates"))
        return _safe_int(non_sectionspan_counts.get("needsEquationAlignmentReviewRows")) or _safe_int(
            by_layer.get("equation_quote")
        )
    if blocker == "table_cell_row_column_bbox_provenance_missing":
        if _safe_int(table_cell_counts_audit.get("auditedTableRegionCandidates")):
            return _safe_int(table_cell_counts_audit.get("auditedTableRegionCandidates"))
        return _safe_int(by_layer.get("table_region"))
    if blocker == "figure_region_link_unverified":
        if _safe_int(figure_counts.get("auditedFigureCaptionCandidates")):
            return _safe_int(figure_counts.get("auditedFigureCaptionCandidates"))
        return _safe_int(by_layer.get("figure_caption"))
    if blocker == "sectionspan_pdf_offsets_require_human_review_before_strict_promotion":
        return _safe_int((summary.get("counts") or {}).get("sectionspanOriginalPdfOffsetReadyForReviewRows")) or _safe_int(
            by_layer.get("sectionspan")
        )
    if blocker == "non_sectionspan_layers_lack_original_pdf_offsets":
        if _safe_int(non_sectionspan_counts.get("totalRows")):
            return _safe_int(non_sectionspan_counts.get("blockedRows"))
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
        if _safe_int(non_sectionspan_counts.get("totalRows")):
            return _safe_int(non_sectionspan_counts.get("needsFigureRegionReviewRows"))
        return _safe_int((summary.get("counts") or {}).get("figureCaptionOriginalPdfOffsetRecoveredRows")) or _safe_int(
            by_layer.get("figure_caption")
        )
    if blocker == "table_caption_pdf_offsets_require_cell_provenance_review":
        if _safe_int(non_sectionspan_counts.get("totalRows")):
            return _safe_int(non_sectionspan_counts.get("needsTableCellProvenanceReviewRows"))
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


def _backlog_item(
    index: int,
    blocker: str,
    summary: dict[str, Any],
    eval_design: dict[str, Any],
    table_cell_result: dict[str, Any] | None = None,
    sectionspan_human_review_gate: dict[str, Any] | None = None,
    sectionspan_selected_decision_proposal: dict[str, Any] | None = None,
    non_sectionspan_pdf_offset_audit: dict[str, Any] | None = None,
    equation_alignment_audit: dict[str, Any] | None = None,
    table_cell_provenance_audit: dict[str, Any] | None = None,
    figure_region_link_audit: dict[str, Any] | None = None,
    candidate_layer_blocker_decision_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule = dict(_BLOCKER_RULES.get(blocker) or {})
    layers = list(rule.get("layers") or [])
    return {
        "backlog_id": f"candidate-layer-blocker-v1-{index:03d}",
        "blocker": blocker,
        "status": "open",
        "priority": str(rule.get("priority") or "P2"),
        "category": str(rule.get("category") or "unknown"),
        "affected_layers": layers,
        "affected_candidate_count": _affected_candidate_count(
            blocker,
            layers,
            summary,
            table_cell_result,
            sectionspan_human_review_gate,
            sectionspan_selected_decision_proposal,
            non_sectionspan_pdf_offset_audit,
            equation_alignment_audit,
            table_cell_provenance_audit,
            figure_region_link_audit,
            candidate_layer_blocker_decision_record,
        ),
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


def _table_cell_result_blockers(result: dict[str, Any]) -> list[str]:
    if not result:
        return []
    blockers: list[str] = []
    counts = dict(result.get("counts") or {})
    gate = dict(result.get("gate") or {})
    status = str(result.get("status") or "")
    if (
        status == "approval_required"
        or gate.get("approvalRequiredBeforeInstallOrRun")
        or _safe_int(counts.get("approvalRequiredRows"))
    ):
        blockers.append("table_cell_isolated_extractor_approval_required")
    if status == "blocked" or _safe_int(counts.get("blockedRows")) or (
        result.get("schema") == TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID
        and gate.get("pilotExecuted")
        and not gate.get("extractorAvailable")
    ):
        blockers.append("table_cell_isolated_extractor_unavailable_or_blocked")
    return blockers


def _sectionspan_human_review_gate_blockers(result: dict[str, Any]) -> list[str]:
    if not result:
        return []
    counts = dict(result.get("counts") or {})
    status = str(result.get("status") or "")
    gate = dict(result.get("gate") or {})
    if (
        status == "review_required"
        or _safe_int(counts.get("pendingHumanReviewRows"))
        or (gate.get("humanReviewGateReady") and not gate.get("humanReviewComplete"))
    ):
        return ["sectionspan_pdf_offset_human_review_pending"]
    return []


def _sectionspan_selected_decision_proposal_blockers(result: dict[str, Any]) -> list[str]:
    if not result:
        return []
    counts = dict(result.get("counts") or {})
    gate = dict(result.get("gate") or {})
    status = str(result.get("status") or "")
    if (
        status == "decision_proposal_ready"
        or _safe_int(counts.get("proposalRows"))
        or (gate.get("decisionProposalReady") and not gate.get("humanReviewComplete"))
    ):
        return ["sectionspan_selected_review_decision_file_required"]
    return []


def _non_sectionspan_pdf_offset_audit_blockers(result: dict[str, Any]) -> list[str]:
    if not result:
        return []
    counts = dict(result.get("counts") or {})
    blockers: list[str] = []
    if _safe_int(counts.get("blockedRows")):
        blockers.append("non_sectionspan_layers_lack_original_pdf_offsets")
    if _safe_int(counts.get("needsFigureRegionReviewRows")):
        blockers.append("figure_caption_pdf_offsets_require_region_link_review")
    if _safe_int(counts.get("needsTableCellProvenanceReviewRows")):
        blockers.append("table_caption_pdf_offsets_require_cell_provenance_review")
    if _safe_int(counts.get("needsEquationAlignmentReviewRows")):
        blockers.append("equation_quote_alignment_missing")
    return blockers


def _downstream_feasibility_audit_blockers(
    *,
    equation_alignment_audit: dict[str, Any],
    table_cell_provenance_audit: dict[str, Any],
    figure_region_link_audit: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    equation_counts = dict(equation_alignment_audit.get("counts") or {})
    if _safe_int(equation_counts.get("auditedEquationQuoteCandidates")) > _safe_int(
        equation_counts.get("canonicalSourceSpanCreatedCandidates")
    ):
        blockers.append("equation_quote_alignment_missing")
    table_counts = dict(table_cell_provenance_audit.get("counts") or {})
    if _safe_int(table_counts.get("auditedTableRegionCandidates")) > _safe_int(
        table_counts.get("tableCellCitationGradeCandidates")
    ):
        blockers.append("table_cell_row_column_bbox_provenance_missing")
    figure_counts = dict(figure_region_link_audit.get("counts") or {})
    if _safe_int(figure_counts.get("auditedFigureCaptionCandidates")) > _safe_int(
        figure_counts.get("figureRegionLinkVerifiedCandidates")
    ):
        blockers.append("figure_region_link_unverified")
    return blockers


def _candidate_layer_blocker_decision_record_blockers(result: dict[str, Any]) -> list[str]:
    if not result:
        return []
    counts = dict(result.get("counts") or {})
    gate = dict(result.get("gate") or {})
    status = str(result.get("status") or "")
    if (
        status == "decision_record_required"
        or _safe_int(counts.get("needsReviewRows"))
        or (gate.get("decisionRecordReady") and not gate.get("allDecisionRowsComplete"))
    ):
        return ["candidate_layer_blocker_decision_record_pending"]
    return []


def _collect_blockers(
    gate: dict[str, Any],
    summary: dict[str, Any],
    eval_design: dict[str, Any],
    table_cell_result: dict[str, Any] | None = None,
    sectionspan_human_review_gate: dict[str, Any] | None = None,
    sectionspan_selected_decision_proposal: dict[str, Any] | None = None,
    non_sectionspan_pdf_offset_audit: dict[str, Any] | None = None,
    equation_alignment_audit: dict[str, Any] | None = None,
    table_cell_provenance_audit: dict[str, Any] | None = None,
    figure_region_link_audit: dict[str, Any] | None = None,
    candidate_layer_blocker_decision_record: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(str(item) for item in list((gate.get("gate") or {}).get("blockers") or []))
    blockers.extend(str(item) for item in list((summary.get("releaseCandidateAssessment") or {}).get("mainBlockers") or []))
    for question in list(eval_design.get("questions") or []):
        blockers.extend(str(item) for item in list(question.get("blocked_by_current_candidates") or []))
    blockers.extend(_table_cell_result_blockers(table_cell_result or {}))
    blockers.extend(_sectionspan_human_review_gate_blockers(sectionspan_human_review_gate or {}))
    blockers.extend(_sectionspan_selected_decision_proposal_blockers(sectionspan_selected_decision_proposal or {}))
    blockers.extend(_non_sectionspan_pdf_offset_audit_blockers(non_sectionspan_pdf_offset_audit or {}))
    blockers.extend(_candidate_layer_blocker_decision_record_blockers(candidate_layer_blocker_decision_record or {}))
    blockers.extend(
        _downstream_feasibility_audit_blockers(
            equation_alignment_audit=equation_alignment_audit or {},
            table_cell_provenance_audit=table_cell_provenance_audit or {},
            figure_region_link_audit=figure_region_link_audit or {},
        )
    )
    return [item for item in dict.fromkeys(blockers) if item]


def build_candidate_layer_blocker_backlog(
    *,
    candidate_layer_review_gate_report: str | Path,
    structured_summary_report: str | Path,
    complex_qa_eval_design_report: str | Path,
    table_cell_isolated_extractor_pilot_result_report: str | Path | None = None,
    sectionspan_pdf_offset_human_review_gate_report: str | Path | None = None,
    sectionspan_pdf_offset_selected_review_decision_proposal_report: str | Path | None = None,
    non_sectionspan_pdf_offset_feasibility_audit_report: str | Path | None = None,
    equation_alignment_feasibility_audit_report: str | Path | None = None,
    table_cell_provenance_feasibility_audit_report: str | Path | None = None,
    figure_region_link_feasibility_audit_report: str | Path | None = None,
    candidate_layer_blocker_decision_record_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only backlog over current candidate-layer blockers."""

    gate_path = Path(str(candidate_layer_review_gate_report)).expanduser()
    summary_path = Path(str(structured_summary_report)).expanduser()
    eval_path = Path(str(complex_qa_eval_design_report)).expanduser()
    gate = _read_json(gate_path)
    summary = _read_json(summary_path)
    eval_design = _read_json(eval_path)
    table_cell_result_path = (
        Path(str(table_cell_isolated_extractor_pilot_result_report)).expanduser()
        if table_cell_isolated_extractor_pilot_result_report
        else None
    )
    table_cell_result = _read_json(table_cell_result_path) if table_cell_result_path else {}
    sectionspan_human_review_gate_path = (
        Path(str(sectionspan_pdf_offset_human_review_gate_report)).expanduser()
        if sectionspan_pdf_offset_human_review_gate_report
        else None
    )
    sectionspan_human_review_gate = (
        _read_json(sectionspan_human_review_gate_path) if sectionspan_human_review_gate_path else {}
    )
    sectionspan_selected_decision_proposal_path = (
        Path(str(sectionspan_pdf_offset_selected_review_decision_proposal_report)).expanduser()
        if sectionspan_pdf_offset_selected_review_decision_proposal_report
        else None
    )
    sectionspan_selected_decision_proposal = (
        _read_json(sectionspan_selected_decision_proposal_path) if sectionspan_selected_decision_proposal_path else {}
    )
    non_sectionspan_pdf_offset_audit_path = (
        Path(str(non_sectionspan_pdf_offset_feasibility_audit_report)).expanduser()
        if non_sectionspan_pdf_offset_feasibility_audit_report
        else None
    )
    non_sectionspan_pdf_offset_audit = (
        _read_json(non_sectionspan_pdf_offset_audit_path) if non_sectionspan_pdf_offset_audit_path else {}
    )
    equation_alignment_audit_path = (
        Path(str(equation_alignment_feasibility_audit_report)).expanduser()
        if equation_alignment_feasibility_audit_report
        else None
    )
    equation_alignment_audit = _read_json(equation_alignment_audit_path) if equation_alignment_audit_path else {}
    table_cell_provenance_audit_path = (
        Path(str(table_cell_provenance_feasibility_audit_report)).expanduser()
        if table_cell_provenance_feasibility_audit_report
        else None
    )
    table_cell_provenance_audit = (
        _read_json(table_cell_provenance_audit_path) if table_cell_provenance_audit_path else {}
    )
    figure_region_link_audit_path = (
        Path(str(figure_region_link_feasibility_audit_report)).expanduser()
        if figure_region_link_feasibility_audit_report
        else None
    )
    figure_region_link_audit = _read_json(figure_region_link_audit_path) if figure_region_link_audit_path else {}
    candidate_layer_blocker_decision_record_path = (
        Path(str(candidate_layer_blocker_decision_record_report)).expanduser()
        if candidate_layer_blocker_decision_record_report
        else None
    )
    candidate_layer_blocker_decision_record = (
        _read_json(candidate_layer_blocker_decision_record_path)
        if candidate_layer_blocker_decision_record_path
        else {}
    )
    schema_violations = _schema_violations(gate, summary, eval_design)
    if table_cell_result and table_cell_result.get("schema") != TABLE_CELL_ISOLATED_EXTRACTOR_PILOT_RESULT_SCHEMA_ID:
        schema_violations.append("table_cell_isolated_extractor_pilot_result_schema_mismatch")
    if (
        sectionspan_human_review_gate
        and sectionspan_human_review_gate.get("schema") != SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID
    ):
        schema_violations.append("sectionspan_pdf_offset_human_review_gate_schema_mismatch")
    if (
        sectionspan_selected_decision_proposal
        and sectionspan_selected_decision_proposal.get("schema")
        != SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_PROPOSAL_SCHEMA_ID
    ):
        schema_violations.append("sectionspan_pdf_offset_selected_review_decision_proposal_schema_mismatch")
    if (
        non_sectionspan_pdf_offset_audit
        and non_sectionspan_pdf_offset_audit.get("schema") != NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID
    ):
        schema_violations.append("non_sectionspan_pdf_offset_feasibility_audit_schema_mismatch")
    if (
        equation_alignment_audit
        and equation_alignment_audit.get("schema") != EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID
    ):
        schema_violations.append("equation_alignment_feasibility_audit_schema_mismatch")
    if (
        table_cell_provenance_audit
        and table_cell_provenance_audit.get("schema") != TABLE_CELL_PROVENANCE_FEASIBILITY_AUDIT_SCHEMA_ID
    ):
        schema_violations.append("table_cell_provenance_feasibility_audit_schema_mismatch")
    if (
        figure_region_link_audit
        and figure_region_link_audit.get("schema") != FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID
    ):
        schema_violations.append("figure_region_link_feasibility_audit_schema_mismatch")
    if (
        candidate_layer_blocker_decision_record
        and candidate_layer_blocker_decision_record.get("schema") != CANDIDATE_LAYER_BLOCKER_DECISION_RECORD_SCHEMA_ID
    ):
        schema_violations.append("candidate_layer_blocker_decision_record_schema_mismatch")
    blockers = _collect_blockers(
        gate,
        summary,
        eval_design,
        table_cell_result,
        sectionspan_human_review_gate,
        sectionspan_selected_decision_proposal,
        non_sectionspan_pdf_offset_audit,
        equation_alignment_audit,
        table_cell_provenance_audit,
        figure_region_link_audit,
        candidate_layer_blocker_decision_record,
    )
    items = [
        _backlog_item(
            index,
            blocker,
            summary,
            eval_design,
            table_cell_result,
            sectionspan_human_review_gate,
            sectionspan_selected_decision_proposal,
            non_sectionspan_pdf_offset_audit,
            equation_alignment_audit,
            table_cell_provenance_audit,
            figure_region_link_audit,
            candidate_layer_blocker_decision_record,
        )
        for index, blocker in enumerate(blockers, start=1)
    ]
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
            "tableCellIsolatedExtractorPilotResultReport": str(table_cell_result_path or ""),
            "sectionspanPdfOffsetHumanReviewGateReport": str(sectionspan_human_review_gate_path or ""),
            "sectionspanPdfOffsetSelectedReviewDecisionProposalReport": str(
                sectionspan_selected_decision_proposal_path or ""
            ),
            "nonSectionspanPdfOffsetFeasibilityAuditReport": str(non_sectionspan_pdf_offset_audit_path or ""),
            "equationAlignmentFeasibilityAuditReport": str(equation_alignment_audit_path or ""),
            "tableCellProvenanceFeasibilityAuditReport": str(table_cell_provenance_audit_path or ""),
            "figureRegionLinkFeasibilityAuditReport": str(figure_region_link_audit_path or ""),
            "candidateLayerBlockerDecisionRecordReport": str(candidate_layer_blocker_decision_record_path or ""),
            "candidateLayerReviewGateSchema": str(gate.get("schema") or ""),
            "structuredSummarySchema": str(summary.get("schema") or ""),
            "complexQaEvalDesignSchema": str(eval_design.get("schema") or ""),
            "tableCellIsolatedExtractorPilotResultSchema": str(table_cell_result.get("schema") or ""),
            "sectionspanPdfOffsetHumanReviewGateSchema": str(sectionspan_human_review_gate.get("schema") or ""),
            "sectionspanPdfOffsetSelectedReviewDecisionProposalSchema": str(
                sectionspan_selected_decision_proposal.get("schema") or ""
            ),
            "nonSectionspanPdfOffsetFeasibilityAuditSchema": str(non_sectionspan_pdf_offset_audit.get("schema") or ""),
            "equationAlignmentFeasibilityAuditSchema": str(equation_alignment_audit.get("schema") or ""),
            "tableCellProvenanceFeasibilityAuditSchema": str(table_cell_provenance_audit.get("schema") or ""),
            "figureRegionLinkFeasibilityAuditSchema": str(figure_region_link_audit.get("schema") or ""),
            "candidateLayerBlockerDecisionRecordSchema": str(
                candidate_layer_blocker_decision_record.get("schema") or ""
            ),
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
            "tableCellIsolatedExtractorTargetRows": _safe_int((table_cell_result.get("counts") or {}).get("targetRows")),
            "tableCellIsolatedExtractorApprovalRequiredRows": _safe_int(
                (table_cell_result.get("counts") or {}).get("approvalRequiredRows")
            ),
            "tableCellIsolatedExtractorBlockedRows": _safe_int((table_cell_result.get("counts") or {}).get("blockedRows")),
            "tableCellIsolatedExtractorProbeAttemptedRows": _safe_int((table_cell_result.get("counts") or {}).get("probeAttemptedRows")),
            "sectionspanHumanReviewGateRows": _safe_int(
                (sectionspan_human_review_gate.get("counts") or {}).get("gateRows")
            ),
            "sectionspanHumanReviewPendingRows": _safe_int(
                (sectionspan_human_review_gate.get("counts") or {}).get("pendingHumanReviewRows")
            ),
            "sectionspanHumanReviewApprovedRows": _safe_int(
                (sectionspan_human_review_gate.get("counts") or {}).get("approvedForLaterPromotionDesignRows")
            ),
            "sectionspanSelectedDecisionProposalRows": _safe_int(
                (sectionspan_selected_decision_proposal.get("counts") or {}).get("proposalRows")
            ),
            "sectionspanSelectedDecisionProposalApproveRows": _safe_int(
                (sectionspan_selected_decision_proposal.get("counts") or {}).get(
                    "proposedApproveForLaterPromotionDesignRows"
                )
            ),
            "sectionspanSelectedDecisionAcceptedRows": _safe_int(
                (sectionspan_selected_decision_proposal.get("counts") or {}).get("acceptedHumanDecisionRows")
            ),
            "nonSectionspanPdfOffsetAuditRows": _safe_int(
                (non_sectionspan_pdf_offset_audit.get("counts") or {}).get("totalRows")
            ),
            "nonSectionspanPdfOffsetRecoveredRows": _safe_int(
                (non_sectionspan_pdf_offset_audit.get("counts") or {}).get("recoveredRows")
            ),
            "nonSectionspanPdfOffsetBlockedRows": _safe_int(
                (non_sectionspan_pdf_offset_audit.get("counts") or {}).get("blockedRows")
            ),
            "nonSectionspanPdfOffsetDiagnosticPageContextRows": _safe_int(
                (non_sectionspan_pdf_offset_audit.get("counts") or {}).get("diagnosticPageContextRows")
            ),
            "nonSectionspanPdfOffsetReadyForRegionReviewRows": _safe_int(
                (non_sectionspan_pdf_offset_audit.get("counts") or {}).get("readyForRegionReviewRows")
            ),
            "equationAlignmentAuditRows": _safe_int(
                (equation_alignment_audit.get("counts") or {}).get("auditedEquationQuoteCandidates")
            ),
            "equationAlignmentCanonicalSourceSpanCreatedRows": _safe_int(
                (equation_alignment_audit.get("counts") or {}).get("canonicalSourceSpanCreatedCandidates")
            ),
            "equationAlignmentDiagnosticTermContextRows": _safe_int(
                (equation_alignment_audit.get("counts") or {}).get("diagnosticTermContextCandidates")
            ),
            "tableCellProvenanceAuditRows": _safe_int(
                (table_cell_provenance_audit.get("counts") or {}).get("auditedTableRegionCandidates")
            ),
            "tableCellProvenanceTotalTableCells": _safe_int(
                (table_cell_provenance_audit.get("counts") or {}).get("totalTableCells")
            ),
            "tableCellProvenanceCellSourceSpanRows": _safe_int(
                (table_cell_provenance_audit.get("counts") or {}).get("cellSourceSpanCandidates")
            ),
            "tableCellProvenanceCitationGradeRows": _safe_int(
                (table_cell_provenance_audit.get("counts") or {}).get("tableCellCitationGradeCandidates")
            ),
            "figureRegionLinkAuditRows": _safe_int(
                (figure_region_link_audit.get("counts") or {}).get("auditedFigureCaptionCandidates")
            ),
            "figureRegionLinkCaptionSourceSpanRows": _safe_int(
                (figure_region_link_audit.get("counts") or {}).get("captionSourceSpanCandidates")
            ),
            "figureRegionLinkVerifiedRows": _safe_int(
                (figure_region_link_audit.get("counts") or {}).get("figureRegionLinkVerifiedCandidates")
            ),
            "candidateLayerBlockerDecisionRecordRows": _safe_int(
                (candidate_layer_blocker_decision_record.get("counts") or {}).get("recordRows")
            ),
            "candidateLayerBlockerDecisionNeedsReviewRows": _safe_int(
                (candidate_layer_blocker_decision_record.get("counts") or {}).get("needsReviewRows")
            ),
            "candidateLayerBlockerManualApprovalRows": _safe_int(
                (candidate_layer_blocker_decision_record.get("counts") or {}).get("manualApprovalRows")
            ),
            "candidateLayerBlockerOperatorApprovedRows": _safe_int(
                (candidate_layer_blocker_decision_record.get("counts") or {}).get("operatorApprovedRows")
            ),
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
    parser.add_argument(
        "--table-cell-isolated-extractor-pilot-result-report",
        default="",
        help="Optional path to the latest TableCell isolated extractor pilot result JSON.",
    )
    parser.add_argument(
        "--sectionspan-pdf-offset-human-review-gate-report",
        default="",
        help="Optional path to the latest SectionSpan PDF offset human review gate JSON.",
    )
    parser.add_argument(
        "--sectionspan-pdf-offset-selected-review-decision-proposal-report",
        default="",
        help="Optional path to the latest SectionSpan selected review decision proposal JSON.",
    )
    parser.add_argument(
        "--non-sectionspan-pdf-offset-feasibility-audit-report",
        default="",
        help="Optional path to the latest non-SectionSpan PDF offset feasibility audit JSON.",
    )
    parser.add_argument(
        "--equation-alignment-feasibility-audit-report",
        default="",
        help="Optional path to the latest EquationQuote alignment feasibility audit JSON.",
    )
    parser.add_argument(
        "--table-cell-provenance-feasibility-audit-report",
        default="",
        help="Optional path to the latest TableCell provenance feasibility audit JSON.",
    )
    parser.add_argument(
        "--figure-region-link-feasibility-audit-report",
        default="",
        help="Optional path to the latest FigureRegion link feasibility audit JSON.",
    )
    parser.add_argument(
        "--candidate-layer-blocker-decision-record-report",
        default="",
        help="Optional path to the latest candidate-layer blocker decision record JSON.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print backlog payload as JSON.")
    args = parser.parse_args(argv)

    report = build_candidate_layer_blocker_backlog(
        candidate_layer_review_gate_report=args.candidate_layer_review_gate_report,
        structured_summary_report=args.structured_summary_report,
        complex_qa_eval_design_report=args.complex_qa_eval_design_report,
        table_cell_isolated_extractor_pilot_result_report=args.table_cell_isolated_extractor_pilot_result_report or None,
        sectionspan_pdf_offset_human_review_gate_report=args.sectionspan_pdf_offset_human_review_gate_report or None,
        sectionspan_pdf_offset_selected_review_decision_proposal_report=(
            args.sectionspan_pdf_offset_selected_review_decision_proposal_report or None
        ),
        non_sectionspan_pdf_offset_feasibility_audit_report=(
            args.non_sectionspan_pdf_offset_feasibility_audit_report or None
        ),
        equation_alignment_feasibility_audit_report=args.equation_alignment_feasibility_audit_report or None,
        table_cell_provenance_feasibility_audit_report=args.table_cell_provenance_feasibility_audit_report or None,
        figure_region_link_feasibility_audit_report=args.figure_region_link_feasibility_audit_report or None,
        candidate_layer_blocker_decision_record_report=args.candidate_layer_blocker_decision_record_report or None,
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
