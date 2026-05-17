"""Report-only TableCell next-action gate.

This helper consolidates the current TableCell diagnostics into explicit next
actions.  It does not choose an extractor, verify cell/text pairing, create
cell source spans, create table-cell evidence, route parsers, write canonical
parsed artifacts, mutate SQLite, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID = "knowledge-hub.paper.table-cell-next-action-gate.v1"
TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-pymupdf-pairing-diagnostic.v1"
)
LOCAL_TABLE_CELL_ALTERNATIVE_EXTRACTOR_AVAILABILITY_SCHEMA_ID = (
    "knowledge-hub.local.table-cell-alternative-extractor-availability.v1"
)
LOCAL_TABLE_CELL_PYMUPDF_OVERLAY_VISUAL_REVIEW_SCHEMA_ID = (
    "knowledge-hub.local.table-cell-pymupdf-overlay-visual-review.v1"
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
        return int(value or 0)
    except Exception:
        return 0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _strict_zero_violations(prefix: str, report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    policy = dict(report.get("policy") or {})
    for key in (
        "cellBboxTextPairingVerifiedRows",
        "cellSourceSpanCreatedRows",
        "cellSourceHashLinkedRows",
        "tableCellEvidenceCreatedRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            violations.append(f"{prefix}_{key}_nonzero")
    for key in (
        "cellBboxTextPairingVerified",
        "cellSourceSpansCreated",
        "cellSourceHashLinked",
        "extractorChoiceMade",
        "tableCellEvidenceReady",
        "tableCellCitationGradeReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            violations.append(f"{prefix}_{key}_true")
    for key in (
        "extractorChoiceMade",
        "tableCellEvidenceCreated",
        "tableCellCitationGradeEvidenceCreated",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            violations.append(f"{prefix}_{key}_true")
    return list(dict.fromkeys(violations))


def _schema_violations(pairing: dict[str, Any], alternative: dict[str, Any], visual: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if pairing.get("schema") != TABLE_CELL_PYMUPDF_PAIRING_DIAGNOSTIC_SCHEMA_ID:
        violations.append("table_cell_pymupdf_pairing_diagnostic_schema_mismatch")
    if pairing.get("status") != "diagnostic_ready":
        violations.append("table_cell_pymupdf_pairing_diagnostic_not_ready")
    if alternative.get("schema") != LOCAL_TABLE_CELL_ALTERNATIVE_EXTRACTOR_AVAILABILITY_SCHEMA_ID:
        violations.append("table_cell_alternative_extractor_availability_schema_mismatch")
    if visual.get("schema") != LOCAL_TABLE_CELL_PYMUPDF_OVERLAY_VISUAL_REVIEW_SCHEMA_ID:
        violations.append("table_cell_pymupdf_overlay_visual_review_schema_mismatch")
    return violations


def _pairing_action_card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_card_id": f"table-cell-next-action:{index:04d}",
        "source_report": "table_cell_pymupdf_pairing_diagnostic",
        "source_row_id": str(row.get("diagnostic_row_id") or ""),
        "source_table_region_candidate_id": str(row.get("source_table_region_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "table_label": _clean_text(row.get("table_label")),
        "action_type": "human_review_pymupdf_pairing_diagnostic",
        "action_status": "ready_for_human_review",
        "priority": "high",
        "reason": "pymupdf produced bbox/text candidates but pairing and source spans are unverified",
        "cell_bbox_candidate_count": _safe_int(row.get("cell_bbox_candidate_count")),
        "cell_text_candidate_count": _safe_int(row.get("cell_text_candidate_count")),
        "cell_pairing_candidate_count": _safe_int(row.get("cell_pairing_candidate_count")),
        "diagnostic_unique_cell_text_matches": _safe_int(row.get("diagnostic_unique_cell_text_matches")),
        "diagnostic_ambiguous_cell_text_matches": _safe_int(row.get("diagnostic_ambiguous_cell_text_matches")),
        "diagnostic_no_match_cell_texts": _safe_int(row.get("diagnostic_no_match_cell_texts")),
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "human_pairing_review_not_completed",
            "cell_bbox_text_pairing_not_verified",
            "cell_source_spans_not_created",
            "table_cell_source_hash_linkage_not_created",
            "strict_promotion_requires_explicit_later_tranche",
        ],
    }


def _alternative_action_card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_card_id": f"table-cell-next-action:{index:04d}",
        "source_report": "table_cell_alternative_extractor_availability",
        "source_row_id": str(row.get("availability_row_id") or ""),
        "source_table_region_candidate_id": str(row.get("source_table_region_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "table_label": _clean_text(row.get("table_label")),
        "action_type": "approve_isolated_alternative_extractor_dependency_pilot",
        "action_status": "blocked_requires_explicit_approval",
        "priority": "medium",
        "reason": str(row.get("blocked_reason") or "local alternative extractor is unavailable"),
        "cell_bbox_candidate_count": 0,
        "cell_text_candidate_count": 0,
        "cell_pairing_candidate_count": 0,
        "diagnostic_unique_cell_text_matches": 0,
        "diagnostic_ambiguous_cell_text_matches": 0,
        "diagnostic_no_match_cell_texts": 0,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "alternative_extractor_not_available",
            "dependency_pilot_requires_explicit_approval",
            "cell_source_spans_not_created",
            "table_cell_source_hash_linkage_not_created",
            "strict_promotion_requires_explicit_later_tranche",
        ],
    }


def _visual_action_card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_card_id": f"table-cell-next-action:{index:04d}",
        "source_report": "table_cell_pymupdf_overlay_visual_review",
        "source_row_id": str(row.get("visual_review_row_id") or ""),
        "source_table_region_candidate_id": str(row.get("source_table_region_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "table_label": _clean_text(row.get("table_label")),
        "action_type": "inspect_visual_overlay_asset",
        "action_status": "ready_for_human_review",
        "priority": "medium",
        "reason": "local rendered image and SVG overlay are available but visual pairing is not completed",
        "page_image_path": str(row.get("page_image_path") or ""),
        "svg_overlay_path": str(row.get("svg_overlay_path") or ""),
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "visual_pairing_review_not_completed",
            "rendered_overlay_is_not_evidence",
            "cell_source_spans_not_created",
            "strict_promotion_requires_explicit_later_tranche",
        ],
    }


def _counts(cards: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in cards:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "nextActionCards": len(cards),
        "humanReviewCards": sum(1 for item in cards if item.get("action_status") == "ready_for_human_review"),
        "dependencyApprovalRequiredCards": sum(
            1 for item in cards if item.get("action_status") == "blocked_requires_explicit_approval"
        ),
        "pymupdfPairingDiagnosticCards": sum(
            1 for item in cards if item.get("action_type") == "human_review_pymupdf_pairing_diagnostic"
        ),
        "visualOverlayReviewCards": sum(1 for item in cards if item.get("action_type") == "inspect_visual_overlay_asset"),
        "alternativeExtractorApprovalCards": sum(
            1 for item in cards if item.get("action_type") == "approve_isolated_alternative_extractor_dependency_pilot"
        ),
        "cellBboxCandidateCount": sum(_safe_int(item.get("cell_bbox_candidate_count")) for item in cards),
        "cellTextCandidateCount": sum(_safe_int(item.get("cell_text_candidate_count")) for item in cards),
        "cellPairingCandidateCount": sum(_safe_int(item.get("cell_pairing_candidate_count")) for item in cards),
        "diagnosticUniqueCellTextMatches": sum(
            _safe_int(item.get("diagnostic_unique_cell_text_matches")) for item in cards
        ),
        "diagnosticAmbiguousCellTextMatches": sum(
            _safe_int(item.get("diagnostic_ambiguous_cell_text_matches")) for item in cards
        ),
        "diagnosticNoMatchCellTexts": sum(_safe_int(item.get("diagnostic_no_match_cell_texts")) for item in cards),
        "strictEligibleCards": 0,
        "citationGradeCards": 0,
        "runtimeEvidenceCards": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byActionType": dict(Counter(str(item.get("action_type") or "") for item in cards)),
        "byActionStatus": dict(Counter(str(item.get("action_status") or "") for item in cards)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_next_action_gate(
    *,
    table_cell_pymupdf_pairing_diagnostic: str | Path,
    table_cell_alternative_extractor_availability: str | Path,
    table_cell_pymupdf_overlay_visual_review: str | Path,
) -> dict[str, Any]:
    """Build a report-only next-action gate for current TableCell diagnostics."""

    pairing_path = Path(str(table_cell_pymupdf_pairing_diagnostic)).expanduser()
    alternative_path = Path(str(table_cell_alternative_extractor_availability)).expanduser()
    visual_path = Path(str(table_cell_pymupdf_overlay_visual_review)).expanduser()
    pairing = _read_json(pairing_path)
    alternative = _read_json(alternative_path)
    visual = _read_json(visual_path)
    violations = [
        *_schema_violations(pairing, alternative, visual),
        *_strict_zero_violations("pairing", pairing),
        *_strict_zero_violations("alternative", alternative),
        *_strict_zero_violations("visual", visual),
    ]
    cards: list[dict[str, Any]] = []
    for row in list(pairing.get("diagnosticRows") or []):
        if isinstance(row, dict):
            cards.append(_pairing_action_card(len(cards) + 1, row))
    for row in list(alternative.get("rows") or []):
        if isinstance(row, dict):
            cards.append(_alternative_action_card(len(cards) + 1, row))
    for row in list(visual.get("rows") or []):
        if isinstance(row, dict):
            cards.append(_visual_action_card(len(cards) + 1, row))
    counts = _counts(cards, violations)
    ready = not violations and bool(cards)
    return {
        "schema": TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID,
        "status": "next_action_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "tableCellPymupdfPairingDiagnostic": str(pairing_path),
            "tableCellAlternativeExtractorAvailability": str(alternative_path),
            "tableCellPymupdfOverlayVisualReview": str(visual_path),
            "tableCellPymupdfPairingDiagnosticSchema": str(pairing.get("schema") or ""),
            "tableCellAlternativeExtractorAvailabilitySchema": str(alternative.get("schema") or ""),
            "tableCellPymupdfOverlayVisualReviewSchema": str(visual.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "nextActionReady": ready,
            "humanReviewRequired": counts["humanReviewCards"] > 0,
            "dependencyApprovalRequired": counts["dependencyApprovalRequiredCards"] > 0,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "extractorChoiceMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "ready_for_operator_next_action" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "human_review_table_cell_pairing_or_approve_isolated_extractor_pilot",
        },
        "policy": {
            "reportOnly": True,
            "gateOnly": True,
            "extractorChoiceMade": False,
            "tableCellEvidenceCreated": False,
            "tableCellCitationGradeEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "next_action_gate_is_not_runtime_evidence",
            "human_review_or_dependency_approval_is_required_before_more_table_cell_progress",
            "no_table_cell_citation_grade_evidence_is_created",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "nextActionCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "warnings",
            "nextActionCards",
        )
        if key in report
    }


def render_table_cell_next_action_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell Next Action Gate",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Human review required: `{bool(gate.get('humanReviewRequired'))}`",
        f"- Dependency approval required: `{bool(gate.get('dependencyApprovalRequired'))}`",
        f"- Next-action cards: `{int(counts.get('nextActionCards') or 0)}`",
        f"- Pairing diagnostic cards: `{int(counts.get('pymupdfPairingDiagnosticCards') or 0)}`",
        f"- Visual overlay review cards: `{int(counts.get('visualOverlayReviewCards') or 0)}`",
        f"- Alternative extractor approval cards: `{int(counts.get('alternativeExtractorApprovalCards') or 0)}`",
        f"- Cell pairing candidates: `{int(counts.get('cellPairingCandidateCount') or 0)}`",
        f"- Strict eligible cards: `{int(counts.get('strictEligibleCards') or 0)}`",
        "",
        "## Boundary",
        "",
        "This gate only selects the next operator action. It does not verify pairing, choose an extractor, create table-cell evidence, promote strict evidence, route parsers, or change answer runtime behavior.",
        "",
        "## Counts",
        "",
        f"- By action type: `{json.dumps(counts.get('byActionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By action status: `{json.dumps(counts.get('byActionStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Next Actions",
        "",
    ]
    for item in list(report.get("nextActionCards") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` "
            f"-> `{item.get('action_type')}` / `{item.get('action_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_next_action_gate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "table-cell-next-action-gate.json"
    summary_path = root / "table-cell-next-action-gate-summary.json"
    markdown_path = root / "table-cell-next-action-gate.md"
    gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_next_action_gate_markdown(report), encoding="utf-8")
    return {"gate": str(gate_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell next-action gate.")
    parser.add_argument("--table-cell-pymupdf-pairing-diagnostic", required=True)
    parser.add_argument("--table-cell-alternative-extractor-availability", required=True)
    parser.add_argument("--table-cell-pymupdf-overlay-visual-review", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_next_action_gate(
        table_cell_pymupdf_pairing_diagnostic=args.table_cell_pymupdf_pairing_diagnostic,
        table_cell_alternative_extractor_availability=args.table_cell_alternative_extractor_availability,
        table_cell_pymupdf_overlay_visual_review=args.table_cell_pymupdf_overlay_visual_review,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_next_action_gate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_NEXT_ACTION_GATE_SCHEMA_ID",
    "build_table_cell_next_action_gate",
    "render_table_cell_next_action_gate_markdown",
    "write_table_cell_next_action_gate_reports",
]
