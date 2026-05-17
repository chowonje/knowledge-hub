"""Report-only PyMuPDF table-cell overlay review pack helpers.

This module consumes TableCell probe result review cards and narrows the rows
where PyMuPDF produced cell bbox candidates.  The output is a human/operator
review pack only: it does not verify cell/text pairing, create cell source
spans, create table-cell evidence, choose an extractor, route parsers, write
canonical parsed artifacts, mutate SQLite, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-pymupdf-overlay-review-pack.v1"
)
TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-probe-result-review-pack.v1"
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


def _schema_violations(report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if report.get("schema") != TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID:
        violations.append("table_cell_probe_result_review_pack_schema_mismatch")
    if report.get("status") != "review_pack_ready":
        violations.append("table_cell_probe_result_review_pack_not_ready")
    return violations


def _unsafe_flags(report: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
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
            unsafe.append(f"{key}_nonzero")
    for key in (
        "extractorChoiceMade",
        "cellBboxTextPairingVerified",
        "cellSourceSpansCreated",
        "cellSourceHashLinked",
        "tableCellEvidenceReady",
        "tableCellCitationGradeReady",
        "strictEvidenceReady",
        "parserRoutingReady",
        "answerIntegrationReady",
        "runtimePromotionAllowed",
    ):
        if bool(gate.get(key)):
            unsafe.append(f"{key}_true")
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
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _overlay_review_status(card: dict[str, Any]) -> str:
    if card.get("review_status") != "pymupdf_overlay_candidate_needs_pairing_review":
        return "held_out_not_pymupdf_overlay_candidate"
    if not bool(card.get("cell_bbox_candidates_observed")):
        return "blocked_no_cell_bbox_candidates"
    text_count = _safe_int(card.get("selected_table_cell_text_count"))
    unique = _safe_int(card.get("diagnostic_unique_cell_text_matches"))
    ambiguous = _safe_int(card.get("diagnostic_ambiguous_cell_text_matches"))
    missing = _safe_int(card.get("diagnostic_no_match_cell_texts"))
    if text_count > 0 and unique == text_count and ambiguous == 0 and missing == 0:
        return "overlay_candidate_ready_for_visual_pairing_review"
    return "manual_overlay_pairing_review_required"


def _recommended_review_action(status: str) -> str:
    if status == "overlay_candidate_ready_for_visual_pairing_review":
        return "visually_review_bbox_text_pairing_before_any_source_span_work"
    if status == "manual_overlay_pairing_review_required":
        return "manually_compare_cell_bboxes_to_page_text_before_any_extractor_choice"
    if status == "blocked_no_cell_bbox_candidates":
        return "return_to_alternative_extractor_review"
    return "keep_out_of_overlay_review"


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _review_card(index: int, card: dict[str, Any]) -> dict[str, Any]:
    status = _overlay_review_status(card)
    text_count = _safe_int(card.get("selected_table_cell_text_count"))
    unique = _safe_int(card.get("diagnostic_unique_cell_text_matches"))
    ambiguous = _safe_int(card.get("diagnostic_ambiguous_cell_text_matches"))
    missing = _safe_int(card.get("diagnostic_no_match_cell_texts"))
    strict_blockers = list(
        dict.fromkeys(
            [
                *[str(value) for value in list(card.get("strict_blockers") or []) if str(value)],
                "table_cell_pymupdf_overlay_review_pack_only",
                "visual_cell_bbox_text_pairing_not_verified",
                "cell_source_spans_not_created",
                "table_cell_source_hash_linkage_not_created",
                "table_cell_citation_grade_evidence_not_created",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                status,
            ]
        )
    )
    return {
        "overlay_review_card_id": f"table-cell-pymupdf-overlay-review:{index:04d}",
        "source_probe_review_card_id": str(card.get("review_card_id") or ""),
        "source_table_region_candidate_id": str(card.get("source_table_region_candidate_id") or ""),
        "paper_id": str(card.get("paper_id") or ""),
        "candidate_type": "table_cell_pymupdf_overlay_review_card",
        "source_parser": "pymupdf_table_probe",
        "table_label": _clean_text(card.get("table_label")),
        "candidate_text": _clean_text(card.get("candidate_text")),
        "page": card.get("page") if isinstance(card.get("page"), int) else None,
        "sourceContentHash": str(card.get("sourceContentHash") or ""),
        "overlay_review_status": status,
        "recommended_review_action": _recommended_review_action(status),
        "selected_table_bbox": card.get("selected_table_bbox"),
        "selected_table_row_count": _safe_int(card.get("selected_table_row_count")),
        "selected_table_column_count": _safe_int(card.get("selected_table_column_count")),
        "selected_table_cell_bbox_count": _safe_int(card.get("selected_table_cell_bbox_count")),
        "selected_table_cell_text_count": text_count,
        "diagnostic_unique_cell_text_matches": unique,
        "diagnostic_ambiguous_cell_text_matches": ambiguous,
        "diagnostic_no_match_cell_texts": missing,
        "diagnostic_unique_match_ratio": _ratio(unique, text_count),
        "diagnostic_ambiguous_match_ratio": _ratio(ambiguous, text_count),
        "diagnostic_no_match_ratio": _ratio(missing, text_count),
        "sample_cell_bboxes": list(card.get("sample_cell_bboxes") or []),
        "sample_cell_text_matches": list(card.get("sample_cell_text_matches") or []),
        "cell_bbox_candidates_observed": bool(card.get("cell_bbox_candidates_observed")),
        "cell_bbox_text_pairing_verified": False,
        "visual_pairing_review_completed": False,
        "cell_source_spans_created": 0,
        "cell_source_hash_linkages_created": 0,
        "table_cell_evidence_created": False,
        "table_cell_citation_grade": False,
        "extractor_choice_made": False,
        "evidence_tier": "table_cell_pymupdf_overlay_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "pymupdf_overlay_review_card_only",
            "visual_cell_bbox_text_pairing_not_verified",
            "diagnostic_cell_text_matches_are_not_cell_source_spans",
            "cell_source_spans_not_created",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(cards: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in cards:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "overlayReviewCardRows": len(cards),
        "manualOverlayPairingReviewRequiredRows": sum(
            1 for item in cards if item.get("overlay_review_status") == "manual_overlay_pairing_review_required"
        ),
        "visualPairingReviewReadyRows": sum(
            1
            for item in cards
            if item.get("overlay_review_status") == "overlay_candidate_ready_for_visual_pairing_review"
        ),
        "blockedRows": sum(1 for item in cards if str(item.get("overlay_review_status") or "").startswith("blocked")),
        "heldOutRows": sum(1 for item in cards if str(item.get("overlay_review_status") or "").startswith("held_out")),
        "cellBboxCandidateRows": sum(1 for item in cards if item.get("cell_bbox_candidates_observed")),
        "selectedTableCellBboxCandidates": sum(
            _safe_int(item.get("selected_table_cell_bbox_count")) for item in cards
        ),
        "selectedTableCellTextCandidates": sum(
            _safe_int(item.get("selected_table_cell_text_count")) for item in cards
        ),
        "diagnosticUniqueCellTextMatches": sum(
            _safe_int(item.get("diagnostic_unique_cell_text_matches")) for item in cards
        ),
        "diagnosticAmbiguousCellTextMatches": sum(
            _safe_int(item.get("diagnostic_ambiguous_cell_text_matches")) for item in cards
        ),
        "diagnosticNoMatchCellTexts": sum(_safe_int(item.get("diagnostic_no_match_cell_texts")) for item in cards),
        "cellBboxTextPairingVerifiedRows": 0,
        "visualPairingReviewCompletedRows": 0,
        "cellSourceSpanCreatedRows": 0,
        "cellSourceHashLinkedRows": 0,
        "tableCellEvidenceCreatedRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "byOverlayReviewStatus": dict(Counter(str(item.get("overlay_review_status") or "") for item in cards)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_pymupdf_overlay_review_pack(
    *,
    table_cell_probe_result_review_pack: str | Path,
) -> dict[str, Any]:
    """Build report-only review cards for PyMuPDF table overlay candidates."""

    path = Path(str(table_cell_probe_result_review_pack)).expanduser()
    report = _read_json(path)
    violations = [*_schema_violations(report), *_unsafe_flags(report)]
    source_cards = [
        dict(card)
        for card in list(report.get("reviewCards") or [])
        if isinstance(card, dict)
        and card.get("review_status") == "pymupdf_overlay_candidate_needs_pairing_review"
    ]
    cards = [_review_card(index + 1, card) for index, card in enumerate(source_cards)]
    counts = _counts(cards, violations)
    blocked = bool(violations) or not cards
    return {
        "schema": TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID,
        "status": "blocked" if blocked else "review_pack_ready",
        "generatedAt": _now(),
        "inputs": {
            "tableCellProbeResultReviewPack": str(path),
            "tableCellProbeResultReviewPackSchema": str(report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "overlayReviewPackReady": not blocked,
            "manualOverlayPairingReviewRequired": counts["manualOverlayPairingReviewRequiredRows"] > 0,
            "visualPairingReviewCompleted": False,
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
            "decision": "ready_for_pymupdf_overlay_human_review" if not blocked else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "manual_or_rendered_overlay_pairing_review_for_pymupdf_candidates",
        },
        "policy": {
            "reportOnly": True,
            "reviewOnly": True,
            "visualReviewOnly": True,
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
            "overlay_review_cards_are_not_runtime_evidence",
            "diagnostic_text_matches_are_not_cell_source_spans",
            "manual_or_rendered_cell_bbox_text_pairing_is_required",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "overlayReviewCards": cards,
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
            "overlayReviewCards",
        )
        if key in report
    }


def render_table_cell_pymupdf_overlay_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell PyMuPDF Overlay Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Overlay review cards: `{int(counts.get('overlayReviewCardRows') or 0)}`",
        f"- Manual overlay pairing review required: `{int(counts.get('manualOverlayPairingReviewRequiredRows') or 0)}`",
        f"- Cell bbox candidate rows: `{int(counts.get('cellBboxCandidateRows') or 0)}`",
        f"- Selected table cell bbox candidates: `{int(counts.get('selectedTableCellBboxCandidates') or 0)}`",
        f"- Diagnostic unique cell text matches: `{int(counts.get('diagnosticUniqueCellTextMatches') or 0)}`",
        f"- Diagnostic ambiguous cell text matches: `{int(counts.get('diagnosticAmbiguousCellTextMatches') or 0)}`",
        f"- Diagnostic no-match cell texts: `{int(counts.get('diagnosticNoMatchCellTexts') or 0)}`",
        f"- Visual pairing review completed rows: `{int(counts.get('visualPairingReviewCompletedRows') or 0)}`",
        f"- Cell source span created rows: `{int(counts.get('cellSourceSpanCreatedRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These cards are for human/operator overlay review only. Sample bboxes and diagnostic text matches do not verify cell/text pairing, create source spans, or authorize table-cell citation-grade evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By overlay review status: `{json.dumps(counts.get('byOverlayReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Overlay Review Cards",
        "",
    ]
    for item in list(report.get("overlayReviewCards") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` page `{item.get('page')}` "
            f"bbox `{item.get('selected_table_cell_bbox_count')}` "
            f"unique `{item.get('diagnostic_unique_cell_text_matches')}` "
            f"ambiguous `{item.get('diagnostic_ambiguous_cell_text_matches')}` "
            f"missing `{item.get('diagnostic_no_match_cell_texts')}` "
            f"-> `{item.get('overlay_review_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_pymupdf_overlay_review_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "table-cell-pymupdf-overlay-review-cards.json"
    summary_path = root / "table-cell-pymupdf-overlay-review-summary.json"
    markdown_path = root / "table-cell-pymupdf-overlay-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_pymupdf_overlay_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only PyMuPDF TableCell overlay review cards.")
    parser.add_argument("--table-cell-probe-result-review-pack", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_pymupdf_overlay_review_pack(
        table_cell_probe_result_review_pack=args.table_cell_probe_result_review_pack,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_pymupdf_overlay_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_PYMUPDF_OVERLAY_REVIEW_PACK_SCHEMA_ID",
    "build_table_cell_pymupdf_overlay_review_pack",
    "render_table_cell_pymupdf_overlay_review_pack_markdown",
    "write_table_cell_pymupdf_overlay_review_pack_reports",
]
