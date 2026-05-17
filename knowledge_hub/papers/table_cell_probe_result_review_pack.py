"""Report-only TableCell probe result review pack helpers.

This module turns the bounded TableCell extractor pilot rows into operator
review cards.  It does not choose a table extractor, verify cell/text pairing,
create cell source spans, create table-cell evidence, route parsers, write
canonical parsed artifacts, mutate SQLite, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-probe-result-review-pack.v1"
)
TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-bbox-source-span-extractor-pilot.v1"
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
    if report.get("schema") != TABLE_CELL_BBOX_SOURCE_SPAN_EXTRACTOR_PILOT_SCHEMA_ID:
        violations.append("table_cell_bbox_source_span_extractor_pilot_schema_mismatch")
    if report.get("status") != "pilot_complete":
        violations.append("table_cell_bbox_source_span_extractor_pilot_not_complete")
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


def _review_status(probe_status: str) -> str:
    if probe_status == "cell_bbox_candidates_detected_non_strict":
        return "pymupdf_overlay_candidate_needs_pairing_review"
    if probe_status == "no_tables_detected":
        return "requires_alternative_table_extractor"
    if probe_status == "held_out_authority_design_not_ready":
        return "held_out_authority_design_not_ready"
    if probe_status == "held_out_max_ready_rows_exceeded":
        return "held_out_max_ready_rows_exceeded"
    if probe_status.startswith("blocked_"):
        return "blocked_probe_input"
    return "blocked_probe_failed_or_unknown"


def _recommended_review_action(review_status: str) -> str:
    if review_status == "pymupdf_overlay_candidate_needs_pairing_review":
        return "review_pymupdf_cell_bbox_overlay_before_source_span_work"
    if review_status == "requires_alternative_table_extractor":
        return "pilot_alternative_table_extractor_for_this_row"
    if review_status == "held_out_authority_design_not_ready":
        return "resolve_caption_source_offset_or_keep_held_out"
    if review_status == "held_out_max_ready_rows_exceeded":
        return "rerun_bounded_probe_with_explicit_row_if_still_needed"
    if review_status == "blocked_probe_input":
        return "fix_probe_input_before_retry"
    return "inspect_probe_failure_before_retry"


def _review_card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    probe_status = str(row.get("probe_status") or "")
    review_status = _review_status(probe_status)
    strict_blockers = list(
        dict.fromkeys(
            [
                *[str(value) for value in list(row.get("strict_blockers") or []) if str(value)],
                "table_cell_probe_result_review_pack_only",
                "extractor_choice_not_made",
                "cell_bbox_text_pairing_not_verified",
                "cell_source_spans_not_created",
                "table_cell_source_hash_linkage_not_created",
                "table_cell_citation_grade_evidence_not_created",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
                review_status,
            ]
        )
    )
    return {
        "review_card_id": f"table-cell-probe-result-review:{index:04d}",
        "source_pilot_row_id": str(row.get("pilot_row_id") or ""),
        "source_authority_design_id": str(row.get("source_authority_design_id") or ""),
        "source_review_card_id": str(row.get("source_review_card_id") or ""),
        "source_table_region_candidate_id": str(row.get("source_table_region_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_type": "table_cell_probe_result_review_card",
        "source_parser": "pymupdf_table_probe",
        "table_label": _clean_text(row.get("table_label")),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "page": row.get("page") if isinstance(row.get("page"), int) else None,
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "probe_status": probe_status,
        "review_status": review_status,
        "recommended_review_action": _recommended_review_action(review_status),
        "probe_attempted": bool(row.get("probe_attempted")),
        "probe_failure_reason": str(row.get("probe_failure_reason") or ""),
        "detected_table_count": _safe_int(row.get("detected_table_count")),
        "selected_table_index": _safe_int(row.get("selected_table_index")),
        "selected_table_bbox": row.get("selected_table_bbox"),
        "selected_table_row_count": _safe_int(row.get("selected_table_row_count")),
        "selected_table_column_count": _safe_int(row.get("selected_table_column_count")),
        "selected_table_cell_bbox_count": _safe_int(row.get("selected_table_cell_bbox_count")),
        "selected_table_cell_text_count": _safe_int(row.get("selected_table_cell_text_count")),
        "diagnostic_unique_cell_text_matches": _safe_int(row.get("diagnostic_unique_cell_text_matches")),
        "diagnostic_ambiguous_cell_text_matches": _safe_int(
            row.get("diagnostic_ambiguous_cell_text_matches")
        ),
        "diagnostic_no_match_cell_texts": _safe_int(row.get("diagnostic_no_match_cell_texts")),
        "sample_cell_bboxes": list(row.get("sample_cell_bboxes") or []),
        "sample_cell_text_matches": list(row.get("sample_cell_text_matches") or []),
        "cell_bbox_candidates_observed": bool(row.get("table_cell_bbox_candidates_found")),
        "cell_bbox_text_pairing_verified": False,
        "cell_source_spans_created": 0,
        "cell_source_hash_linkages_created": 0,
        "table_cell_evidence_created": False,
        "table_cell_citation_grade": False,
        "extractor_choice_made": False,
        "evidence_tier": "table_cell_probe_result_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "probe_result_review_card_only",
            "extractor_choice_not_made",
            "cell_bbox_text_pairing_not_verified",
            "cell_source_spans_not_created",
            "table_cell_source_hash_linkage_not_created",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(cards: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in cards:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "reviewCardRows": len(cards),
        "pymupdfOverlayCandidateRows": sum(
            1
            for item in cards
            if item.get("review_status") == "pymupdf_overlay_candidate_needs_pairing_review"
        ),
        "alternativeExtractorRequiredRows": sum(
            1 for item in cards if item.get("review_status") == "requires_alternative_table_extractor"
        ),
        "heldOutRows": sum(1 for item in cards if str(item.get("review_status") or "").startswith("held_out")),
        "blockedRows": sum(1 for item in cards if str(item.get("review_status") or "").startswith("blocked")),
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
        "cellBboxTextPairingVerifiedRows": 0,
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
        "byProbeStatus": dict(Counter(str(item.get("probe_status") or "") for item in cards)),
        "byReviewStatus": dict(Counter(str(item.get("review_status") or "") for item in cards)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_probe_result_review_pack(
    *,
    table_cell_extractor_pilot_report: str | Path,
) -> dict[str, Any]:
    """Build report-only review cards from the TableCell extractor pilot."""

    path = Path(str(table_cell_extractor_pilot_report)).expanduser()
    report = _read_json(path)
    violations = [*_schema_violations(report), *_unsafe_flags(report)]
    cards = [
        _review_card(index + 1, dict(row))
        for index, row in enumerate(list(report.get("pilotRows") or []))
        if isinstance(row, dict)
    ]
    counts = _counts(cards, violations)
    blocked = bool(violations) or not cards
    return {
        "schema": TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID,
        "status": "blocked" if blocked else "review_pack_ready",
        "generatedAt": _now(),
        "inputs": {
            "tableCellExtractorPilotReport": str(path),
            "tableCellExtractorPilotSchema": str(report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": not blocked,
            "pymupdfOverlayCandidateObserved": counts["pymupdfOverlayCandidateRows"] > 0,
            "alternativeExtractorRequired": counts["alternativeExtractorRequiredRows"] > 0,
            "extractorChoiceMade": False,
            "cellBboxTextPairingVerified": False,
            "cellSourceSpansCreated": False,
            "cellSourceHashLinked": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "ready_for_table_cell_probe_human_review" if not blocked else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "choose_or_pilot_alternative_table_cell_extractor_for_no_table_rows",
        },
        "policy": {
            "reportOnly": True,
            "reviewOnly": True,
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
            "probe_review_cards_are_not_runtime_evidence",
            "pymupdf_cell_bbox_candidates_need_manual_pairing_review",
            "no_table_detected_rows_need_alternative_extractor_pilot_before_any_table_cell_evidence",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "reviewCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "reviewCards")
        if key in report
    }


def render_table_cell_probe_result_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell Probe Result Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardRows') or 0)}`",
        f"- PyMuPDF overlay candidates: `{int(counts.get('pymupdfOverlayCandidateRows') or 0)}`",
        f"- Alternative extractor required rows: `{int(counts.get('alternativeExtractorRequiredRows') or 0)}`",
        f"- Held out rows: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Cell bbox candidate rows: `{int(counts.get('cellBboxCandidateRows') or 0)}`",
        f"- Selected table cell bbox candidates: `{int(counts.get('selectedTableCellBboxCandidates') or 0)}`",
        f"- Diagnostic unique cell text matches: `{int(counts.get('diagnosticUniqueCellTextMatches') or 0)}`",
        f"- Diagnostic ambiguous cell text matches: `{int(counts.get('diagnosticAmbiguousCellTextMatches') or 0)}`",
        f"- Cell source span created rows: `{int(counts.get('cellSourceSpanCreatedRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These cards are for human/operator review only. PyMuPDF table probe results can suggest overlay work or alternative extractor work, but they do not verify cell/text pairing, create original source spans, create cell-level source hash linkage, or authorize table-cell citation-grade evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By probe status: `{json.dumps(counts.get('byProbeStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By review status: `{json.dumps(counts.get('byReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Review Cards",
        "",
    ]
    for item in list(report.get("reviewCards") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('table_label')}` "
            f"probe `{item.get('probe_status')}` -> `{item.get('review_status')}` "
            f"action `{item.get('recommended_review_action')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_table_cell_probe_result_review_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "table-cell-probe-result-review-cards.json"
    summary_path = root / "table-cell-probe-result-review-summary.json"
    markdown_path = root / "table-cell-probe-result-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_probe_result_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell probe result review cards.")
    parser.add_argument("--table-cell-extractor-pilot-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_probe_result_review_pack(
        table_cell_extractor_pilot_report=args.table_cell_extractor_pilot_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_probe_result_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_PROBE_RESULT_REVIEW_PACK_SCHEMA_ID",
    "build_table_cell_probe_result_review_pack",
    "render_table_cell_probe_result_review_pack_markdown",
    "write_table_cell_probe_result_review_pack_reports",
]
