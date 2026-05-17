"""Report-only TableCell bbox/source-span authority design.

This helper documents what would be required before TableCell candidates could
become citation-grade evidence.  It consumes the TableCell provenance review
pack and emits design rows/options only; it does not extract cells, choose
authority, create strict evidence, route parsers, write artifacts, mutate DB,
reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-bbox-source-span-authority-design.v1"
)
TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.table-cell-provenance-review-pack.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _schema_violations(review_pack: dict[str, Any]) -> list[str]:
    if review_pack.get("schema") != TABLE_CELL_PROVENANCE_REVIEW_PACK_SCHEMA_ID:
        return ["table_cell_provenance_review_pack_schema_mismatch"]
    return []


def _unsafe_flags(review_pack: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(review_pack.get("counts") or {})
    gate = dict(review_pack.get("gate") or {})
    policy = dict(review_pack.get("policy") or {})
    if review_pack.get("status") != "review_pack_ready":
        unsafe.append("table_cell_provenance_review_pack_not_ready")
    for key in (
        "tableCellEvidenceVerifiedRows",
        "tableCellCitationGradeRows",
        "strictEligibleRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    for key in (
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


def _design_status(card: dict[str, Any]) -> str:
    if card.get("review_status") == "ready_for_cell_provenance_review":
        return "ready_for_cell_bbox_source_span_authority_design"
    if card.get("review_status") == "held_out_caption_source_offset_missing":
        return "blocked_caption_source_offset_missing"
    if not card.get("table_structure_available"):
        return "blocked_table_structure_missing"
    if not card.get("row_column_text_available"):
        return "blocked_row_column_text_missing"
    return "blocked_table_cell_review_not_ready"


def _design_row(index: int, card: dict[str, Any]) -> dict[str, Any]:
    status = _design_status(card)
    blockers = list(
        dict.fromkeys(
            [
                *[str(value) for value in list(card.get("strict_blockers") or []) if str(value)],
                "table_cell_bbox_source_span_authority_design_only",
                "authority_decision_not_made",
                "per_cell_bbox_source_span_and_hash_not_available",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
            ]
        )
    )
    return {
        "design_id": f"table-cell-authority-design:{index:04d}",
        "source_review_card_id": str(card.get("review_card_id") or ""),
        "source_table_region_candidate_id": str(card.get("source_table_region_candidate_id") or ""),
        "paper_id": str(card.get("paper_id") or ""),
        "candidate_type": "table_cell_bbox_source_span_authority_design_row",
        "source_parser": "mineru+pymupdf_alignment",
        "table_label": _clean_text(card.get("table_label")),
        "candidate_text": _clean_text(card.get("candidate_text")),
        "caption_text": _clean_text(card.get("caption_text")),
        "caption_original_pdf_offset_recovered": bool(card.get("caption_original_pdf_offset_recovered")),
        "original_pdf_span": dict(card.get("original_pdf_span") or {}),
        "table_structure_available": bool(card.get("table_structure_available")),
        "row_column_text_available": bool(card.get("row_column_text_available")),
        "table_row_count": _safe_int(card.get("table_row_count")),
        "table_cell_count": _safe_int(card.get("table_cell_count")),
        "non_empty_table_cell_count": _safe_int(card.get("non_empty_table_cell_count")),
        "cell_bbox_count": _safe_int(card.get("cell_bbox_count")),
        "cell_source_span_count": _safe_int(card.get("cell_source_span_count")),
        "cell_source_hash_count": _safe_int(card.get("cell_source_hash_count")),
        "source_review_status": str(card.get("review_status") or ""),
        "authority_design_status": status,
        "recommended_authority_path": (
            "require_per_cell_bbox_source_span_and_hash_before_strict_table_evidence"
            if status == "ready_for_cell_bbox_source_span_authority_design"
            else "hold_until_caption_source_offset_and_table_structure_are_available"
        ),
        "required_before_strict_table_cell_use": [
            "per_cell_bbox_coordinates",
            "per_cell_original_pdf_chars_start_end",
            "per_cell_sourceContentHash_linkage",
            "verified_row_column_or_cell_index_mapping",
            "caption_and_table_region_link_review",
            "ambiguous_cell_text_match_rejection",
            "protecting_tests_for_no_generated_markdown_cell_promotion",
            "later_explicit_strict_promotion_tranche",
        ],
        "blocked_actions": [
            "strict_evidence_promotion",
            "runtime_answer_citation",
            "parser_routing",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "authority_decision_made": False,
        "table_cell_evidence_ready": False,
        "strict_promotion_ready": False,
        "runtime_promotion_allowed": False,
        "evidence_tier": "table_cell_bbox_source_span_authority_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "this_is_an_authority_design_not_a_cell_evidence_artifact",
            "generated_table_rows_are_not_original_source_cell_spans",
            "per_cell_bbox_source_span_and_hash_linkage_are_required",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _option(
    *,
    option_id: str,
    title: str,
    recommendation: str,
    affected_rows: int,
    pros: list[str],
    cons: list[str],
    required_before_use: list[str],
) -> dict[str, Any]:
    return {
        "option_id": option_id,
        "title": title,
        "authority_posture": "table_cell_authority_design_only",
        "recommendation": recommendation,
        "affected_row_count": affected_rows,
        "pros": pros,
        "cons": cons,
        "required_before_use": required_before_use,
        "blocked_actions": [
            "strict_evidence_promotion",
            "runtime_answer_citation",
            "parser_routing",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "authority_decision_made": False,
        "strict_promotion_ready": False,
        "runtime_promotion_allowed": False,
        "evidence_tier": "table_cell_authority_option_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }


def _options(ready_rows: int) -> list[dict[str, Any]]:
    return [
        _option(
            option_id="keep_table_region_caption_only",
            title="Keep tables as caption/region candidates only",
            recommendation="safe_default_until_cell_authority_exists",
            affected_rows=ready_rows,
            pros=[
                "preserves current strict evidence policy",
                "keeps table regions useful for operator review",
                "requires no runtime or parser routing change",
            ],
            cons=[
                "does not answer table-cell numeric questions safely",
                "does not create row/column/cell citation evidence",
            ],
            required_before_use=["none_for_report_only_review"],
        ),
        _option(
            option_id="recover_per_cell_bbox_source_spans",
            title="Recover per-cell bbox, source spans, and source hash linkage",
            recommendation="recommended_before_any_table_cell_strict_promotion",
            affected_rows=ready_rows,
            pros=[
                "keeps table-cell citations tied to original source provenance",
                "creates a clear gate for numeric table QA",
                "avoids treating generated Markdown cell text as source authority",
            ],
            cons=[
                "requires a table cell extractor or overlay aligner",
                "must handle merged cells, repeated values, and ambiguous numeric text",
            ],
            required_before_use=[
                "bounded_cell_extractor_or_overlay_pilot",
                "per_cell_bbox_coordinates",
                "per_cell_original_pdf_chars_start_end",
                "per_cell_sourceContentHash_linkage",
                "row_column_index_mapping",
                "cell_ambiguity_and_merged_cell_rules",
                "later_explicit_strict_promotion_tranche",
            ],
        ),
        _option(
            option_id="explicitly_authorize_generated_markdown_cells",
            title="Authorize generated Markdown table cells as table-cell authority",
            recommendation="not_recommended_without_durable_policy_decision",
            affected_rows=ready_rows,
            pros=[
                "could expose table structure sooner",
                "reuses existing MinerU-generated rows",
            ],
            cons=[
                "changes evidence authority semantics",
                "does not prove original source cell position",
                "risks citation-grade numeric claims from generated parser output",
            ],
            required_before_use=[
                "explicit_authority_decision",
                "ADR_or_project_state_policy_record",
                "runtime_evidence_policy_update",
                "tests_preventing_summary_or_paraphrase_cell_promotion",
                "later_explicit_strict_promotion_tranche",
            ],
        ),
    ]


def _counts(rows: list[dict[str, Any]], options: list[dict[str, Any]], violations: list[str]) -> dict[str, Any]:
    ready = [
        row for row in rows if row.get("authority_design_status") == "ready_for_cell_bbox_source_span_authority_design"
    ]
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputReviewCards": len(rows),
        "authorityDesignRows": len(rows),
        "readyForCellAuthorityDesignRows": len(ready),
        "heldOutRows": len(rows) - len(ready),
        "optionCount": len(options),
        "authorityDecisionMadeOptions": 0,
        "strictPromotionReadyOptions": 0,
        "runtimePromotionAllowedOptions": 0,
        "totalTableRows": sum(_safe_int(item.get("table_row_count")) for item in rows),
        "totalTableCells": sum(_safe_int(item.get("table_cell_count")) for item in rows),
        "nonEmptyTableCells": sum(_safe_int(item.get("non_empty_table_cell_count")) for item in rows),
        "cellBboxRows": sum(1 for item in rows if _safe_int(item.get("cell_bbox_count")) > 0),
        "cellSourceSpanRows": sum(1 for item in rows if _safe_int(item.get("cell_source_span_count")) > 0),
        "cellSourceHashRows": sum(1 for item in rows if _safe_int(item.get("cell_source_hash_count")) > 0),
        "tableCellEvidenceReadyRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byAuthorityDesignStatus": dict(Counter(str(item.get("authority_design_status") or "") for item in rows)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_cell_bbox_source_span_authority_design(
    *,
    table_cell_provenance_review_pack: str | Path,
) -> dict[str, Any]:
    """Build a report-only TableCell bbox/source-span authority design."""

    path = Path(str(table_cell_provenance_review_pack)).expanduser()
    review_pack = _read_json(path)
    violations = [*_schema_violations(review_pack), *_unsafe_flags(review_pack)]
    source_cards = [dict(item) for item in list(review_pack.get("reviewCards") or []) if isinstance(item, dict)]
    rows = [_design_row(index, card) for index, card in enumerate(source_cards, start=1)]
    ready_rows = sum(
        1 for row in rows if row.get("authority_design_status") == "ready_for_cell_bbox_source_span_authority_design"
    )
    options = _options(ready_rows)
    counts = _counts(rows, options, violations)
    ready = not violations and bool(rows)
    return {
        "schema": TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID,
        "status": "design_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "tableCellProvenanceReviewPack": str(path),
            "tableCellProvenanceReviewPackSchema": str(review_pack.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "authorityDesignReady": ready,
            "authorityDecisionMade": False,
            "tableCellEvidenceReady": False,
            "tableCellCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "table_cell_authority_design_ready_no_decision_made" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "bounded_table_cell_bbox_source_span_extractor_pilot",
        },
        "policy": {
            "reportOnly": True,
            "authorityDecisionMade": False,
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
        "designPrinciples": [
            "table_caption_offsets_do_not_create_cell_level_evidence",
            "generated_markdown_table_cells_are_not_original_source_spans",
            "strict_table_cell_evidence_requires_per_cell_bbox_source_span_and_hash",
            "ambiguous_or_repeated_cell_values_must_fail_closed",
        ],
        "warnings": [
            "this_report_does_not_extract_or_materialize_table_cells",
            "no_table_cell_evidence_or_strict_runtime_citations_are_created",
            "parser_routing_and_answer_integration_remain_disabled",
        ],
        "authorityOptions": options,
        "designRows": rows,
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
            "designPrinciples",
            "warnings",
            "authorityOptions",
            "designRows",
        )
        if key in report
    }


def render_table_cell_bbox_source_span_authority_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableCell BBox / Source-Span Authority Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Authority design rows: `{int(counts.get('authorityDesignRows') or 0)}`",
        f"- Ready for cell authority design: `{int(counts.get('readyForCellAuthorityDesignRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Total table cells under review: `{int(counts.get('totalTableCells') or 0)}`",
        f"- Cell bbox rows: `{int(counts.get('cellBboxRows') or 0)}`",
        f"- Cell source span rows: `{int(counts.get('cellSourceSpanRows') or 0)}`",
        f"- Table-cell citation-grade rows: `{int(counts.get('tableCellCitationGradeRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a design report only. It does not extract per-cell bboxes, create source spans, choose source authority, create strict evidence, route parsers, or wire runtime answers to table cells.",
        "",
        "## Required Before Strict Table-Cell Evidence",
        "",
        "- per-cell bbox coordinates",
        "- per-cell original-PDF `chars:start-end`",
        "- per-cell `sourceContentHash` linkage",
        "- verified row/column or cell index mapping",
        "- ambiguity handling for repeated numeric/text values",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By authority design status: `{json.dumps(counts.get('byAuthorityDesignStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_table_cell_bbox_source_span_authority_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "table-cell-bbox-source-span-authority-design.json"
    summary_path = root / "table-cell-bbox-source-span-authority-design-summary.json"
    markdown_path = root / "table-cell-bbox-source-span-authority-design.md"
    design_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_cell_bbox_source_span_authority_design_markdown(report), encoding="utf-8")
    return {"design": str(design_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableCell bbox/source-span authority design.")
    parser.add_argument("--table-cell-provenance-review-pack", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_cell_bbox_source_span_authority_design(
        table_cell_provenance_review_pack=args.table_cell_provenance_review_pack,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_cell_bbox_source_span_authority_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_CELL_BBOX_SOURCE_SPAN_AUTHORITY_DESIGN_SCHEMA_ID",
    "build_table_cell_bbox_source_span_authority_design",
    "render_table_cell_bbox_source_span_authority_design_markdown",
    "write_table_cell_bbox_source_span_authority_design_reports",
]
