"""Report-only audit for non-SectionSpan original PDF offset feasibility.

This module consolidates FigureCaption, TableRegion, and EquationQuote PDF
offset feasibility reports. It does not read PDFs, mutate SQLite, write
canonical parsed artifacts, change parser routing, or promote candidates into
strict evidence.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.non-sectionspan-pdf-offset-feasibility-audit.v1"
)
FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1"
)
TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
)
EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1"
)

_LAYER_CONFIG = {
    "figure_caption": {
        "schema": FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
        "sourceCandidateKey": "source_figure_caption_candidate_id",
        "regionVerifiedKey": "figure_region_link_verified",
        "readyAction": "figure_caption_region_link_review",
        "blockedAction": "recover_figure_caption_original_pdf_offset_before_region_review",
        "tier": "figure_caption_pdf_offset_feasibility_only",
    },
    "table_region": {
        "schema": TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
        "sourceCandidateKey": "source_table_region_candidate_id",
        "regionVerifiedKey": "table_region_link_verified",
        "readyAction": "table_cell_provenance_review",
        "blockedAction": "recover_table_caption_original_pdf_offset_before_cell_review",
        "tier": "table_region_pdf_offset_feasibility_only",
    },
    "equation_quote": {
        "schema": EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
        "sourceCandidateKey": "source_equation_quote_candidate_id",
        "regionVerifiedKey": "equation_region_link_verified",
        "readyAction": "equation_quote_review",
        "blockedAction": "equation_quote_alignment_feasibility_review",
        "tier": "equation_quote_pdf_offset_feasibility_only",
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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _span_page(row: dict[str, Any]) -> int | None:
    span = row.get("original_pdf_span")
    if not isinstance(span, dict):
        return None
    page = span.get("page")
    try:
        return int(page) if page is not None else None
    except Exception:
        return None


def _source_hash(row: dict[str, Any]) -> str:
    span = row.get("original_pdf_span")
    if isinstance(span, dict) and span.get("sourceContentHash"):
        return str(span.get("sourceContentHash") or "")
    return str(row.get("sourceContentHash") or "")


def _recommended_action(layer: str, row: dict[str, Any]) -> str:
    config = _LAYER_CONFIG[layer]
    if bool(row.get("original_pdf_offset_recovered")):
        return str(config["readyAction"])
    if layer == "equation_quote" and str(row.get("feasibility_status") or "") == "diagnostic_page_context_candidate_only":
        return "equation_quote_normalization_or_layout_review"
    return str(config["blockedAction"])


def _readiness(layer: str, row: dict[str, Any]) -> str:
    recovered = bool(row.get("original_pdf_offset_recovered"))
    if recovered and layer in {"figure_caption", "table_region"}:
        return "ready_for_region_review_non_strict"
    if recovered and layer == "equation_quote":
        return "ready_for_quote_review_non_strict"
    if layer == "equation_quote" and str(row.get("feasibility_status") or "") == "diagnostic_page_context_candidate_only":
        return "diagnostic_page_context_only_non_strict"
    return "blocked_original_pdf_offset_not_recovered"


def _audit_row(layer: str, row: dict[str, Any]) -> dict[str, Any]:
    config = _LAYER_CONFIG[layer]
    source_row_id = str(row.get("feasibility_row_id") or "")
    original_span = row.get("original_pdf_span") if isinstance(row.get("original_pdf_span"), dict) else {}
    diagnostic_candidates = _as_list(row.get("diagnostic_page_candidates"))
    strict_blockers = list(dict.fromkeys(str(item) for item in _as_list(row.get("strict_blockers")) if item))
    non_strict_reason = list(dict.fromkeys(str(item) for item in _as_list(row.get("non_strict_reason")) if item))
    if "non_sectionspan_pdf_offset_feasibility_audit_only" not in non_strict_reason:
        non_strict_reason.append("non_sectionspan_pdf_offset_feasibility_audit_only")
    if "runtime_promotion_disabled_for_tranche" not in strict_blockers:
        strict_blockers.append("runtime_promotion_disabled_for_tranche")
    return {
        "audit_row_id": f"nonsectionspan-pdf-offset:{layer}:{source_row_id}",
        "candidate_layer": layer,
        "candidate_type": f"{layer}_candidate",
        "source_parser": "mineru+pymupdf_alignment",
        "source_feasibility_row_id": source_row_id,
        "source_candidate_id": str(row.get(str(config["sourceCandidateKey"])) or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "feasibility_status": str(row.get("feasibility_status") or ""),
        "readiness": _readiness(layer, row),
        "original_pdf_offset_recovered": bool(row.get("original_pdf_offset_recovered")),
        "original_pdf_span": original_span,
        "page": _span_page(row),
        "sourceContentHash": _source_hash(row),
        "page_agrees_with_canonical": bool(row.get("page_agrees_with_canonical")),
        "source_hash_agrees_with_canonical": bool(row.get("source_hash_agrees_with_canonical")),
        "layout_region_candidate_present": bool(row.get("layout_region_candidate_present")),
        "region_link_verified": bool(row.get(str(config["regionVerifiedKey"]))),
        "table_cell_evidence_available": bool(row.get("table_cell_evidence_available")),
        "table_cell_citation_grade": bool(row.get("table_cell_citation_grade")),
        "equation_semantics_interpreted": bool(row.get("equation_semantics_interpreted")),
        "diagnostic_page_context_available": bool(diagnostic_candidates),
        "diagnostic_best_page_coverage": row.get("diagnostic_best_page_coverage"),
        "evidence_tier": "non_sectionspan_pdf_offset_feasibility_audit_only",
        "source_evidence_tier": str(row.get("evidence_tier") or config["tier"]),
        "report_only": True,
        "runtime_promotion_allowed": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": non_strict_reason,
        "recommended_next_action": _recommended_action(layer, row),
    }


def _schema_violation(layer: str, payload: dict[str, Any]) -> str:
    expected = str(_LAYER_CONFIG[layer]["schema"])
    actual = str(payload.get("schema") or "")
    if actual != expected:
        return f"{layer}_schema_mismatch"
    if str(payload.get("status") or "") != "feasibility_complete":
        return f"{layer}_feasibility_not_complete"
    if _safe_int((payload.get("counts") or {}).get("schemaViolationCount")):
        return f"{layer}_upstream_schema_violations"
    return ""


def _layer_source(layer: str, path: str | Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = dict(payload.get("counts") or {})
    return {
        "layer": layer,
        "path": str(Path(str(path)).expanduser()),
        "schema": str(payload.get("schema") or ""),
        "status": str(payload.get("status") or ""),
        "rowCount": len(rows),
        "recoveredRows": sum(1 for item in rows if item.get("original_pdf_offset_recovered")),
        "blockedRows": sum(1 for item in rows if not item.get("original_pdf_offset_recovered")),
        "strictEligibleRows": _safe_int(counts.get("strictEligibleRows")),
        "citationGradeRows": _safe_int(counts.get("citationGradeRows")),
        "runtimeEvidenceRows": _safe_int(counts.get("runtimeEvidenceRows")),
        "byPaper": dict(counts.get("byPaper") or {}),
        "byFeasibilityStatus": dict(counts.get("byFeasibilityStatus") or {}),
        "strictBlockerSummary": dict(counts.get("strictBlockerSummary") or {}),
    }


def _counter_dict(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "") for row in rows))


def build_non_sectionspan_pdf_offset_feasibility_audit(
    *,
    figure_caption_pdf_offset_feasibility_report: str | Path,
    table_region_pdf_offset_feasibility_report: str | Path,
    equation_quote_pdf_offset_feasibility_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only non-SectionSpan PDF offset feasibility audit."""

    input_paths = {
        "figure_caption": str(Path(str(figure_caption_pdf_offset_feasibility_report)).expanduser()),
        "table_region": str(Path(str(table_region_pdf_offset_feasibility_report)).expanduser()),
        "equation_quote": str(Path(str(equation_quote_pdf_offset_feasibility_report)).expanduser()),
    }
    payloads = {
        "figure_caption": _read_json(figure_caption_pdf_offset_feasibility_report),
        "table_region": _read_json(table_region_pdf_offset_feasibility_report),
        "equation_quote": _read_json(equation_quote_pdf_offset_feasibility_report),
    }
    schema_violations = [
        violation
        for layer, payload in payloads.items()
        for violation in [_schema_violation(layer, payload)]
        if violation
    ]
    rows_by_layer = {
        layer: [_audit_row(layer, row) for row in _as_list(payload.get("feasibilityRows")) if isinstance(row, dict)]
        for layer, payload in payloads.items()
    }
    rows = [row for layer in ("figure_caption", "table_region", "equation_quote") for row in rows_by_layer[layer]]
    recovered = [row for row in rows if row.get("original_pdf_offset_recovered")]
    blocked = [row for row in rows if not row.get("original_pdf_offset_recovered")]
    layer_sources = [
        _layer_source(layer, input_paths[layer], payloads[layer], rows_by_layer[layer])
        for layer in ("figure_caption", "table_region", "equation_quote")
    ]
    strict_blocker_summary = Counter()
    for row in rows:
        strict_blocker_summary.update(str(item) for item in _as_list(row.get("strict_blockers")) if item)
    counts = {
        "inputReportCount": len(input_paths),
        "schemaViolationCount": len(schema_violations),
        "totalRows": len(rows),
        "recoveredRows": len(recovered),
        "blockedRows": len(blocked),
        "pageRecoveredRows": sum(1 for row in rows if row.get("page") is not None),
        "sourceHashLinkedRows": sum(1 for row in rows if row.get("sourceContentHash")),
        "pageAgreementRows": sum(1 for row in rows if row.get("page_agrees_with_canonical")),
        "sourceHashAgreementRows": sum(1 for row in rows if row.get("source_hash_agrees_with_canonical")),
        "diagnosticPageContextRows": sum(
            1 for row in rows if row.get("readiness") == "diagnostic_page_context_only_non_strict"
        ),
        "readyForRegionReviewRows": sum(
            1 for row in rows if row.get("readiness") == "ready_for_region_review_non_strict"
        ),
        "needsFigureRegionReviewRows": sum(
            1 for row in rows if row.get("candidate_layer") == "figure_caption" and row.get("original_pdf_offset_recovered")
        ),
        "needsTableCellProvenanceReviewRows": sum(
            1 for row in rows if row.get("candidate_layer") == "table_region" and row.get("original_pdf_offset_recovered")
        ),
        "needsEquationAlignmentReviewRows": sum(1 for row in rows if row.get("candidate_layer") == "equation_quote"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "byLayerRows": _counter_dict(rows, "candidate_layer"),
        "byLayerRecoveredRows": _counter_dict(recovered, "candidate_layer"),
        "byLayerBlockedRows": _counter_dict(blocked, "candidate_layer"),
        "byPaper": _counter_dict(rows, "paper_id"),
        "byFeasibilityStatus": _counter_dict(rows, "feasibility_status"),
        "byReadiness": _counter_dict(rows, "readiness"),
        "strictBlockerSummary": dict(strict_blocker_summary.most_common()),
    }
    return {
        "schema": NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID,
        "status": "ok" if not schema_violations else "blocked_upstream_schema",
        "generatedAt": _now(),
        "inputs": {
            "figure_caption_pdf_offset_feasibility": input_paths["figure_caption"],
            "table_region_pdf_offset_feasibility": input_paths["table_region"],
            "equation_quote_pdf_offset_feasibility": input_paths["equation_quote"],
        },
        "counts": counts,
        "policy": {
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "gate": {
            "auditComplete": not schema_violations,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "non_sectionspan_pdf_offset_audit_ready" if not schema_violations else "blocked_upstream_schema",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "review_non_sectionspan_offset_rows_before_any_strict_promotion",
        },
        "warnings": [
            "audit_rows_are_not_runtime_evidence",
            "bbox_or_generated_markdown_offsets_are_not_strict_evidence",
            "figure_caption_rows_need_region_link_review",
            "table_region_rows_need_cell_provenance_review",
            "equation_quote_rows_need_alignment_or_normalization_review",
            "parser_routing_and_answer_integration_remain_out_of_scope",
        ],
        "layerSources": layer_sources,
        "auditRows": rows,
    }


def render_non_sectionspan_pdf_offset_feasibility_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Non-SectionSpan PDF Offset Feasibility Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Total rows: `{int(counts.get('totalRows') or 0)}`",
        f"- Recovered original PDF offsets: `{int(counts.get('recoveredRows') or 0)}`",
        f"- Blocked rows: `{int(counts.get('blockedRows') or 0)}`",
        f"- Page recovered rows: `{int(counts.get('pageRecoveredRows') or 0)}`",
        f"- Source-hash linked rows: `{int(counts.get('sourceHashLinkedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Citation-grade: `{int(counts.get('citationGradeRows') or 0)}`",
        f"- Runtime evidence rows: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Policy",
        "",
        "All rows remain report-only, non-strict candidates. This audit does not create runtime citations.",
        "",
        "## Layer Counts",
        "",
    ]
    for source in list(report.get("layerSources") or []):
        lines.extend(
            [
                f"### `{source.get('layer', '')}`",
                "",
                f"- Rows: `{source.get('rowCount')}`",
                f"- Recovered: `{source.get('recoveredRows')}`",
                f"- Blocked: `{source.get('blockedRows')}`",
                f"- Strict eligible: `{source.get('strictEligibleRows')}`",
                f"- Citation-grade: `{source.get('citationGradeRows')}`",
                f"- Feasibility status: `{json.dumps(source.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Gate",
            "",
            f"- Audit complete: `{bool(gate.get('auditComplete'))}`",
            f"- Strict evidence ready: `{bool(gate.get('strictEvidenceReady'))}`",
            f"- Parser routing ready: `{bool(gate.get('parserRoutingReady'))}`",
            f"- Recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
            "",
            "## Strict Blockers",
            "",
            f"`{json.dumps(counts.get('strictBlockerSummary') or {}, ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_non_sectionspan_pdf_offset_feasibility_audit_reports(
    report: dict[str, Any], output_dir: str | Path
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "non-sectionspan-pdf-offset-feasibility-audit.json"
    summary_path = root / "non-sectionspan-pdf-offset-feasibility-summary.json"
    markdown_path = root / "non-sectionspan-pdf-offset-feasibility-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_payload = {key: report[key] for key in ("schema", "status", "generatedAt", "inputs", "counts", "policy", "gate", "warnings")}
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_non_sectionspan_pdf_offset_feasibility_audit_markdown(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only non-SectionSpan PDF offset feasibility audit.")
    parser.add_argument("--figure-caption-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--table-region-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--equation-quote-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_non_sectionspan_pdf_offset_feasibility_audit(
        figure_caption_pdf_offset_feasibility_report=args.figure_caption_pdf_offset_feasibility_report,
        table_region_pdf_offset_feasibility_report=args.table_region_pdf_offset_feasibility_report,
        equation_quote_pdf_offset_feasibility_report=args.equation_quote_pdf_offset_feasibility_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_non_sectionspan_pdf_offset_feasibility_audit_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NON_SECTIONSPAN_PDF_OFFSET_FEASIBILITY_AUDIT_SCHEMA_ID",
    "build_non_sectionspan_pdf_offset_feasibility_audit",
    "render_non_sectionspan_pdf_offset_feasibility_audit_markdown",
    "write_non_sectionspan_pdf_offset_feasibility_audit_reports",
]
