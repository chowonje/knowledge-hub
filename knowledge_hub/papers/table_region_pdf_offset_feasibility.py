"""Report-only TableRegion caption original PDF offset feasibility audit.

This helper checks whether TableRegionCandidate caption rows can be located in
the original local source PDF text.  Recovered caption offsets remain non-strict
because table-cell row/column/bbox/source-span provenance is still missing.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _clean_text,
    _empty_span,
    _exact_matches,
    _normalized_matches,
    _paper_context,
    _safe_int,
)


TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.table-region-pdf-offset-feasibility.v1"
)
TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.table-region-candidate-report.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("candidates") or []):
        if isinstance(item, dict) and str(item.get("candidate_type") or "") == "table_region_candidate":
            rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")),
            _safe_int(item.get("chars_start")),
            str(item.get("candidate_id") or ""),
        )
    )
    return rows


def _source_text_options(row: dict[str, Any]) -> list[tuple[str, str, float]]:
    options: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for label, value, confidence in (
        ("full_candidate_text", row.get("candidate_text"), 1.0),
        ("caption_body_text", row.get("caption_text"), 0.9),
    ):
        text = _clean_text(value)
        if text and text not in seen:
            options.append((label, text, confidence))
            seen.add(text)
    return options


def _find_unique_pdf_match(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    pages = list(context.get("pages") or [])
    attempted: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for target, text, target_confidence in _source_text_options(row):
        exact = _exact_matches(pages, text)
        attempted.append({"target": target, "method": "exact", "matchCount": len(exact)})
        if len(exact) == 1:
            match = dict(exact[0])
            match["match_target"] = target
            match["matched_text"] = text
            match["match_confidence"] = min(float(match.get("match_confidence") or 0.0), target_confidence)
            return {"status": "ok", "match": match, "attempted": attempted}
        if len(exact) > 1:
            ambiguous.append({"target": target, "method": "exact", "matchCount": len(exact)})
            continue
        normalized = _normalized_matches(pages, text)
        attempted.append({"target": target, "method": "normalized_whitespace_case", "matchCount": len(normalized)})
        if len(normalized) == 1:
            match = dict(normalized[0])
            match["match_target"] = target
            match["matched_text"] = text
            match["match_confidence"] = min(float(match.get("match_confidence") or 0.0), target_confidence)
            return {"status": "ok", "match": match, "attempted": attempted}
        if len(normalized) > 1:
            ambiguous.append(
                {"target": target, "method": "normalized_whitespace_case", "matchCount": len(normalized)}
            )
    if ambiguous:
        return {"status": "ambiguous", "ambiguous": ambiguous, "attempted": attempted}
    return {"status": "not_found", "attempted": attempted}


def _base_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    source_hash = str(context.get("sourceContentHash") or "").strip()
    return {
        "feasibility_row_id": f"tableregion-pdf-offset:{row.get('paper_id')}:{row.get('candidate_id')}",
        "source_table_region_candidate_id": str(row.get("candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "caption_text": _clean_text(row.get("caption_text")),
        "table_label": _clean_text(row.get("table_label")),
        "canonical_alignment_status": str(row.get("canonical_alignment_status") or ""),
        "canonical_alignment_method": str(row.get("alignment_method") or ""),
        "canonical_span": {
            "chars_start": row.get("chars_start"),
            "chars_end": row.get("chars_end"),
            "page": row.get("page"),
            "sourceContentHash": str(row.get("sourceContentHash") or ""),
        },
        "layout_element_ids": [str(item) for item in list(row.get("layout_element_ids") or [])],
        "bbox": row.get("bbox"),
        "layout_region_candidate_present": bool(row.get("layout_region_candidate_present")),
        "table_region_link_verified": False,
        "table_cell_evidence_available": False,
        "table_cell_citation_grade": False,
        "source_pdf_path": str(context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(context.get("manifestPath") or ""),
        "sourceContentHash": source_hash,
        "evidence_tier": "table_region_pdf_offset_feasibility_only",
        "report_only": True,
        "runtime_promotion_allowed": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }


def _blocked_row(
    row: dict[str, Any],
    context: dict[str, Any],
    *,
    status: str,
    reason: str,
    match_count: int = 0,
    attempted_matches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_hash = str(context.get("sourceContentHash") or "").strip()
    return {
        **_base_row(row, context),
        "feasibility_status": status,
        "original_pdf_offset_recovered": False,
        "original_pdf_span": _empty_span(source_hash),
        "match_target": "",
        "matched_text": "",
        "match_count": match_count,
        "attempted_matches": attempted_matches or [],
        "page_agrees_with_canonical": False,
        "source_hash_agrees_with_canonical": False,
        "feasibility_failure_reason": reason,
        "strict_blockers": [
            "report_only",
            "original_pdf_offset_not_recovered",
            "table_cell_row_column_bbox_provenance_missing",
            "table_cell_provenance_missing",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "table_region_pdf_offset_feasibility_only",
            "no_runtime_or_strict_evidence_created",
            reason,
        ],
    }


def _feasibility_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if str(context.get("status") or "") != "ok":
        status = str(context.get("status") or "blocked")
        return _blocked_row(row, context, status=status, reason=status)
    candidate_hash = str(row.get("sourceContentHash") or "").strip()
    source_hash = str(context.get("sourceContentHash") or "").strip()
    if candidate_hash and source_hash and candidate_hash != source_hash:
        return _blocked_row(row, context, status="blocked_source_hash_mismatch", reason="source_hash_mismatch")
    if not _source_text_options(row):
        return _blocked_row(row, context, status="blocked_no_candidate_text", reason="candidate_text_missing")

    match_result = _find_unique_pdf_match(row, context)
    if match_result["status"] == "ambiguous":
        ambiguous = list(match_result.get("ambiguous") or [])
        return _blocked_row(
            row,
            context,
            status="blocked_ambiguous_match",
            reason="original_pdf_table_caption_match_ambiguous",
            match_count=sum(int(item.get("matchCount") or 0) for item in ambiguous),
            attempted_matches=list(match_result.get("attempted") or []),
        )
    if match_result["status"] != "ok":
        return _blocked_row(
            row,
            context,
            status="blocked_no_match",
            reason="unique_original_pdf_table_caption_match_not_found",
            attempted_matches=list(match_result.get("attempted") or []),
        )

    match = dict(match_result.get("match") or {})
    canonical_page = row.get("page")
    recovered_page = match.get("page")
    page_agrees = canonical_page is not None and _safe_int(canonical_page) == _safe_int(recovered_page)
    source_hash_agrees = bool(candidate_hash and source_hash and candidate_hash == source_hash)
    blockers = [
        "report_only",
        "table_cell_row_column_bbox_provenance_missing",
        "table_cell_provenance_missing",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_explicit_later_tranche",
    ]
    if str(match.get("match_target") or "") != "full_candidate_text":
        blockers.append("full_table_caption_label_not_recovered_in_original_pdf_match")
    return {
        **_base_row(row, context),
        "feasibility_status": f"recovered_{match.get('match_method')}",
        "original_pdf_offset_recovered": True,
        "original_pdf_span": {
            "originalPdfCharsStart": match.get("chars_start"),
            "originalPdfCharsEnd": match.get("chars_end"),
            "page": recovered_page,
            "sourceContentHash": source_hash,
            "matchMethod": match.get("match_method"),
            "matchConfidence": match.get("match_confidence"),
        },
        "match_target": str(match.get("match_target") or ""),
        "matched_text": str(match.get("matched_text") or ""),
        "match_count": 1,
        "attempted_matches": list(match_result.get("attempted") or []),
        "page_agrees_with_canonical": page_agrees,
        "source_hash_agrees_with_canonical": source_hash_agrees,
        "feasibility_failure_reason": "",
        "strict_blockers": blockers,
        "non_strict_reason": [
            "original_pdf_table_caption_offset_recovered_in_report_only_feasibility_audit",
            "table_cell_provenance_still_missing",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    recovered = [item for item in rows if item.get("original_pdf_offset_recovered")]
    blocked = [item for item in rows if not item.get("original_pdf_offset_recovered")]
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputTableRegionCandidates": len(rows),
        "feasibilityRows": len(rows),
        "originalPdfOffsetRecoveredRows": len(recovered),
        "blockedRows": len(blocked),
        "exactRecoveredRows": sum(
            1 for item in recovered if (item.get("original_pdf_span") or {}).get("matchMethod") == "exact"
        ),
        "normalizedRecoveredRows": sum(
            1
            for item in recovered
            if (item.get("original_pdf_span") or {}).get("matchMethod") == "normalized_whitespace_case"
        ),
        "fullCandidateTextRecoveredRows": sum(1 for item in recovered if item.get("match_target") == "full_candidate_text"),
        "captionBodyTextRecoveredRows": sum(1 for item in recovered if item.get("match_target") == "caption_body_text"),
        "ambiguousRows": sum(1 for item in rows if item.get("feasibility_status") == "blocked_ambiguous_match"),
        "noMatchRows": sum(1 for item in rows if item.get("feasibility_status") == "blocked_no_match"),
        "pageAgreementRows": sum(1 for item in recovered if item.get("page_agrees_with_canonical")),
        "sourceHashAgreementRows": sum(1 for item in recovered if item.get("source_hash_agrees_with_canonical")),
        "tableCellEvidenceRows": 0,
        "tableCellCitationGradeRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byFeasibilityStatus": dict(Counter(str(item.get("feasibility_status") or "") for item in rows)),
        "byMatchTarget": dict(Counter(str(item.get("match_target") or "") for item in rows)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_table_region_pdf_offset_feasibility_report(
    *,
    table_region_candidate_report: str | Path,
    pymupdf_parsed_root: str | Path,
    pdf_page_text_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a report-only TableRegion caption original PDF offset feasibility report."""

    report_path = Path(str(table_region_candidate_report)).expanduser()
    parsed_root = Path(str(pymupdf_parsed_root)).expanduser()
    candidate_report = _read_json(report_path)
    schema_violations: list[str] = []
    if candidate_report.get("schema") != TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID:
        schema_violations.append("table_region_candidate_report_schema_mismatch")
    if candidate_report.get("status") not in {"ok", "empty"}:
        schema_violations.append("table_region_candidate_report_status_unexpected")
    source_rows = _candidate_rows(candidate_report)
    loader = pdf_page_text_loader
    if loader is None:
        from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import _extract_pdf_pages

        loader = _extract_pdf_pages
    contexts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in contexts:
            contexts[paper_id] = _paper_context(paper_id=paper_id, parsed_root=parsed_root, page_loader=loader)
        rows.append(_feasibility_row(row, contexts[paper_id]))
    counts = _counts(rows, schema_violations=schema_violations)
    blocked = bool(schema_violations) or not rows
    return {
        "schema": TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
        "status": "blocked" if blocked else "feasibility_complete",
        "generatedAt": _now(),
        "inputs": {
            "tableRegionCandidateReport": str(report_path),
            "tableRegionCandidateSchema": str(candidate_report.get("schema") or ""),
            "pymupdfParsedRoot": str(parsed_root),
        },
        "counts": counts,
        "gate": {
            "feasibilityComplete": not blocked,
            "applyExecuted": False,
            "tableCellEvidenceReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "feasibility_complete_non_strict" if not blocked else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "review_table_caption_offsets_and_cell_provenance_before_any_strict_promotion",
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "applyExecuted": False,
            "tableCellEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "table_caption_pdf_offsets_are_report_only_feasibility_results",
            "recovered_table_caption_offsets_do_not_create_table_cell_evidence",
            "row_column_cell_bbox_source_span_provenance_remains_required_before_citation_grade_tables",
            "runtime_promotion_remains_disabled",
        ],
        "paperContexts": {key: {k: v for k, v in value.items() if k != "pages"} for key, value in contexts.items()},
        "feasibilityRows": rows,
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
            "paperContexts",
            "feasibilityRows",
        )
        if key in report
    }


def render_table_region_pdf_offset_feasibility_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TableRegion Caption Original PDF Offset Feasibility",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Feasibility rows: `{int(counts.get('feasibilityRows') or 0)}`",
        f"- Original PDF offsets recovered: `{int(counts.get('originalPdfOffsetRecoveredRows') or 0)}`",
        f"- Blocked rows: `{int(counts.get('blockedRows') or 0)}`",
        f"- Table-cell evidence rows: `{int(counts.get('tableCellEvidenceRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a report-only feasibility audit. It reads local source PDFs and records unique table-caption matches, but does not create table-cell evidence, write canonical parsed artifacts, mutate SQLite, reindex, reembed, create strict evidence, allow runtime citations, or change parser routing.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By feasibility status: `{json.dumps(counts.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By match target: `{json.dumps(counts.get('byMatchTarget') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_table_region_pdf_offset_feasibility_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "table-region-pdf-offset-feasibility.json"
    summary_path = root / "table-region-pdf-offset-feasibility-summary.json"
    markdown_path = root / "table-region-pdf-offset-feasibility.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_table_region_pdf_offset_feasibility_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only TableRegion caption original PDF offset feasibility.")
    parser.add_argument("--table-region-candidate-report", required=True)
    parser.add_argument("--pymupdf-parsed-root", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_table_region_pdf_offset_feasibility_report(
        table_region_candidate_report=args.table_region_candidate_report,
        pymupdf_parsed_root=args.pymupdf_parsed_root,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_table_region_pdf_offset_feasibility_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TABLE_REGION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID",
    "build_table_region_pdf_offset_feasibility_report",
    "render_table_region_pdf_offset_feasibility_markdown",
    "write_table_region_pdf_offset_feasibility_reports",
]
