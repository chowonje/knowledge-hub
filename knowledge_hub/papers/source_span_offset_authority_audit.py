"""Report-only source span offset authority audit for paper candidates.

This helper checks whether structured candidate layers carry canonical parsed
text spans or only generated Markdown/layout locators. It deliberately does not
promote any row to strict evidence and does not touch parser routing, answers,
SQLite, indexes, embeddings, or canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.source-span-offset-authority-audit.v1"

_LAYER_SCHEMA_BY_NAME = {
    "sectionspan": "knowledge-hub.paper.sectionspan-candidate-report.v1",
    "figure_caption": "knowledge-hub.paper.figure-caption-candidate-report.v1",
    "equation_quote": "knowledge-hub.paper.equation-quote-candidate-report.v1",
    "table_region": "knowledge-hub.paper.table-region-candidate-report.v1",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _locator_kind(item: dict[str, Any]) -> str:
    source_locator = dict(item.get("source_span_locator") or item.get("source_locator") or {})
    markdown_locator = dict(item.get("markdown_locator") or {})
    return str(source_locator.get("locatorKind") or markdown_locator.get("locatorKind") or "")


def _has_bbox_only_locator(item: dict[str, Any]) -> bool:
    return bool(item.get("bbox") or item.get("layout_element_ids"))


def _span_status(item: dict[str, Any]) -> tuple[str, list[str]]:
    chars_start = _safe_int(item.get("chars_start"))
    chars_end = _safe_int(item.get("chars_end"))
    page = _safe_int(item.get("page"))
    source_hash = str(item.get("sourceContentHash") or "").strip()
    locator_kind = _locator_kind(item)
    blockers: list[str] = [
        "source_span_offset_authority_audit_only",
        "runtime_promotion_disabled_for_tranche",
    ]
    if chars_start is None or chars_end is None or chars_end <= chars_start:
        blockers.append("missing_canonical_chars_start_end")
    if page is None:
        blockers.append("missing_page")
    if not source_hash:
        blockers.append("missing_source_content_hash")
    if locator_kind in {"generated_markdown"}:
        blockers.append("generated_markdown_locator_only")
    if locator_kind in {"canonical_generated_markdown", "canonical_parsed_text"}:
        blockers.append("canonical_text_span_is_candidate_only")
    else:
        blockers.append("original_pdf_offset_not_available")
    if _has_bbox_only_locator(item):
        blockers.append("layout_bbox_is_not_source_span")
    if "missing_canonical_chars_start_end" in blockers:
        if locator_kind == "generated_markdown":
            return "generated_markdown_locator_only", blockers
        if _has_bbox_only_locator(item):
            return "layout_or_bbox_only", blockers
        return "missing_source_span", blockers
    if "missing_page" in blockers:
        return "canonical_chars_without_page_non_strict", blockers
    if "missing_source_content_hash" in blockers:
        return "canonical_chars_without_source_hash_non_strict", blockers
    if locator_kind == "canonical_generated_markdown":
        return "canonical_generated_markdown_span_non_strict", blockers
    if locator_kind == "canonical_parsed_text":
        return "canonical_parsed_text_span_non_strict", blockers
    return "canonical_chars_unknown_locator_non_strict", blockers


def _row(layer: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    chars_start = _safe_int(item.get("chars_start"))
    chars_end = _safe_int(item.get("chars_end"))
    page = _safe_int(item.get("page"))
    locator_kind = _locator_kind(item)
    status, blockers = _span_status(item)
    source_hash = str(item.get("sourceContentHash") or "").strip()
    return {
        "audit_id": f"source-span-offset-authority:{layer}:{index:04d}",
        "candidate_id": str(item.get("candidate_id") or ""),
        "candidate_type": str(item.get("candidate_type") or ""),
        "layer": layer,
        "paper_id": str(item.get("paper_id") or ""),
        "source_parser": str(item.get("source_parser") or "mineru+pymupdf_alignment"),
        "candidate_text": str(item.get("candidate_text") or ""),
        "canonical_alignment_status": str(item.get("canonical_alignment_status") or ""),
        "alignment_method": str(item.get("alignment_method") or ""),
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "sourceContentHash": source_hash,
        "locatorKind": locator_kind,
        "source_span_authority_status": status,
        "canonical_parsed_text_span_available": status in {
            "canonical_generated_markdown_span_non_strict",
            "canonical_parsed_text_span_non_strict",
            "canonical_chars_unknown_locator_non_strict",
        },
        "original_pdf_offset_available": False,
        "layout_or_bbox_only": status == "layout_or_bbox_only",
        "markdown_offset_only": status == "generated_markdown_locator_only",
        "evidence_tier": "source_span_offset_authority_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": list(dict.fromkeys(blockers)),
        "non_strict_reason": [
            "source_span_offset_authority_audit_is_report_only",
            "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _read_layer(layer: str, path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    return payload, [dict(item) for item in list(payload.get("candidates") or []) if isinstance(item, dict)]


def _schema_violations(payloads: dict[str, dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for layer, expected in _LAYER_SCHEMA_BY_NAME.items():
        if payloads.get(layer, {}).get("schema") != expected:
            violations.append(f"{layer}_schema_mismatch")
    return violations


def _counts(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(item.get("source_span_authority_status") or "") for item in rows)
    by_layer = Counter(str(item.get("layer") or "") for item in rows)
    by_locator = Counter(str(item.get("locatorKind") or "missing") for item in rows)
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputCandidateCount": len(rows),
        "auditedCandidateCount": len(rows),
        "canonicalParsedTextSpanCandidates": sum(1 for item in rows if item.get("canonical_parsed_text_span_available")),
        "canonicalGeneratedMarkdownSpanCandidates": _safe_int(by_status.get("canonical_generated_markdown_span_non_strict")) or 0,
        "generatedMarkdownOnlyCandidates": _safe_int(by_status.get("generated_markdown_locator_only")) or 0,
        "layoutOrBboxOnlyCandidates": _safe_int(by_status.get("layout_or_bbox_only")) or 0,
        "missingSourceSpanCandidates": _safe_int(by_status.get("missing_source_span")) or 0,
        "originalPdfOffsetCandidates": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "schemaViolationCount": len(schema_violations),
        "byLayer": dict(by_layer),
        "byLocatorKind": dict(by_locator),
        "byAuthorityStatus": dict(by_status),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_source_span_offset_authority_audit(
    *,
    sectionspan_report: str | Path,
    figure_caption_report: str | Path,
    equation_quote_report: str | Path,
    table_region_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only audit of candidate source span offset authority."""

    inputs = {
        "sectionspan": str(Path(str(sectionspan_report)).expanduser()),
        "figure_caption": str(Path(str(figure_caption_report)).expanduser()),
        "equation_quote": str(Path(str(equation_quote_report)).expanduser()),
        "table_region": str(Path(str(table_region_report)).expanduser()),
    }
    payloads: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for layer, path in inputs.items():
        payload, candidates = _read_layer(layer, path)
        payloads[layer] = payload
        for item in candidates:
            rows.append(_row(layer, len(rows) + 1, item))
    schema_violations = _schema_violations(payloads)
    counts = _counts(rows, schema_violations=schema_violations)
    return {
        "schema": SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "inputs": {
            **inputs,
            "sectionspanSchema": str(payloads.get("sectionspan", {}).get("schema") or ""),
            "figureCaptionSchema": str(payloads.get("figure_caption", {}).get("schema") or ""),
            "equationQuoteSchema": str(payloads.get("equation_quote", {}).get("schema") or ""),
            "tableRegionSchema": str(payloads.get("table_region", {}).get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "canonicalParsedTextSpanCandidateLayerReady": bool(counts["canonicalParsedTextSpanCandidates"]),
            "originalPdfOffsetReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "source_span_authority_audit_ready" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "equation_quote_alignment_feasibility_audit",
        },
        "policy": {
            "auditOnly": True,
            "allRowsNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "canonical_generated_markdown_offsets_are_candidate_spans_not_original_pdf_offsets",
            "sourceContentHash_plus_chars_start_end_does_not_create_strict_evidence",
            "bbox_only_and_markdown_offset_only_rows_remain_non_strict",
        ],
        "rows": rows,
    }


def render_source_span_offset_authority_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Source Span Offset Authority Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Audited candidates: `{int(counts.get('auditedCandidateCount') or 0)}`",
        f"- Canonical parsed-text span candidates: `{int(counts.get('canonicalParsedTextSpanCandidates') or 0)}`",
        f"- Original PDF offset candidates: `{int(counts.get('originalPdfOffsetCandidates') or 0)}`",
        f"- Strict eligible candidates: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence candidates: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        "",
        "## Policy",
        "",
        "This audit is report-only. Canonical generated Markdown offsets are not original PDF byte offsets and do not create strict evidence.",
        "",
        "## Counts",
        "",
        f"- By layer: `{json.dumps(counts.get('byLayer') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By locator kind: `{json.dumps(counts.get('byLocatorKind') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By authority status: `{json.dumps(counts.get('byAuthorityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_source_span_offset_authority_audit_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "source-span-offset-authority-audit.json"
    markdown_path = root / "source-span-offset-authority-audit.md"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_source_span_offset_authority_audit_markdown(report), encoding="utf-8")
    return {
        "audit": str(audit_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only source span offset authority audit.")
    parser.add_argument("--sectionspan-report", required=True, help="Path to sectionspan-candidates.json.")
    parser.add_argument("--figure-caption-report", required=True, help="Path to figure-caption-candidates.json.")
    parser.add_argument("--equation-quote-report", required=True, help="Path to equation-quote-candidates.json.")
    parser.add_argument("--table-region-report", required=True, help="Path to table-region-candidates.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print audit payload as JSON.")
    args = parser.parse_args(argv)

    report = build_source_span_offset_authority_audit(
        sectionspan_report=args.sectionspan_report,
        figure_caption_report=args.figure_caption_report,
        equation_quote_report=args.equation_quote_report,
        table_region_report=args.table_region_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_source_span_offset_authority_audit_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID",
    "build_source_span_offset_authority_audit",
    "render_source_span_offset_authority_audit_markdown",
    "write_source_span_offset_authority_audit_reports",
]
