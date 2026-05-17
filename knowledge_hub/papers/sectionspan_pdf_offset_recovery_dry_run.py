"""Report-only SectionSpan original PDF offset recovery dry-run.

This helper attempts to locate SectionSpan heading candidates in local source
PDF text. It records whether a unique original-PDF text span can be found, but
it does not write canonical parsed artifacts, create strict evidence, route
parsers, wire answer citations, mutate SQLite, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
import unicodedata

from knowledge_hub.papers.source_text import source_hash_for_path


SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-dry-run.v1"
)
SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-design.v1"
)


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    indexes: list[int]


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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_with_indexes(value: str) -> _NormalizedText:
    parts: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(str(value or "")):
        folded = unicodedata.normalize("NFKC", char).casefold()
        for item in folded:
            if item.isspace():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                    indexes.append(index)
                continue
            parts.append(item)
            indexes.append(index)
    start = 0
    end = len(parts)
    while start < end and parts[start].isspace():
        start += 1
    while end > start and parts[end - 1].isspace():
        end -= 1
    return _NormalizedText("".join(parts[start:end]), indexes[start:end])


def _find_all(value: str, needle: str) -> list[int]:
    if not value or not needle:
        return []
    offsets: list[int] = []
    cursor = 0
    while True:
        index = value.find(needle, cursor)
        if index < 0:
            break
        offsets.append(index)
        cursor = index + max(1, len(needle))
    return offsets


def _source_pdf_from_manifest(manifest: dict[str, Any]) -> str:
    parser_meta = dict(manifest.get("parser_meta") or {})
    for key in ("source_pdf", "extracted_from", "pdf_path", "sourcePath"):
        value = str(parser_meta.get(key) or manifest.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_pdf_pages(source_pdf: str | Path) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    path = Path(str(source_pdf)).expanduser()
    try:
        document = fitz.open(str(path))
    except Exception:
        return []
    pages: list[dict[str, Any]] = []
    try:
        page_total = int(getattr(document, "page_count", 0) or 0)
        for page_index in range(page_total):
            try:
                page = document.load_page(page_index)
                text = str(page.get_text("text") or "")
            except Exception:
                text = ""
            pages.append({"page": page_index + 1, "text": text})
    finally:
        try:
            document.close()
        except Exception:
            pass
    return pages


def _with_offsets(raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    cursor = 0
    for item in raw_pages:
        text = str(item.get("text") or "")
        start = cursor
        end = start + len(text)
        pages.append(
            {
                "page": _safe_int(item.get("page")),
                "text": text,
                "chars_start": start,
                "chars_end": end,
                "normalized": _normalize_with_indexes(text),
            }
        )
        cursor = end + 2
    return pages


def _exact_matches(pages: list[dict[str, Any]], candidate_text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for page in pages:
        text = str(page.get("text") or "")
        page_start = _safe_int(page.get("chars_start"))
        for start in _find_all(text, candidate_text):
            matches.append(
                {
                    "page": _safe_int(page.get("page")),
                    "chars_start": page_start + start,
                    "chars_end": page_start + start + len(candidate_text),
                    "match_method": "exact",
                    "match_confidence": 1.0,
                }
            )
    return matches


def _normalized_matches(pages: list[dict[str, Any]], candidate_text: str) -> list[dict[str, Any]]:
    needle = _normalize_with_indexes(candidate_text).text
    if not needle:
        return []
    matches: list[dict[str, Any]] = []
    for page in pages:
        normalized = page.get("normalized")
        if not isinstance(normalized, _NormalizedText) or not normalized.text:
            continue
        page_start = _safe_int(page.get("chars_start"))
        for start in _find_all(normalized.text, needle):
            end = start + len(needle)
            if end > len(normalized.indexes):
                continue
            raw_start = normalized.indexes[start]
            raw_end = normalized.indexes[end - 1] + 1
            matches.append(
                {
                    "page": _safe_int(page.get("page")),
                    "chars_start": page_start + raw_start,
                    "chars_end": page_start + raw_end,
                    "match_method": "normalized_whitespace_case",
                    "match_confidence": 0.95,
                }
            )
    return matches


def _empty_span(source_hash: str) -> dict[str, Any]:
    return {
        "originalPdfCharsStart": None,
        "originalPdfCharsEnd": None,
        "page": None,
        "sourceContentHash": source_hash,
        "matchMethod": "",
        "matchConfidence": 0.0,
    }


def _paper_context(
    *,
    paper_id: str,
    parsed_root: Path,
    page_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    manifest_path = parsed_root / paper_id / "manifest.json"
    manifest = _read_json(manifest_path)
    source_pdf = _source_pdf_from_manifest(manifest)
    if not manifest:
        return {"status": "blocked_manifest_missing", "manifestPath": str(manifest_path), "sourcePdfPath": ""}
    if not source_pdf:
        return {"status": "blocked_source_pdf_unregistered", "manifestPath": str(manifest_path), "sourcePdfPath": ""}
    source_pdf_path = Path(source_pdf).expanduser()
    if not source_pdf_path.exists():
        return {"status": "blocked_source_pdf_missing", "manifestPath": str(manifest_path), "sourcePdfPath": str(source_pdf_path)}
    source_hash = source_hash_for_path(str(source_pdf_path))
    if not source_hash:
        return {"status": "blocked_source_hash_unavailable", "manifestPath": str(manifest_path), "sourcePdfPath": str(source_pdf_path)}
    pages = _with_offsets(page_loader(source_pdf_path))
    pages_with_text = sum(1 for item in pages if str(item.get("text") or "").strip())
    if not pages or pages_with_text <= 0:
        return {
            "status": "blocked_pdf_text_extraction_unavailable",
            "manifestPath": str(manifest_path),
            "sourcePdfPath": str(source_pdf_path),
            "sourceContentHash": source_hash,
        }
    return {
        "status": "ok",
        "manifestPath": str(manifest_path),
        "sourcePdfPath": str(source_pdf_path),
        "sourceContentHash": source_hash,
        "pages": pages,
        "pageCount": len(pages),
        "pagesWithText": pages_with_text,
    }


def _recover_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    candidate_text = _clean_text(row.get("candidate_text"))
    canonical_span = dict(row.get("canonical_span") or {})
    planned_hash = str(canonical_span.get("sourceContentHash") or "").strip()
    source_hash = str(context.get("sourceContentHash") or "").strip()
    base = {
        "recovery_plan_id": str(row.get("recovery_plan_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": candidate_text,
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "canonical_span": canonical_span,
        "source_pdf_path": str(context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(context.get("manifestPath") or ""),
        "sourceContentHash": source_hash,
        "evidence_tier": "sectionspan_pdf_offset_recovery_dry_run_only",
        "dry_run_only": True,
        "runtime_promotion_allowed": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }

    def blocked(status: str, reason: str, match_count: int = 0) -> dict[str, Any]:
        return {
            **base,
            "recovery_status": status,
            "original_pdf_offset_recovered": False,
            "original_pdf_span": _empty_span(source_hash),
            "match_count": match_count,
            "recovery_failure_reason": reason,
            "strict_blockers": [
                "dry_run_only",
                "original_pdf_offset_not_recovered",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_explicit_later_tranche",
            ],
            "non_strict_reason": [
                "dry_run_report_only",
                "no_runtime_or_strict_evidence_created",
                reason,
            ],
        }

    if str(context.get("status") or "") != "ok":
        status = str(context.get("status") or "blocked")
        return blocked(status, status)
    if not candidate_text:
        return blocked("blocked_no_candidate_text", "candidate_text_missing")
    if planned_hash and source_hash and planned_hash != source_hash:
        return blocked("blocked_source_hash_mismatch", "source_hash_mismatch")

    pages = list(context.get("pages") or [])
    exact = _exact_matches(pages, candidate_text)
    if len(exact) == 1:
        match = exact[0]
    elif len(exact) > 1:
        return blocked("blocked_ambiguous_match", "exact_match_ambiguous", len(exact))
    else:
        normalized = _normalized_matches(pages, candidate_text)
        if len(normalized) == 1:
            match = normalized[0]
        elif len(normalized) > 1:
            return blocked("blocked_ambiguous_match", "normalized_match_ambiguous", len(normalized))
        else:
            return blocked("blocked_no_match", "unique_original_pdf_text_match_not_found", 0)

    return {
        **base,
        "recovery_status": f"recovered_{match['match_method']}",
        "original_pdf_offset_recovered": True,
        "original_pdf_span": {
            "originalPdfCharsStart": match["chars_start"],
            "originalPdfCharsEnd": match["chars_end"],
            "page": match["page"],
            "sourceContentHash": source_hash,
            "matchMethod": match["match_method"],
            "matchConfidence": match["match_confidence"],
        },
        "match_count": 1,
        "recovery_failure_reason": "",
        "strict_blockers": [
            "dry_run_only",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "original_pdf_offset_recovered_in_report_only_dry_run",
            "no_runtime_or_strict_evidence_created",
            "later_explicit_apply_and_promotion_tranches_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    recovered = [item for item in rows if item.get("original_pdf_offset_recovered")]
    blocked = [item for item in rows if not item.get("original_pdf_offset_recovered")]
    return {
        "inputRecoveryPlanRows": len(rows),
        "dryRunRows": len(rows),
        "originalPdfOffsetRecoveredRows": len(recovered),
        "blockedRows": len(blocked),
        "exactRecoveredRows": sum(1 for item in recovered if (item.get("original_pdf_span") or {}).get("matchMethod") == "exact"),
        "normalizedRecoveredRows": sum(
            1 for item in recovered if (item.get("original_pdf_span") or {}).get("matchMethod") == "normalized_whitespace_case"
        ),
        "ambiguousRows": sum(1 for item in rows if item.get("recovery_status") == "blocked_ambiguous_match"),
        "noMatchRows": sum(1 for item in rows if item.get("recovery_status") == "blocked_no_match"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byRecoveryStatus": dict(Counter(str(item.get("recovery_status") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_recovery_dry_run(
    *,
    sectionspan_pdf_offset_recovery_design_report: str | Path,
    pymupdf_parsed_root: str | Path,
    pdf_page_text_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a report-only original-PDF-offset recovery dry-run."""

    design_path = Path(str(sectionspan_pdf_offset_recovery_design_report)).expanduser()
    parsed_root = Path(str(pymupdf_parsed_root)).expanduser()
    design = _read_json(design_path)
    schema_violations: list[str] = []
    if design.get("schema") != SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID:
        schema_violations.append("sectionspan_pdf_offset_recovery_design_schema_mismatch")
    if design.get("status") != "design_ready":
        schema_violations.append("sectionspan_pdf_offset_recovery_design_not_ready")
    source_rows = [dict(item) for item in list(design.get("recoveryPlanRows") or []) if isinstance(item, dict)]
    loader = pdf_page_text_loader or _extract_pdf_pages
    contexts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in contexts:
            contexts[paper_id] = _paper_context(paper_id=paper_id, parsed_root=parsed_root, page_loader=loader)
        rows.append(_recover_row(row, contexts[paper_id]))

    counts = _counts(rows, schema_violations=schema_violations)
    blocked = bool(schema_violations) or not rows
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID,
        "status": "blocked" if blocked else "dry_run_complete",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetRecoveryDesignReport": str(design_path),
            "sectionspanPdfOffsetRecoveryDesignSchema": str(design.get("schema") or ""),
            "pymupdfParsedRoot": str(parsed_root),
        },
        "counts": counts,
        "gate": {
            "dryRunComplete": not blocked,
            "applyExecuted": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "dry_run_complete_non_strict" if not blocked else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "review_recovered_offsets_before_any_strict_promotion",
        },
        "policy": {
            "reportOnly": True,
            "dryRunOnly": True,
            "applyExecuted": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "original_pdf_offsets_are_report_only_dry_run_results",
            "recovered_offsets_are_not_runtime_or_strict_evidence",
            "ambiguous_or_missing_matches_remain_blocked",
        ],
        "paperContexts": {key: {item_key: item_value for item_key, item_value in value.items() if item_key != "pages"} for key, value in contexts.items()},
        "recoveryRows": rows,
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
            "recoveryRows",
        )
        if key in report
    }


def render_sectionspan_pdf_offset_recovery_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Original PDF Offset Recovery Dry-Run",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Dry-run rows: `{int(counts.get('dryRunRows') or 0)}`",
        f"- Original PDF offsets recovered: `{int(counts.get('originalPdfOffsetRecoveredRows') or 0)}`",
        f"- Blocked rows: `{int(counts.get('blockedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a report-only dry-run. It reads local source PDFs and records unique heading matches, but does not write canonical parsed artifacts, mutate SQLite, reindex, reembed, create strict evidence, allow runtime citations, or change parser routing.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recovery status: `{json.dumps(counts.get('byRecoveryStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_recovery_dry_run_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "sectionspan-pdf-offset-recovery-dry-run.json"
    summary_path = root / "sectionspan-pdf-offset-recovery-dry-run-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-recovery-dry-run.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_recovery_dry_run_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only SectionSpan original PDF offset recovery dry-run.")
    parser.add_argument("--sectionspan-pdf-offset-recovery-design-report", required=True)
    parser.add_argument("--pymupdf-parsed-root", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_recovery_dry_run(
        sectionspan_pdf_offset_recovery_design_report=args.sectionspan_pdf_offset_recovery_design_report,
        pymupdf_parsed_root=args.pymupdf_parsed_root,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_recovery_dry_run_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID",
    "build_sectionspan_pdf_offset_recovery_dry_run",
    "render_sectionspan_pdf_offset_recovery_dry_run_markdown",
    "write_sectionspan_pdf_offset_recovery_dry_run_reports",
]
