"""Report-only figure-region link feasibility audit.

This helper checks whether FigureCaptionCandidate rows can be linked to actual
figure/image regions.  It keeps caption spans and MinerU layout bboxes as
candidate-only signals and deliberately does not create verified figure-region
evidence, runtime citations, parser routing, or canonical artifact writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.figure-region-link-feasibility-audit.v1"
FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.figure-caption-candidate-report.v1"
MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-source-alignment-audit.v1"
MINERU_NORMALIZER_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-normalizer-audit.v1"


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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    coords: list[float] = []
    for item in value[:4]:
        try:
            coords.append(float(item))
        except Exception:
            return None
    return coords


def _candidate_rows(figure_caption_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in list(figure_caption_report.get("candidates") or []):
        if isinstance(item, dict) and item.get("candidate_type") == "figure_caption_candidate":
            rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")) or 0,
            _safe_int(item.get("chars_start")) or 0,
            str(item.get("candidate_id") or ""),
        )
    )
    return rows


def _normalizer_paths(source_alignment_report: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for paper in list(source_alignment_report.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paperId") or "")
        path = str((paper.get("input") or {}).get("mineruNormalizerCandidatesPath") or "")
        if paper_id and path:
            paths[paper_id] = path
    return paths


def _normalizer_figure_captions(paths_by_paper: dict[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    captions: dict[str, dict[str, Any]] = {}
    schemas: dict[str, str] = {}
    for paper_id, path in sorted(paths_by_paper.items()):
        payload = _read_json(path)
        schemas[paper_id] = str(payload.get("schema") or "")
        for item in list(payload.get("candidates") or []):
            if not isinstance(item, dict) or item.get("candidate_type") != "figure_caption_candidate":
                continue
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id:
                row = dict(item)
                row["_normalizer_path"] = path
                row["_normalizer_schema"] = schemas[paper_id]
                captions[candidate_id] = row
    return captions, schemas


def _has_caption_source_span(item: dict[str, Any]) -> bool:
    return (
        item.get("canonical_alignment_status") == "aligned"
        and item.get("chars_start") is not None
        and item.get("chars_end") is not None
        and item.get("page") is not None
        and bool(str(item.get("sourceContentHash") or "").strip())
    )


def _layout_ids(normalizer: dict[str, Any] | None) -> list[str]:
    if normalizer is None:
        return []
    return [str(value) for value in list(normalizer.get("layout_element_ids") or []) if str(value)]


def _feasibility_status(
    *,
    normalizer_match: bool,
    layout_region_candidate_present: bool,
    caption_source_span_available: bool,
    region_page_recovered: bool,
    figure_region_type_verified: bool,
    caption_alignment_exact: bool,
) -> str:
    if not normalizer_match:
        return "blocked_missing_normalizer_caption"
    if not layout_region_candidate_present:
        return "blocked_missing_layout_region_candidate"
    if not caption_source_span_available:
        return "figure_region_candidate_caption_alignment_blocked"
    if not caption_alignment_exact:
        return "figure_region_candidate_caption_fuzzy_non_strict"
    if not region_page_recovered:
        return "figure_region_candidate_no_region_page"
    if not figure_region_type_verified:
        return "figure_region_candidate_type_unverified"
    return "figure_region_link_candidate_non_strict"


def _strict_blockers(
    item: dict[str, Any],
    *,
    normalizer_match: bool,
    layout_region_candidate_present: bool,
    caption_source_span_available: bool,
    region_page_recovered: bool,
    figure_region_type_verified: bool,
) -> list[str]:
    blockers = [str(value) for value in list(item.get("strict_blockers") or []) if str(value)]
    required = [
        "figure_region_link_feasibility_audit_only",
        "runtime_promotion_disabled_for_tranche",
        "figure_region_link_incomplete",
        "generated_mineru_markdown_layout_is_candidate_only",
        "markdown_offsets_are_generated_not_original_pdf_offsets",
    ]
    if not normalizer_match:
        required.append("missing_normalizer_figure_caption_candidate")
    if not layout_region_candidate_present:
        required.append("missing_layout_region_candidate")
    if not caption_source_span_available:
        required.append("caption_source_span_incomplete")
    if not region_page_recovered:
        required.append("figure_region_page_missing")
    if not figure_region_type_verified:
        required.append("figure_region_type_unverified")
    return list(dict.fromkeys([*blockers, *required]))


def _row(index: int, item: dict[str, Any], normalizer_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_candidate_id = str(item.get("source_candidate_id") or "")
    normalizer = normalizer_by_id.get(source_candidate_id)
    layout_ids = _layout_ids(normalizer)
    bbox = _bbox((normalizer or {}).get("bbox"))
    normalizer_page = _safe_int((normalizer or {}).get("page")) if normalizer is not None else None
    normalizer_match = normalizer is not None
    layout_region_candidate_present = bool(layout_ids and bbox)
    caption_source_span_available = _has_caption_source_span(item)
    caption_alignment_exact = str(item.get("alignment_method") or "") == "exact"
    region_page_recovered = normalizer_page is not None
    figure_region_type_verified = False
    status = _feasibility_status(
        normalizer_match=normalizer_match,
        layout_region_candidate_present=layout_region_candidate_present,
        caption_source_span_available=caption_source_span_available,
        region_page_recovered=region_page_recovered,
        figure_region_type_verified=figure_region_type_verified,
        caption_alignment_exact=caption_alignment_exact,
    )
    blockers = _strict_blockers(
        item,
        normalizer_match=normalizer_match,
        layout_region_candidate_present=layout_region_candidate_present,
        caption_source_span_available=caption_source_span_available,
        region_page_recovered=region_page_recovered,
        figure_region_type_verified=figure_region_type_verified,
    )
    return {
        "audit_id": f"figure-region-link-feasibility:{index:04d}",
        "figure_caption_candidate_id": str(item.get("candidate_id") or ""),
        "source_candidate_id": source_candidate_id,
        "paper_id": str(item.get("paper_id") or ""),
        "candidate_type": "figure_region_link_feasibility_candidate",
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": _clean_text(item.get("candidate_text")),
        "figure_label": str(item.get("figure_label") or ""),
        "caption_text": _clean_text(item.get("caption_text")),
        "caption_alignment_status": str(item.get("canonical_alignment_status") or ""),
        "caption_alignment_method": str(item.get("alignment_method") or ""),
        "caption_chars_start": item.get("chars_start"),
        "caption_chars_end": item.get("chars_end"),
        "caption_page": item.get("page"),
        "sourceContentHash": str(item.get("sourceContentHash") or ""),
        "sourceContentHashSource": str(item.get("sourceContentHashSource") or ""),
        "caption_source_span_available": caption_source_span_available,
        "normalizer_candidate_id": str((normalizer or {}).get("candidate_id") or ""),
        "normalizer_report_path": str((normalizer or {}).get("_normalizer_path") or ""),
        "normalizer_schema": str((normalizer or {}).get("_normalizer_schema") or ""),
        "normalizer_match": normalizer_match,
        "layout_element_ids": layout_ids,
        "layout_element_count": len(layout_ids),
        "bbox": bbox,
        "layout_link_reason": str((normalizer or {}).get("link_reason") or item.get("layout_link_reason") or ""),
        "layout_region_candidate_present": layout_region_candidate_present,
        "normalizer_region_page": normalizer_page,
        "region_page_recovered": region_page_recovered,
        "caption_page_matches_region_page": (
            bool(item.get("page") == normalizer_page) if normalizer_page is not None else False
        ),
        "figure_region_type_verified": figure_region_type_verified,
        "figure_region_link_verified": False,
        "feasibility_status": status,
        "confidence": float(item.get("confidence") or 0.0),
        "evidence_tier": "figure_region_link_feasibility_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "figure_region_link_feasibility_rows_are_not_evidence",
            "caption_span_does_not_verify_figure_image_region",
            "layout_bbox_without_verified_page_and_region_type_is_non_strict",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _schema_violations(
    figure_caption_report: dict[str, Any],
    source_alignment_report: dict[str, Any],
    normalizer_schemas: dict[str, str],
) -> list[str]:
    violations: list[str] = []
    if figure_caption_report.get("schema") != FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID:
        violations.append("figure_caption_candidate_report_schema_mismatch")
    if source_alignment_report.get("schema") != MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID:
        violations.append("mineru_source_alignment_report_schema_mismatch")
    for paper_id, schema in sorted(normalizer_schemas.items()):
        if schema and schema != MINERU_NORMALIZER_AUDIT_SCHEMA_ID:
            violations.append(f"mineru_normalizer_schema_mismatch:{paper_id}")
    return violations


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(item.get("feasibility_status") or "") for item in rows)
    by_paper = Counter(str(item.get("paper_id") or "") for item in rows)
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputFigureCaptionCandidates": len(rows),
        "auditedFigureCaptionCandidates": len(rows),
        "normalizerFigureCaptionMatches": sum(1 for item in rows if item.get("normalizer_match")),
        "captionSourceSpanCandidates": sum(1 for item in rows if item.get("caption_source_span_available")),
        "layoutRegionCandidates": sum(1 for item in rows if item.get("layout_region_candidate_present")),
        "regionPageRecoveredCandidates": sum(1 for item in rows if item.get("region_page_recovered")),
        "captionRegionPageMatchCandidates": sum(1 for item in rows if item.get("caption_page_matches_region_page")),
        "figureRegionTypeVerifiedCandidates": 0,
        "figureRegionLinkVerifiedCandidates": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byFeasibilityStatus": dict(by_status),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_figure_region_link_feasibility_audit(
    *,
    figure_caption_report: str | Path,
    mineru_source_alignment_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only feasibility audit for figure-region linking."""

    figure_caption_path = Path(str(figure_caption_report)).expanduser()
    source_alignment_path = Path(str(mineru_source_alignment_report)).expanduser()
    figure_caption_payload = _read_json(figure_caption_path)
    source_alignment_payload = _read_json(source_alignment_path)
    normalizer_paths = _normalizer_paths(source_alignment_payload)
    normalizer_captions, normalizer_schemas = _normalizer_figure_captions(normalizer_paths)
    schema_violations = _schema_violations(figure_caption_payload, source_alignment_payload, normalizer_schemas)
    candidates = _candidate_rows(figure_caption_payload)
    rows = [_row(index + 1, item, normalizer_captions) for index, item in enumerate(candidates)]
    counts = _counts(rows, schema_violations)
    status = "blocked" if schema_violations else "ok"
    return {
        "schema": FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "figureCaptionReport": str(figure_caption_path),
            "mineruSourceAlignmentReport": str(source_alignment_path),
            "figureCaptionSchema": str(figure_caption_payload.get("schema") or ""),
            "mineruSourceAlignmentSchema": str(source_alignment_payload.get("schema") or ""),
            "mineruNormalizerReportPaths": normalizer_paths,
            "mineruNormalizerSchemas": normalizer_schemas,
        },
        "counts": counts,
        "gate": {
            "figureRegionLinkReviewed": not schema_violations,
            "figureRegionCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "blocked" if schema_violations else "figure_region_link_feasibility_reviewed",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "candidate_layer_promotion_policy_draft",
        },
        "policy": {
            "auditOnly": True,
            "figureRegionCandidateOnly": True,
            "figureRegionEvidenceCreated": False,
            "figureRegionLinkVerified": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "caption_source_span_does_not_verify_figure_image_region",
            "mineru_layout_bbox_is_candidate_only_without_region_page_and_type_verification",
            "figure_region_linking_requires_explicit_visual_or_layout_region_authority_before_runtime_use",
            "no_parser_routing_or_runtime_answer_integration_is_changed",
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_figure_region_link_feasibility_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Figure Region Link Feasibility Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Audited figure-caption candidates: `{int(counts.get('auditedFigureCaptionCandidates') or 0)}`",
        f"- Caption source spans: `{int(counts.get('captionSourceSpanCandidates') or 0)}`",
        f"- Layout region candidates: `{int(counts.get('layoutRegionCandidates') or 0)}`",
        f"- Region page recovered: `{int(counts.get('regionPageRecoveredCandidates') or 0)}`",
        f"- Region type verified: `{int(counts.get('figureRegionTypeVerifiedCandidates') or 0)}`",
        f"- Figure-region links verified: `{int(counts.get('figureRegionLinkVerifiedCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All rows are `figure_region_link_feasibility_candidate_only`. They are not strict evidence.",
        "A canonical caption span can support caption review, but it does not verify the visual figure/image region.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By feasibility status: `{json.dumps(counts.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for item in list(report.get("rows") or []):
        lines.append(
            f"- `{item.get('paper_id')}` `{item.get('figure_label')}` page `{item.get('caption_page')}` "
            f"layout ids `{item.get('layout_element_count')}` -> `{item.get('feasibility_status')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_figure_region_link_feasibility_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "figure-region-link-feasibility-audit.json"
    summary_path = root / "figure-region-link-feasibility-summary.json"
    markdown_path = root / "figure-region-link-feasibility-audit.md"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_figure_region_link_feasibility_audit_markdown(report), encoding="utf-8")
    return {"audit": str(audit_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only figure-region link feasibility audit.")
    parser.add_argument("--figure-caption-report", required=True, help="Path to figure-caption-candidates.json.")
    parser.add_argument(
        "--mineru-source-alignment-report",
        required=True,
        help="Path to mineru-source-alignment-report.json.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_figure_region_link_feasibility_audit(
        figure_caption_report=args.figure_caption_report,
        mineru_source_alignment_report=args.mineru_source_alignment_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_figure_region_link_feasibility_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID",
    "build_figure_region_link_feasibility_audit",
    "render_figure_region_link_feasibility_audit_markdown",
    "write_figure_region_link_feasibility_audit_reports",
]
