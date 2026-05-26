"""Report-only segmented matching design for multiline TeX equations.

This helper consumes remaining-window diagnostics and evaluates whether long
TeX equation rows become easier to align when split into row-like segments. It
records canonical generated-Markdown and PDF-region candidate signals only. It
does not create source spans, strict evidence, runtime evidence, parser
routing, answer integration, or canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from knowledge_hub.papers.tex_equation_canonical_text_normalizer_design import (
    _canonical_tokens,
    _ordered_windows,
    _profile_terms,
    _read_text,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    _extract_pdf_blocks,
    _pdf_region_candidates,
    _select_region,
    _source_context,
)
from knowledge_hub.papers.tex_equation_remaining_window_diagnostic import (
    TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID,
)
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import DEFAULT_PARSED_ROOT


TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-segmented-multiline-matching-design.v1"
)

DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-remaining-window-diagnostic-10paper"
    / "01-tex-equation-remaining-window-diagnostic"
    / "tex-equation-remaining-window-diagnostic-report.json"
)

_SPACE_RE = re.compile(r"\s+")
_SEGMENT_SPLIT_RE = re.compile(r"\\\\+")
_LEADING_ALIGNMENT_RE = re.compile(r"^\s*(?:&|\+|=|\s)+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _paper_document_path(parsed_root: Path, paper_id: str) -> Path:
    return parsed_root / paper_id / "document.md"


def _segments(text: str) -> list[str]:
    parts = [_clean_text(_LEADING_ALIGNMENT_RE.sub("", part)) for part in _SEGMENT_SPLIT_RE.split(text)]
    return [part for part in parts if part]


def _segment_terms(text: str) -> tuple[str, list[str]]:
    compact_terms = _profile_terms("canonical_math_compaction_v1", text)
    if len(compact_terms) >= 2:
        return "canonical_math_compaction_v1", compact_terms
    return "ordered_bridge_terms_v1", _profile_terms("ordered_bridge_terms_v1", text)


def _canonical_status(
    *,
    document_text: str,
    terms: list[str],
) -> tuple[str, int]:
    if not document_text:
        return "blocked_missing_canonical_document", 0
    if len(terms) < 2:
        return "insufficient_segment_terms", 0
    windows = _ordered_windows(terms, _canonical_tokens(document_text), max_gap_tokens=35)
    if len(windows) == 1:
        return "unique_segment_canonical_window_candidate_only", 1
    if len(windows) > 1:
        return "ambiguous_segment_canonical_window_candidate_only", len(windows)
    return "no_segment_canonical_window", 0


def _pdf_status(
    *,
    row: dict[str, Any],
    terms: list[str],
    source_context: dict[str, Any],
) -> tuple[str, int, dict[str, Any]]:
    source_status = str(source_context.get("status") or "blocked")
    if source_status != "ok":
        return source_status, 0, {}
    if len(terms) < 2:
        return "insufficient_segment_terms", 0, {}
    segment_row = dict(row)
    segment_row["normalized_terms"] = terms
    segment_row["window_details"] = []
    candidates = _pdf_region_candidates(segment_row, source_context)
    selected, unique = _select_region(candidates)
    if selected and unique:
        return "unique_segment_pdf_region_candidate_only", len(candidates), selected
    if candidates:
        return "ambiguous_segment_pdf_region_candidate_only", len(candidates), selected or {}
    return "no_segment_pdf_region_candidate", 0, {}


def _segment(
    *,
    parent_index: int,
    segment_index: int,
    row: dict[str, Any],
    text: str,
    document_text: str,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    profile, terms = _segment_terms(text)
    canonical_status, canonical_window_count = _canonical_status(document_text=document_text, terms=terms)
    pdf_status, pdf_candidate_count, selected_pdf = _pdf_status(
        row=row,
        terms=terms,
        source_context=source_context,
    )
    segment_ready = canonical_status.startswith("unique_") or pdf_status.startswith("unique_")
    blockers = list(
        dict.fromkeys(
            [
                "segmented_multiline_matching_design_only",
                "segment_matches_do_not_create_source_spans",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "pdf_region_bbox_is_not_source_span",
                "equation_semantics_not_interpreted",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "segment_id": f"tex-equation-segmented-multiline-design:{parent_index:04d}:{segment_index:02d}",
        "segment_index": segment_index,
        "segment_text": text,
        "segment_text_length": len(text),
        "term_profile": profile,
        "normalized_terms": terms,
        "normalized_term_count": len(terms),
        "canonical_match_status": canonical_status,
        "canonical_window_count": canonical_window_count,
        "pdf_region_match_status": pdf_status,
        "pdf_region_candidate_count": pdf_candidate_count,
        "selected_pdf_region": {
            "page": selected_pdf.get("page"),
            "bbox": selected_pdf.get("bbox") or [],
            "blockIndexes": selected_pdf.get("block_indexes") or [],
            "matchedTerms": selected_pdf.get("matched_terms") or [],
            "coverage": selected_pdf.get("coverage", 0.0),
            "formulaScore": selected_pdf.get("formula_score", 0.0),
            "textPreview": selected_pdf.get("text_preview", ""),
        },
        "segment_candidate_ready": segment_ready,
        "source_span_created": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _row(
    index: int,
    source_row: dict[str, Any],
    *,
    parsed_root: Path,
    document_cache: dict[str, str],
    source_context_cache: dict[str, dict[str, Any]],
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    paper_id = str(source_row.get("paper_id") or "")
    document_path = _paper_document_path(parsed_root, paper_id)
    document_key = str(document_path)
    if document_key not in document_cache:
        document_cache[document_key] = _read_text(document_path)
    if paper_id not in source_context_cache:
        source_context_cache[paper_id] = _source_context(
            paper_id=paper_id,
            parsed_root=parsed_root,
            pdf_block_loader=pdf_block_loader,
        )
    source_context = source_context_cache[paper_id]
    segment_rows = [
        _segment(
            parent_index=index,
            segment_index=segment_index,
            row=source_row,
            text=segment_text,
            document_text=document_cache[document_key],
            source_context=source_context,
        )
        for segment_index, segment_text in enumerate(_segments(str(source_row.get("candidate_text") or "")), start=1)
    ]
    ready_segments = [segment for segment in segment_rows if bool(segment.get("segment_candidate_ready"))]
    blockers = list(
        dict.fromkeys(
            [
                "segmented_multiline_matching_design_only",
                "parent_equation_row_remains_non_strict",
                "source_span_creation_disabled_for_tranche",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(source_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "design_id": f"tex-equation-segmented-multiline-design:{index:04d}",
        "candidate_type": "tex_equation_segmented_multiline_matching_design",
        "source_diagnostic_id": str(source_row.get("diagnostic_id") or ""),
        "source_pdf_region_anchor_id": str(source_row.get("source_pdf_region_anchor_id") or ""),
        "source_line_local_anchor_id": str(source_row.get("source_line_local_anchor_id") or ""),
        "source_design_id": str(source_row.get("source_design_id") or ""),
        "source_candidate_id": str(source_row.get("source_candidate_id") or ""),
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": str(source_row.get("source_file") or ""),
        "equation_environment": str(source_row.get("equation_environment") or ""),
        "candidate_text": _clean_text(source_row.get("candidate_text")),
        "segment_count": len(segment_rows),
        "candidate_ready_segment_count": len(ready_segments),
        "source_context_status": str(source_context.get("status") or "blocked"),
        "sourceContentHash": str(source_context.get("sourceContentHash") or ""),
        "source_pdf_path": str(source_context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(source_context.get("manifestPath") or ""),
        "segments": segment_rows,
        "evidence_tier": "segmented_multiline_matching_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "recommended_action": (
            "review_segment_candidates_before_any_later_promotion_design"
            if ready_segments
            else "keep_blocked_pending_rendered_macro_or_alternative_extractor_design"
        ),
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(
    *,
    input_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    segments = [segment for row in rows for segment in list(row.get("segments") or [])]
    return {
        "inputRows": len(input_rows),
        "targetRows": len(rows),
        "segmentRows": len(segments),
        "rowsWithCandidateReadySegments": sum(1 for row in rows if int(row.get("candidate_ready_segment_count") or 0) > 0),
        "candidateReadySegmentRows": sum(1 for segment in segments if bool(segment.get("segment_candidate_ready"))),
        "uniqueCanonicalSegmentRows": sum(
            1 for segment in segments if str(segment.get("canonical_match_status") or "").startswith("unique_")
        ),
        "ambiguousCanonicalSegmentRows": sum(
            1 for segment in segments if str(segment.get("canonical_match_status") or "").startswith("ambiguous_")
        ),
        "uniquePdfRegionSegmentRows": sum(
            1 for segment in segments if str(segment.get("pdf_region_match_status") or "").startswith("unique_")
        ),
        "ambiguousPdfRegionSegmentRows": sum(
            1 for segment in segments if str(segment.get("pdf_region_match_status") or "").startswith("ambiguous_")
        ),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byEnvironment": dict(Counter(str(row.get("equation_environment") or "") for row in rows)),
        "byCanonicalMatchStatus": dict(Counter(str(segment.get("canonical_match_status") or "") for segment in segments)),
        "byPdfRegionMatchStatus": dict(Counter(str(segment.get("pdf_region_match_status") or "") for segment in segments)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_tex_equation_segmented_multiline_matching_design(
    remaining_window_diagnostic_report: str | Path = DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT,
    *,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]] = _extract_pdf_blocks,
) -> dict[str, Any]:
    input_path = Path(str(remaining_window_diagnostic_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    parent_schema = str(payload.get("schema") or "")
    schema_violations = [] if parent_schema == TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID else [
        "tex_equation_remaining_window_diagnostic_schema_mismatch"
    ]
    input_rows = [
        dict(row)
        for row in list(payload.get("rows") or [])
        if not schema_violations
        and isinstance(row, dict)
        and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    target_rows = [
        row
        for row in input_rows
        if str(row.get("recommended_action") or "") == "design_segmented_multiline_equation_matching"
    ]
    document_cache: dict[str, str] = {}
    source_context_cache: dict[str, dict[str, Any]] = {}
    rows = [
        _row(
            index + 1,
            row,
            parsed_root=parsed_root_path,
            document_cache=document_cache,
            source_context_cache=source_context_cache,
            pdf_block_loader=pdf_block_loader,
        )
        for index, row in enumerate(target_rows)
    ]
    counts = _counts(input_rows=input_rows, rows=rows, schema_violations=schema_violations)
    return {
        "schema": TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "remainingWindowDiagnosticReportPath": str(input_path),
            "remainingWindowDiagnosticReportSchema": parent_schema,
            "parsedRoot": str(parsed_root_path),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "designReady": bool(rows) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "segmented_multiline_matching_design_ready" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "segmented_multiline_candidate_review_or_rendered_macro_profile_design",
        },
        "policy": {
            "allRowsNonStrict": True,
            "reportOnly": True,
            "designOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
            "equationInterpretationAllowed": False,
        },
        "warnings": [
            "segment_matches_are_not_source_spans",
            "pdf_region_matches_are_layout_candidates_only",
            "recommended_actions_are_nonbinding",
            "do_not_promote_without_later_explicit_tranche",
            *schema_violations,
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_tex_equation_segmented_multiline_matching_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX Equation Segmented Multiline Matching Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Segment rows: `{int(counts.get('segmentRows') or 0)}`",
        f"- Candidate-ready segments: `{int(counts.get('candidateReadySegmentRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This report evaluates segment-level matching signals only. It does not create evidence or connect to runtime answering.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Canonical status: `{json.dumps(counts.get('byCanonicalMatchStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- PDF-region status: `{json.dumps(counts.get('byPdfRegionMatchStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Recommended action: `{json.dumps(counts.get('byRecommendedAction') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('equation_environment')}` "
            f"segments `{row.get('segment_count')}` ready `{row.get('candidate_ready_segment_count')}`"
        )
    return "\n".join(lines)


def write_tex_equation_segmented_multiline_matching_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-segmented-multiline-matching-design-report.json"
    summary_path = root / "tex-equation-segmented-multiline-matching-design-summary.json"
    markdown_path = root / "tex-equation-segmented-multiline-matching-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_segmented_multiline_matching_design_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only segmented multiline TeX equation matching design.")
    parser.add_argument("--remaining-window-diagnostic-report", default=str(DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT), help="Parsed paper artifact root.")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_segmented_multiline_matching_design(
        remaining_window_diagnostic_report=args.remaining_window_diagnostic_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_segmented_multiline_matching_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID",
    "build_tex_equation_segmented_multiline_matching_design",
    "render_tex_equation_segmented_multiline_matching_design_markdown",
    "write_tex_equation_segmented_multiline_matching_design_reports",
]
