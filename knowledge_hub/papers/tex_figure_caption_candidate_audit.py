"""Report-only TeX-derived FigureCaptionCandidate audit helpers.

This module consumes the TeX structure candidate alignment audit and projects
exactly aligned TeX figure captions into a non-strict FigureCaption candidate
layer. It does not create strict evidence, route parsers, mutate SQLite,
reindex, reembed, write canonical parsed artifacts, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.tex_structure_candidate_alignment_audit import (
    TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID,
)


TEX_FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-figure-caption-candidate-report.v1"
)

DEFAULT_TEX_STRUCTURE_ALIGNMENT_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-structure-candidate-alignment-audit"
    / "tex-structure-candidate-alignment-report.json"
)

_FIGURE_CAPTION_TYPE = "figure_caption"
_GENERIC_CAPTION_TYPE = "caption"


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
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _hold_reason(row: dict[str, Any]) -> str | None:
    structure_type = str(row.get("structure_type") or "")
    if structure_type == _GENERIC_CAPTION_TYPE:
        return "generic_caption_excluded"
    if structure_type != _FIGURE_CAPTION_TYPE:
        return "not_figure_caption"
    if not _clean_text(row.get("candidate_text")):
        return "empty_caption_text"
    status = str(row.get("alignment_status") or "")
    method = str(row.get("alignment_method") or "")
    if status == "ambiguous":
        return "ambiguous_canonical_match"
    if status != "aligned":
        return "canonical_alignment_not_available"
    if method != "exact":
        return "non_exact_alignment"
    if row.get("chars_start") is None or row.get("chars_end") is None:
        return "missing_chars_start_end"
    if row.get("page") is None:
        return "missing_page"
    if not str(row.get("sourceContentHash") or "").strip():
        return "missing_source_content_hash"
    if row.get("source_span_candidate_ready") is not True:
        return "source_span_candidate_not_ready"
    return None


def _candidate(index: int, row: dict[str, Any]) -> dict[str, Any]:
    strict_blockers = list(
        dict.fromkeys(
            [
                "source_structure_candidate_only",
                "figure_caption_candidate_layer_not_runtime_evidence",
                "figure_region_link_unverified",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "tex_offsets_are_not_canonical_source_spans",
                *[str(item) for item in list(row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "candidate_id": f"tex-figurecaption:{row.get('paper_id')}:{index:04d}",
        "candidate_type": "figure_caption_candidate",
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": str(row.get("source_file") or ""),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "caption_text": _clean_text(row.get("candidate_text")),
        "caption_source": "tex_figure_caption",
        "figure_label": "",
        "canonical_alignment_status": "aligned",
        "alignment_method": "exact",
        "chars_start": _safe_int(row.get("chars_start")),
        "chars_end": _safe_int(row.get("chars_end")),
        "page": _safe_int(row.get("page")),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "sourceContentHashSource": str(row.get("sourceContentHashSource") or ""),
        "confidence": float(row.get("confidence") or 0.0),
        "source_span_locator": dict(row.get("source_span_locator") or {}),
        "tex_locator": dict(row.get("tex_locator") or {}),
        "review": {
            "sourceAuditCandidateId": str(row.get("candidate_id") or ""),
            "classification": str(row.get("classification") or ""),
            "sourceSpanCandidateReady": bool(row.get("source_span_candidate_ready")),
            "mineruLayoutLinkStatus": str(row.get("mineru_layout_link_status") or ""),
            "mineruLayoutLinkMethod": str(row.get("mineru_layout_link_method") or ""),
            "mineruCandidateIds": [str(item) for item in list(row.get("mineru_candidate_ids") or [])],
            "mineruBboxLinkCount": _safe_int(row.get("mineru_bbox_link_count")) or 0,
        },
        "figure_region_verified": False,
        "evidence_tier": "figure_caption_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": strict_blockers,
    }


def _held_out(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "sourceCandidateId": str(row.get("candidate_id") or ""),
        "paperId": str(row.get("paper_id") or ""),
        "candidateText": _clean_text(row.get("candidate_text")),
        "structureType": str(row.get("structure_type") or ""),
        "alignmentStatus": str(row.get("alignment_status") or ""),
        "alignmentMethod": str(row.get("alignment_method") or ""),
        "reason": reason,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
    }


def _counts(
    candidates: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    *,
    input_rows: int,
    figure_caption_rows: int,
    generic_caption_rows: int,
) -> dict[str, Any]:
    return {
        "inputRows": input_rows,
        "figureCaptionRows": figure_caption_rows,
        "genericCaptionRows": generic_caption_rows,
        "figureCaptionCandidates": len(candidates),
        "heldOutCandidates": len(held_out),
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "figureRegionVerifiedCandidates": 0,
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in candidates)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_tex_figure_caption_candidate_report(
    alignment_report_path: str | Path = DEFAULT_TEX_STRUCTURE_ALIGNMENT_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a non-strict FigureCaptionCandidate report from TeX alignment rows."""

    input_path = Path(str(alignment_report_path)).expanduser()
    alignment_report = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    parent_schema = str(alignment_report.get("schema") or "")
    parent_contract_valid = parent_schema == TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID
    rows = [
        dict(row)
        for row in list(alignment_report.get("candidates") or [])
        if parent_contract_valid and isinstance(row, dict) and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    candidates: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    figure_caption_rows = sum(1 for row in rows if str(row.get("structure_type") or "") == _FIGURE_CAPTION_TYPE)
    generic_caption_rows = sum(1 for row in rows if str(row.get("structure_type") or "") == _GENERIC_CAPTION_TYPE)

    for row in rows:
        structure_type = str(row.get("structure_type") or "")
        reason = _hold_reason(row)
        if reason:
            if structure_type in {_FIGURE_CAPTION_TYPE, _GENERIC_CAPTION_TYPE}:
                held_out.append(_held_out(row, reason))
            continue
        candidates.append(_candidate(len(candidates) + 1, row))

    counts = _counts(
        candidates,
        held_out,
        input_rows=len(rows),
        figure_caption_rows=figure_caption_rows,
        generic_caption_rows=generic_caption_rows,
    )
    return {
        "schema": TEX_FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
        "status": "ok" if candidates else "empty",
        "generatedAt": _now(),
        "input": {
            "alignmentReportPath": str(input_path),
            "alignmentReportSchema": parent_schema,
            "paperIds": requested,
        },
        "counts": counts,
        "policy": {
            "allCandidatesNonStrict": True,
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
            "figureRegionVerificationRequired": True,
        },
        "promotionRules": [
            "require_expected_tex_structure_alignment_schema",
            "emit_only_tex_figure_caption_rows",
            "exclude_generic_caption_rows_until_table_figure_disambiguation_exists",
            "require_parent_source_span_candidate_ready",
            "require_exact_canonical_generated_markdown_alignment",
            "require_page_chars_and_source_hash",
            "keep_figure_caption_candidates_non_strict",
            "require_later_figure_region_link_verification_before_strict_promotion",
        ],
        "warnings": [
            "tex_figure_caption_candidates_are_not_runtime_evidence",
            "figure_region_link_is_not_verified",
            "generic_caption_rows_are_held_out_to_avoid_table_caption_contamination",
            "canonical_generated_markdown_offsets_are_not_original_pdf_byte_offsets",
            "source_hash_page_and_chars_do_not_imply_strict_eligibility",
            *([] if parent_contract_valid else ["alignment_report_schema_mismatch"]),
        ],
        "candidates": candidates,
        "heldOut": held_out,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "policy", "promotionRules", "warnings", "heldOut")
        if key in report
    }


def render_tex_figure_caption_candidate_report_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX FigureCaptionCandidate Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Figure caption rows: `{int(counts.get('figureCaptionRows') or 0)}`",
        f"- Generic caption rows held out: `{int(counts.get('genericCaptionRows') or 0)}`",
        f"- FigureCaption candidates: `{int(counts.get('figureCaptionCandidates') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        f"- Figure region verified: `{int(counts.get('figureRegionVerifiedCandidates') or 0)}`",
        "",
        "## Boundary",
        "",
        "All rows are `figure_caption_candidate_only`. They are not strict evidence, runtime evidence, or answer citations.",
        "Figure region links are explicitly unverified in this tranche.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Held out by reason: `{json.dumps(counts.get('heldOutByReason') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Candidates",
        "",
    ]
    for item in list(report.get("candidates") or []):
        caption = _clean_text(item.get("caption_text"))
        lines.append(f"- `{item.get('paper_id')}` page `{item.get('page')}` {caption}")
    return "\n".join(lines)


def write_tex_figure_caption_candidate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "tex-figure-caption-candidates.json"
    summary_path = root / "tex-figure-caption-candidate-summary.json"
    markdown_path = root / "tex-figure-caption-candidate-audit.md"
    candidates_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_figure_caption_candidate_report_markdown(report), encoding="utf-8")
    return {"candidates": str(candidates_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX FigureCaptionCandidate audit.")
    parser.add_argument("--alignment-report", default=str(DEFAULT_TEX_STRUCTURE_ALIGNMENT_REPORT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_figure_caption_candidate_report(
        alignment_report_path=args.alignment_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_figure_caption_candidate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID",
    "build_tex_figure_caption_candidate_report",
    "render_tex_figure_caption_candidate_report_markdown",
    "write_tex_figure_caption_candidate_reports",
]
