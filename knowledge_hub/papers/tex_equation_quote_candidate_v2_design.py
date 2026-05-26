"""Report-only TeX EquationQuoteCandidate v2 design helper.

This helper consumes the TeX equation PDF-region anchor audit and sketches a
non-runtime v2 candidate layer that combines TeX equation text, canonical
line-local context, source hash, and PDF-region bbox candidates. It does not
create strict evidence, source spans, runtime citations, parser routing, or
answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)


TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-quote-candidate-v2-design.v1"
)

DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-pdf-region-anchor-audit"
    / "tex-equation-pdf-region-anchor-report.json"
)

_TAG_RE = re.compile(r"\\tag\s*\{\s*([^}]+?)\s*\}")


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _equation_label(text: str, region: dict[str, Any]) -> str:
    match = _TAG_RE.search(str(text or ""))
    if match:
        return f"tag:{_clean_text(match.group(1))}"
    numbers = [str(item) for item in list(region.get("equation_numbers") or []) if str(item).strip()]
    if len(numbers) == 1:
        return f"pdf-equation-number:{numbers[0]}"
    return ""


def _held_out_reason(row: dict[str, Any]) -> str | None:
    if not _clean_text(row.get("candidate_text")):
        return "empty_equation_text"
    if str(row.get("pdf_region_anchor_status") or "").startswith("blocked_"):
        return str(row.get("pdf_region_anchor_status") or "blocked")
    if str(row.get("pdf_region_anchor_status") or "") == "empty_equation_text":
        return "empty_equation_text"
    if str(row.get("pdf_region_anchor_status") or "") == "insufficient_normalized_terms":
        return "insufficient_normalized_terms"
    if not bool(row.get("pdf_region_anchor_unique")):
        return "pdf_region_anchor_not_unique"
    selected = dict(row.get("selected_pdf_region") or {})
    if not selected.get("bbox"):
        return "missing_pdf_region_bbox"
    if selected.get("page") is None:
        return "missing_pdf_region_page"
    if not str(row.get("sourceContentHash") or "").strip():
        return "missing_source_content_hash"
    if not list(selected.get("matched_terms") or []):
        return "missing_matched_terms"
    return None


def _strict_blockers(row: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "equation_quote_candidate_v2_design_only",
                "pdf_region_bbox_is_not_source_span",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "tex_offsets_are_not_canonical_source_spans",
                "equation_semantics_not_interpreted",
                "equation_region_not_runtime_verified",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(row.get("strict_blockers") or [])],
            ]
        )
    )


def _candidate(index: int, row: dict[str, Any]) -> dict[str, Any]:
    selected = dict(row.get("selected_pdf_region") or {})
    text = _clean_text(row.get("candidate_text"))
    blockers = _strict_blockers(row)
    confidence = min(
        0.82,
        0.48
        + (_safe_float(selected.get("coverage")) * 0.18)
        + (_safe_float(selected.get("formula_score")) * 0.05)
        + (0.08 if bool(row.get("line_local_ambiguity_resolved_by_pdf_region")) else 0.0),
    )
    return {
        "candidate_id": f"tex-equationquote-v2-design:{row.get('paper_id')}:{index:04d}",
        "candidate_type": "equation_quote_candidate_v2_design",
        "source_pdf_region_anchor_id": str(row.get("pdf_region_anchor_id") or ""),
        "source_line_local_anchor_id": str(row.get("source_line_local_anchor_id") or ""),
        "source_design_id": str(row.get("source_design_id") or ""),
        "source_candidate_id": str(row.get("source_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": str(row.get("source_file") or ""),
        "equation_environment": str(row.get("equation_environment") or ""),
        "candidate_text": text,
        "equation_text": text,
        "equation_label": _equation_label(text, selected),
        "normalized_terms": [str(item) for item in list(row.get("normalized_terms") or [])],
        "canonical_context": {
            "lineLocalAnchorStatus": str(row.get("line_local_anchor_status") or ""),
            "lineLocalAnchorMethod": str(row.get("line_local_anchor_method") or ""),
            "canonicalPageMarkers": [_safe_int(item) for item in list(row.get("canonical_page_markers") or [])],
            "normalizedWindowCount": int(row.get("normalized_window_count") or 0),
            "canonicalCharsStart": None,
            "canonicalCharsEnd": None,
            "canonicalPage": None,
            "canonicalSourceSpanCreated": False,
        },
        "pdf_region": {
            "pdfRegionAnchorStatus": str(row.get("pdf_region_anchor_status") or ""),
            "pdfRegionAnchorMethod": str(row.get("pdf_region_anchor_method") or ""),
            "page": selected.get("page"),
            "bbox": selected.get("bbox") or [],
            "blockIndexes": selected.get("block_indexes") or [],
            "matchedTerms": selected.get("matched_terms") or [],
            "coverage": selected.get("coverage", 0.0),
            "formulaScore": selected.get("formula_score", 0.0),
            "equationNumbers": selected.get("equation_numbers") or [],
            "textPreview": selected.get("text_preview", ""),
        },
        "source_pdf_path": str(row.get("source_pdf_path") or ""),
        "source_manifest_path": str(row.get("source_manifest_path") or ""),
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "sourceContentHashSource": "pymupdf_parsed_manifest_source_pdf",
        "source_hash_agrees_with_input": bool(row.get("source_hash_agrees_with_input")),
        "text_source_authority": "arxiv_tex_equation_environment",
        "location_source_authority": "pymupdf_pdf_region_bbox_candidate",
        "source_span_created": False,
        "equation_region_verified": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_quote_candidate_v2_design_only",
        "confidence": round(confidence, 6),
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
        "recommended_action": "keep_as_report_only_v2_design_candidate",
    }


def _held_out(row: dict[str, Any], reason: str) -> dict[str, Any]:
    selected = dict(row.get("selected_pdf_region") or {})
    return {
        "sourcePdfRegionAnchorId": str(row.get("pdf_region_anchor_id") or ""),
        "sourceLineLocalAnchorId": str(row.get("source_line_local_anchor_id") or ""),
        "paperId": str(row.get("paper_id") or ""),
        "candidateText": _clean_text(row.get("candidate_text")),
        "pdfRegionAnchorStatus": str(row.get("pdf_region_anchor_status") or ""),
        "lineLocalAnchorStatus": str(row.get("line_local_anchor_status") or ""),
        "pdfRegionCandidateCount": int(row.get("pdf_region_candidate_count") or 0),
        "selectedPdfRegionPage": selected.get("page"),
        "selectedPdfRegionBbox": selected.get("bbox") or [],
        "reason": reason,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
    }


def _counts(
    *,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "pdfRegionAnchorRows": len(rows),
        "uniquePdfRegionAnchorRows": sum(1 for row in rows if bool(row.get("pdf_region_anchor_unique"))),
        "pdfRegionResolvedAmbiguousRows": sum(
            1 for row in rows if bool(row.get("line_local_ambiguity_resolved_by_pdf_region"))
        ),
        "v2DesignCandidates": len(candidates),
        "heldOutRows": len(held_out),
        "sourceSpanCreatedRows": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "equationRegionVerifiedCandidates": 0,
        "equationSemanticsInterpretedCandidates": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in candidates)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_tex_equation_quote_candidate_v2_design(
    pdf_region_anchor_report: str | Path = DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    input_path = Path(str(pdf_region_anchor_report)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    parent_schema = str(payload.get("schema") or "")
    schema_violations = [] if parent_schema == TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID else [
        "tex_equation_pdf_region_anchor_audit_schema_mismatch"
    ]
    rows = [
        dict(row)
        for row in list(payload.get("rows") or [])
        if not schema_violations
        and isinstance(row, dict)
        and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    candidates: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for row in rows:
        reason = _held_out_reason(row)
        if reason:
            held_out.append(_held_out(row, reason))
            continue
        candidates.append(_candidate(len(candidates) + 1, row))
    counts = _counts(rows=rows, candidates=candidates, held_out=held_out, schema_violations=schema_violations)
    return {
        "schema": TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID,
        "status": "ok" if candidates and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "pdfRegionAnchorReportPath": str(input_path),
            "pdfRegionAnchorReportSchema": parent_schema,
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "v2DesignReady": bool(candidates) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_quote_candidate_v2_design_ready" if candidates and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "equation_quote_candidate_v2_review_pack_or_10paper_pilot",
        },
        "policy": {
            "allCandidatesNonStrict": True,
            "reportOnly": True,
            "v2DesignOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
            "equationInterpretationAllowed": False,
            "equationRegionVerificationRequired": True,
        },
        "designRules": [
            "require_expected_pdf_region_anchor_schema",
            "emit_only_rows_with_unique_pdf_region_anchor",
            "require_candidate_text_source_hash_page_and_bbox",
            "carry_tex_equation_text_as_quote_text_only",
            "carry_pdf_bbox_as_region_candidate_only",
            "do_not_interpret_equations",
            "do_not_create_source_spans",
            "keep_every_v2_design_candidate_non_strict",
            "require_later_explicit_tranche_before_runtime_or_strict_promotion",
        ],
        "warnings": [
            "equation_quote_candidate_v2_design_rows_are_not_runtime_evidence",
            "pdf_region_bboxes_are_not_source_spans",
            "equation_semantics_are_not_interpreted",
            "source_hash_page_and_bbox_do_not_imply_strict_eligibility",
            "canonical_generated_markdown_offsets_remain_unavailable_for_these_equations",
            *([] if not schema_violations else schema_violations),
        ],
        "candidates": candidates,
        "heldOut": held_out,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "designRules", "warnings", "heldOut")
        if key in report
    }


def render_tex_equation_quote_candidate_v2_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX EquationQuoteCandidate v2 Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Unique PDF-region anchor rows: `{int(counts.get('uniquePdfRegionAnchorRows') or 0)}`",
        f"- PDF-region resolved ambiguous rows: `{int(counts.get('pdfRegionResolvedAmbiguousRows') or 0)}`",
        f"- v2 design candidates: `{int(counts.get('v2DesignCandidates') or 0)}`",
        f"- Held out rows: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceCandidates') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a v2 design report only. TeX equation text and PDF bboxes are kept as non-strict quote/location candidates.",
        "No equation is interpreted, no source span is created, and no answer/runtime path consumes these rows.",
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
        region = dict(item.get("pdf_region") or {})
        lines.append(
            f"- `{item.get('paper_id')}` page `{region.get('page')}` bbox `{region.get('bbox')}` "
            f"{item.get('equation_text')}"
        )
    return "\n".join(lines)


def write_tex_equation_quote_candidate_v2_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-quote-candidate-v2-design-report.json"
    summary_path = root / "tex-equation-quote-candidate-v2-design-summary.json"
    markdown_path = root / "tex-equation-quote-candidate-v2-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_quote_candidate_v2_design_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX EquationQuoteCandidate v2 design report.")
    parser.add_argument("--pdf-region-anchor-report", default=str(DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_quote_candidate_v2_design(
        pdf_region_anchor_report=args.pdf_region_anchor_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_quote_candidate_v2_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_QUOTE_CANDIDATE_V2_DESIGN_SCHEMA_ID",
    "build_tex_equation_quote_candidate_v2_design",
    "render_tex_equation_quote_candidate_v2_design_markdown",
    "write_tex_equation_quote_candidate_v2_design_reports",
]
