"""Report-only TeX equation source-span promotion readiness audit.

This helper consolidates the latest report-only TeX equation design/anchor
outputs and classifies which candidates are currently ready for a later
SourceSpan review tranche.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.tex_equation_label_number_pdf_region_disambiguation_design import (
    TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_segmented_multiline_matching_design import (
    TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
)


TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-source-span-promotion-readiness-audit.v1"
)

DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "tex-equation-label-number-pdf-region-disambiguation-design"
    / "01-tex-equation-label-number-pdf-region-disambiguation-design"
    / "tex-equation-label-number-pdf-region-disambiguation-design-report.json"
)
DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "tex-equation-rendered-macro-term-profile-design"
    / "01-tex-equation-rendered-macro-term-profile-design"
    / "tex-equation-rendered-macro-term-profile-design-report.json"
)
DEFAULT_TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "tex-equation-segmented-multiline-matching-design"
    / "01-tex-equation-segmented-multiline-matching-design"
    / "tex-equation-segmented-multiline-matching-design-report.json"
)
DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "tex-equation-pdf-region-anchor-audit"
    / "tex-equation-pdf-region-anchor-report.json"
)


_READINESS_READY = "promotion_review_ready_candidate_only"
_READINESS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
_READINESS_BLOCKED_AMBIGUOUS_PDF_REGION = "blocked_ambiguous_pdf_region"
_READINESS_BLOCKED_MISSING_LABEL_NUMBER = "blocked_missing_label_number_disambiguation"
_READINESS_BLOCKED_PDF_REGION_ONLY = "blocked_pdf_region_only_not_source_span"
_READINESS_BLOCKED_MANUAL_OR_EXTRACTOR = "blocked_requires_manual_or_later_extractor_review"


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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_text(value: Any) -> str:
    return " ".join(_safe_text(value).split())


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _schema_violations(
    *,
    rendered_report: dict[str, Any],
    label_report: dict[str, Any],
    segmented_report: dict[str, Any],
    pdf_region_report: dict[str, Any],
) -> list[str]:
    violations = []
    if _safe_text(rendered_report.get("schema")) != TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID:
        violations.append("tex_equation_rendered_macro_term_profile_design_schema_mismatch")
    if (
        _safe_text(label_report.get("schema"))
        != TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID
    ):
        violations.append("tex_equation_label_number_pdf_region_disambiguation_design_schema_mismatch")
    if (
        _safe_text(segmented_report.get("schema"))
        != TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID
    ):
        violations.append("tex_equation_segmented_multiline_matching_design_schema_mismatch")
    if _safe_text(pdf_region_report.get("schema")) != TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID:
        violations.append("tex_equation_pdf_region_anchor_audit_schema_mismatch")
    return violations


def _selected_region(row: dict[str, Any], key: str = "selected_pdf_region") -> dict[str, Any]:
    return dict(row.get(key) or {})


def _selected_segment_region(segments: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, bool]:
    ready_regions = [dict(item.get("selected_pdf_region") or {}) for item in segments if bool(item.get("segment_candidate_ready"))]
    if ready_regions:
        return ready_regions[0], True, False
    for segment in segments:
        segment_status = _safe_text(segment.get("pdf_region_match_status"))
        if segment_status.startswith("ambiguous_"):
            return dict(segment.get("selected_pdf_region") or {}), False, True
    if segments:
        return dict(segments[0].get("selected_pdf_region") or {}), False, False
    return {}, False, False


def _has_region(page: Any, bbox: list[Any]) -> bool:
    return page is not None and bool(_safe_int(page) >= 0) and bool(list(bbox))


def _extract_row(
    *,
    source_id: str,
    rendered_row: dict[str, Any] | None,
    label_row: dict[str, Any] | None,
    segmented_row: dict[str, Any] | None,
    pdf_anchor_row: dict[str, Any] | None,
) -> dict[str, Any]:
    rendered = rendered_row or {}
    label = label_row or {}
    segmented = segmented_row or {}
    pdf_anchor = pdf_anchor_row or {}
    paper_id = _safe_text(
        next(
            value
            for value in (
                rendered_row.get("paper_id") if rendered_row else "",
                label_row.get("paper_id") if label_row else "",
                segmented_row.get("paper_id") if segmented_row else "",
                pdf_anchor_row.get("paper_id") if pdf_anchor_row else "",
            )
            if _safe_text(value)
        )
        or ""
    )
    candidate_text = _clean_text(
        next(
            value
            for value in (
                _safe_text(segmented_row.get("candidate_text") if segmented_row else ""),
                _safe_text(label_row.get("candidate_text") if label_row else ""),
                _safe_text(rendered_row.get("candidate_text") if rendered_row else ""),
                _safe_text(pdf_anchor_row.get("candidate_text") if pdf_anchor_row else ""),
            )
            if _safe_text(value)
        )
        or ""
    )
    source_hash = _safe_text(
        next(
            (
                value
                for value in (
                    label_row.get("sourceContentHash") if label_row else "",
                    segmented_row.get("sourceContentHash") if segmented_row else "",
                    pdf_anchor_row.get("sourceContentHash") if pdf_anchor_row else "",
                    rendered_row.get("sourceContentHash") if rendered_row else "",
                )
                if _safe_text(value)
            ),
            "",
        )
    )
    segments = list(segmented_row.get("segments") or []) if isinstance(segmented_row, dict) else []
    segment_region, segment_ready, segment_ambiguous = _selected_segment_region(segments)
    label_region = _selected_region(label_row or {})
    pdf_region = _selected_region(pdf_anchor_row or {})
    rendered_region = _selected_region(rendered_row or {})
    selected_region = label_region or segment_region or pdf_region or rendered_region
    page = selected_region.get("page")
    bbox = selected_region.get("bbox") or []
    block_indexes = selected_region.get("blockIndexes") or selected_region.get("block_indexes") or []

    label_hint = dict(label_row.get("source_label_number_hint") or {}) if isinstance(label_row, dict) else {}
    label_status = _safe_text(label_row.get("disambiguation_status") if label_row else "")
    if label_status == "unique_label_number_pdf_region_candidate_only":
        label_ready = True
    else:
        label_ready = False
    label_ambiguous = label_status.startswith("ambiguous_")
    label_missing_hint = label_status in {"", "blocked_no_label_number_hint", "blocked_no_source_candidates"}

    segment_ready_flag = _safe_int(segmented_row.get("candidate_ready_segment_count") if segmented_row else 0) > 0
    segment_ambiguous = bool(segment_ambiguous)

    pdf_status = _safe_text(pdf_anchor_row.get("pdf_region_anchor_status") if pdf_anchor_row else "")
    pdf_unique = pdf_status.startswith("unique_")
    pdf_ambiguous = pdf_status.startswith("ambiguous_")

    if not source_hash:
        readiness = _READINESS_BLOCKED_MISSING_SOURCE_HASH
    elif label_row is not None:
        if label_missing_hint:
            readiness = _READINESS_BLOCKED_MISSING_LABEL_NUMBER
        elif label_ambiguous:
            readiness = _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        elif label_ready and _has_region(page, bbox):
            readiness = _READINESS_READY
        elif label_status in {"no_label_number_matching_pdf_region_candidate", "blocked_no_label_number_matching_pdf_region_candidate"}:
            readiness = _READINESS_BLOCKED_MISSING_LABEL_NUMBER
        elif pdf_ambiguous:
            readiness = _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        else:
            readiness = _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR
    elif segment_row := segmented_row:
        if segment_ready_flag and _has_region(page, bbox):
            readiness = _READINESS_READY
        elif segment_ambiguous:
            readiness = _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        elif _has_region(page, bbox):
            readiness = _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR
        else:
            readiness = _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR
    elif pdf_anchor_row is not None:
        if pdf_ambiguous:
            readiness = _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        elif pdf_unique and _has_region(page, bbox):
            readiness = _READINESS_BLOCKED_PDF_REGION_ONLY
        elif _safe_int(label_row.get("pdf_region_candidate_count") if label_row else 0) > 0:
            readiness = _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        else:
            readiness = _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR
    elif rendered_row is not None:
        readiness = _READINESS_BLOCKED_MISSING_LABEL_NUMBER
    else:
        readiness = _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR

    if readiness == _READINESS_READY:
        strict_blockers = [
            "source_span_promotion_readiness_audit_only",
            "candidate_ready_for_later_source_span_review_not_executed",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "candidate is review-only and must enter later SourceSpan promotion tranche",
            "no source span has been created in this tranche",
        ]
    elif readiness == _READINESS_BLOCKED_MISSING_SOURCE_HASH:
        strict_blockers = [
            "source_content_hash_missing",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "sourceContentHash required for later SourceSpan promotion review",
            "row remains diagnostic",
        ]
    elif readiness == _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION:
        strict_blockers = [
            "pdf_region_ambiguity_not_resolved",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "ambiguous PDF-region candidates remain non-strict",
            "source span creation remains a later tranche",
        ]
    elif readiness == _READINESS_BLOCKED_MISSING_LABEL_NUMBER:
        strict_blockers = [
            "label_or_number_disambiguation_missing",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "label/number disambiguation evidence missing from latest design stage",
            "row remains diagnostic",
        ]
    elif readiness == _READINESS_BLOCKED_PDF_REGION_ONLY:
        strict_blockers = [
            "pdf_region_candidate_only",
            "source_span_not_materialized",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "only PDF-region anchors are available in this tranche",
            "manual or extraction recovery still required before source span promotion",
        ]
    else:
        strict_blockers = [
            "manual_or_later_extractor_review_required",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_later_explicit_tranche",
        ]
        non_strict_reason = [
            "late-stage extractor recovery or manual mapping required",
        ]

    upstream_blockers = []
    for source_row in (rendered_row, label_row, segmented_row, pdf_anchor_row):
        if isinstance(source_row, dict):
            upstream_blockers.extend(str(item) for item in list(source_row.get("strict_blockers") or []))
    strict_blockers = _dedupe([*strict_blockers, *upstream_blockers])
    non_strict_reason = _dedupe([*non_strict_reason])

    if readiness == _READINESS_READY:
        recommended_next_action = "review_in_source_span_promotion_tranche"
    elif readiness == _READINESS_BLOCKED_MISSING_SOURCE_HASH:
        recommended_next_action = "recover_source_content_hash_from_manifest_before_source_span_review"
    elif readiness == _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION:
        recommended_next_action = "resolve_pdf_region_ambiguity_before_source_span_review"
    elif readiness == _READINESS_BLOCKED_MISSING_LABEL_NUMBER:
        recommended_next_action = "derive_label_or_number_disambiguation_before_source_span_review"
    elif readiness == _READINESS_BLOCKED_PDF_REGION_ONLY:
        recommended_next_action = "manual_or_extractor_review_for_pdf_region_only_rows"
    else:
        recommended_next_action = "manual_or_later_extractor_review_before_source_span_readiness"

    return {
        "readiness_audit_id": f"tex-equation-source-span-promotion-readiness-audit:{paper_id}:{source_id}",
        "candidate_type": "tex_equation_source_span_promotion_readiness_audit",
        "source_rendered_macro_design_id": _safe_text(rendered.get("design_id")),
        "source_label_number_disambiguation_design_id": _safe_text(label.get("design_id")),
        "source_segmented_multiline_design_id": _safe_text(segmented.get("design_id")),
        "source_pdf_region_anchor_id": _safe_text(
            pdf_anchor.get("pdf_region_anchor_id")
            or segmented.get("source_pdf_region_anchor_id")
            or label.get("source_pdf_region_anchor_id")
            or ""
        ),
        "source_line_local_anchor_id": _safe_text(
            pdf_anchor.get("source_line_local_anchor_id")
            or segmented.get("source_line_local_anchor_id", "")
            or label.get("source_line_local_anchor_id", "")
        ),
        "source_diagnostic_id": _safe_text(
            label.get("source_diagnostic_id")
            or segmented.get("source_diagnostic_id", "")
            or rendered.get("source_diagnostic_id", "")
            or pdf_anchor.get("source_diagnostic_id", "")
        ),
        "source_design_id": _safe_text(
            label.get("source_design_id")
            or segmented.get("source_design_id", "")
            or rendered.get("source_design_id", "")
            or pdf_anchor.get("source_design_id", "")
        ),
        "source_candidate_id": source_id,
        "paper_id": paper_id,
        "source_parser": _safe_text(
            rendered.get("source_parser")
            or label.get("source_parser")
            or segmented.get("source_parser")
            or pdf_anchor.get("source_parser")
            or ""
        ),
        "source_file": _safe_text(
            rendered.get("source_file")
            or label.get("source_file")
            or segmented.get("source_file")
            or pdf_anchor.get("source_file", "")
            or ""
        ),
        "equation_environment": _safe_text(
            rendered.get("equation_environment")
            or label.get("equation_environment")
            or segmented.get("equation_environment")
            or pdf_anchor.get("equation_environment", "")
            or ""
        ),
        "candidate_text": candidate_text,
        "source_tex_row_id": _safe_text(
            label_hint.get("sourceStructureRowId")
            or rendered.get("source_structure_row_id", "")
            or rendered.get("sourceStructureRowId", "")
        ),
        "source_pdf_path": _safe_text(
            label.get("source_pdf_path")
            or segmented.get("source_pdf_path")
            or pdf_anchor.get("source_pdf_path")
            or ""
        ),
        "source_manifest_path": _safe_text(
            label.get("source_manifest_path")
            or segmented.get("source_manifest_path")
            or pdf_anchor.get("source_manifest_path")
            or ""
        ),
        "sourceContentHash": source_hash,
        "pdf_region": {
            "page": _safe_int(page) if page is not None else None,
            "bbox": [float(item) for item in bbox],
            "blockIndexes": [_safe_int(item) for item in list(block_indexes)],
            "matchedTerms": [str(item) for item in list(selected_region.get("matchedTerms") or selected_region.get("matched_terms") or [])],
            "coverage": _safe_float(selected_region.get("coverage", 0.0)),
            "formulaScore": _safe_float(selected_region.get("formulaScore", selected_region.get("formula_score", 0.0))),
            "textPreview": _safe_text(selected_region.get("textPreview", selected_region.get("text_preview", ""))),
        },
        "label_number_disambiguation_evidence": {
            "sourceStructureRowId": _safe_text(label_hint.get("sourceStructureRowId")),
            "texEnvironment": _safe_text(label_hint.get("texEnvironment")),
            "latexLabels": [str(item) for item in list(label_hint.get("latexLabels") or [])],
            "inferredEquationNumbers": [str(item) for item in list(label_hint.get("inferredEquationNumbers") or [])],
            "method": _safe_text(label_hint.get("method")),
            "status": _safe_text(label_hint.get("status")),
        },
        "readiness_category": readiness,
        "recommended_next_action": recommended_next_action,
        "source_span_created": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_eligible": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": non_strict_reason,
    }


def _collect_rows(
    *,
    payload: dict[str, Any],
    paper_ids: set[str],
    action_filter: str | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in list(payload.get("rows") or []):
        if not isinstance(row, dict):
            continue
        paper_id = _safe_text(row.get("paper_id"))
        source_candidate_id = _safe_text(row.get("source_candidate_id"))
        if not source_candidate_id:
            continue
        if paper_ids and paper_id not in paper_ids:
            continue
        if action_filter and _safe_text(row.get("recommended_action")) != action_filter:
            continue
        rows[source_candidate_id] = dict(row)
    return rows


def _counts(
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "targetRows": len(rows),
        "promotionReviewReadyCandidateOnlyRows": sum(
            1 for row in rows if str(row.get("readiness_category") or "") == _READINESS_READY
        ),
        "blockedMissingSourceHashRows": sum(
            1
            for row in rows
            if str(row.get("readiness_category") or "") == _READINESS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedAmbiguousPdfRegionRows": sum(
            1
            for row in rows
            if str(row.get("readiness_category") or "") == _READINESS_BLOCKED_AMBIGUOUS_PDF_REGION
        ),
        "blockedMissingLabelNumberRows": sum(
            1
            for row in rows
            if str(row.get("readiness_category") or "") == _READINESS_BLOCKED_MISSING_LABEL_NUMBER
        ),
        "blockedPdfRegionOnlyRows": sum(
            1
            for row in rows
            if str(row.get("readiness_category") or "") == _READINESS_BLOCKED_PDF_REGION_ONLY
        ),
        "blockedManualOrLaterExtractorRows": sum(
            1
            for row in rows
            if str(row.get("readiness_category") or "") == _READINESS_BLOCKED_MANUAL_OR_EXTRACTOR
        ),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byReadiness": dict(Counter(str(row.get("readiness_category") or "") for row in rows)),
        "byEnvironment": dict(Counter(str(row.get("equation_environment") or "") for row in rows)),
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings") if key in report}


def _render_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX Equation Source-Span Promotion Readiness Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Promotion-ready candidate rows: `{int(counts.get('promotionReviewReadyCandidateOnlyRows') or 0)}`",
        f"- Blocked (source hash missing): `{int(counts.get('blockedMissingSourceHashRows') or 0)}`",
        f"- Blocked (PDF region ambiguous): `{int(counts.get('blockedAmbiguousPdfRegionRows') or 0)}`",
        f"- Blocked (missing label/number disambiguation): `{int(counts.get('blockedMissingLabelNumberRows') or 0)}`",
        f"- Blocked (pdf-region only): `{int(counts.get('blockedPdfRegionOnlyRows') or 0)}`",
        f"- Blocked (manual/later extractor): `{int(counts.get('blockedManualOrLaterExtractorRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        "",
        "## Row breakdown",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('readiness_category')}` "
            f"`{row.get('source_candidate_id')}` `{row.get('source_tex_row_id')}`"
        )
    return "\n".join(lines)


def write_tex_equation_source_span_promotion_readiness_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-source-span-promotion-readiness-audit-report.json"
    summary_path = root / "tex-equation-source-span-promotion-readiness-audit-summary.json"
    markdown_path = root / "tex-equation-source-span-promotion-readiness-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def build_tex_equation_source_span_promotion_readiness_audit(
    rendered_macro_term_profile_design_report: str | Path = DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT,
    label_number_pdf_region_disambiguation_design_report: str | Path = DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT,
    segmented_multiline_matching_design_report: str | Path = DEFAULT_TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_REPORT,
    pdf_region_anchor_audit_report: str | Path = DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    rendered_path = Path(str(rendered_macro_term_profile_design_report)).expanduser()
    label_path = Path(str(label_number_pdf_region_disambiguation_design_report)).expanduser()
    segmented_path = Path(str(segmented_multiline_matching_design_report)).expanduser()
    pdf_anchor_path = Path(str(pdf_region_anchor_audit_report)).expanduser()

    rendered = _read_json(rendered_path)
    label = _read_json(label_path)
    segmented = _read_json(segmented_path)
    pdf_anchor = _read_json(pdf_anchor_path)

    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    violations = _schema_violations(
        rendered_report=rendered,
        label_report=label,
        segmented_report=segmented,
        pdf_region_report=pdf_anchor,
    )

    rendered_rows = (
        _collect_rows(
            payload=rendered,
            paper_ids=allowed,
            action_filter="requires_equation_number_or_label_to_pdf_region_disambiguation_design",
        )
        if not violations
        else {}
    )
    label_rows = _collect_rows(payload=label, paper_ids=allowed) if not violations else {}
    segmented_rows = _collect_rows(payload=segmented, paper_ids=allowed) if not violations else {}
    pdf_anchor_rows = _collect_rows(payload=pdf_anchor, paper_ids=allowed) if not violations else {}

    source_candidate_ids = set(rendered_rows) | set(label_rows) | set(segmented_rows) | set(pdf_anchor_rows)
    rows = [
        _extract_row(
            source_id=source_id,
            rendered_row=rendered_rows.get(source_id),
            label_row=label_rows.get(source_id),
            segmented_row=segmented_rows.get(source_id),
            pdf_anchor_row=pdf_anchor_rows.get(source_id),
        )
        for source_id in sorted(source_candidate_ids)
    ]

    counts = _counts(rows=rows, schema_violations=violations)
    return {
        "schema": TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "renderedMacroTermProfileDesignReportPath": str(rendered_path),
            "renderedMacroTermProfileDesignReportSchema": str(rendered.get("schema") or ""),
            "labelNumberPdfRegionDisambiguationDesignReportPath": str(label_path),
            "labelNumberPdfRegionDisambiguationDesignReportSchema": str(label.get("schema") or ""),
            "segmentedMultilineMatchingDesignReportPath": str(segmented_path),
            "segmentedMultilineMatchingDesignReportSchema": str(segmented.get("schema") or ""),
            "pdfRegionAnchorAuditReportPath": str(pdf_anchor_path),
            "pdfRegionAnchorAuditReportSchema": str(pdf_anchor.get("schema") or ""),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "readinessAuditRows": bool(rows) and not violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "source_span_promotion_readiness_audit_ready" if rows and not violations else "blocked",
            "schemaViolations": violations,
            "recommendedNextTranche": "source_span_promotion_review_pack",
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
        "warnings": _dedupe(
            [
                "source-span readiness classifications are audit-only and do not create source spans",
                "any source span/strict/evidence state requires a later explicit promotion tranche",
                *[item.replace("_", " ") for item in violations],
            ]
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Build a report-only TeX equation source-span promotion readiness audit.")
    parser.add_argument(
        "--rendered-macro-term-profile-design-report",
        default=str(DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT),
    )
    parser.add_argument(
        "--label-number-disambiguation-design-report",
        default=str(DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT),
    )
    parser.add_argument(
        "--segmented-multiline-matching-design-report",
        default=str(DEFAULT_TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_REPORT),
    )
    parser.add_argument(
        "--pdf-region-anchor-audit-report",
        default=str(DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT),
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated report files.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_source_span_promotion_readiness_audit(
        rendered_macro_term_profile_design_report=args.rendered_macro_term_profile_design_report,
        label_number_pdf_region_disambiguation_design_report=args.label_number_disambiguation_design_report,
        segmented_multiline_matching_design_report=args.segmented_multiline_matching_design_report,
        pdf_region_anchor_audit_report=args.pdf_region_anchor_audit_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_source_span_promotion_readiness_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_SOURCE_SPAN_PROMOTION_READINESS_AUDIT_SCHEMA_ID",
    "build_tex_equation_source_span_promotion_readiness_audit",
    "write_tex_equation_source_span_promotion_readiness_audit_reports",
    "main",
]
