"""Report-only equation authority readiness audit.

This helper aggregates existing TeX/equation diagnostic reports and classifies
whether rows have enough equation-native authority for a future strict
structured-evidence tranche. It does not create EquationArtifact,
StrictEvidence, SourceSpan rows, DB/index state, vault content, parser routing,
or answer integration.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.tex_equation_canonical_alignment_diagnostic_audit import (
    TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_label_number_pdf_region_disambiguation_design import (
    TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    DEFAULT_TEX_EQUATION_LINE_LOCAL_ANCHOR_REPORT,
    TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT,
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
)


EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.equation-authority-readiness-audit.v1"
)

DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT = (
    DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT.parents[2]
    / "tex-equation-rendered-macro-term-profile-design-10paper"
    / "01-tex-equation-rendered-macro-term-profile-design"
    / "tex-equation-rendered-macro-term-profile-design-report.json"
)
DEFAULT_TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-canonical-alignment-diagnostic-audit"
    / "tex-equation-canonical-alignment-diagnostic-report.json"
)
DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT = (
    DEFAULT_TEX_EQUATION_LINE_LOCAL_ANCHOR_REPORT.parents[1]
    / "tex-equation-pdf-region-anchor-audit"
    / "tex-equation-pdf-region-anchor-report.json"
)
DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "tex-equation-label-number-pdf-region-disambiguation-design-10paper"
    / "01-tex-equation-label-number-pdf-region-disambiguation-design"
    / "tex-equation-label-number-pdf-region-disambiguation-design-report.json"
)

STATUS_CANDIDATE_ONLY = "equation_authority_candidate_only"
STATUS_BLOCKED_MISSING_EQUATION_IDENTITY = "blocked_missing_equation_identity"
STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH = "blocked_missing_tex_or_mathml_hash"
STATUS_BLOCKED_MISSING_PDF_REGION = "blocked_missing_pdf_region"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH = "blocked_ambiguous_equation_match"
STATUS_BLOCKED_RASTER_ONLY_EQUATION = "blocked_raster_only_equation"
STATUS_BLOCKED_INPUT_REPORT_MISSING = "blocked_input_report_missing"
STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION = "blocked_input_schema_violation"

_STATUS_VALUES = {
    STATUS_CANDIDATE_ONLY,
    STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
    STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
    STATUS_BLOCKED_MISSING_PDF_REGION,
    STATUS_BLOCKED_MISSING_SOURCE_HASH,
    STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
    STATUS_BLOCKED_RASTER_ONLY_EQUATION,
    STATUS_BLOCKED_INPUT_REPORT_MISSING,
    STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
}


@dataclass(frozen=True)
class _InputSpec:
    key: str
    label: str
    path: Path
    schema_id: str


@dataclass(frozen=True)
class _LoadedInput:
    spec: _InputSpec
    payload: dict[str, Any]
    missing: bool = False
    unreadable: bool = False

    @property
    def schema(self) -> str:
        return _safe_text(self.payload.get("schema"))


def _default_specs(
    *,
    rendered_macro_term_profile_design_report: str | Path,
    canonical_alignment_diagnostic_report: str | Path,
    pdf_region_anchor_audit_report: str | Path,
    label_number_pdf_region_disambiguation_design_report: str | Path,
) -> list[_InputSpec]:
    return [
        _InputSpec(
            key="renderedMacroTermProfileDesign",
            label="TeX rendered macro term-profile design report",
            path=Path(str(rendered_macro_term_profile_design_report)).expanduser(),
            schema_id=TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
        ),
        _InputSpec(
            key="canonicalAlignmentDiagnostic",
            label="TeX equation canonical alignment diagnostic report",
            path=Path(str(canonical_alignment_diagnostic_report)).expanduser(),
            schema_id=TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
        ),
        _InputSpec(
            key="pdfRegionAnchorAudit",
            label="TeX equation PDF-region anchor audit report",
            path=Path(str(pdf_region_anchor_audit_report)).expanduser(),
            schema_id=TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
        ),
        _InputSpec(
            key="labelNumberPdfRegionDisambiguationDesign",
            label="TeX equation label/number PDF-region disambiguation design report",
            path=Path(str(label_number_pdf_region_disambiguation_design_report)).expanduser(),
            schema_id=TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        ),
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_text(value: Any) -> str:
    return " ".join(_safe_text(value).split())


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_safe_text(value)] if _safe_text(value) else []
    if not isinstance(value, (list, tuple)):
        return []
    return [_safe_text(item) for item in value if _safe_text(item)]


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in list(value):
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[int] = []
    for item in list(value):
        parsed = _safe_int(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _load_input(spec: _InputSpec) -> _LoadedInput:
    if not spec.path.is_file():
        return _LoadedInput(spec=spec, payload={}, missing=True)
    try:
        payload = json.loads(spec.path.read_text(encoding="utf-8"))
    except Exception:
        return _LoadedInput(spec=spec, payload={}, unreadable=True)
    if not isinstance(payload, dict):
        return _LoadedInput(spec=spec, payload={}, unreadable=True)
    return _LoadedInput(spec=spec, payload=payload)


def _input_metadata(loaded: list[_LoadedInput], paper_ids: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"paperIds": paper_ids}
    for item in loaded:
        payload[f"{item.spec.key}ReportPath"] = str(item.spec.path)
        payload[f"{item.spec.key}ReportSchema"] = item.schema
        payload[f"{item.spec.key}ReportExists"] = not item.missing
    return payload


def _input_blockers(loaded: list[_LoadedInput]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for item in loaded:
        if item.missing:
            blockers.append(
                {
                    "inputName": item.spec.key,
                    "inputLabel": item.spec.label,
                    "path": str(item.spec.path),
                    "readiness_status": STATUS_BLOCKED_INPUT_REPORT_MISSING,
                    "detail": "expected input report is missing",
                }
            )
        elif item.unreadable:
            blockers.append(
                {
                    "inputName": item.spec.key,
                    "inputLabel": item.spec.label,
                    "path": str(item.spec.path),
                    "readiness_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                    "detail": "expected input report is unreadable or not a JSON object",
                }
            )
        elif item.schema != item.spec.schema_id:
            blockers.append(
                {
                    "inputName": item.spec.key,
                    "inputLabel": item.spec.label,
                    "path": str(item.spec.path),
                    "readiness_status": STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION,
                    "detail": f"schema mismatch: expected {item.spec.schema_id}, got {item.schema or 'missing'}",
                }
            )
    return blockers


def _rows_by_key(payload: dict[str, Any], *, paper_ids: set[str], source: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(list(payload.get("rows") or []), start=1):
        if not isinstance(row_value, dict):
            continue
        row = dict(row_value)
        paper_id = _safe_text(row.get("paper_id") or row.get("paperId"))
        if paper_ids and paper_id not in paper_ids:
            continue
        key = _candidate_key(row)
        if not key:
            key = f"__missing_identity__:{source}:{index:04d}"
        rows.setdefault(key, row)
    return rows


def _candidate_key(row: dict[str, Any]) -> str:
    for key in (
        "source_candidate_id",
        "sourceCandidateId",
        "candidate_id",
        "candidateId",
        "source_tex_row_id",
        "sourceStructureRowId",
    ):
        value = _safe_text(row.get(key))
        if value:
            return value
    hint = dict(row.get("source_label_number_hint") or {})
    return _safe_text(hint.get("sourceStructureRowId"))


def _first_text(*values: Any) -> str:
    for value in values:
        text = _safe_text(value)
        if text:
            return text
    return ""


def _source_hash(*rows: dict[str, Any]) -> str:
    for row in rows:
        value = _safe_text(row.get("sourceContentHash") or row.get("source_content_hash"))
        if value:
            return value
    return ""


def _first_row_with_value(field: str, *rows: dict[str, Any]) -> dict[str, Any]:
    for row in rows:
        if _safe_text(row.get(field)):
            return row
    return {}


def _nested_first_text(field: str, *rows: dict[str, Any]) -> str:
    for row in rows:
        value = _safe_text(row.get(field))
        if value:
            return value
    return ""


def _nested_first_text_for_fields(fields: tuple[str, ...], *rows: dict[str, Any]) -> str:
    for field in fields:
        value = _nested_first_text(field, *rows)
        if value:
            return value
    return ""


def _best_rendered_profile(row: dict[str, Any]) -> dict[str, Any]:
    profiles = [dict(item) for item in list(row.get("profile_results") or []) if isinstance(item, dict)]
    preferred = _safe_text(row.get("recommended_profile"))
    for profile in profiles:
        if _safe_text(profile.get("profile_name")) == preferred:
            return profile
    return profiles[0] if profiles else {}


def _region_from_payload(region: dict[str, Any]) -> dict[str, Any]:
    page = _safe_int(region.get("page"))
    bbox = _float_list(region.get("bbox"))
    return {
        "page": page,
        "bbox": bbox,
        "blockIndexes": _int_list(region.get("blockIndexes") or region.get("block_indexes")),
        "matchedTerms": _string_list(region.get("matchedTerms") or region.get("matched_terms")),
        "coverage": _safe_float(region.get("coverage")),
        "formulaScore": _safe_float(region.get("formulaScore", region.get("formula_score"))),
        "equationNumbers": _string_list(region.get("equationNumbers") or region.get("equation_numbers")),
        "textPreview": _clean_text(region.get("textPreview") or region.get("text_preview")),
        "available": page is not None and len(bbox) >= 4,
    }


def _selected_pdf_region(
    *,
    label_row: dict[str, Any],
    pdf_row: dict[str, Any],
    rendered_row: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        dict(label_row.get("selected_pdf_region") or {}),
        dict(pdf_row.get("selected_pdf_region") or {}),
        dict(_best_rendered_profile(rendered_row).get("selected_pdf_region") or {}),
    ]
    fallback = _region_from_payload({})
    for candidate in candidates:
        region = _region_from_payload(candidate)
        if region["available"]:
            return region
        if candidate and not fallback["available"]:
            fallback = region
    return fallback


def _equation_identity(
    *,
    source_candidate_id: str,
    label_row: dict[str, Any],
    rendered_row: dict[str, Any],
    pdf_row: dict[str, Any],
    canonical_row: dict[str, Any],
) -> dict[str, Any]:
    label_hint = dict(label_row.get("source_label_number_hint") or {})
    latex_labels = _dedupe(
        [
            *_string_list(label_row.get("latex_labels")),
            *_string_list(label_hint.get("latexLabels")),
            *_string_list(rendered_row.get("latex_labels")),
        ]
    )
    equation_numbers = _dedupe(
        [
            *_string_list(label_hint.get("inferredEquationNumbers")),
            *_string_list(dict(label_row.get("selected_pdf_region") or {}).get("equationNumbers")),
            *_string_list(dict(pdf_row.get("selected_pdf_region") or {}).get("equation_numbers")),
        ]
    )
    equation_id = _nested_first_text("equationId", label_row, rendered_row, pdf_row, canonical_row)
    source_tex_row_id = _first_text(
        label_hint.get("sourceStructureRowId"),
        label_row.get("source_tex_row_id"),
        rendered_row.get("source_tex_row_id"),
        pdf_row.get("source_tex_row_id"),
        canonical_row.get("source_candidate_id"),
    )
    identity_basis = "missing"
    if equation_id:
        identity_basis = "equationId"
    elif source_tex_row_id:
        identity_basis = "source_tex_row_id"
    elif source_candidate_id:
        identity_basis = "source_candidate_id"
    elif latex_labels or equation_numbers:
        identity_basis = "label_or_number_hint"
    return {
        "equationId": equation_id,
        "sourceCandidateId": source_candidate_id,
        "sourceTexRowId": source_tex_row_id,
        "latexLabels": latex_labels,
        "equationNumbers": equation_numbers,
        "identityBasis": identity_basis,
        "available": identity_basis != "missing",
    }


def _tex_mathml_hash_availability(
    *,
    label_row: dict[str, Any],
    rendered_row: dict[str, Any],
    pdf_row: dict[str, Any],
    canonical_row: dict[str, Any],
) -> dict[str, Any]:
    source = _first_row_with_value("candidate_text", label_row, rendered_row, pdf_row, canonical_row)
    candidate_text = _clean_text(source.get("candidate_text"))
    equation_tex_hash = _nested_first_text("equationTeXHash", label_row, rendered_row, pdf_row, canonical_row)
    mathml = _nested_first_text_for_fields(("mathml", "mathML"), label_row, rendered_row, pdf_row, canonical_row)
    mathml_hash = _nested_first_text_for_fields(
        ("mathmlHash", "mathMLHash"),
        label_row,
        rendered_row,
        pdf_row,
        canonical_row,
    )
    return {
        "texAvailable": bool(candidate_text),
        "mathmlAvailable": bool(mathml),
        "equationTeXHash": equation_tex_hash,
        "mathmlHash": mathml_hash,
        "hashAvailable": bool(equation_tex_hash or mathml_hash),
        "candidateText": candidate_text,
        "renderedAliasText": _clean_text(rendered_row.get("rendered_alias_text")),
    }


def _canonical_alignment(rendered_row: dict[str, Any], canonical_row: dict[str, Any]) -> dict[str, Any]:
    best_profile = _best_rendered_profile(rendered_row)
    diagnosis = _safe_text(canonical_row.get("diagnosis"))
    profile_status = _safe_text(best_profile.get("canonical_match_status"))
    raw_count = int(canonical_row.get("raw_tex_match_count") or 0)
    compact_count = int(canonical_row.get("compact_tex_match_count") or 0)
    plain_count = int(canonical_row.get("plain_text_match_count") or 0)
    term_coverage = _safe_float(canonical_row.get("diagnostic_term_coverage"))
    status = diagnosis or profile_status or "missing_canonical_alignment_report_row"
    non_unique = (
        "ambiguous" in status
        or raw_count > 1
        or compact_count > 1
        or plain_count > 1
        or _safe_text(rendered_row.get("recommended_status")).startswith("ambiguous")
        or _safe_text(best_profile.get("profile_status")).startswith("ambiguous")
    )
    matched = bool(
        raw_count == 1
        or compact_count == 1
        or plain_count == 1
        or term_coverage > 0.0
        or profile_status.startswith("unique")
    )
    return {
        "status": status,
        "renderedProfileCanonicalStatus": profile_status,
        "rawTexMatchCount": raw_count,
        "compactTexMatchCount": compact_count,
        "plainTextMatchCount": plain_count,
        "diagnosticTermCoverage": term_coverage,
        "canonicalDocumentAvailable": _safe_bool(canonical_row.get("canonical_document_available")),
        "matched": matched,
        "nonUnique": non_unique,
    }


def _ambiguity(
    *,
    label_row: dict[str, Any],
    rendered_row: dict[str, Any],
    pdf_row: dict[str, Any],
    canonical_alignment: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    label_status = _safe_text(label_row.get("disambiguation_status"))
    if label_status.startswith("ambiguous"):
        reasons.append(f"label_number_disambiguation:{label_status}")
    rendered_status = _safe_text(rendered_row.get("recommended_status"))
    if rendered_status.startswith("ambiguous"):
        reasons.append(f"rendered_macro_profile:{rendered_status}")
    pdf_status = _safe_text(pdf_row.get("pdf_region_anchor_status"))
    if pdf_status.startswith("ambiguous"):
        reasons.append(f"pdf_region_anchor:{pdf_status}")
    if _safe_int(pdf_row.get("pdf_region_candidate_count")) and not _safe_bool(pdf_row.get("pdf_region_anchor_unique")):
        if int(pdf_row.get("pdf_region_candidate_count") or 0) > 1:
            reasons.append("pdf_region_candidate_count_non_unique")
    if _safe_bool(canonical_alignment.get("nonUnique")):
        reasons.append(f"canonical_alignment:{canonical_alignment.get('status')}")
    best_profile = _best_rendered_profile(rendered_row)
    if _safe_text(best_profile.get("pdf_region_match_status")).startswith("ambiguous"):
        reasons.append(f"rendered_profile_pdf_region:{best_profile.get('pdf_region_match_status')}")
    return {
        "nonUniqueMatch": bool(reasons),
        "reasons": _dedupe(reasons),
        "pdfRegionCandidateCount": int(
            label_row.get("pdf_region_candidate_count")
            or pdf_row.get("pdf_region_candidate_count")
            or best_profile.get("pdf_region_candidate_count")
            or 0
        ),
        "labelNumberMatchingCandidateCount": int(label_row.get("label_number_matching_candidate_count") or 0),
    }


def _raster_image_only_blocker(*rows: dict[str, Any]) -> bool:
    fragments: list[str] = []
    for row in rows:
        for key in (
            "source_context_status",
            "pdf_region_anchor_status",
            "diagnosis",
            "feasibility_failure_reason",
            "recommended_action",
        ):
            fragments.append(_safe_text(row.get(key)))
        fragments.extend(_string_list(row.get("strict_blockers")))
        fragments.extend(_string_list(row.get("non_strict_reason")))
    text = " ".join(fragments).casefold()
    return any(
        token in text
        for token in (
            "raster_only",
            "raster-only",
            "image_only",
            "image-only",
            "ocr_only",
            "ocr-only",
            "pdf_block_extraction_unavailable",
            "image_equation",
        )
    )


def _classify(
    *,
    identity: dict[str, Any],
    tex_mathml_hash: dict[str, Any],
    pdf_region: dict[str, Any],
    source_hash: str,
    ambiguity: dict[str, Any],
    raster_only: bool,
) -> tuple[str, list[str], str]:
    if not _safe_bool(identity.get("available")):
        return (
            STATUS_BLOCKED_MISSING_EQUATION_IDENTITY,
            ["equation_identity_missing"],
            "recover_equation_identity_before_strict_equation_authority_review",
        )
    if not source_hash:
        return (
            STATUS_BLOCKED_MISSING_SOURCE_HASH,
            ["sourceContentHash_missing"],
            "recover_source_content_hash_before_equation_authority_review",
        )
    if raster_only:
        return (
            STATUS_BLOCKED_RASTER_ONLY_EQUATION,
            ["raster_or_image_only_equation_signal"],
            "route_to_image_or_ocr_equation_extractor_before_equation_authority_review",
        )
    if _safe_bool(ambiguity.get("nonUniqueMatch")):
        return (
            STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH,
            _string_list(ambiguity.get("reasons")) or ["equation_match_non_unique"],
            "resolve_non_unique_equation_match_before_equation_authority_review",
        )
    if not _safe_bool(pdf_region.get("available")):
        return (
            STATUS_BLOCKED_MISSING_PDF_REGION,
            ["page_or_bbox_pdf_region_missing"],
            "recover_page_bbox_pdf_region_before_equation_authority_review",
        )
    if not _safe_bool(tex_mathml_hash.get("hashAvailable")):
        return (
            STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH,
            ["equationTeXHash_or_mathmlHash_missing"],
            "derive_equation_tex_or_mathml_hash_before_strict_structured_evidence_review",
        )
    return (
        STATUS_CANDIDATE_ONLY,
        ["equation_authority_readiness_audit_only"],
        "queue_for_later_explicit_equation_authority_promotion_review",
    )


def _row(
    index: int,
    source_candidate_id: str,
    *,
    label_row: dict[str, Any],
    rendered_row: dict[str, Any],
    pdf_row: dict[str, Any],
    canonical_row: dict[str, Any],
) -> dict[str, Any]:
    public_source_candidate_id = "" if source_candidate_id.startswith("__missing_identity__") else source_candidate_id
    paper_id = _first_text(
        label_row.get("paper_id"),
        rendered_row.get("paper_id"),
        pdf_row.get("paper_id"),
        canonical_row.get("paper_id"),
    )
    identity = _equation_identity(
        source_candidate_id=public_source_candidate_id,
        label_row=label_row,
        rendered_row=rendered_row,
        pdf_row=pdf_row,
        canonical_row=canonical_row,
    )
    tex_mathml_hash = _tex_mathml_hash_availability(
        label_row=label_row,
        rendered_row=rendered_row,
        pdf_row=pdf_row,
        canonical_row=canonical_row,
    )
    pdf_region = _selected_pdf_region(
        label_row=label_row,
        pdf_row=pdf_row,
        rendered_row=rendered_row,
    )
    canonical = _canonical_alignment(rendered_row, canonical_row)
    ambiguity = _ambiguity(
        label_row=label_row,
        rendered_row=rendered_row,
        pdf_row=pdf_row,
        canonical_alignment=canonical,
    )
    source_hash = _source_hash(label_row, rendered_row, pdf_row, canonical_row)
    raster_only = _raster_image_only_blocker(label_row, rendered_row, pdf_row, canonical_row)
    readiness_status, blockers, recommended_action = _classify(
        identity=identity,
        tex_mathml_hash=tex_mathml_hash,
        pdf_region=pdf_region,
        source_hash=source_hash,
        ambiguity=ambiguity,
        raster_only=raster_only,
    )
    source_file = _first_text(
        label_row.get("source_file"),
        rendered_row.get("source_file"),
        pdf_row.get("source_file"),
        canonical_row.get("source_file"),
    )
    equation_environment = _first_text(
        label_row.get("equation_environment"),
        rendered_row.get("equation_environment"),
        pdf_row.get("equation_environment"),
        canonical_row.get("equation_environment"),
    )
    best_profile = _best_rendered_profile(rendered_row)
    strict_blockers = _dedupe(
        [
            *blockers,
            *[str(item) for item in list(label_row.get("strict_blockers") or [])],
            *[str(item) for item in list(rendered_row.get("strict_blockers") or [])],
            *[str(item) for item in list(pdf_row.get("strict_blockers") or [])],
            *[str(item) for item in list(canonical_row.get("strict_blockers") or [])],
            "equation_authority_readiness_audit_only",
            "no_equation_artifact_created",
            "no_strict_evidence_created",
            "no_source_span_mutation",
        ]
    )
    return {
        "audit_row_id": f"equation-authority-readiness-audit:{paper_id}:{index:04d}",
        "candidate_type": "equation_authority_readiness_audit",
        "paper_id": paper_id,
        "source_candidate_id": public_source_candidate_id,
        "source_file": source_file,
        "equation_environment": equation_environment,
        "source_diagnostic_ids": {
            "canonicalAlignmentDiagnosticId": _safe_text(canonical_row.get("diagnostic_id")),
            "renderedMacroDesignId": _safe_text(rendered_row.get("design_id")),
            "pdfRegionAnchorId": _safe_text(pdf_row.get("pdf_region_anchor_id")),
            "labelNumberDisambiguationDesignId": _safe_text(label_row.get("design_id")),
        },
        "equation_identity": identity,
        "tex_mathml_hash_availability": tex_mathml_hash,
        "pdf_region": pdf_region,
        "canonical_text_alignment": canonical,
        "sourceContentHash": source_hash,
        "source_hash_available": bool(source_hash),
        "surrounding_context": {
            "canonicalDiagnosticTerms": _string_list(canonical_row.get("diagnostic_terms")),
            "canonicalDiagnosticTermMatches": _string_list(canonical_row.get("diagnostic_term_matches")),
            "renderedProfileTerms": _string_list(best_profile.get("normalized_terms")),
            "pdfTextPreview": _safe_text(pdf_region.get("textPreview")),
            "renderedAliasText": _safe_text(tex_mathml_hash.get("renderedAliasText")),
        },
        "ambiguity": ambiguity,
        "raster_image_only_blocker": raster_only,
        "readiness_status": readiness_status,
        "blockers": strict_blockers,
        "recommended_action": recommended_action,
        "equationArtifactCreated": False,
        "strictEvidenceCreated": False,
        "sourceSpanMutated": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "parserRoutingChanged": False,
        "answerIntegrationChanged": False,
    }


def _count_status(rows: list[dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if _safe_text(row.get("readiness_status")) == status)


def _counts(rows: list[dict[str, Any]], input_blockers: list[dict[str, str]]) -> dict[str, Any]:
    by_status = Counter(_safe_text(row.get("readiness_status")) for row in rows)
    input_statuses = Counter(_safe_text(item.get("readiness_status")) for item in input_blockers)
    for status, count in input_statuses.items():
        by_status[status] += count
    return {
        "inputRows": len(rows),
        "targetRows": len(rows),
        "equationAuthorityCandidateOnlyRows": _count_status(rows, STATUS_CANDIDATE_ONLY),
        "blockedMissingEquationIdentityRows": _count_status(rows, STATUS_BLOCKED_MISSING_EQUATION_IDENTITY),
        "blockedMissingTexOrMathmlHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH),
        "blockedMissingPdfRegionRows": _count_status(rows, STATUS_BLOCKED_MISSING_PDF_REGION),
        "blockedMissingSourceHashRows": _count_status(rows, STATUS_BLOCKED_MISSING_SOURCE_HASH),
        "blockedAmbiguousEquationMatchRows": _count_status(rows, STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH),
        "blockedRasterOnlyEquationRows": _count_status(rows, STATUS_BLOCKED_RASTER_ONLY_EQUATION),
        "blockedInputReportMissingRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_REPORT_MISSING, 0)),
        "blockedInputSchemaViolationRows": int(input_statuses.get(STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION, 0)),
        "equationArtifactCreatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "sourceSpanMutatedRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultScanRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "inputBlockerCount": len(input_blockers),
        "byReadinessStatus": dict(by_status),
        "byPaper": dict(Counter(_safe_text(row.get("paper_id")) for row in rows)),
        "byEnvironment": dict(Counter(_safe_text(row.get("equation_environment")) for row in rows)),
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings", "inputBlockers")
        if key in report
    }


def render_equation_authority_readiness_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Equation Authority Readiness Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Candidate-only rows: `{int(counts.get('equationAuthorityCandidateOnlyRows') or 0)}`",
        f"- Blocked missing equation identity rows: `{int(counts.get('blockedMissingEquationIdentityRows') or 0)}`",
        f"- Blocked missing TeX/MathML hash rows: `{int(counts.get('blockedMissingTexOrMathmlHashRows') or 0)}`",
        f"- Blocked missing PDF region rows: `{int(counts.get('blockedMissingPdfRegionRows') or 0)}`",
        f"- Blocked missing source hash rows: `{int(counts.get('blockedMissingSourceHashRows') or 0)}`",
        f"- Blocked ambiguous equation match rows: `{int(counts.get('blockedAmbiguousEquationMatchRows') or 0)}`",
        f"- Blocked raster/image-only rows: `{int(counts.get('blockedRasterOnlyEquationRows') or 0)}`",
        f"- Input blockers: `{int(counts.get('inputBlockerCount') or 0)}`",
        f"- EquationArtifact created rows: `{int(counts.get('equationArtifactCreatedRows') or 0)}`",
        f"- StrictEvidence created rows: `{int(counts.get('strictEvidenceCreatedRows') or 0)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        identity = dict(row.get("equation_identity") or {})
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('readiness_status')}` "
            f"`{identity.get('identityBasis', '')}` `{row.get('source_candidate_id', '')}`"
        )
    if report.get("inputBlockers"):
        lines.extend(["", "## Input blockers", ""])
        for blocker in list(report.get("inputBlockers") or []):
            lines.append(
                f"- `{blocker.get('readiness_status')}` `{blocker.get('inputName')}` {blocker.get('detail')}"
            )
    return "\n".join(lines)


def write_equation_authority_readiness_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-authority-readiness-audit-report.json"
    summary_path = root / "equation-authority-readiness-audit-summary.json"
    markdown_path = root / "equation-authority-readiness-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_authority_readiness_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def build_equation_authority_readiness_audit(
    rendered_macro_term_profile_design_report: str | Path = DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT,
    canonical_alignment_diagnostic_report: str | Path = DEFAULT_TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_REPORT,
    pdf_region_anchor_audit_report: str | Path = DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT,
    label_number_pdf_region_disambiguation_design_report: str | Path = DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT,
    *,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    specs = _default_specs(
        rendered_macro_term_profile_design_report=rendered_macro_term_profile_design_report,
        canonical_alignment_diagnostic_report=canonical_alignment_diagnostic_report,
        pdf_region_anchor_audit_report=pdf_region_anchor_audit_report,
        label_number_pdf_region_disambiguation_design_report=label_number_pdf_region_disambiguation_design_report,
    )
    loaded = [_load_input(spec) for spec in specs]
    input_blockers = _input_blockers(loaded)
    rows: list[dict[str, Any]] = []
    if not input_blockers:
        by_input = {item.spec.key: item.payload for item in loaded}
        rendered_rows = _rows_by_key(
            by_input["renderedMacroTermProfileDesign"],
            paper_ids=allowed,
            source="renderedMacroTermProfileDesign",
        )
        canonical_rows = _rows_by_key(
            by_input["canonicalAlignmentDiagnostic"],
            paper_ids=allowed,
            source="canonicalAlignmentDiagnostic",
        )
        pdf_rows = _rows_by_key(
            by_input["pdfRegionAnchorAudit"],
            paper_ids=allowed,
            source="pdfRegionAnchorAudit",
        )
        label_rows = _rows_by_key(
            by_input["labelNumberPdfRegionDisambiguationDesign"],
            paper_ids=allowed,
            source="labelNumberPdfRegionDisambiguationDesign",
        )
        source_ids = sorted(set(rendered_rows) | set(canonical_rows) | set(pdf_rows) | set(label_rows))
        rows = [
            _row(
                index,
                source_id,
                label_row=label_rows.get(source_id, {}),
                rendered_row=rendered_rows.get(source_id, {}),
                pdf_row=pdf_rows.get(source_id, {}),
                canonical_row=canonical_rows.get(source_id, {}),
            )
            for index, source_id in enumerate(source_ids, start=1)
        ]
    counts = _counts(rows, input_blockers)
    status = "ok" if rows and not input_blockers else "blocked"
    decision = "equation_authority_readiness_audit_ready" if status == "ok" else (
        STATUS_BLOCKED_INPUT_REPORT_MISSING
        if any(item.get("readiness_status") == STATUS_BLOCKED_INPUT_REPORT_MISSING for item in input_blockers)
        else STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION
        if input_blockers
        else "blocked_no_equation_rows"
    )
    warnings = _dedupe(
        [
            "equation authority readiness is classification only",
            "missing TeX/MathML hashes block strict structured equation evidence",
            "PDF-region page/bbox is diagnostic context and not an EquationArtifact",
            "no SourceSpan, StrictEvidence, DB/index, vault, parser routing, or answer integration mutation occurs",
            *[item.get("detail", "") for item in input_blockers],
        ]
    )
    return {
        "schema": EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "input": _input_metadata(loaded, requested),
        "counts": counts,
        "gate": {
            "readinessAuditRows": bool(rows) and not input_blockers,
            "equationAuthorityPromotionReady": False,
            "equationArtifactCreationReady": False,
            "strictEvidenceReady": False,
            "sourceSpanMutationReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "recommendedNextTranche": "equation_authority_hash_and_identity_design",
            "inputBlockers": input_blockers,
        },
        "policy": {
            "reportOnly": True,
            "classificationOnly": True,
            "equationArtifactCreated": False,
            "strictEvidenceCreated": False,
            "sourceSpanMutation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "equationInterpretationAllowed": False,
        },
        "warnings": warnings,
        "inputBlockers": input_blockers,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Build a report-only equation authority readiness audit.")
    parser.add_argument(
        "--rendered-macro-term-profile-design-report",
        default=str(DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT),
    )
    parser.add_argument(
        "--canonical-alignment-diagnostic-report",
        default=str(DEFAULT_TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_REPORT),
    )
    parser.add_argument("--pdf-region-anchor-audit-report", default=str(DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT))
    parser.add_argument(
        "--label-number-disambiguation-design-report",
        default=str(DEFAULT_TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_REPORT),
    )
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated report files.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_authority_readiness_audit(
        rendered_macro_term_profile_design_report=args.rendered_macro_term_profile_design_report,
        canonical_alignment_diagnostic_report=args.canonical_alignment_diagnostic_report,
        pdf_region_anchor_audit_report=args.pdf_region_anchor_audit_report,
        label_number_pdf_region_disambiguation_design_report=args.label_number_disambiguation_design_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_equation_authority_readiness_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_AUTHORITY_READINESS_AUDIT_SCHEMA_ID",
    "STATUS_CANDIDATE_ONLY",
    "STATUS_BLOCKED_MISSING_EQUATION_IDENTITY",
    "STATUS_BLOCKED_MISSING_TEX_OR_MATHML_HASH",
    "STATUS_BLOCKED_MISSING_PDF_REGION",
    "STATUS_BLOCKED_MISSING_SOURCE_HASH",
    "STATUS_BLOCKED_AMBIGUOUS_EQUATION_MATCH",
    "STATUS_BLOCKED_RASTER_ONLY_EQUATION",
    "STATUS_BLOCKED_INPUT_REPORT_MISSING",
    "STATUS_BLOCKED_INPUT_SCHEMA_VIOLATION",
    "build_equation_authority_readiness_audit",
    "render_equation_authority_readiness_audit_markdown",
    "write_equation_authority_readiness_audit_reports",
    "main",
]
