"""Report-only parsed-artifact structured evidence readiness audit.

This helper evaluates existing local report artifacts and classifies candidates by
readiness buckets for the next source-span-promotion tranche.

The helper is intentionally report-only and never changes runtime state.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge_hub.papers.arxiv_source_tex_availability_audit import DEFAULT_PARSED_ROOT
from knowledge_hub.papers.equation_quote_candidate_audit import (
    EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
)
from knowledge_hub.papers.figure_caption_candidate_audit import (
    FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
)
from knowledge_hub.papers.sectionspan_candidate_audit import (
    SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
)
from knowledge_hub.papers.table_region_candidate_audit import TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID
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


PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-structured-evidence-readiness-audit.v1"
)

STATUS_READY = "promotion_review_ready_candidate_only"
STATUS_BLOCKED_MISSING_SOURCE_HASH = "blocked_missing_source_hash"
STATUS_BLOCKED_AMBIGUOUS_PDF_REGION = "blocked_ambiguous_pdf_region"
STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION = "blocked_missing_label_number_disambiguation"
STATUS_BLOCKED_PDF_REGION_ONLY = "blocked_pdf_region_only_not_source_span"
STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR = "blocked_requires_manual_or_later_extractor_review"

RECOMMENDED_ACTION_BY_STATUS = {
    STATUS_READY: "queue_for_explicit_source_span_promotion_executor_review",
    STATUS_BLOCKED_MISSING_SOURCE_HASH: "recover_source_content_hash_before_promotion",
    STATUS_BLOCKED_AMBIGUOUS_PDF_REGION: "resolve_ambiguous_pdf_region_before_promotion",
    STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION: "run_label_number_disambiguation_before_promotion",
    STATUS_BLOCKED_PDF_REGION_ONLY: "recover_original_source_span_or_offset_authority_before_promotion",
    STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR: "manual_or_later_extractor_review_required",
}

_ARTIFACT_INPUTS = {
    "sectionspan": {
        "schema": SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
        "schema_aliases": (
            "knowledge-hub.paper.tex-sectionspan-candidate-report.v1",
        ),
        "row_keys": ("candidates",),
        "artifact_type": "section",
    },
    "table_region": {
        "schema": TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID,
        "schema_aliases": (),
        "row_keys": ("candidates",),
        "artifact_type": "table",
    },
    "figure_caption": {
        "schema": FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
        "schema_aliases": (
            "knowledge-hub.paper.tex-figure-caption-candidate-report.v1",
        ),
        "row_keys": ("candidates",),
        "artifact_type": "figure",
    },
    "equation_quote": {
        "schema": EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
        "schema_aliases": (
            "knowledge-hub.paper.tex-equation-quote-candidate-report.v1",
        ),
        "row_keys": ("candidates",),
        "artifact_type": "equation",
    },
    "tex_rendered_macro_term_profile_design": {
        "schema": TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
        "row_keys": ("rows",),
        "artifact_type": "equation",
    },
    "tex_segmented_multiline_matching_design": {
        "schema": TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
        "row_keys": ("rows",),
        "artifact_type": "equation",
    },
    "tex_label_number_pdf_region_disambiguation_design": {
        "schema": TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        "row_keys": ("rows",),
        "artifact_type": "equation",
    },
    "tex_equation_pdf_region_anchor_audit": {
        "schema": TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
        "row_keys": ("rows",),
        "artifact_type": "equation",
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    if not payload_path.exists():
        return {}
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in list(value)[:4]:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _normalize_indexes(value: Any) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for item in list(value):
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _extract_rows(payload: dict[str, Any], row_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    for key in row_keys:
        rows = payload.get(key)
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, dict)]
    return []


def _manifest_paths(parsed_root: Path, paper_id: str) -> dict[str, bool]:
    paper_dir = parsed_root / paper_id
    return {
        "manifest": (paper_dir / "manifest.json").is_file(),
        "document_md": (paper_dir / "document.md").is_file(),
        "document_json": (paper_dir / "document.json").is_file(),
        "source_artifact_present": paper_dir.is_dir(),
    }


def _contains_text(haystack: Any, *needles: str) -> bool:
    text = _safe_text(haystack).lower()
    return any(needle in text for needle in needles)


def _contains_tokens(tokens: list[str], *needles: str) -> bool:
    lowered = " ".join(item.lower() for item in tokens)
    return any(needle in lowered for needle in needles)


def _extract_label_number_hint(row: dict[str, Any]) -> Any:
    if row.get("labelNumberHint") is not None:
        return row.get("labelNumberHint")
    if row.get("source_label_number_hint") is not None:
        return row.get("source_label_number_hint")
    equation_numbers = row.get("equation_numbers")
    if isinstance(equation_numbers, (list, tuple)):
        return {
            "equation_numbers": [str(item).strip() for item in equation_numbers if str(item).strip()],
        }
    return None


def _read_source_tex_row_id(row: dict[str, Any], artifact_type: str) -> str:
    preferred = [
        "source_tex_row_id",
        "source_row_id",
        "source_quote_row_id",
        "source_diagnostic_id",
        "source_design_id",
        "source_candidate_id",
        "candidate_id",
        "sourceCandidateId",
    ]
    if artifact_type == "equation":
        preferred.insert(0, "source_equation_row_id")
    for key in preferred:
        value = _safe_text(row.get(key))
        if value:
            return value
    return ""


def _classify_row(row: dict[str, Any], artifact_type: str) -> str:
    blockers = _normalize_string_list(row.get("strict_blockers"))
    if not blockers:
        blockers = _normalize_string_list(row.get("non_strict_reason"))

    source_hash = _safe_text(row.get("sourceContentHash"))
    if not source_hash:
        return STATUS_BLOCKED_MISSING_SOURCE_HASH

    status_hint = _safe_text(row.get("readiness_status") or row.get("readiness"))
    action_hint = _safe_text(row.get("recommended_action") or row.get("next_action"))

    if _contains_tokens(blockers, "ambiguous", "pdf_region_ambiguous", "ambiguous_pdf_region"):
        return STATUS_BLOCKED_AMBIGUOUS_PDF_REGION
    if _contains_text(
        status_hint,
        "ambiguous_pdf_region",
        "ambiguous",
    ) or _contains_text(action_hint, "ambiguous"):
        return STATUS_BLOCKED_AMBIGUOUS_PDF_REGION

    if _contains_tokens(blockers, "label_number", "equation_number", "missing_label", "missing_number"):
        return STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION
    if _contains_text(
        status_hint,
        "missing_label_number",
        "missing_equation_number",
        "label_number_disambiguation",
    ):
        return STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION

    chars_start = _safe_int(row.get("chars_start"))
    chars_end = _safe_int(row.get("chars_end"))
    page = _safe_int(row.get("page"))
    bbox = _normalize_bbox(row.get("bbox"))
    block_indexes = _normalize_indexes(
        row.get("blockIndexes")
        or row.get("block_indexes")
        or (row.get("selected_pdf_region") or {}).get("block_indexes")
    )

    if _contains_tokens(blockers, "manual", "later_extractor", "needs_manual"):
        return STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR
    if _contains_text(
        action_hint,
        "manual",
        "later_extractor",
        "later\nextractor",
        "manual_extractor",
    ):
        return STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR

    if artifact_type == "equation" and chars_start is None and chars_end is None:
        if page is not None or bbox or block_indexes:
            return STATUS_BLOCKED_PDF_REGION_ONLY

    if artifact_type != "equation" and chars_start is None and chars_end is None:
        if page is not None or bbox or block_indexes:
            return STATUS_BLOCKED_PDF_REGION_ONLY

    return STATUS_READY


def _as_row(
    *,
    source_row: dict[str, Any],
    layer_name: str,
    source_path: str,
    parsed_root: Path,
    by_parsed_artifact: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    layer = _ARTIFACT_INPUTS[layer_name]
    artifact_type = layer["artifact_type"]
    paper_id = _safe_text(source_row.get("paper_id"))
    source_content_hash = _safe_text(source_row.get("sourceContentHash"))

    selected_pdf_region = source_row.get("selected_pdf_region") or {}
    bbox = _normalize_bbox(
        source_row.get("bbox")
        or source_row.get("selected_bbox")
        or selected_pdf_region.get("bbox")
    )
    block_indexes = _normalize_indexes(
        source_row.get("blockIndexes")
        or source_row.get("block_indexes")
        or selected_pdf_region.get("block_indexes")
    )

    strict_blockers = _normalize_string_list(source_row.get("strict_blockers"))
    non_strict_reason = _normalize_string_list(source_row.get("non_strict_reason"))
    if not non_strict_reason:
        non_strict_reason = list(strict_blockers)

    source_candidate_id = _safe_text(
        source_row.get("source_candidate_id")
        or source_row.get("sourceCandidateId")
        or source_row.get("candidate_id")
        or source_row.get("row_id")
    )

    source_file = _safe_text(
        source_row.get("source_file")
        or source_row.get("sourceFile")
        or source_row.get("source_pdf_path")
        or source_row.get("sourcePdfPath")
    )

    source_span_created = _safe_bool(source_row.get("source_span_created"))
    strict_eligible = _safe_bool(source_row.get("strict_eligible"))
    citation_grade = _safe_bool(source_row.get("citation_grade"))
    runtime_evidence = _safe_bool(source_row.get("runtime_evidence"))
    parser_routing_changed = _safe_bool(source_row.get("parser_routing_changed"))
    answer_integration_changed = _safe_bool(source_row.get("answer_integration_changed"))

    row = {
        "artifact_type": artifact_type,
        "paper_id": paper_id,
        "source_candidate_id": source_candidate_id,
        "sourceContentHash": source_content_hash,
        "source_tex_row_id": _read_source_tex_row_id(source_row, artifact_type),
        "source_file": source_file,
        "page": _safe_int(source_row.get("page")) if _safe_int(source_row.get("page")) is not None else None,
        "bbox": bbox,
        "blockIndexes": block_indexes,
        "labelNumberHint": _extract_label_number_hint(source_row),
        "strict_blockers": strict_blockers,
        "non_strict_reason": non_strict_reason,
        "source_span_created": source_span_created,
        "strict_eligible": strict_eligible,
        "citation_grade": citation_grade,
        "runtime_evidence": runtime_evidence,
        "parser_routing_changed": parser_routing_changed,
        "answer_integration_changed": answer_integration_changed,
        "readiness_status": STATUS_READY,
        "recommended_action": "",
        "inputPath": source_path,
    }

    readiness = _classify_row(source_row, artifact_type)
    row["readiness_status"] = readiness
    row["recommended_action"] = RECOMMENDED_ACTION_BY_STATUS.get(readiness, "")

    if paper_id:
        presence = by_parsed_artifact.setdefault(paper_id, _manifest_paths(parsed_root, paper_id))
        row["parsedManifestPresent"] = bool(presence["manifest"])
        row["parsedDocumentPresent"] = bool(presence["document_md"])
    else:
        row["parsedManifestPresent"] = False
        row["parsedDocumentPresent"] = False

    return row


def _read_report_input(
    *,
    layer_name: str,
    path: str | Path | None,
    parsed_root: Path,
    all_input_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    schema_violations: list[str],
    warnings: list[str],
    by_parsed_artifact: dict[str, dict[str, bool]],
) -> None:
    config = _ARTIFACT_INPUTS[layer_name]
    if not path:
        warnings.append(f"{layer_name}_report_not_provided")
        return

    report_path = Path(str(path)).expanduser()
    payload = _read_json(report_path)
    if not payload:
        warnings.append(f"{layer_name}_report_unreadable_or_missing")
        return

    expected_schema = config["schema"]
    schema_aliases = tuple(config.get("schema_aliases", ()))
    accepted_schemas = {str(expected_schema), *[str(item) for item in schema_aliases]}
    if _safe_text(payload.get("schema")) not in accepted_schemas:
        warnings.append(f"{layer_name}_schema_missing_or_mismatch")
        schema_violations.append(f"{layer_name}_schema_mismatch")
        return

    rows = _extract_rows(payload, config["row_keys"])
    if not rows:
        warnings.append(f"{layer_name}_rows_missing")

    for row in rows:
        all_input_rows.append(row)
        target_rows.append(
            _as_row(
                source_row=row,
                layer_name=layer_name,
                source_path=str(report_path),
                parsed_root=parsed_root,
                by_parsed_artifact=by_parsed_artifact,
            )
        )


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    return {
        "inputRows": len(rows),
        "targetRows": len(rows),
        "promotionReviewReadyCandidateOnlyRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_READY
        ),
        "blockedMissingSourceHashRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_MISSING_SOURCE_HASH
        ),
        "blockedAmbiguousPdfRegionRows": sum(
            1
            for row in rows
            if row.get("readiness_status") == STATUS_BLOCKED_AMBIGUOUS_PDF_REGION
        ),
        "blockedMissingLabelNumberRows": sum(
            1
            for row in rows
            if row.get("readiness_status") == STATUS_BLOCKED_MISSING_LABEL_NUMBER_DISAMBIGUATION
        ),
        "blockedPdfRegionOnlyRows": sum(
            1 for row in rows if row.get("readiness_status") == STATUS_BLOCKED_PDF_REGION_ONLY
        ),
        "blockedManualOrLaterExtractorRows": sum(
            1
            for row in rows
            if row.get("readiness_status") == STATUS_BLOCKED_MANUAL_OR_LATER_EXTRACTOR
        ),
        "sourceSpanCreatedRows": sum(1 for row in rows if _safe_bool(row.get("source_span_created"))),
        "strictEligibleRows": sum(1 for row in rows if _safe_bool(row.get("strict_eligible"))),
        "citationGradeRows": sum(1 for row in rows if _safe_bool(row.get("citation_grade"))),
        "runtimeEvidenceRows": sum(1 for row in rows if _safe_bool(row.get("runtime_evidence"))),
        "parserRoutingChangedRows": sum(1 for row in rows if _safe_bool(row.get("parser_routing_changed"))),
        "answerIntegrationChangedRows": sum(
            1 for row in rows if _safe_bool(row.get("answer_integration_changed"))
        ),
        "schemaViolationCount": len(schema_violations),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in rows)),
        "byReadinessStatus": dict(Counter(str(row.get("readiness_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def _build_counts_by_artifact(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    parsed_doc_present = Counter(row.get("parsedDocumentPresent") for row in rows)
    parsed_manifest_present = Counter(row.get("parsedManifestPresent") for row in rows)
    return {
        "parsedDocumentRows": int(parsed_doc_present.get(True, 0)),
        "missingParsedDocumentRows": int(parsed_doc_present.get(False, 0)),
        "parsedManifestRows": int(parsed_manifest_present.get(True, 0)),
        "missingParsedManifestRows": int(parsed_manifest_present.get(False, 0)),
    }


def build_parsed_artifact_structured_evidence_readiness_audit(
    *,
    sectionspan_candidate_report: str | Path | None = None,
    table_region_candidate_report: str | Path | None = None,
    figure_caption_candidate_report: str | Path | None = None,
    equation_quote_candidate_report: str | Path | None = None,
    tex_rendered_macro_term_profile_design_report: str | Path | None = None,
    tex_segmented_multiline_matching_design_report: str | Path | None = None,
    tex_label_number_pdf_region_disambiguation_design_report: str | Path | None = None,
    tex_equation_pdf_region_anchor_audit_report: str | Path | None = None,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    parsed_root_path = Path(str(parsed_root)).expanduser()
    requested = {str(item).strip() for item in (paper_ids or []) if str(item).strip()}

    schema_violations: list[str] = []
    warnings: list[str] = []
    all_input_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    by_parsed_artifact: dict[str, dict[str, bool]] = {}

    _read_report_input(
        layer_name="sectionspan",
        path=sectionspan_candidate_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="table_region",
        path=table_region_candidate_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="figure_caption",
        path=figure_caption_candidate_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="equation_quote",
        path=equation_quote_candidate_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="tex_rendered_macro_term_profile_design",
        path=tex_rendered_macro_term_profile_design_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="tex_segmented_multiline_matching_design",
        path=tex_segmented_multiline_matching_design_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="tex_label_number_pdf_region_disambiguation_design",
        path=tex_label_number_pdf_region_disambiguation_design_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )
    _read_report_input(
        layer_name="tex_equation_pdf_region_anchor_audit",
        path=tex_equation_pdf_region_anchor_audit_report,
        parsed_root=parsed_root_path,
        all_input_rows=all_input_rows,
        target_rows=normalized_rows,
        schema_violations=schema_violations,
        warnings=warnings,
        by_parsed_artifact=by_parsed_artifact,
    )

    if requested:
        filtered_rows: list[dict[str, Any]] = [
            row for row in normalized_rows if (not row.get("paper_id") or str(row.get("paper_id") or "") in requested)
        ]
        normalized_rows = filtered_rows

    if not all_input_rows:
        warnings.append("no_input_rows_found")

    status = "ok" if (normalized_rows and not schema_violations) else "blocked"
    if status != "ok":
        warnings.append("report_not_ready_for_promotion_audit")

    counts = _counts(normalized_rows, schema_violations)
    parsed_paper_coverage = _build_counts_by_artifact(normalized_rows)
    parsed_papers = len(by_parsed_artifact)
    parsed_missing = sum(
        1
        for presence in by_parsed_artifact.values()
        if not (presence["source_artifact_present"] and presence["manifest"])
    )

    return {
        "schema": PARSED_ARTIFACT_STRUCTURED_EVIDENCE_READINESS_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "sectionspanCandidateReport": str(Path(sectionspan_candidate_report).expanduser()) if sectionspan_candidate_report is not None else "",
            "tableRegionCandidateReport": str(Path(table_region_candidate_report).expanduser()) if table_region_candidate_report is not None else "",
            "figureCaptionCandidateReport": str(Path(figure_caption_candidate_report).expanduser()) if figure_caption_candidate_report is not None else "",
            "equationQuoteCandidateReport": str(Path(equation_quote_candidate_report).expanduser()) if equation_quote_candidate_report is not None else "",
            "texRenderedMacroTermProfileDesignReport": str(
                Path(tex_rendered_macro_term_profile_design_report).expanduser()
            )
            if tex_rendered_macro_term_profile_design_report is not None
            else "",
            "texSegmentedMultilineMatchingDesignReport": str(
                Path(tex_segmented_multiline_matching_design_report).expanduser()
            )
            if tex_segmented_multiline_matching_design_report is not None
            else "",
            "texLabelNumberPdfRegionDisambiguationDesignReport": str(
                Path(tex_label_number_pdf_region_disambiguation_design_report).expanduser()
            )
            if tex_label_number_pdf_region_disambiguation_design_report is not None
            else "",
            "texEquationPdfRegionAnchorAuditReport": str(
                Path(tex_equation_pdf_region_anchor_audit_report).expanduser()
            )
            if tex_equation_pdf_region_anchor_audit_report is not None
            else "",
            "parsedRoot": str(parsed_root_path),
            "requestedPaperIds": sorted(requested),
            "sectionspanSchema": SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
            "tableRegionSchema": TABLE_REGION_CANDIDATE_REPORT_SCHEMA_ID,
            "figureCaptionSchema": FIGURE_CAPTION_CANDIDATE_REPORT_SCHEMA_ID,
            "equationQuoteSchema": EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
            "texRenderedMacroTermProfileDesignSchema": TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
            "texSegmentedMultilineMatchingDesignSchema": TEX_EQUATION_SEGMENTED_MULTILINE_MATCHING_DESIGN_SCHEMA_ID,
            "texLabelNumberPdfRegionDisambiguationDesignSchema": TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
            "texPdfRegionAnchorSchema": TEX_EQUATION_PDF_REGION_ANCHOR_AUDIT_SCHEMA_ID,
        },
        "counts": {
            **counts,
            "inputPaperCount": parsed_papers,
            "papersMissingParsedArtifacts": parsed_missing,
            "parsedPaperCoverage": parsed_paper_coverage,
            "inputRows": len(all_input_rows),
            "targetRows": len(normalized_rows),
        },
        "gate": {
            "readyForPromotionReadinessReview": status == "ok" and not schema_violations,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "schemaViolations": schema_violations,
            "decision": "parsed_artifact_structured_evidence_readiness_audit_ready" if status == "ok" else "blocked",
            "recommendedNextTranche": "parsed_artifact_structured_evidence_readiness_execution",
        },
        "policy": {
            "reportOnly": True,
            "designOnly": True,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
        },
        "warnings": warnings,
        "rows": normalized_rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "status": report["status"],
        "generatedAt": report["generatedAt"],
        "input": report["input"],
        "counts": report["counts"],
        "gate": report["gate"],
        "policy": report["policy"],
        "warnings": report["warnings"],
        "rows": report["rows"],
    }


def render_parsed_artifact_structured_evidence_readiness_audit_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    rows = list(report.get("rows", []))
    return "\n".join(
        [
            "# Parsed Artifact Structured Evidence Readiness Audit",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- design-only: {json.dumps(report.get('policy', {}).get('designOnly'))}",
            f"- input rows: {int(counts.get('inputRows') or 0)}",
            f"- target rows: {int(counts.get('targetRows') or 0)}",
            f"- ready rows: {int(counts.get('promotionReviewReadyCandidateOnlyRows') or 0)}",
            f"- blocked missing source hash rows: {int(counts.get('blockedMissingSourceHashRows') or 0)}",
            f"- blocked ambiguous PDF-region rows: {int(counts.get('blockedAmbiguousPdfRegionRows') or 0)}",
            f"- blocked missing label/number rows: {int(counts.get('blockedMissingLabelNumberRows') or 0)}",
            f"- blocked PDF-region-only rows: {int(counts.get('blockedPdfRegionOnlyRows') or 0)}",
            f"- blocked manual/later rows: {int(counts.get('blockedManualOrLaterExtractorRows') or 0)}",
            "",
            f"- schema violations: {int(counts.get('schemaViolationCount') or 0)}",
            "",
            "## Rows",
            *(f"- paper={row.get('paper_id','')} artifact={row.get('artifact_type','')} status={row.get('readiness_status','')}"
              for row in rows),
        ]
    )


def write_parsed_artifact_structured_evidence_readiness_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-structured-evidence-readiness-audit.json"
    summary_path = root / "parsed-artifact-structured-evidence-readiness-audit-summary.json"
    markdown_path = root / "parsed-artifact-structured-evidence-readiness-audit.md"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = _summary_payload(report)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown_path.write_text(render_parsed_artifact_structured_evidence_readiness_audit_markdown(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Generate a report-only parsed-artifact structured-evidence readiness audit."
        )
    )
    parser.add_argument("--sectionspan-candidate-report", default="")
    parser.add_argument("--table-region-candidate-report", default="")
    parser.add_argument("--figure-caption-candidate-report", default="")
    parser.add_argument("--equation-quote-candidate-report", default="")
    parser.add_argument("--tex-rendered-macro-term-profile-design-report", default="")
    parser.add_argument("--tex-segmented-multiline-matching-design-report", default="")
    parser.add_argument("--tex-label-number-pdf-region-disambiguation-design-report", default="")
    parser.add_argument("--tex-equation-pdf-region-anchor-audit-report", default="")
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to paper id; repeatable")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_structured_evidence_readiness_audit(
        sectionspan_candidate_report=args.sectionspan_candidate_report or None,
        table_region_candidate_report=args.table_region_candidate_report or None,
        figure_caption_candidate_report=args.figure_caption_candidate_report or None,
        equation_quote_candidate_report=args.equation_quote_candidate_report or None,
        tex_rendered_macro_term_profile_design_report=args.tex_rendered_macro_term_profile_design_report or None,
        tex_segmented_multiline_matching_design_report=args.tex_segmented_multiline_matching_design_report or None,
        tex_label_number_pdf_region_disambiguation_design_report=args.tex_label_number_pdf_region_disambiguation_design_report or None,
        tex_equation_pdf_region_anchor_audit_report=args.tex_equation_pdf_region_anchor_audit_report or None,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id,
    )

    if args.output_dir:
        paths = write_parsed_artifact_structured_evidence_readiness_audit_reports(report, args.output_dir)
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")
    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
