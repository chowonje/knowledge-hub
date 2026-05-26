"""Report-only label/number PDF-region disambiguation for TeX equations.

This helper consumes rendered macro term-profile design rows that are still
ambiguous across canonical/PDF candidates. It uses TeX source row order and
labels as a non-authoritative equation-number hint, then filters recomputed
PyMuPDF PDF-region candidates by visible equation numbers. It records design
signals only and does not create source spans, strict evidence, parser routing,
answer integration, DB/index writes, vault scans, or canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from knowledge_hub.papers.arxiv_source_tex_availability_audit import (
    ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_equation_pdf_region_anchor_audit import (
    _extract_pdf_blocks,
    _pdf_region_candidates,
    _safe_float,
    _safe_int,
    _select_region,
    _source_context,
)
from knowledge_hub.papers.tex_equation_rendered_macro_term_profile_design import (
    TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
    _clean_text,
    _latex_labels,
)
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import DEFAULT_PARSED_ROOT


TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-label-number-pdf-region-disambiguation-design.v1"
)

DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-rendered-macro-term-profile-design-10paper"
    / "01-tex-equation-rendered-macro-term-profile-design"
    / "tex-equation-rendered-macro-term-profile-design-report.json"
)

DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-v2-10paper-pilot"
    / "01-arxiv-source-tex-availability"
    / "arxiv-source-tex-availability-report.json"
)

_TAG_RE = re.compile(r"\\tag\s*\{([^{}]+)\}")
_NUMBERED_SINGLE_ENVIRONMENTS = {
    "equation",
    "gather",
    "multline",
}


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


def _explicit_tex_number_hints(candidate_text: str) -> list[str]:
    return list(dict.fromkeys(_clean_text(item) for item in _TAG_RE.findall(str(candidate_text or "")) if _clean_text(item)))


def _source_rows(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in list(source_report.get("structureRows") or []) if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            str(row.get("paper_id") or ""),
            str(row.get("source_file") or ""),
            _safe_int(row.get("tex_chars_start")),
            str(row.get("structure_row_id") or ""),
        )
    )
    return rows


def _numbered_environment_number(env: str, counters: defaultdict[tuple[str, str], int], key: tuple[str, str]) -> str:
    clean_env = str(env or "").strip()
    if clean_env.endswith("*"):
        return ""
    if clean_env in _NUMBERED_SINGLE_ENVIRONMENTS:
        counters[key] += 1
        return str(counters[key])
    return ""


def _source_hint_index(source_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    hints: dict[str, dict[str, Any]] = {}
    for row in _source_rows(source_report):
        row_id = str(row.get("structure_row_id") or "")
        if not row_id:
            continue
        paper_id = str(row.get("paper_id") or "")
        source_file = str(row.get("source_file") or "")
        env = str(row.get("tex_environment") or "")
        candidate_text = str(row.get("candidate_text") or "")
        explicit_numbers = _explicit_tex_number_hints(candidate_text)
        inferred_number = _numbered_environment_number(env, counters, (paper_id, source_file))
        labels = _latex_labels(candidate_text)
        numbers = explicit_numbers or ([inferred_number] if inferred_number else [])
        if explicit_numbers:
            status = "explicit_tex_tag_number_hint"
        elif inferred_number and labels:
            status = "inferred_label_number_hint"
        elif inferred_number:
            status = "inferred_source_order_number_hint"
        else:
            status = "no_supported_label_number_hint"
        hints[row_id] = {
            "sourceStructureRowId": row_id,
            "paperId": paper_id,
            "sourceFile": source_file,
            "texEnvironment": env,
            "latexLabels": labels,
            "inferredEquationNumbers": numbers,
            "method": "source_order_equation_environment_counter_v1",
            "status": status,
        }
    return hints


def _best_profile(row: dict[str, Any]) -> dict[str, Any]:
    preferred = str(row.get("recommended_profile") or "")
    profiles = [dict(profile) for profile in list(row.get("profile_results") or []) if isinstance(profile, dict)]
    for profile in profiles:
        if str(profile.get("profile_name") or "") == preferred:
            return profile
    return profiles[0] if profiles else {}


def _candidate_payload(candidate: dict[str, Any], expected_numbers: set[str]) -> dict[str, Any]:
    equation_numbers = [str(item) for item in list(candidate.get("equation_numbers") or [])]
    return {
        "rank": _safe_int(candidate.get("rank")),
        "page": _safe_int(candidate.get("page")),
        "bbox": [float(item) for item in list(candidate.get("bbox") or [])],
        "blockIndexes": [_safe_int(item) for item in list(candidate.get("block_indexes") or [])],
        "matchedTerms": [str(item) for item in list(candidate.get("matched_terms") or [])],
        "coverage": _safe_float(candidate.get("coverage")),
        "formulaScore": _safe_float(candidate.get("formula_score")),
        "equationNumbers": equation_numbers,
        "labelNumberMatch": bool(expected_numbers and expected_numbers.intersection(equation_numbers)),
        "textPreview": str(candidate.get("text_preview") or ""),
    }


def _selected_region_payload(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {
            "page": None,
            "bbox": [],
            "blockIndexes": [],
            "matchedTerms": [],
            "coverage": 0.0,
            "formulaScore": 0.0,
            "equationNumbers": [],
            "textPreview": "",
        }
    return {
        "page": _safe_int(candidate.get("page")),
        "bbox": [float(item) for item in list(candidate.get("bbox") or [])],
        "blockIndexes": [_safe_int(item) for item in list(candidate.get("block_indexes") or [])],
        "matchedTerms": [str(item) for item in list(candidate.get("matched_terms") or [])],
        "coverage": _safe_float(candidate.get("coverage")),
        "formulaScore": _safe_float(candidate.get("formula_score")),
        "equationNumbers": [str(item) for item in list(candidate.get("equation_numbers") or [])],
        "textPreview": str(candidate.get("text_preview") or ""),
    }


def _status(
    *,
    source_context_status: str,
    terms: list[str],
    expected_numbers: set[str],
    pdf_candidate_count: int,
    matching_candidate_count: int,
    selected_unique: bool,
) -> str:
    if source_context_status != "ok":
        return source_context_status
    if len(terms) < 2:
        return "insufficient_rendered_macro_terms"
    if not expected_numbers:
        return "blocked_no_label_number_hint"
    if pdf_candidate_count <= 0:
        return "no_pdf_region_candidates_for_label_number_filter"
    if matching_candidate_count == 1 or (matching_candidate_count > 1 and selected_unique):
        return "unique_label_number_pdf_region_candidate_only"
    if matching_candidate_count > 1:
        return "ambiguous_label_number_pdf_region_candidate_only"
    return "no_label_number_matching_pdf_region_candidate"


def _recommended_action(status: str) -> str:
    if status == "unique_label_number_pdf_region_candidate_only":
        return "review_label_number_disambiguated_candidate_before_source_span_promotion_audit"
    if status == "ambiguous_label_number_pdf_region_candidate_only":
        return "requires_stronger_tex_pdf_sync_or_manual_disambiguation_design"
    if status == "no_label_number_matching_pdf_region_candidate":
        return "keep_blocked_label_number_conflict_or_pdf_parser_alignment_review"
    if status == "blocked_no_label_number_hint":
        return "requires_source_label_number_inference_input"
    return "keep_blocked_pending_alternative_extractor_or_manual_review"


def _row(
    index: int,
    source_row: dict[str, Any],
    *,
    source_hint: dict[str, Any],
    parsed_root: Path,
    source_context_cache: dict[str, dict[str, Any]],
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]],
) -> dict[str, Any]:
    paper_id = str(source_row.get("paper_id") or "")
    if paper_id not in source_context_cache:
        source_context_cache[paper_id] = _source_context(
            paper_id=paper_id,
            parsed_root=parsed_root,
            pdf_block_loader=pdf_block_loader,
        )
    source_context = source_context_cache[paper_id]
    best_profile = _best_profile(source_row)
    terms = [str(item) for item in list(best_profile.get("normalized_terms") or []) if str(item).strip()]
    expected_numbers = set(str(item) for item in list(source_hint.get("inferredEquationNumbers") or []) if str(item))
    pdf_source_row = dict(source_row)
    pdf_source_row["normalized_terms"] = terms
    pdf_source_row["window_details"] = []
    candidates = (
        _pdf_region_candidates(pdf_source_row, source_context)
        if str(source_context.get("status") or "blocked") == "ok" and len(terms) >= 2
        else []
    )
    matching_candidates = [
        candidate
        for candidate in candidates
        if expected_numbers.intersection(str(item) for item in list(candidate.get("equation_numbers") or []))
    ]
    selected, selected_unique = _select_region(matching_candidates)
    source_context_status = str(source_context.get("status") or "blocked")
    status = _status(
        source_context_status=source_context_status,
        terms=terms,
        expected_numbers=expected_numbers,
        pdf_candidate_count=len(candidates),
        matching_candidate_count=len(matching_candidates),
        selected_unique=bool(selected_unique),
    )
    ready = status == "unique_label_number_pdf_region_candidate_only" and bool(selected_unique)
    blockers = list(
        dict.fromkeys(
            [
                "label_number_pdf_region_disambiguation_design_only",
                "tex_source_order_number_hint_is_not_authoritative",
                "pdf_region_bbox_is_not_source_span",
                "equation_number_match_is_not_runtime_evidence",
                "equation_semantics_not_interpreted",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(source_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "design_id": f"tex-equation-label-number-pdf-region-disambiguation-design:{index:04d}",
        "candidate_type": "tex_equation_label_number_pdf_region_disambiguation_design",
        "source_rendered_macro_design_id": str(source_row.get("design_id") or ""),
        "source_diagnostic_id": str(source_row.get("source_diagnostic_id") or ""),
        "source_pdf_region_anchor_id": str(source_row.get("source_pdf_region_anchor_id") or ""),
        "source_line_local_anchor_id": str(source_row.get("source_line_local_anchor_id") or ""),
        "source_design_id": str(source_row.get("source_design_id") or ""),
        "source_candidate_id": str(source_row.get("source_candidate_id") or ""),
        "paper_id": paper_id,
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": str(source_row.get("source_file") or ""),
        "equation_environment": str(source_row.get("equation_environment") or ""),
        "candidate_text": _clean_text(source_row.get("candidate_text")),
        "latex_labels": [str(item) for item in list(source_row.get("latex_labels") or [])],
        "recommended_profile": str(best_profile.get("profile_name") or ""),
        "profile_normalized_terms": terms,
        "profile_normalized_term_count": len(terms),
        "source_context_status": source_context_status,
        "sourceContentHash": str(source_context.get("sourceContentHash") or ""),
        "source_pdf_path": str(source_context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(source_context.get("manifestPath") or ""),
        "source_label_number_hint": {
            "sourceStructureRowId": str(source_hint.get("sourceStructureRowId") or ""),
            "texEnvironment": str(source_hint.get("texEnvironment") or ""),
            "latexLabels": [str(item) for item in list(source_hint.get("latexLabels") or [])],
            "inferredEquationNumbers": [str(item) for item in list(source_hint.get("inferredEquationNumbers") or [])],
            "method": str(source_hint.get("method") or ""),
            "status": str(source_hint.get("status") or "missing_source_hint"),
        },
        "pdf_region_candidates": [_candidate_payload(candidate, expected_numbers) for candidate in candidates],
        "pdf_region_candidate_count": len(candidates),
        "label_number_matching_candidate_count": len(matching_candidates),
        "selected_pdf_region": _selected_region_payload(selected if ready else None),
        "disambiguation_status": status,
        "candidate_ready": ready,
        "evidence_tier": "label_number_pdf_region_disambiguation_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "recommended_action": _recommended_action(status),
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(
    *,
    input_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    return {
        "inputRows": len(input_rows),
        "targetRows": len(rows),
        "sourceLabelNumberHintRows": sum(
            1
            for row in rows
            if list(dict(row.get("source_label_number_hint") or {}).get("inferredEquationNumbers") or [])
        ),
        "candidateReadyRows": sum(1 for row in rows if bool(row.get("candidate_ready"))),
        "uniqueDisambiguatedRows": sum(
            1
            for row in rows
            if str(row.get("disambiguation_status") or "") == "unique_label_number_pdf_region_candidate_only"
        ),
        "ambiguousDisambiguatedRows": sum(
            1
            for row in rows
            if str(row.get("disambiguation_status") or "") == "ambiguous_label_number_pdf_region_candidate_only"
        ),
        "noMatchingPdfRegionRows": sum(
            1
            for row in rows
            if str(row.get("disambiguation_status") or "") == "no_label_number_matching_pdf_region_candidate"
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
        "byDisambiguationStatus": dict(Counter(str(row.get("disambiguation_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
    }


def build_tex_equation_label_number_pdf_region_disambiguation_design(
    rendered_macro_term_profile_design_report: str | Path = DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT,
    *,
    arxiv_source_tex_availability_report: str | Path = DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
    pdf_block_loader: Callable[[str | Path], list[dict[str, Any]]] = _extract_pdf_blocks,
) -> dict[str, Any]:
    input_path = Path(str(rendered_macro_term_profile_design_report)).expanduser()
    source_report_path = Path(str(arxiv_source_tex_availability_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    rendered_payload = _read_json(input_path)
    source_payload = _read_json(source_report_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    parent_schema = str(rendered_payload.get("schema") or "")
    source_schema = str(source_payload.get("schema") or "")
    schema_violations: list[str] = []
    if parent_schema != TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID:
        schema_violations.append("tex_equation_rendered_macro_term_profile_design_schema_mismatch")
    if source_schema != ARXIV_SOURCE_TEX_AVAILABILITY_AUDIT_SCHEMA_ID:
        schema_violations.append("arxiv_source_tex_availability_audit_schema_mismatch")
    input_rows = [
        dict(row)
        for row in list(rendered_payload.get("rows") or [])
        if not schema_violations
        and isinstance(row, dict)
        and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    target_rows = [
        row
        for row in input_rows
        if str(row.get("recommended_action") or "")
        == "requires_equation_number_or_label_to_pdf_region_disambiguation_design"
    ]
    source_hints = _source_hint_index(source_payload) if not schema_violations else {}
    source_context_cache: dict[str, dict[str, Any]] = {}
    rows = [
        _row(
            index + 1,
            row,
            source_hint=source_hints.get(str(row.get("source_candidate_id") or ""), {}),
            parsed_root=parsed_root_path,
            source_context_cache=source_context_cache,
            pdf_block_loader=pdf_block_loader,
        )
        for index, row in enumerate(target_rows)
    ]
    counts = _counts(input_rows=input_rows, rows=rows, schema_violations=schema_violations)
    return {
        "schema": TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "renderedMacroTermProfileDesignReportPath": str(input_path),
            "renderedMacroTermProfileDesignReportSchema": parent_schema,
            "arxivSourceTexAvailabilityReportPath": str(source_report_path),
            "arxivSourceTexAvailabilityReportSchema": source_schema,
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
            "decision": "label_number_pdf_region_disambiguation_design_ready"
            if rows and not schema_violations
            else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "source_span_promotion_readiness_audit",
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
            "label_number_disambiguation_matches_are_not_source_spans",
            "source_order_equation_numbers_are_design_hints_only",
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


def render_tex_equation_label_number_pdf_region_disambiguation_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX Equation Label/Number PDF-Region Disambiguation Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Source label/number hint rows: `{int(counts.get('sourceLabelNumberHintRows') or 0)}`",
        f"- Candidate-ready rows: `{int(counts.get('candidateReadyRows') or 0)}`",
        f"- Unique disambiguated rows: `{int(counts.get('uniqueDisambiguatedRows') or 0)}`",
        f"- Ambiguous disambiguated rows: `{int(counts.get('ambiguousDisambiguatedRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This report evaluates TeX label/number PDF-region disambiguation only. It does not create evidence or connect to runtime answering.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Disambiguation status: `{json.dumps(counts.get('byDisambiguationStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Recommended action: `{json.dumps(counts.get('byRecommendedAction') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        hint = dict(row.get("source_label_number_hint") or {})
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('equation_environment')}` "
            f"`{','.join(hint.get('inferredEquationNumbers') or [])}` "
            f"`{row.get('disambiguation_status')}`"
        )
    return "\n".join(lines)


def write_tex_equation_label_number_pdf_region_disambiguation_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-label-number-pdf-region-disambiguation-design-report.json"
    summary_path = root / "tex-equation-label-number-pdf-region-disambiguation-design-summary.json"
    markdown_path = root / "tex-equation-label-number-pdf-region-disambiguation-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_tex_equation_label_number_pdf_region_disambiguation_design_markdown(report),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX equation label/number PDF-region disambiguation design.")
    parser.add_argument("--rendered-macro-term-profile-design-report", default=str(DEFAULT_TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_REPORT))
    parser.add_argument("--arxiv-source-tex-availability-report", default=str(DEFAULT_ARXIV_SOURCE_TEX_AVAILABILITY_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT), help="Parsed paper artifact root.")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_label_number_pdf_region_disambiguation_design(
        rendered_macro_term_profile_design_report=args.rendered_macro_term_profile_design_report,
        arxiv_source_tex_availability_report=args.arxiv_source_tex_availability_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_label_number_pdf_region_disambiguation_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_LABEL_NUMBER_PDF_REGION_DISAMBIGUATION_DESIGN_SCHEMA_ID",
    "build_tex_equation_label_number_pdf_region_disambiguation_design",
    "render_tex_equation_label_number_pdf_region_disambiguation_design_markdown",
    "write_tex_equation_label_number_pdf_region_disambiguation_design_reports",
]
