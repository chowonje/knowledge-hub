"""Report-only rendered macro term-profile design for TeX equations.

This helper consumes remaining-window diagnostics and evaluates whether rows
blocked by rendered/custom LaTeX macros become matchable after expanding common
math macros into their rendered token aliases. It records canonical
generated-Markdown and PDF-region candidate signals only. It does not create
source spans, strict evidence, runtime evidence, parser routing, answer
integration, or canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

from knowledge_hub.papers.tex_equation_canonical_text_normalizer_design import (
    _canonical_tokens,
    _ordered_windows,
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


TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-rendered-macro-term-profile-design.v1"
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
_LABEL_RE = re.compile(r"\\label\s*\{([^{}]*)\}")
_GROUP_MACRO_RE = re.compile(r"\\([A-Za-z]+)\*?\s*\{([^{}]*)\}")
_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")
_SUBSCRIPT_BRACE_RE = re.compile(r"([A-Za-z])_\{([A-Za-z0-9]+)\}")
_SUBSCRIPT_PLAIN_RE = re.compile(r"([A-Za-z])_([A-Za-z0-9]+)")
_TOKEN_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|[0-9]+(?:\.[0-9]+)?")
_STOP_TERMS = {
    "begin",
    "end",
    "eq",
    "equation",
    "label",
}
_RENDERED_ALIAS_MACROS = {
    "bm",
    "boldsymbol",
    "mathbb",
    "mathbf",
    "mathcal",
    "mathit",
    "mathrm",
    "operatorname",
    "text",
    "textbf",
    "textrm",
    "ve",
    "vec",
}


@dataclass(frozen=True)
class _Profile:
    name: str
    description: str
    proposed_rules: tuple[str, ...]
    source: str


_PROFILES = (
    _Profile(
        name="diagnostic_normalized_terms_v1",
        description="Use the remaining-window diagnostic's current normalized terms as a baseline.",
        proposed_rules=(
            "preserve_parent_diagnostic_terms",
            "no_rendered_macro_alias_expansion",
        ),
        source="diagnostic",
    ),
    _Profile(
        name="rendered_macro_alias_terms_v1",
        description="Expand common rendered math macros into the visible tokens expected in generated Markdown/PDF text.",
        proposed_rules=(
            "latex_label_elision",
            "rendered_vector_macro_argument_alias",
            "rendered_mathcal_argument_alias",
            "subscript_superscript_alnum_compaction",
            "single_symbol_math_token_retention",
            "ordered_anchor_token_window",
        ),
        source="rendered_alias",
    ),
)


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


def _latex_labels(text: str) -> list[str]:
    return list(dict.fromkeys(_LABEL_RE.findall(str(text or ""))))


def _replace_group_macro(match: re.Match[str]) -> str:
    macro = match.group(1)
    value = match.group(2)
    if macro in _RENDERED_ALIAS_MACROS:
        return f" {value} "
    return " "


def _rendered_alias_text(text: str) -> str:
    value = _LABEL_RE.sub(" ", str(text or ""))
    value = value.replace(r"\{", " ").replace(r"\}", " ")
    previous = ""
    while previous != value:
        previous = value
        value = _GROUP_MACRO_RE.sub(_replace_group_macro, value)
    value = _SUBSCRIPT_BRACE_RE.sub(r"\1\2", value)
    value = _SUBSCRIPT_PLAIN_RE.sub(r"\1\2", value)
    value = _COMMAND_RE.sub(" ", value)
    value = value.replace("\\\\", " ")
    return _clean_text(value)


def _diagnostic_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in list(row.get("normalized_terms") or []):
        term = _clean_text(item)
        normalized = re.sub(r"[^A-Za-z0-9]", "", term).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
    return terms


def _rendered_alias_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(_rendered_alias_text(text)):
        compact = re.sub(r"[^A-Za-z0-9]", "", token)
        normalized = compact.casefold()
        if not normalized or normalized in _STOP_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(compact)
    return terms[:32]


def _profile_terms(profile: _Profile, row: dict[str, Any]) -> list[str]:
    if profile.source == "diagnostic":
        return _diagnostic_terms(row)
    return _rendered_alias_terms(str(row.get("candidate_text") or ""))


def _canonical_status(*, document_text: str, terms: list[str]) -> tuple[str, int]:
    if not document_text:
        return "blocked_missing_canonical_document", 0
    if len(terms) < 2:
        return "insufficient_rendered_macro_terms", 0
    windows = _ordered_windows(terms, _canonical_tokens(document_text), max_gap_tokens=20)
    if len(windows) == 1:
        return "unique_rendered_macro_canonical_window_candidate_only", 1
    if len(windows) > 1:
        return "ambiguous_rendered_macro_canonical_window_candidate_only", len(windows)
    return "no_rendered_macro_canonical_window", 0


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
        return "insufficient_rendered_macro_terms", 0, {}
    profile_row = dict(row)
    profile_row["normalized_terms"] = terms
    profile_row["window_details"] = []
    candidates = _pdf_region_candidates(profile_row, source_context)
    selected, unique = _select_region(candidates)
    if selected and unique:
        return "unique_rendered_macro_pdf_region_candidate_only", len(candidates), selected
    if candidates:
        return "ambiguous_rendered_macro_pdf_region_candidate_only", len(candidates), selected or {}
    return "no_rendered_macro_pdf_region_candidate", 0, {}


def _profile_status(canonical_status: str, pdf_status: str) -> str:
    if canonical_status.startswith("unique_") or pdf_status.startswith("unique_"):
        return "unique_rendered_macro_profile_candidate_only"
    if canonical_status.startswith("ambiguous_") or pdf_status.startswith("ambiguous_"):
        return "ambiguous_rendered_macro_profile_candidate_only"
    if canonical_status == "insufficient_rendered_macro_terms" or pdf_status == "insufficient_rendered_macro_terms":
        return "insufficient_rendered_macro_terms"
    return "no_rendered_macro_profile_candidate"


def _profile_result(
    profile: _Profile,
    *,
    row: dict[str, Any],
    document_text: str,
    source_context: dict[str, Any],
) -> dict[str, Any]:
    terms = _profile_terms(profile, row)
    canonical_status, canonical_window_count = _canonical_status(document_text=document_text, terms=terms)
    pdf_status, pdf_candidate_count, selected_pdf = _pdf_status(
        row=row,
        terms=terms,
        source_context=source_context,
    )
    status = _profile_status(canonical_status, pdf_status)
    return {
        "profile_name": profile.name,
        "description": profile.description,
        "proposed_rules": list(profile.proposed_rules),
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
        "profile_status": status,
        "profile_candidate_ready": status == "unique_rendered_macro_profile_candidate_only",
    }


def _best_profile(profile_results: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {
        "unique_rendered_macro_profile_candidate_only": 0,
        "ambiguous_rendered_macro_profile_candidate_only": 1,
        "no_rendered_macro_profile_candidate": 2,
        "insufficient_rendered_macro_terms": 3,
    }
    return min(
        profile_results,
        key=lambda result: (
            priority.get(str(result.get("profile_status") or ""), 99),
            0 if str(result.get("profile_name") or "") == "rendered_macro_alias_terms_v1" else 1,
            -int(result.get("normalized_term_count") or 0),
        ),
    )


def _recommended_action(best: dict[str, Any]) -> str:
    status = str(best.get("profile_status") or "")
    canonical_status = str(best.get("canonical_match_status") or "")
    pdf_status = str(best.get("pdf_region_match_status") or "")
    if status == "unique_rendered_macro_profile_candidate_only":
        return "review_rendered_macro_unique_candidate_before_any_later_promotion_design"
    if status == "ambiguous_rendered_macro_profile_candidate_only":
        if canonical_status.startswith("ambiguous_") and pdf_status.startswith("ambiguous_"):
            return "requires_equation_number_or_label_to_pdf_region_disambiguation_design"
        return "requires_additional_rendered_macro_disambiguation_design"
    return "keep_blocked_pending_alternative_extractor_or_manual_review"


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
    profile_results = [
        _profile_result(
            profile,
            row=source_row,
            document_text=document_cache[document_key],
            source_context=source_context,
        )
        for profile in _PROFILES
    ]
    best = _best_profile(profile_results)
    ready = bool(best.get("profile_candidate_ready"))
    blockers = list(
        dict.fromkeys(
            [
                "rendered_macro_term_profile_design_only",
                "rendered_macro_matches_do_not_create_source_spans",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "pdf_region_bbox_is_not_source_span",
                "equation_label_to_number_mapping_not_authoritative",
                "equation_semantics_not_interpreted",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(source_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "design_id": f"tex-equation-rendered-macro-term-profile-design:{index:04d}",
        "candidate_type": "tex_equation_rendered_macro_term_profile_design",
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
        "rendered_alias_text": _rendered_alias_text(str(source_row.get("candidate_text") or "")),
        "latex_labels": _latex_labels(str(source_row.get("candidate_text") or "")),
        "custom_latex_macros": [str(item) for item in list(source_row.get("custom_latex_macros") or [])],
        "source_context_status": str(source_context.get("status") or "blocked"),
        "sourceContentHash": str(source_context.get("sourceContentHash") or ""),
        "source_pdf_path": str(source_context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(source_context.get("manifestPath") or ""),
        "profile_results": profile_results,
        "recommended_profile": str(best.get("profile_name") or ""),
        "recommended_status": str(best.get("profile_status") or ""),
        "candidate_ready": ready,
        "evidence_tier": "rendered_macro_term_profile_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "recommended_action": _recommended_action(best),
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(
    *,
    input_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    profile_results = [profile for row in rows for profile in list(row.get("profile_results") or [])]
    return {
        "inputRows": len(input_rows),
        "targetRows": len(rows),
        "profileRows": len(profile_results),
        "candidateReadyRows": sum(1 for row in rows if bool(row.get("candidate_ready"))),
        "uniqueProfileRows": sum(
            1 for profile in profile_results if str(profile.get("profile_status") or "").startswith("unique_")
        ),
        "ambiguousProfileRows": sum(
            1 for profile in profile_results if str(profile.get("profile_status") or "").startswith("ambiguous_")
        ),
        "uniqueCanonicalProfileRows": sum(
            1
            for profile in profile_results
            if str(profile.get("canonical_match_status") or "").startswith("unique_")
        ),
        "ambiguousCanonicalProfileRows": sum(
            1
            for profile in profile_results
            if str(profile.get("canonical_match_status") or "").startswith("ambiguous_")
        ),
        "uniquePdfRegionProfileRows": sum(
            1
            for profile in profile_results
            if str(profile.get("pdf_region_match_status") or "").startswith("unique_")
        ),
        "ambiguousPdfRegionProfileRows": sum(
            1
            for profile in profile_results
            if str(profile.get("pdf_region_match_status") or "").startswith("ambiguous_")
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
        "byRecommendedStatus": dict(Counter(str(row.get("recommended_status") or "") for row in rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in rows)),
        "byProfileStatus": dict(Counter(str(profile.get("profile_status") or "") for profile in profile_results)),
        "byCanonicalMatchStatus": dict(
            Counter(str(profile.get("canonical_match_status") or "") for profile in profile_results)
        ),
        "byPdfRegionMatchStatus": dict(
            Counter(str(profile.get("pdf_region_match_status") or "") for profile in profile_results)
        ),
    }


def build_tex_equation_rendered_macro_term_profile_design(
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
        if str(row.get("recommended_action") or "") == "design_rendered_macro_term_profile"
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
        "schema": TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID,
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
            "decision": "rendered_macro_term_profile_design_ready" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "rendered_macro_candidate_review_or_equation_label_number_disambiguation_design",
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
        "profiles": [
            {
                "profile_name": profile.name,
                "description": profile.description,
                "proposed_rules": list(profile.proposed_rules),
            }
            for profile in _PROFILES
        ],
        "warnings": [
            "rendered_macro_profile_matches_are_not_source_spans",
            "pdf_region_matches_are_layout_candidates_only",
            "equation_label_to_number_mapping_is_not_authoritative_in_this_tranche",
            "recommended_actions_are_nonbinding",
            "do_not_promote_without_later_explicit_tranche",
            *schema_violations,
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "profiles", "warnings")
        if key in report
    }


def render_tex_equation_rendered_macro_term_profile_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX Equation Rendered Macro Term Profile Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Target rows: `{int(counts.get('targetRows') or 0)}`",
        f"- Profile rows: `{int(counts.get('profileRows') or 0)}`",
        f"- Candidate-ready rows: `{int(counts.get('candidateReadyRows') or 0)}`",
        f"- Unique canonical profile rows: `{int(counts.get('uniqueCanonicalProfileRows') or 0)}`",
        f"- Unique PDF-region profile rows: `{int(counts.get('uniquePdfRegionProfileRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This report evaluates rendered macro term profiles only. It does not create evidence or connect to runtime answering.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Recommended status: `{json.dumps(counts.get('byRecommendedStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Profile status: `{json.dumps(counts.get('byProfileStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Canonical status: `{json.dumps(counts.get('byCanonicalMatchStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- PDF-region status: `{json.dumps(counts.get('byPdfRegionMatchStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('equation_environment')}` "
            f"`{row.get('recommended_profile')}` `{row.get('recommended_status')}`"
        )
    return "\n".join(lines)


def write_tex_equation_rendered_macro_term_profile_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-rendered-macro-term-profile-design-report.json"
    summary_path = root / "tex-equation-rendered-macro-term-profile-design-summary.json"
    markdown_path = root / "tex-equation-rendered-macro-term-profile-design.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_rendered_macro_term_profile_design_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only rendered macro term-profile TeX equation design.")
    parser.add_argument("--remaining-window-diagnostic-report", default=str(DEFAULT_TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT), help="Parsed paper artifact root.")
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_rendered_macro_term_profile_design(
        remaining_window_diagnostic_report=args.remaining_window_diagnostic_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_rendered_macro_term_profile_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_RENDERED_MACRO_TERM_PROFILE_DESIGN_SCHEMA_ID",
    "build_tex_equation_rendered_macro_term_profile_design",
    "render_tex_equation_rendered_macro_term_profile_design_markdown",
    "write_tex_equation_rendered_macro_term_profile_design_reports",
]
