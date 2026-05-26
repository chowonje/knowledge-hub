"""Report-only diagnostic for unresolved TeX equation window rows.

This helper consumes the TeX equation PDF-region anchor audit and classifies
rows that still lack a canonical line-local normalized window. It is a planning
diagnostic only: it does not create source spans, strict evidence, runtime
evidence, parser routing, answer integration, or canonical parsed artifacts.
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


TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-remaining-window-diagnostic.v1"
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

_SPACE_RE = re.compile(r"\s+")
_LATEX_MACRO_RE = re.compile(r"\\([A-Za-z]+)")
_COMMON_LATEX_MACROS = {
    "begin",
    "cdot",
    "cos",
    "end",
    "frac",
    "hat",
    "label",
    "left",
    "log",
    "max",
    "min",
    "right",
    "sqrt",
    "sum",
    "tag",
    "text",
    "times",
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


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _macro_names(text: str) -> list[str]:
    return list(dict.fromkeys(_LATEX_MACRO_RE.findall(str(text or ""))))


def _custom_macro_names(text: str) -> list[str]:
    return [name for name in _macro_names(text) if name not in _COMMON_LATEX_MACROS]


def _is_multiline(text: str, environment: str) -> bool:
    env = str(environment or "").casefold()
    return env in {"align", "align*", "multline", "multline*", "gather", "gather*"} or "\\\\" in text or "&" in text


def _diagnoses(row: dict[str, Any]) -> list[str]:
    text = _clean_text(row.get("candidate_text"))
    environment = str(row.get("equation_environment") or "")
    terms = [str(item) for item in list(row.get("normalized_terms") or []) if str(item).strip()]
    custom_macros = _custom_macro_names(text)
    diagnoses = [
        "line_local_normalized_window_missing",
        "pdf_region_anchor_missing",
    ]
    if _is_multiline(text, environment):
        diagnoses.append("multiline_equation_environment")
    if len(text) >= 280:
        diagnoses.append("large_equation_text_window")
    if custom_macros:
        diagnoses.append("custom_or_rendered_latex_macro_gap")
    if "\\tt" in text or "[CLS]" in text or "[MASK]" in text or "[SEP]" in text:
        diagnoses.append("text_heavy_equation_environment")
    if len(terms) >= 12:
        diagnoses.append("many_normalized_terms_need_segmented_matching")
    if not text:
        diagnoses.append("empty_equation_text")
    return list(dict.fromkeys(diagnoses))


def _recommended_action(diagnoses: list[str]) -> str:
    items = set(diagnoses)
    if "text_heavy_equation_environment" in items:
        return "keep_blocked_or_handle_as_text_example_not_equation_quote"
    if "large_equation_text_window" in items or "multiline_equation_environment" in items:
        return "design_segmented_multiline_equation_matching"
    if "custom_or_rendered_latex_macro_gap" in items:
        return "design_rendered_macro_term_profile"
    return "keep_blocked_pending_alternative_extractor_or_manual_review"


def _row(index: int, source_row: dict[str, Any]) -> dict[str, Any]:
    text = _clean_text(source_row.get("candidate_text"))
    diagnoses = _diagnoses(source_row)
    custom_macros = _custom_macro_names(text)
    action = _recommended_action(diagnoses)
    blockers = list(
        dict.fromkeys(
            [
                "remaining_window_rows_are_diagnostic_only",
                "canonical_line_local_normalized_window_missing",
                "pdf_region_anchor_missing",
                "source_span_creation_disabled_for_tranche",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                *[str(item) for item in list(source_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "diagnostic_id": f"tex-equation-remaining-window-diagnostic:{index:04d}",
        "candidate_type": "tex_equation_remaining_window_diagnostic",
        "source_pdf_region_anchor_id": str(source_row.get("pdf_region_anchor_id") or ""),
        "source_line_local_anchor_id": str(source_row.get("source_line_local_anchor_id") or ""),
        "source_design_id": str(source_row.get("source_design_id") or ""),
        "source_candidate_id": str(source_row.get("source_candidate_id") or ""),
        "paper_id": str(source_row.get("paper_id") or ""),
        "source_parser": "arxiv_tex+pymupdf_pdf_blocks",
        "source_file": str(source_row.get("source_file") or ""),
        "equation_environment": str(source_row.get("equation_environment") or ""),
        "candidate_text": text,
        "candidate_text_length": len(text),
        "normalized_terms": [str(item) for item in list(source_row.get("normalized_terms") or [])],
        "normalized_term_count": len(list(source_row.get("normalized_terms") or [])),
        "custom_latex_macros": custom_macros,
        "custom_latex_macro_count": len(custom_macros),
        "line_local_anchor_status": str(source_row.get("line_local_anchor_status") or ""),
        "pdf_region_anchor_status": str(source_row.get("pdf_region_anchor_status") or ""),
        "normalized_window_count": _safe_int(source_row.get("normalized_window_count")),
        "pdf_region_candidate_count": _safe_int(source_row.get("pdf_region_candidate_count")),
        "diagnoses": diagnoses,
        "recommended_action": action,
        "evidence_tier": "remaining_window_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "source_span_created": False,
        "parser_routing_changed": False,
        "answer_integration_changed": False,
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(
    *,
    input_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    schema_violations: list[str],
) -> dict[str, Any]:
    diagnoses = Counter()
    actions = Counter()
    for row in rows:
        diagnoses.update(str(item) for item in list(row.get("diagnoses") or []))
        actions.update([str(row.get("recommended_action") or "")])
    return {
        "inputRows": len(input_rows),
        "diagnosticRows": len(rows),
        "blockedNoLineLocalWindowRows": len(rows),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(row.get("paper_id") or "") for row in rows)),
        "byEnvironment": dict(Counter(str(row.get("equation_environment") or "") for row in rows)),
        "byDiagnosis": dict(diagnoses),
        "byRecommendedAction": dict(actions),
    }


def build_tex_equation_remaining_window_diagnostic(
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
    input_rows = [
        dict(row)
        for row in list(payload.get("rows") or [])
        if not schema_violations
        and isinstance(row, dict)
        and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    blocked_rows = [
        row
        for row in input_rows
        if str(row.get("pdf_region_anchor_status") or "") == "blocked_no_line_local_normalized_window"
    ]
    rows = [_row(index + 1, row) for index, row in enumerate(blocked_rows)]
    counts = _counts(input_rows=input_rows, rows=rows, schema_violations=schema_violations)
    return {
        "schema": TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "pdfRegionAnchorReportPath": str(input_path),
            "pdfRegionAnchorReportSchema": parent_schema,
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "diagnosticReady": bool(rows) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "remaining_window_diagnostic_ready" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "segmented_multiline_equation_matching_design",
        },
        "policy": {
            "allRowsNonStrict": True,
            "reportOnly": True,
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
            "remaining_window_diagnostics_are_not_evidence",
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


def render_tex_equation_remaining_window_diagnostic_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# TeX Equation Remaining Window Diagnostic",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input rows: `{int(counts.get('inputRows') or 0)}`",
        f"- Diagnostic rows: `{int(counts.get('diagnosticRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        f"- Runtime evidence: `{int(counts.get('runtimeEvidenceRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This report only classifies unresolved TeX equation rows. It does not create evidence or connect to runtime answering.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By environment: `{json.dumps(counts.get('byEnvironment') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By diagnosis: `{json.dumps(counts.get('byDiagnosis') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recommended action: `{json.dumps(counts.get('byRecommendedAction') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Rows",
        "",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            f"- `{row.get('paper_id')}` `{row.get('equation_environment')}` "
            f"`{row.get('recommended_action')}`: {row.get('candidate_text')}"
        )
    return "\n".join(lines)


def write_tex_equation_remaining_window_diagnostic_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-remaining-window-diagnostic-report.json"
    summary_path = root / "tex-equation-remaining-window-diagnostic-summary.json"
    markdown_path = root / "tex-equation-remaining-window-diagnostic.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_remaining_window_diagnostic_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only diagnostic for unresolved TeX equation rows.")
    parser.add_argument("--pdf-region-anchor-report", default=str(DEFAULT_TEX_EQUATION_PDF_REGION_ANCHOR_REPORT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_remaining_window_diagnostic(
        pdf_region_anchor_report=args.pdf_region_anchor_report,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_remaining_window_diagnostic_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_REMAINING_WINDOW_DIAGNOSTIC_SCHEMA_ID",
    "build_tex_equation_remaining_window_diagnostic",
    "render_tex_equation_remaining_window_diagnostic_markdown",
    "write_tex_equation_remaining_window_diagnostic_reports",
]
