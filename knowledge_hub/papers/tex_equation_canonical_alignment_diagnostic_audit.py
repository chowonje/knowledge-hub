"""Report-only TeX equation canonical alignment diagnostics.

This helper investigates why TeX equation environment text does not align to
PyMuPDF-generated canonical Markdown text. It is diagnostic only: it does not
create source spans, interpret equations, route parsers, mutate SQLite, reindex,
reembed, write canonical parsed artifacts, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.tex_structure_candidate_alignment_audit import (
    DEFAULT_PARSED_ROOT,
    TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID,
)


TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-canonical-alignment-diagnostic-audit.v1"
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

_EQUATION_STRUCTURE_TYPE = "equation_environment"
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")
_LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")
_LATEX_GROUP_COMMAND_RE = re.compile(
    r"\\(?:mathrm|mathtt|textrm|textbf|textit|text|operatorname\*?|mathbf|mathit|mathbbm|mathlarger)\s*\{([^{}]*)\}"
)
_STOP_TERMS = {
    "begin",
    "end",
    "equation",
    "align",
    "array",
    "cases",
    "text",
    "mathrm",
    "mathtt",
    "textrm",
    "textbf",
    "operatorname",
    "operatorname",
    "mathlarger",
    "mathbbm",
    "scriptsize",
    "displaystyle",
    "left",
    "right",
    "qquad",
    "quad",
    "cdot",
    "sum",
    "frac",
    "sqrt",
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


def _read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(str(path)).expanduser().read_text(encoding="utf-8")
    except Exception:
        return ""


def _clean_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def _compact(value: Any) -> str:
    return _WHITESPACE_RE.sub("", str(value or ""))


def _find_count(haystack: str, needle: str, *, min_length: int = 1) -> int:
    if len(needle) < min_length:
        return 0
    return haystack.count(needle)


def _tex_to_plain(value: Any) -> str:
    text = str(value or "")
    previous = ""
    while previous != text:
        previous = text
        text = _LATEX_GROUP_COMMAND_RE.sub(r" \1 ", text)
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r" \1 / \2 ", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" sqrt \1 ", text)
    text = _LATEX_COMMAND_RE.sub(r" \1 ", text)
    text = text.replace("\\\\", " ")
    text = re.sub(r"[{}&_^]", " ", text)
    text = re.sub(r"[\[\](),=+*/<>|:;.-]+", " ", text)
    return _clean_text(text)


def _diagnostic_terms(value: Any) -> list[str]:
    plain = _tex_to_plain(value)
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(plain):
        normalized = token.strip("_").casefold()
        if len(normalized) < 2 or normalized in _STOP_TERMS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token.strip("_"))
    return terms[:24]


def _term_matches(terms: list[str], canonical_text: str) -> list[str]:
    haystack = canonical_text.casefold()
    return [term for term in terms if term.casefold() in haystack]


def _diagnosis(
    *,
    candidate_text: str,
    canonical_text: str,
    raw_count: int,
    compact_count: int,
    plain_count: int,
    term_coverage: float,
    tex_has_macros: bool,
) -> tuple[str, float]:
    if not candidate_text:
        return "empty_equation_text", 1.0
    if not canonical_text:
        return "canonical_document_missing", 1.0
    if raw_count == 1:
        return "raw_tex_unique_match_candidate_only", 0.92
    if raw_count > 1:
        return "raw_tex_ambiguous_match_candidate_only", 0.4
    if compact_count == 1:
        return "compact_tex_unique_match_candidate_only", 0.82
    if compact_count > 1:
        return "compact_tex_ambiguous_match_candidate_only", 0.35
    if plain_count == 1:
        return "plain_text_unique_match_candidate_only", 0.72
    if plain_count > 1:
        return "plain_text_ambiguous_match_candidate_only", 0.3
    if term_coverage >= 0.45 and tex_has_macros:
        return "tex_to_canonical_normalization_gap_candidate_only", 0.62
    if term_coverage >= 0.45:
        return "diagnostic_term_context_candidate_only", 0.55
    if term_coverage > 0.0:
        return "weak_term_context_candidate_only", 0.25
    return "likely_canonical_equation_text_missing", 0.1


def _paper_document_path(parsed_root: Path, paper_id: str) -> Path:
    return parsed_root / paper_id / "document.md"


def _row(
    index: int,
    row: dict[str, Any],
    *,
    parsed_root: Path,
    document_cache: dict[str, str],
) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "")
    document_path = _paper_document_path(parsed_root, paper_id)
    document_path_key = str(document_path)
    if document_path_key not in document_cache:
        document_cache[document_path_key] = _read_text(document_path)
    canonical_text = document_cache[document_path_key]
    candidate_text = _clean_text(row.get("candidate_text"))
    compact_candidate = _compact(candidate_text)
    compact_canonical = _compact(canonical_text)
    plain_text = _tex_to_plain(candidate_text)
    compact_plain = _compact(plain_text)
    terms = _diagnostic_terms(candidate_text)
    matches = _term_matches(terms, canonical_text)
    term_coverage = round(len(matches) / len(terms), 6) if terms else 0.0
    raw_count = _find_count(canonical_text, candidate_text, min_length=8)
    compact_count = _find_count(compact_canonical, compact_candidate, min_length=12)
    plain_count = _find_count(compact_canonical.casefold(), compact_plain.casefold(), min_length=12)
    tex_has_macros = "\\" in candidate_text
    diagnosis, confidence = _diagnosis(
        candidate_text=candidate_text,
        canonical_text=canonical_text,
        raw_count=raw_count,
        compact_count=compact_count,
        plain_count=plain_count,
        term_coverage=term_coverage,
        tex_has_macros=tex_has_macros,
    )
    blockers = list(
        dict.fromkeys(
            [
                "tex_equation_alignment_diagnostic_only",
                "diagnostic_matches_do_not_create_source_spans",
                "equation_semantics_not_interpreted",
                "equation_region_link_unverified",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "tex_offsets_are_not_canonical_source_spans",
                *[str(item) for item in list(row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "diagnostic_id": f"tex-equation-alignment-diagnostic:{index:04d}",
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_canonical_alignment_diagnostic",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": str(row.get("source_file") or ""),
        "equation_environment": str(row.get("tex_environment") or ""),
        "candidate_text": candidate_text,
        "plain_text_candidate": plain_text,
        "canonical_document_path": document_path_key,
        "canonical_document_available": bool(canonical_text),
        "existing_alignment_status": str(row.get("alignment_status") or ""),
        "existing_alignment_method": str(row.get("alignment_method") or ""),
        "existing_alignment_reason": str(row.get("alignment_reason") or ""),
        "tex_has_macros": tex_has_macros,
        "raw_tex_match_count": raw_count,
        "compact_tex_match_count": compact_count,
        "plain_text_match_count": plain_count,
        "diagnostic_terms": terms,
        "diagnostic_term_matches": matches,
        "diagnostic_term_coverage": term_coverage,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_alignment_diagnostic_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "diagnostic_rows_are_not_evidence",
            "diagnostic_matches_do_not_create_source_spans",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_diagnosis = Counter(str(row.get("diagnosis") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    return {
        "equationEnvironmentRows": len(rows),
        "textBearingEquationEnvironmentRows": sum(1 for row in rows if row.get("candidate_text")),
        "emptyEquationTextRows": sum(1 for row in rows if not row.get("candidate_text")),
        "rawTexMatchRows": sum(1 for row in rows if int(row.get("raw_tex_match_count") or 0) > 0),
        "compactTexMatchRows": sum(1 for row in rows if int(row.get("compact_tex_match_count") or 0) > 0),
        "plainTextMatchRows": sum(1 for row in rows if int(row.get("plain_text_match_count") or 0) > 0),
        "diagnosticTermContextRows": sum(
            1
            for row in rows
            if str(row.get("diagnosis") or "")
            in {
                "tex_to_canonical_normalization_gap_candidate_only",
                "diagnostic_term_context_candidate_only",
                "weak_term_context_candidate_only",
            }
        ),
        "likelyCanonicalEquationTextMissingRows": int(by_diagnosis.get("likely_canonical_equation_text_missing", 0)),
        "normalizationGapRows": int(by_diagnosis.get("tex_to_canonical_normalization_gap_candidate_only", 0)),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byDiagnosis": dict(by_diagnosis),
    }


def build_tex_equation_canonical_alignment_diagnostic_audit(
    *,
    alignment_report: str | Path = DEFAULT_TEX_STRUCTURE_ALIGNMENT_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    input_path = Path(str(alignment_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    schema = str(payload.get("schema") or "")
    schema_violations = [] if schema == TEX_STRUCTURE_CANDIDATE_ALIGNMENT_AUDIT_SCHEMA_ID else [
        "tex_structure_candidate_alignment_report_schema_mismatch"
    ]
    source_rows = [
        dict(row)
        for row in list(payload.get("candidates") or [])
        if isinstance(row, dict)
        and str(row.get("structure_type") or "") == _EQUATION_STRUCTURE_TYPE
        and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    document_cache: dict[str, str] = {}
    rows = [
        _row(index + 1, row, parsed_root=parsed_root_path, document_cache=document_cache)
        for index, row in enumerate(source_rows)
        if not schema_violations
    ]
    counts = _counts(rows, schema_violations)
    return {
        "schema": TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "alignmentReportPath": str(input_path),
            "alignmentReportSchema": schema,
            "parsedRoot": str(parsed_root_path),
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
            "decision": "equation_canonical_alignment_diagnosed" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "tex_equation_canonical_normalizer_design",
        },
        "policy": {
            "reportOnly": True,
            "diagnosticOnly": True,
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
            "diagnostic_matches_do_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
            "equation_candidates_remain_non_strict_until_an_explicit_promotion_tranche",
        ],
        "rows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_tex_equation_canonical_alignment_diagnostic_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TeX Equation Canonical Alignment Diagnostic Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Equation environment rows: `{int(counts.get('equationEnvironmentRows') or 0)}`",
        f"- Text-bearing equation rows: `{int(counts.get('textBearingEquationEnvironmentRows') or 0)}`",
        f"- Empty equation rows: `{int(counts.get('emptyEquationTextRows') or 0)}`",
        f"- Raw TeX match rows: `{int(counts.get('rawTexMatchRows') or 0)}`",
        f"- Compact TeX match rows: `{int(counts.get('compactTexMatchRows') or 0)}`",
        f"- Plain-text match rows: `{int(counts.get('plainTextMatchRows') or 0)}`",
        f"- Diagnostic term-context rows: `{int(counts.get('diagnosticTermContextRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Policy",
        "",
        "This audit is diagnostic only. It does not interpret equations, create source spans, route parsers, mutate DB/index state, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By diagnosis: `{json.dumps(counts.get('byDiagnosis') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_tex_equation_canonical_alignment_diagnostic_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-canonical-alignment-diagnostic-report.json"
    summary_path = root / "tex-equation-canonical-alignment-diagnostic-summary.json"
    markdown_path = root / "tex-equation-canonical-alignment-diagnostic-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_canonical_alignment_diagnostic_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX equation canonical alignment diagnostic audit.")
    parser.add_argument("--alignment-report", default=str(DEFAULT_TEX_STRUCTURE_ALIGNMENT_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_canonical_alignment_diagnostic_audit(
        alignment_report=args.alignment_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_canonical_alignment_diagnostic_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID",
    "build_tex_equation_canonical_alignment_diagnostic_audit",
    "render_tex_equation_canonical_alignment_diagnostic_audit_markdown",
    "write_tex_equation_canonical_alignment_diagnostic_audit_reports",
]
