"""Report-only equation quote alignment feasibility audit.

This helper investigates why MinerU equation candidates did not align to
PyMuPDF canonical parsed text.  It may find raw, compact, or term-level
diagnostic matches, but it never creates source spans, strict evidence, parser
routing, answer integration, or canonical artifact writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.equation-alignment-feasibility-audit.v1"
EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.equation-quote-candidate-report.v1"
MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID = "knowledge-hub.paper.mineru-source-alignment-audit.v1"

_LATEX_COMMAND_STOP = {
    "begin",
    "end",
    "array",
    "frac",
    "sqrt",
    "tag",
    "left",
    "right",
    "operatorname",
    "operatorname*",
    "mathrm",
    "mathtt",
    "displaystyle",
    "qquad",
    "cdot",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text(path: str | Path) -> str:
    try:
        return Path(str(path)).expanduser().read_text(encoding="utf-8")
    except Exception:
        return ""


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _paper_document_paths(source_alignment: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for paper in list(source_alignment.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paperId") or "")
        document_path = str((paper.get("input") or {}).get("pymupdfDocumentMarkdownPath") or "")
        if paper_id and document_path:
            paths[paper_id] = document_path
    return paths


def _style_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"\\(?:mathrm|mathtt|operatorname\*?)\s*\{\s*([^{}]+?)\s*\}", text or ""):
        candidate = "".join(str(match.group(1)).split())
        if len(candidate) >= 2:
            terms.append(candidate)
    return terms


def _diagnostic_terms(text: str) -> list[str]:
    terms = _style_terms(text)
    for spaced in re.findall(r"(?:[A-Za-z]\s+){2,}[A-Za-z]", text or ""):
        joined = "".join(spaced.split())
        if len(joined) >= 3:
            terms.append(joined)
    for command in re.findall(r"\\([A-Za-z*]+)", text or ""):
        if command not in _LATEX_COMMAND_STOP and len(command.replace("*", "")) >= 2:
            terms.append(command.replace("*", ""))
    for word in re.findall(r"[A-Za-z][A-Za-z_]{1,}", text or ""):
        if word not in _LATEX_COMMAND_STOP and len(word) >= 2:
            terms.append(word)
    cleaned: list[str] = []
    for term in terms:
        normalized = term.strip("_")
        if len(normalized) >= 2 and normalized.casefold() not in {item.casefold() for item in cleaned}:
            cleaned.append(normalized)
    return cleaned[:16]


def _term_matches(terms: list[str], document_text: str) -> list[str]:
    haystack = document_text.casefold()
    return [term for term in terms if term.casefold() in haystack]


def _match_counts(candidate_text: str, document_text: str) -> tuple[bool, int, list[str], list[str], float]:
    raw_match = bool(candidate_text and candidate_text in document_text)
    compact_candidate = _compact(candidate_text)
    compact_document = _compact(document_text)
    compact_count = compact_document.count(compact_candidate) if len(compact_candidate) >= 12 else 0
    terms = _diagnostic_terms(candidate_text)
    matches = _term_matches(terms, document_text)
    coverage = round(len(matches) / len(terms), 6) if terms else 0.0
    return raw_match, compact_count, terms, matches, coverage


def _status(
    *,
    raw_match: bool,
    compact_count: int,
    term_coverage: float,
    document_text: str,
) -> str:
    if not document_text:
        return "blocked_missing_canonical_document_text"
    if raw_match:
        return "raw_text_match_candidate_only"
    if compact_count == 1:
        return "compact_text_unique_match_candidate_only"
    if compact_count > 1:
        return "compact_text_ambiguous_match_candidate_only"
    if term_coverage >= 0.5:
        return "diagnostic_term_context_candidate_only"
    return "blocked_no_canonical_equation_text_match"


def _row(index: int, item: dict[str, Any], document_paths: dict[str, str], document_cache: dict[str, str]) -> dict[str, Any]:
    paper_id = str(item.get("paper_id") or "")
    document_path = document_paths.get(paper_id, "")
    if document_path not in document_cache:
        document_cache[document_path] = _read_text(document_path) if document_path else ""
    document_text = document_cache.get(document_path, "")
    candidate_text = str(item.get("candidate_text") or "")
    raw_match, compact_count, terms, matches, coverage = _match_counts(candidate_text, document_text)
    status = _status(
        raw_match=raw_match,
        compact_count=compact_count,
        term_coverage=coverage,
        document_text=document_text,
    )
    blockers = list(dict.fromkeys([
        "equation_alignment_feasibility_audit_only",
        "runtime_promotion_disabled_for_tranche",
        "equation_quote_candidate_layer_not_runtime_evidence",
        "equation_semantics_not_interpreted",
        "no_canonical_chars_start_end_created",
        "markdown_offsets_are_generated_not_original_pdf_offsets",
        *[str(value) for value in list(item.get("strict_blockers") or [])],
    ]))
    return {
        "audit_id": f"equation-alignment-feasibility:{index:04d}",
        "candidate_id": str(item.get("candidate_id") or ""),
        "paper_id": paper_id,
        "candidate_type": str(item.get("candidate_type") or ""),
        "candidate_text": candidate_text,
        "source_parser": str(item.get("source_parser") or "mineru+pymupdf_alignment"),
        "canonical_document_path": document_path,
        "canonical_document_available": bool(document_text),
        "existing_alignment_status": str(item.get("canonical_alignment_status") or ""),
        "existing_alignment_method": str(item.get("alignment_method") or ""),
        "raw_text_match": raw_match,
        "compact_text_match_count": compact_count,
        "diagnostic_terms": terms,
        "diagnostic_term_matches": matches,
        "diagnostic_term_coverage": coverage,
        "feasibility_status": status,
        "layout_element_count": len(list(item.get("layout_element_ids") or [])),
        "bbox_available": bool(item.get("bbox")),
        "sourceContentHash": str(item.get("sourceContentHash") or ""),
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "evidence_tier": "equation_alignment_feasibility_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "feasibility_rows_are_not_evidence",
            "diagnostic_matches_do_not_create_source_spans",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _schema_violations(equation_report: dict[str, Any], source_alignment_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if equation_report.get("schema") != EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID:
        violations.append("equation_quote_candidate_report_schema_mismatch")
    if source_alignment_report.get("schema") != MINERU_SOURCE_ALIGNMENT_AUDIT_SCHEMA_ID:
        violations.append("mineru_source_alignment_report_schema_mismatch")
    return violations


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(item.get("feasibility_status") or "") for item in rows)
    by_paper = Counter(str(item.get("paper_id") or "") for item in rows)
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputEquationQuoteCandidates": len(rows),
        "auditedEquationQuoteCandidates": len(rows),
        "rawTextMatchCandidates": sum(1 for item in rows if item.get("raw_text_match")),
        "compactUniqueMatchCandidates": _safe_count(by_status, "compact_text_unique_match_candidate_only"),
        "diagnosticTermContextCandidates": _safe_count(by_status, "diagnostic_term_context_candidate_only"),
        "canonicalSourceSpanCreatedCandidates": 0,
        "equationSemanticsInterpretedCandidates": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "runtimeEvidenceCandidates": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byFeasibilityStatus": dict(by_status),
        "strictBlockerSummary": dict(blocker_counts),
    }


def _safe_count(counter: Counter[str], key: str) -> int:
    try:
        return int(counter.get(key, 0))
    except Exception:
        return 0


def build_equation_alignment_feasibility_audit(
    *,
    equation_quote_report: str | Path,
    mineru_source_alignment_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only feasibility audit for equation quote alignment."""

    equation_path = Path(str(equation_quote_report)).expanduser()
    source_path = Path(str(mineru_source_alignment_report)).expanduser()
    equation_report = _read_json(equation_path)
    source_alignment = _read_json(source_path)
    document_paths = _paper_document_paths(source_alignment)
    document_cache: dict[str, str] = {}
    candidates = [dict(item) for item in list(equation_report.get("candidates") or []) if isinstance(item, dict)]
    rows = [_row(index + 1, item, document_paths, document_cache) for index, item in enumerate(candidates)]
    schema_violations = _schema_violations(equation_report, source_alignment)
    counts = _counts(rows, schema_violations)
    return {
        "schema": EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteReport": str(equation_path),
            "mineruSourceAlignmentReport": str(source_path),
            "equationQuoteSchema": str(equation_report.get("schema") or ""),
            "mineruSourceAlignmentSchema": str(source_alignment.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "equationAlignmentFeasibilityReviewed": bool(rows),
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "decision": "equation_alignment_feasibility_reviewed" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "table_cell_provenance_feasibility_audit",
        },
        "policy": {
            "auditOnly": True,
            "quoteOnly": True,
            "equationSemanticsInterpreted": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "diagnostic_matches_do_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "equation_candidates_remain_non_strict_until_an_explicit_promotion_tranche",
        ],
        "rows": rows,
    }


def render_equation_alignment_feasibility_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Equation Alignment Feasibility Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Audited equation candidates: `{int(counts.get('auditedEquationQuoteCandidates') or 0)}`",
        f"- Raw text matches: `{int(counts.get('rawTextMatchCandidates') or 0)}`",
        f"- Compact unique matches: `{int(counts.get('compactUniqueMatchCandidates') or 0)}`",
        f"- Diagnostic term-context candidates: `{int(counts.get('diagnosticTermContextCandidates') or 0)}`",
        f"- Canonical source spans created: `{int(counts.get('canonicalSourceSpanCreatedCandidates') or 0)}`",
        f"- Strict eligible candidates: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        "",
        "## Policy",
        "",
        "This audit is diagnostic only. It does not interpret equations or create source spans, strict evidence, parser routing, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By feasibility status: `{json.dumps(counts.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_alignment_feasibility_audit_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    audit_path = root / "equation-alignment-feasibility-audit.json"
    markdown_path = root / "equation-alignment-feasibility-audit.md"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_alignment_feasibility_audit_markdown(report), encoding="utf-8")
    return {
        "audit": str(audit_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only equation alignment feasibility audit.")
    parser.add_argument("--equation-quote-report", required=True, help="Path to equation-quote-candidates.json.")
    parser.add_argument("--mineru-source-alignment-report", required=True, help="Path to mineru-source-alignment-report.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print audit payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_alignment_feasibility_audit(
        equation_quote_report=args.equation_quote_report,
        mineru_source_alignment_report=args.mineru_source_alignment_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_alignment_feasibility_audit_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_ALIGNMENT_FEASIBILITY_AUDIT_SCHEMA_ID",
    "build_equation_alignment_feasibility_audit",
    "render_equation_alignment_feasibility_audit_markdown",
    "write_equation_alignment_feasibility_audit_reports",
]
