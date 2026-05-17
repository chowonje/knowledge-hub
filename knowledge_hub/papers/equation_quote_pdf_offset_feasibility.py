"""Report-only EquationQuote original PDF offset feasibility audit.

This helper checks whether EquationQuoteCandidate text can be located in the
original local source PDF text. Diagnostic term/page context may be recorded,
but it does not interpret equations, create strict evidence, mutate SQLite,
route parsers, write canonical parsed artifacts, reindex, or reembed.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
import unicodedata

from knowledge_hub.papers.equation_alignment_feasibility_audit import _diagnostic_terms
from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import (
    _clean_text,
    _empty_span,
    _exact_matches,
    _find_all,
    _normalized_matches,
    _paper_context,
    _safe_int,
)


EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.equation-quote-pdf-offset-feasibility.v1"
)
EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.equation-quote-candidate-report.v1"


@dataclass(frozen=True)
class _CompactText:
    text: str
    indexes: list[int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact_with_indexes(value: str) -> _CompactText:
    parts: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(str(value or "")):
        folded = unicodedata.normalize("NFKC", char).casefold()
        for item in folded:
            if item.isspace():
                continue
            parts.append(item)
            indexes.append(index)
    return _CompactText("".join(parts), indexes)


def _candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("candidates") or []):
        if isinstance(item, dict) and str(item.get("candidate_type") or "") == "equation_quote_candidate":
            rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")),
            _safe_int(item.get("chars_start")),
            str(item.get("candidate_id") or ""),
        )
    )
    return rows


def _source_text_options(row: dict[str, Any]) -> list[tuple[str, str, float]]:
    options: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for label, value, confidence in (
        ("full_equation_text", row.get("equation_text") or row.get("candidate_text"), 1.0),
        ("candidate_text", row.get("candidate_text"), 1.0),
    ):
        text = _clean_text(value)
        if text and text not in seen:
            options.append((label, text, confidence))
            seen.add(text)
    return options


def _compact_matches(pages: list[dict[str, Any]], candidate_text: str) -> list[dict[str, Any]]:
    needle = _compact_with_indexes(candidate_text)
    if len(needle.text) < 4:
        return []
    matches: list[dict[str, Any]] = []
    for page in pages:
        compact = _compact_with_indexes(str(page.get("text") or ""))
        if not compact.text:
            continue
        page_start = _safe_int(page.get("chars_start"))
        for start in _find_all(compact.text, needle.text):
            end = start + len(needle.text)
            if end > len(compact.indexes):
                continue
            raw_start = compact.indexes[start]
            raw_end = compact.indexes[end - 1] + 1
            matches.append(
                {
                    "page": _safe_int(page.get("page")),
                    "chars_start": page_start + raw_start,
                    "chars_end": page_start + raw_end,
                    "match_method": "compact_whitespace_removed",
                    "match_confidence": 0.9,
                }
            )
    return matches


def _find_unique_pdf_match(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    pages = list(context.get("pages") or [])
    attempted: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for target, text, target_confidence in _source_text_options(row):
        for method, finder in (
            ("exact", _exact_matches),
            ("normalized_whitespace_case", _normalized_matches),
            ("compact_whitespace_removed", _compact_matches),
        ):
            matches = finder(pages, text)
            attempted.append({"target": target, "method": method, "matchCount": len(matches)})
            if len(matches) == 1:
                match = dict(matches[0])
                match["match_target"] = target
                match["matched_text"] = text
                match["match_confidence"] = min(float(match.get("match_confidence") or 0.0), target_confidence)
                return {"status": "ok", "match": match, "attempted": attempted}
            if len(matches) > 1:
                ambiguous.append({"target": target, "method": method, "matchCount": len(matches)})
                break
    if ambiguous:
        return {"status": "ambiguous", "ambiguous": ambiguous, "attempted": attempted}
    return {"status": "not_found", "attempted": attempted}


def _diagnostic_page_context(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    terms = _diagnostic_terms(str(row.get("equation_text") or row.get("candidate_text") or ""))
    if not terms:
        return {"terms": [], "pageCandidates": [], "bestCoverage": 0.0}
    page_candidates: list[dict[str, Any]] = []
    for page in list(context.get("pages") or []):
        haystack = str(page.get("text") or "").casefold()
        matches = [term for term in terms if term.casefold() in haystack]
        if not matches:
            continue
        coverage = round(len(matches) / len(terms), 6)
        page_candidates.append(
            {
                "page": _safe_int(page.get("page")),
                "matchedTerms": matches,
                "termCount": len(terms),
                "matchCount": len(matches),
                "coverage": coverage,
            }
        )
    page_candidates.sort(key=lambda item: (-float(item.get("coverage") or 0.0), _safe_int(item.get("page"))))
    best = float(page_candidates[0].get("coverage") or 0.0) if page_candidates else 0.0
    return {"terms": terms, "pageCandidates": page_candidates[:5], "bestCoverage": best}


def _base_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    source_hash = str(context.get("sourceContentHash") or "").strip()
    return {
        "feasibility_row_id": f"equationquote-pdf-offset:{row.get('paper_id')}:{row.get('candidate_id')}",
        "source_equation_quote_candidate_id": str(row.get("candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "equation_text": _clean_text(row.get("equation_text") or row.get("candidate_text")),
        "equation_label": str(row.get("equation_label") or ""),
        "canonical_alignment_status": str(row.get("canonical_alignment_status") or ""),
        "canonical_alignment_method": str(row.get("alignment_method") or ""),
        "canonical_span": {
            "chars_start": row.get("chars_start"),
            "chars_end": row.get("chars_end"),
            "page": row.get("page"),
            "sourceContentHash": str(row.get("sourceContentHash") or ""),
        },
        "layout_element_ids": [str(item) for item in list(row.get("layout_element_ids") or [])],
        "bbox": row.get("bbox"),
        "layout_region_candidate_present": bool(row.get("layout_region_candidate_present")),
        "equation_region_link_verified": False,
        "equation_semantics_interpreted": False,
        "source_pdf_path": str(context.get("sourcePdfPath") or ""),
        "source_manifest_path": str(context.get("manifestPath") or ""),
        "sourceContentHash": source_hash,
        "evidence_tier": "equation_quote_pdf_offset_feasibility_only",
        "report_only": True,
        "quote_only": True,
        "runtime_promotion_allowed": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
    }


def _blocked_row(
    row: dict[str, Any],
    context: dict[str, Any],
    *,
    status: str,
    reason: str,
    match_count: int = 0,
    attempted_matches: list[dict[str, Any]] | None = None,
    diagnostic_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_hash = str(context.get("sourceContentHash") or "").strip()
    diagnostic = diagnostic_context or {"terms": [], "pageCandidates": [], "bestCoverage": 0.0}
    return {
        **_base_row(row, context),
        "feasibility_status": status,
        "original_pdf_offset_recovered": False,
        "original_pdf_span": _empty_span(source_hash),
        "match_target": "",
        "matched_text": "",
        "match_count": match_count,
        "attempted_matches": attempted_matches or [],
        "diagnostic_terms": list(diagnostic.get("terms") or []),
        "diagnostic_page_candidates": list(diagnostic.get("pageCandidates") or []),
        "diagnostic_best_page_coverage": float(diagnostic.get("bestCoverage") or 0.0),
        "page_agrees_with_canonical": False,
        "source_hash_agrees_with_canonical": False,
        "feasibility_failure_reason": reason,
        "source_span_created": False,
        "strict_blockers": [
            "report_only",
            "original_pdf_offset_not_recovered",
            "equation_alignment_missing",
            "equation_semantics_not_interpreted",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "equation_quote_pdf_offset_feasibility_only",
            "no_runtime_or_strict_evidence_created",
            reason,
        ],
    }


def _feasibility_row(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if str(context.get("status") or "") != "ok":
        status = str(context.get("status") or "blocked")
        return _blocked_row(row, context, status=status, reason=status)
    candidate_hash = str(row.get("sourceContentHash") or "").strip()
    source_hash = str(context.get("sourceContentHash") or "").strip()
    if candidate_hash and source_hash and candidate_hash != source_hash:
        return _blocked_row(row, context, status="blocked_source_hash_mismatch", reason="source_hash_mismatch")
    if not _source_text_options(row):
        return _blocked_row(row, context, status="blocked_no_candidate_text", reason="candidate_text_missing")

    diagnostic = _diagnostic_page_context(row, context)
    match_result = _find_unique_pdf_match(row, context)
    if match_result["status"] == "ambiguous":
        ambiguous = list(match_result.get("ambiguous") or [])
        return _blocked_row(
            row,
            context,
            status="blocked_ambiguous_match",
            reason="original_pdf_equation_quote_match_ambiguous",
            match_count=sum(int(item.get("matchCount") or 0) for item in ambiguous),
            attempted_matches=list(match_result.get("attempted") or []),
            diagnostic_context=diagnostic,
        )
    if match_result["status"] != "ok":
        status = (
            "diagnostic_page_context_candidate_only"
            if float(diagnostic.get("bestCoverage") or 0.0) >= 0.5
            else "blocked_no_match"
        )
        return _blocked_row(
            row,
            context,
            status=status,
            reason="unique_original_pdf_equation_quote_match_not_found",
            attempted_matches=list(match_result.get("attempted") or []),
            diagnostic_context=diagnostic,
        )

    match = dict(match_result.get("match") or {})
    canonical_page = row.get("page")
    recovered_page = match.get("page")
    page_agrees = canonical_page is not None and _safe_int(canonical_page) == _safe_int(recovered_page)
    source_hash_agrees = bool(candidate_hash and source_hash and candidate_hash == source_hash)
    return {
        **_base_row(row, context),
        "feasibility_status": f"recovered_{match.get('match_method')}",
        "original_pdf_offset_recovered": True,
        "original_pdf_span": {
            "originalPdfCharsStart": match.get("chars_start"),
            "originalPdfCharsEnd": match.get("chars_end"),
            "page": recovered_page,
            "sourceContentHash": source_hash,
            "matchMethod": match.get("match_method"),
            "matchConfidence": match.get("match_confidence"),
        },
        "match_target": str(match.get("match_target") or ""),
        "matched_text": str(match.get("matched_text") or ""),
        "match_count": 1,
        "attempted_matches": list(match_result.get("attempted") or []),
        "diagnostic_terms": list(diagnostic.get("terms") or []),
        "diagnostic_page_candidates": list(diagnostic.get("pageCandidates") or []),
        "diagnostic_best_page_coverage": float(diagnostic.get("bestCoverage") or 0.0),
        "page_agrees_with_canonical": page_agrees,
        "source_hash_agrees_with_canonical": source_hash_agrees,
        "feasibility_failure_reason": "",
        "source_span_created": False,
        "strict_blockers": [
            "report_only",
            "equation_quote_offset_recovered_but_not_promoted",
            "equation_semantics_not_interpreted",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "original_pdf_equation_quote_offset_recovered_in_report_only_feasibility_audit",
            "equation_semantics_not_interpreted",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(rows: list[dict[str, Any]], *, schema_violations: list[str]) -> dict[str, Any]:
    recovered = [item for item in rows if item.get("original_pdf_offset_recovered")]
    blocked = [item for item in rows if not item.get("original_pdf_offset_recovered")]
    blocker_counts: Counter[str] = Counter()
    for item in rows:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputEquationQuoteCandidates": len(rows),
        "feasibilityRows": len(rows),
        "originalPdfOffsetRecoveredRows": len(recovered),
        "blockedRows": len(blocked),
        "exactRecoveredRows": sum(
            1 for item in recovered if (item.get("original_pdf_span") or {}).get("matchMethod") == "exact"
        ),
        "normalizedRecoveredRows": sum(
            1
            for item in recovered
            if (item.get("original_pdf_span") or {}).get("matchMethod") == "normalized_whitespace_case"
        ),
        "compactRecoveredRows": sum(
            1
            for item in recovered
            if (item.get("original_pdf_span") or {}).get("matchMethod") == "compact_whitespace_removed"
        ),
        "diagnosticPageContextRows": sum(
            1 for item in rows if item.get("feasibility_status") == "diagnostic_page_context_candidate_only"
        ),
        "ambiguousRows": sum(1 for item in rows if item.get("feasibility_status") == "blocked_ambiguous_match"),
        "noMatchRows": sum(1 for item in rows if item.get("feasibility_status") == "blocked_no_match"),
        "pageAgreementRows": sum(1 for item in recovered if item.get("page_agrees_with_canonical")),
        "sourceHashAgreementRows": sum(1 for item in recovered if item.get("source_hash_agrees_with_canonical")),
        "sourceSpanCreatedRows": 0,
        "equationSemanticsInterpretedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byFeasibilityStatus": dict(Counter(str(item.get("feasibility_status") or "") for item in rows)),
        "byMatchTarget": dict(Counter(str(item.get("match_target") or "") for item in rows)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_equation_quote_pdf_offset_feasibility_report(
    *,
    equation_quote_report: str | Path,
    pymupdf_parsed_root: str | Path,
    pdf_page_text_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a report-only EquationQuote original PDF offset feasibility report."""

    report_path = Path(str(equation_quote_report)).expanduser()
    parsed_root = Path(str(pymupdf_parsed_root)).expanduser()
    candidate_report = _read_json(report_path)
    schema_violations: list[str] = []
    if candidate_report.get("schema") != EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID:
        schema_violations.append("equation_quote_candidate_report_schema_mismatch")
    if candidate_report.get("status") not in {"ok", "empty"}:
        schema_violations.append("equation_quote_candidate_report_status_unexpected")
    source_rows = _candidate_rows(candidate_report)
    loader = pdf_page_text_loader
    if loader is None:
        from knowledge_hub.papers.sectionspan_pdf_offset_recovery_dry_run import _extract_pdf_pages

        loader = _extract_pdf_pages
    contexts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in contexts:
            contexts[paper_id] = _paper_context(paper_id=paper_id, parsed_root=parsed_root, page_loader=loader)
        rows.append(_feasibility_row(row, contexts[paper_id]))
    counts = _counts(rows, schema_violations=schema_violations)
    blocked = bool(schema_violations) or not rows
    return {
        "schema": EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID,
        "status": "blocked" if blocked else "feasibility_complete",
        "generatedAt": _now(),
        "inputs": {
            "equationQuoteReport": str(report_path),
            "equationQuoteSchema": str(candidate_report.get("schema") or ""),
            "pymupdfParsedRoot": str(parsed_root),
        },
        "counts": counts,
        "gate": {
            "feasibilityComplete": not blocked,
            "applyExecuted": False,
            "sourceSpanCreationReady": False,
            "equationSemanticsReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "feasibility_complete_non_strict" if not blocked else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "equation_quote_normalization_or_layout_review_before_any_strict_promotion",
        },
        "policy": {
            "reportOnly": True,
            "quoteOnly": True,
            "dryRunOnly": True,
            "applyExecuted": False,
            "sourceSpanCreated": False,
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
            "equation_quote_pdf_offsets_are_report_only_feasibility_results",
            "diagnostic_page_context_does_not_create_source_spans",
            "equation_semantics_are_not_interpreted",
            "runtime_promotion_remains_disabled",
        ],
        "paperContexts": {key: {k: v for k, v in value.items() if k != "pages"} for key, value in contexts.items()},
        "feasibilityRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "inputs",
            "counts",
            "gate",
            "policy",
            "warnings",
            "paperContexts",
            "feasibilityRows",
        )
        if key in report
    }


def render_equation_quote_pdf_offset_feasibility_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# EquationQuote Original PDF Offset Feasibility",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Feasibility rows: `{int(counts.get('feasibilityRows') or 0)}`",
        f"- Original PDF offsets recovered: `{int(counts.get('originalPdfOffsetRecoveredRows') or 0)}`",
        f"- Diagnostic page-context rows: `{int(counts.get('diagnosticPageContextRows') or 0)}`",
        f"- Blocked rows: `{int(counts.get('blockedRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a report-only feasibility audit. It reads local source PDFs and records unique equation-quote matches or diagnostic page context, but does not interpret equations, create source spans, write canonical parsed artifacts, mutate SQLite, reindex, reembed, create strict evidence, allow runtime citations, or change parser routing.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By feasibility status: `{json.dumps(counts.get('byFeasibilityStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By match target: `{json.dumps(counts.get('byMatchTarget') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_equation_quote_pdf_offset_feasibility_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "equation-quote-pdf-offset-feasibility.json"
    summary_path = root / "equation-quote-pdf-offset-feasibility-summary.json"
    markdown_path = root / "equation-quote-pdf-offset-feasibility.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_pdf_offset_feasibility_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only EquationQuote original PDF offset feasibility.")
    parser.add_argument("--equation-quote-report", required=True)
    parser.add_argument("--pymupdf-parsed-root", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_pdf_offset_feasibility_report(
        equation_quote_report=args.equation_quote_report,
        pymupdf_parsed_root=args.pymupdf_parsed_root,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_pdf_offset_feasibility_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_PDF_OFFSET_FEASIBILITY_SCHEMA_ID",
    "build_equation_quote_pdf_offset_feasibility_report",
    "render_equation_quote_pdf_offset_feasibility_markdown",
    "write_equation_quote_pdf_offset_feasibility_reports",
]
