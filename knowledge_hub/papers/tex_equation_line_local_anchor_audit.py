"""Report-only TeX equation line-local anchor audit.

This helper checks whether TeX equation normalizer-design rows can be narrowed
with line-local canonical Markdown context or nearby equation-number markers.
It is still an audit only: it does not create source spans, interpret equations,
route parsers, mutate SQLite, reindex, reembed, write canonical parsed
artifacts, or change answer behavior.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.tex_equation_canonical_text_normalizer_design import (
    TEX_EQUATION_CANONICAL_TEXT_NORMALIZER_DESIGN_SCHEMA_ID,
)
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import DEFAULT_PARSED_ROOT


TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-line-local-anchor-audit.v1"
)

DEFAULT_TEX_EQUATION_NORMALIZER_DESIGN_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-canonical-text-normalizer-design"
    / "tex-equation-canonical-text-normalizer-design-report.json"
)

_TOKEN_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|[0-9]+(?:\.[0-9]+)?")
_PAGE_MARKER_RE = re.compile(r"^## Page\s+(\d+)\s*$")
_EQUATION_NUMBER_RE = re.compile(r"\((\d{1,3})\)")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Line:
    number: int
    start: int
    end: int
    text: str
    page_marker: int | None


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


def _normalize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", token).casefold()


def _canonical_bridge_text_with_offsets(value: str) -> tuple[str, list[int]]:
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "×": " x ",
        "∗": " * ",
        "−": " - ",
        "–": " - ",
        "—": " - ",
        "√": " sqrt ",
    }
    chars: list[str] = []
    offsets: list[int] = []
    for original_offset, char in enumerate(value):
        replacement = replacements.get(char, char)
        chars.append(replacement)
        offsets.extend([original_offset] * len(replacement))
    return "".join(chars), offsets


def _canonical_tokens(document_text: str) -> list[_Token]:
    bridge_text, offsets = _canonical_bridge_text_with_offsets(document_text)
    tokens: list[_Token] = []
    for match in _TOKEN_RE.finditer(bridge_text):
        start_index = match.start()
        end_index = match.end() - 1
        if start_index >= len(offsets) or end_index >= len(offsets):
            continue
        tokens.append(_Token(match.group(0), offsets[start_index], offsets[end_index] + 1))
    return tokens


def _matches_anchor(token: str, anchor: str) -> bool:
    token_norm = _normalize_token(token)
    anchor_norm = _normalize_token(anchor)
    if token_norm == anchor_norm:
        return True
    if len(anchor_norm) < 2:
        return False
    return token_norm.startswith(anchor_norm) or anchor_norm in token_norm


def _ordered_windows(
    anchors: list[str],
    canonical_tokens: list[_Token],
    *,
    max_gap_tokens: int = 50,
) -> list[tuple[int, int]]:
    if len(anchors) < 2:
        return []
    windows: list[tuple[int, int]] = []
    for start_index, token in enumerate(canonical_tokens):
        if not _matches_anchor(token.text, anchors[0]):
            continue
        current_index = start_index
        matched = 1
        for anchor in anchors[1:]:
            next_index = None
            search_stop = min(len(canonical_tokens), current_index + max_gap_tokens + 1)
            for candidate_index in range(current_index + 1, search_stop):
                if _matches_anchor(canonical_tokens[candidate_index].text, anchor):
                    next_index = candidate_index
                    break
            if next_index is None:
                break
            current_index = next_index
            matched += 1
        if matched == len(anchors):
            windows.append((start_index, current_index))
    return windows


def _line_spans(document_text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    page_marker: int | None = None
    for number, raw_line in enumerate(document_text.splitlines(keepends=True), start=1):
        text = raw_line.rstrip("\n")
        match = _PAGE_MARKER_RE.match(text.strip())
        if match:
            page_marker = int(match.group(1))
        lines.append(_Line(number=number, start=offset, end=offset + len(raw_line), text=text, page_marker=page_marker))
        offset += len(raw_line)
    if document_text and (not lines or lines[-1].end < len(document_text)):
        lines.append(
            _Line(number=len(lines) + 1, start=offset, end=len(document_text), text=document_text[offset:], page_marker=page_marker)
        )
    return lines


def _line_for_char(lines: list[_Line], char_offset: int) -> _Line | None:
    for line in lines:
        if line.start <= char_offset < line.end:
            return line
    return lines[-1] if lines else None


def _equation_numbers(value: str) -> list[str]:
    return list(dict.fromkeys(_EQUATION_NUMBER_RE.findall(value)))


def _window_preview(document_text: str, start: int, end: int) -> str:
    return _clean_text(document_text[max(0, start - 80) : min(len(document_text), end + 80)])


def _window_details(
    *,
    document_text: str,
    canonical_tokens: list[_Token],
    lines: list[_Line],
    windows: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, (start_token, end_token) in enumerate(windows, start=1):
        start = canonical_tokens[start_token].start
        end = canonical_tokens[end_token].end
        line = _line_for_char(lines, start)
        line_text = line.text if line else ""
        context = document_text[max(0, start - 160) : min(len(document_text), end + 160)]
        details.append(
            {
                "window_index": index,
                "line_number": line.number if line else None,
                "page_marker": line.page_marker if line else None,
                "line_equation_numbers": _equation_numbers(line_text),
                "context_equation_numbers": _equation_numbers(context),
                "line_preview": _clean_text(line_text[:500]),
                "context_preview": _window_preview(document_text, start, end),
            }
        )
    return details


def _anchor_status(
    *,
    candidate_text: str,
    normalized_terms: list[str],
    window_details: list[dict[str, Any]],
) -> tuple[str, str, float]:
    if not candidate_text:
        return "empty_equation_text", "none", 1.0
    if len(normalized_terms) < 2:
        return "insufficient_normalized_terms", "none", 0.15
    if not window_details:
        return "no_normalized_windows", "none", 0.1
    if len(window_details) == 1:
        return "unique_line_local_anchor_candidate_only", "single_normalized_window", 0.72
    line_numbered = [
        detail
        for detail in window_details
        if detail.get("line_equation_numbers")
    ]
    if len(line_numbered) == 1:
        return "unique_equation_number_anchor_candidate_only", "single_numbered_window", 0.66
    context_numbered = [
        detail
        for detail in window_details
        if detail.get("context_equation_numbers")
    ]
    if not line_numbered and len(context_numbered) == 1:
        return "unique_equation_number_anchor_candidate_only", "single_context_numbered_window", 0.6
    line_signatures = {
        (
            detail.get("page_marker"),
            detail.get("line_number"),
            tuple(detail.get("line_equation_numbers") or []),
            tuple(detail.get("context_equation_numbers") or []),
        )
        for detail in window_details
    }
    if len(line_signatures) == 1:
        return "ambiguous_same_line_local_anchor_candidate_only", "same_line_or_equation_number", 0.3
    return "ambiguous_line_local_anchor_candidate_only", "multiple_line_or_equation_number_windows", 0.35


def _recommended_action(status: str) -> str:
    if status in {
        "unique_line_local_anchor_candidate_only",
        "unique_equation_number_anchor_candidate_only",
    }:
        return "keep_as_line_local_anchor_candidate_no_source_span"
    if status.startswith("ambiguous"):
        return "requires_stronger_pdf_or_equation_number_pairing_before_source_span"
    if status == "no_normalized_windows":
        return "requires_canonical_equation_rendering_or_pdf_layout_anchor"
    if status == "insufficient_normalized_terms":
        return "requires_non_equation_label_filter_or_manual_design_review"
    if status == "empty_equation_text":
        return "hold_out_empty_equation_environment"
    return "blocked"


def _paper_document_path(parsed_root: Path, paper_id: str) -> Path:
    return parsed_root / paper_id / "document.md"


def _recommended_profile_result(row: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(row.get("recommended_profile") or "")
    for result in list(row.get("profile_results") or []):
        if isinstance(result, dict) and str(result.get("profile_name") or "") == profile_name:
            return dict(result)
    for result in list(row.get("profile_results") or []):
        if isinstance(result, dict):
            return dict(result)
    return {}


def _row(
    index: int,
    design_row: dict[str, Any],
    *,
    parsed_root: Path,
    document_cache: dict[str, str],
) -> dict[str, Any]:
    paper_id = str(design_row.get("paper_id") or "")
    document_path = _paper_document_path(parsed_root, paper_id)
    document_key = str(document_path)
    if document_key not in document_cache:
        document_cache[document_key] = _read_text(document_path)
    document_text = document_cache[document_key]
    canonical_tokens = _canonical_tokens(document_text)
    lines = _line_spans(document_text)
    profile_result = _recommended_profile_result(design_row)
    normalized_terms = [str(item) for item in list(profile_result.get("normalized_terms") or [])]
    windows = _ordered_windows(normalized_terms, canonical_tokens) if document_text else []
    details = _window_details(
        document_text=document_text,
        canonical_tokens=canonical_tokens,
        lines=lines,
        windows=windows,
    )
    candidate_text = _clean_text(design_row.get("candidate_text"))
    status, method, confidence = _anchor_status(
        candidate_text=candidate_text,
        normalized_terms=normalized_terms,
        window_details=details,
    )
    blockers = list(
        dict.fromkeys(
            [
                "tex_equation_line_local_anchor_audit_only",
                "line_local_anchors_do_not_create_source_spans",
                "equation_semantics_not_interpreted",
                "equation_region_link_unverified",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "tex_offsets_are_not_canonical_source_spans",
                "line_local_windows_are_diagnostic_not_provenance",
                *[str(item) for item in list(design_row.get("strict_blockers") or [])],
            ]
        )
    )
    equation_numbers = sorted(
        {
            number
            for detail in details
            for number in list(detail.get("line_equation_numbers") or []) + list(detail.get("context_equation_numbers") or [])
        },
        key=lambda value: int(value) if value.isdigit() else value,
    )
    return {
        "anchor_id": f"tex-equation-line-local-anchor:{index:04d}",
        "source_design_id": str(design_row.get("design_id") or ""),
        "source_diagnostic_id": str(design_row.get("source_diagnostic_id") or ""),
        "source_candidate_id": str(design_row.get("source_candidate_id") or ""),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_line_local_anchor_candidate",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": str(design_row.get("source_file") or ""),
        "equation_environment": str(design_row.get("equation_environment") or ""),
        "candidate_text": candidate_text,
        "recommended_profile": str(profile_result.get("profile_name") or ""),
        "normalized_terms": normalized_terms,
        "normalized_term_count": len(normalized_terms),
        "canonical_document_path": document_key,
        "canonical_document_available": bool(document_text),
        "normalized_window_count": len(details),
        "distinct_line_count": len({detail.get("line_number") for detail in details if detail.get("line_number") is not None}),
        "distinct_page_marker_count": len({detail.get("page_marker") for detail in details if detail.get("page_marker") is not None}),
        "equation_number_candidates": equation_numbers,
        "window_details": details,
        "line_local_anchor_status": status,
        "line_local_anchor_method": method,
        "recommended_action": _recommended_action(status),
        "sourceContentHash": str(design_row.get("sourceContentHash") or ""),
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_line_local_anchor_candidate_only",
        "confidence": confidence,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "line_local_anchor_rows_are_not_evidence",
            "line_local_windows_do_not_create_source_spans",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("line_local_anchor_status") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    return {
        "normalizerDesignRows": len(rows),
        "textBearingEquationRows": sum(1 for row in rows if row.get("candidate_text")),
        "emptyEquationTextRows": sum(1 for row in rows if not row.get("candidate_text")),
        "normalizedWindowRows": sum(1 for row in rows if int(row.get("normalized_window_count") or 0) > 0),
        "uniqueLineLocalAnchorRows": int(by_status.get("unique_line_local_anchor_candidate_only", 0)),
        "uniqueEquationNumberAnchorRows": int(by_status.get("unique_equation_number_anchor_candidate_only", 0)),
        "ambiguousAnchorRows": sum(1 for status, count in by_status.items() if status.startswith("ambiguous") for _ in range(count)),
        "failedAnchorRows": int(by_status.get("no_normalized_windows", 0)),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byLineLocalAnchorStatus": dict(by_status),
    }


def build_tex_equation_line_local_anchor_audit(
    *,
    normalizer_design_report: str | Path = DEFAULT_TEX_EQUATION_NORMALIZER_DESIGN_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    input_path = Path(str(normalizer_design_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    schema = str(payload.get("schema") or "")
    schema_violations = [] if schema == TEX_EQUATION_CANONICAL_TEXT_NORMALIZER_DESIGN_SCHEMA_ID else [
        "tex_equation_canonical_text_normalizer_design_schema_mismatch"
    ]
    source_rows = [
        dict(row)
        for row in list(payload.get("rows") or [])
        if isinstance(row, dict) and (not allowed or str(row.get("paper_id") or "") in allowed)
    ]
    document_cache: dict[str, str] = {}
    rows = [
        _row(index + 1, row, parsed_root=parsed_root_path, document_cache=document_cache)
        for index, row in enumerate(source_rows)
        if not schema_violations
    ]
    counts = _counts(rows, schema_violations)
    return {
        "schema": TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "normalizerDesignReportPath": str(input_path),
            "normalizerDesignReportSchema": schema,
            "parsedRoot": str(parsed_root_path),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "lineLocalAnchorAuditReady": bool(rows) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_line_local_anchor_audited" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "equation_pdf_region_or_tex_pdf_sync_anchor_audit",
        },
        "policy": {
            "reportOnly": True,
            "lineLocalAnchorAuditOnly": True,
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
            "line_local_anchor_windows_do_not_create_source_spans",
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


def render_tex_equation_line_local_anchor_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TeX Equation Line-Local Anchor Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Normalizer design rows: `{int(counts.get('normalizerDesignRows') or 0)}`",
        f"- Text-bearing equation rows: `{int(counts.get('textBearingEquationRows') or 0)}`",
        f"- Normalized-window rows: `{int(counts.get('normalizedWindowRows') or 0)}`",
        f"- Unique line-local anchor rows: `{int(counts.get('uniqueLineLocalAnchorRows') or 0)}`",
        f"- Unique equation-number anchor rows: `{int(counts.get('uniqueEquationNumberAnchorRows') or 0)}`",
        f"- Ambiguous anchor rows: `{int(counts.get('ambiguousAnchorRows') or 0)}`",
        f"- Failed anchor rows: `{int(counts.get('failedAnchorRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Policy",
        "",
        "This report is a line-local anchor audit only. Line-local windows and equation numbers are not source spans or evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By line-local anchor status: `{json.dumps(counts.get('byLineLocalAnchorStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_tex_equation_line_local_anchor_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-line-local-anchor-report.json"
    summary_path = root / "tex-equation-line-local-anchor-summary.json"
    markdown_path = root / "tex-equation-line-local-anchor-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_line_local_anchor_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX equation line-local anchor audit.")
    parser.add_argument("--normalizer-design-report", default=str(DEFAULT_TEX_EQUATION_NORMALIZER_DESIGN_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_line_local_anchor_audit(
        normalizer_design_report=args.normalizer_design_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_line_local_anchor_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_LINE_LOCAL_ANCHOR_AUDIT_SCHEMA_ID",
    "build_tex_equation_line_local_anchor_audit",
    "render_tex_equation_line_local_anchor_audit_markdown",
    "write_tex_equation_line_local_anchor_audit_reports",
]
