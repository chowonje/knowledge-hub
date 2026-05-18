"""Report-only TeX equation normalization bridge audit.

This helper checks whether text-bearing TeX equation diagnostics can be
connected to nearby canonical Markdown equation context after a conservative
token normalization pass. It is still a bridge audit only: it does not create
source spans, interpret equations, route parsers, mutate SQLite, reindex,
reembed, write canonical parsed artifacts, or change answer behavior.
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

from knowledge_hub.papers.tex_equation_canonical_alignment_diagnostic_audit import (
    TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID,
)
from knowledge_hub.papers.tex_structure_candidate_alignment_audit import DEFAULT_PARSED_ROOT


TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.tex-equation-normalization-bridge-audit.v1"
)

DEFAULT_TEX_EQUATION_DIAGNOSTIC_REPORT = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-18"
    / "tex-equation-canonical-alignment-diagnostic-audit"
    / "tex-equation-canonical-alignment-diagnostic-report.json"
)

_WHITESPACE_RE = re.compile(r"\s+")
_GROUP_COMMAND_RE = re.compile(
    r"\\(?:mathrm|mathtt|textrm|textbf|textit|text|operatorname\*?|mathbf|mathit|mathbbm|mathlarger)\s*\{([^{}]*)\}"
)
_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_COMMAND_RE = re.compile(r"\\([A-Za-z]+)\*?")
_TOKEN_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|[0-9]+(?:\.[0-9]+)?")
_STOP_TOKENS = {
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
    "where",
    "label",
    "eq",
    "and",
    "assuming",
    "by",
    "expectation",
    "independence",
    "indepedence",
    "linearity",
    "of",
}


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


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


def _tex_to_bridge_text(value: Any) -> str:
    text = str(value or "")
    previous = ""
    while previous != text:
        previous = text
        text = _GROUP_COMMAND_RE.sub(r" \1 ", text)
    text = _FRAC_RE.sub(r" \1 \2 ", text)
    text = _SQRT_RE.sub(r" sqrt \1 ", text)
    text = _COMMAND_RE.sub(r" \1 ", text)
    text = text.replace("\\\\", " ")
    text = re.sub(r"[{}&_^]", " ", text)
    text = re.sub(r"[\[\](),=+*/<>|:;.-]+", " ", text)
    return _clean_text(text)


def _canonical_bridge_text(value: str) -> str:
    text = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace("×", " x ").replace("∗", " * ")
    text = text.replace("−", " - ").replace("–", " - ").replace("—", " - ")
    text = text.replace("√", " sqrt ")
    return text


def _tokens_with_offsets(value: str) -> list[_Token]:
    return [_Token(match.group(0), match.start(), match.end()) for match in _TOKEN_RE.finditer(value)]


def _normalize_token(token: str) -> str:
    return token.strip("_").casefold()


def _bridge_anchor_terms(candidate_text: str) -> list[str]:
    bridge_text = _tex_to_bridge_text(candidate_text)
    anchors: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(bridge_text):
        normalized = _normalize_token(token)
        if normalized in _STOP_TOKENS or len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        anchors.append(token.strip("_"))
    return anchors[:32]


def _canonical_tokens(document_text: str) -> list[_Token]:
    return _tokens_with_offsets(_canonical_bridge_text(document_text))


def _matches_anchor(token: str, anchor: str) -> bool:
    if token == anchor:
        return True
    if len(anchor) < 2:
        return False
    return token.startswith(anchor) or anchor in token


def _ordered_windows(
    anchors: list[str],
    canonical_tokens: list[_Token],
    *,
    max_gap_tokens: int = 40,
) -> list[tuple[int, int, int]]:
    normalized_anchors = [_normalize_token(anchor) for anchor in anchors]
    normalized_tokens = [_normalize_token(token.text) for token in canonical_tokens]
    windows: list[tuple[int, int, int]] = []
    if len(normalized_anchors) < 2:
        return windows
    first = normalized_anchors[0]
    for start_index, token in enumerate(normalized_tokens):
        if not _matches_anchor(token, first):
            continue
        current_index = start_index
        matched = 1
        for anchor in normalized_anchors[1:]:
            search_stop = min(len(normalized_tokens), current_index + max_gap_tokens + 1)
            next_index = None
            for candidate_index in range(current_index + 1, search_stop):
                if _matches_anchor(normalized_tokens[candidate_index], anchor):
                    next_index = candidate_index
                    break
            if next_index is None:
                break
            current_index = next_index
            matched += 1
        if matched == len(normalized_anchors):
            windows.append((start_index, current_index, matched))
    return windows


def _preview(document_text: str, canonical_tokens: list[_Token], window: tuple[int, int, int] | None) -> str:
    if not window:
        return ""
    start_token, end_token, _ = window
    start = max(0, canonical_tokens[start_token].start - 120)
    end = min(len(document_text), canonical_tokens[end_token].end + 120)
    return _clean_text(document_text[start:end])


def _status(
    *,
    candidate_text: str,
    canonical_available: bool,
    anchors: list[str],
    windows: list[tuple[int, int, int]],
) -> tuple[str, str, float]:
    if not candidate_text:
        return "empty_equation_text", "none", 1.0
    if not canonical_available:
        return "canonical_document_missing", "none", 1.0
    if len(anchors) < 2:
        return "insufficient_bridge_anchor_terms", "none", 0.15
    if len(windows) == 1:
        return "unique_ordered_token_window_candidate_only", "ordered_anchor_token_window", 0.68
    if len(windows) > 1:
        return "ambiguous_ordered_token_window_candidate_only", "ordered_anchor_token_window", 0.35
    return "ordered_token_window_not_found", "ordered_anchor_token_window", 0.1


def _paper_document_path(parsed_root: Path, paper_id: str) -> Path:
    return parsed_root / paper_id / "document.md"


def _row(
    index: int,
    diagnostic_row: dict[str, Any],
    *,
    parsed_root: Path,
    document_cache: dict[str, str],
) -> dict[str, Any]:
    paper_id = str(diagnostic_row.get("paper_id") or "")
    document_path = _paper_document_path(parsed_root, paper_id)
    document_key = str(document_path)
    if document_key not in document_cache:
        document_cache[document_key] = _read_text(document_path)
    document_text = document_cache[document_key]
    candidate_text = _clean_text(diagnostic_row.get("candidate_text"))
    anchors = _bridge_anchor_terms(candidate_text)
    tokens = _canonical_tokens(document_text)
    windows = _ordered_windows(anchors, tokens) if document_text else []
    bridge_status, bridge_method, confidence = _status(
        candidate_text=candidate_text,
        canonical_available=bool(document_text),
        anchors=anchors,
        windows=windows,
    )
    unique_window = windows[0] if len(windows) == 1 else None
    blockers = list(
        dict.fromkeys(
            [
                "tex_equation_normalization_bridge_candidate_only",
                "normalization_bridge_does_not_create_source_spans",
                "equation_semantics_not_interpreted",
                "equation_region_link_unverified",
                "runtime_promotion_disabled_for_tranche",
                "strict_promotion_requires_later_explicit_tranche",
                "canonical_generated_markdown_offsets_are_not_original_pdf_offsets",
                "tex_offsets_are_not_canonical_source_spans",
                "ordered_token_windows_are_diagnostic_not_provenance",
                *[str(item) for item in list(diagnostic_row.get("strict_blockers") or [])],
            ]
        )
    )
    return {
        "bridge_candidate_id": f"tex-equation-normalization-bridge:{index:04d}",
        "source_diagnostic_id": str(diagnostic_row.get("diagnostic_id") or ""),
        "source_candidate_id": str(diagnostic_row.get("source_candidate_id") or ""),
        "paper_id": paper_id,
        "candidate_type": "tex_equation_normalization_bridge_candidate",
        "source_parser": "arxiv_tex+pymupdf_alignment",
        "source_file": str(diagnostic_row.get("source_file") or ""),
        "equation_environment": str(diagnostic_row.get("equation_environment") or ""),
        "candidate_text": candidate_text,
        "bridge_anchor_terms": anchors,
        "bridge_anchor_term_count": len(anchors),
        "canonical_document_path": document_key,
        "canonical_document_available": bool(document_text),
        "bridge_status": bridge_status,
        "bridge_method": bridge_method,
        "bridge_window_count": len(windows),
        "canonical_context_preview": _preview(document_text, tokens, unique_window),
        "sourceContentHash": str(diagnostic_row.get("sourceContentHash") or ""),
        "chars_start": None,
        "chars_end": None,
        "page": None,
        "source_span_created": False,
        "equation_semantics_interpreted": False,
        "equation_region_verified": False,
        "evidence_tier": "tex_equation_normalization_bridge_candidate_only",
        "confidence": confidence,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "normalization_bridge_rows_are_not_evidence",
            "ordered_token_windows_do_not_create_source_spans",
            "later_explicit_promotion_tranche_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    by_status = Counter(str(row.get("bridge_status") or "") for row in rows)
    by_paper = Counter(str(row.get("paper_id") or "") for row in rows)
    return {
        "equationDiagnosticRows": len(rows),
        "textBearingEquationRows": sum(1 for row in rows if row.get("candidate_text")),
        "emptyEquationTextRows": sum(1 for row in rows if not row.get("candidate_text")),
        "bridgeWindowRows": sum(1 for row in rows if int(row.get("bridge_window_count") or 0) > 0),
        "uniqueBridgeWindowRows": int(by_status.get("unique_ordered_token_window_candidate_only", 0)),
        "ambiguousBridgeWindowRows": int(by_status.get("ambiguous_ordered_token_window_candidate_only", 0)),
        "failedBridgeWindowRows": int(by_status.get("ordered_token_window_not_found", 0)),
        "sourceSpanCreatedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(by_paper),
        "byBridgeStatus": dict(by_status),
    }


def build_tex_equation_normalization_bridge_audit(
    *,
    diagnostic_report: str | Path = DEFAULT_TEX_EQUATION_DIAGNOSTIC_REPORT,
    parsed_root: str | Path = DEFAULT_PARSED_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    input_path = Path(str(diagnostic_report)).expanduser()
    parsed_root_path = Path(str(parsed_root)).expanduser()
    payload = _read_json(input_path)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    allowed = set(requested)
    schema = str(payload.get("schema") or "")
    schema_violations = [] if schema == TEX_EQUATION_CANONICAL_ALIGNMENT_DIAGNOSTIC_AUDIT_SCHEMA_ID else [
        "tex_equation_canonical_alignment_diagnostic_report_schema_mismatch"
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
        "schema": TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID,
        "status": "ok" if rows and not schema_violations else "blocked",
        "generatedAt": _now(),
        "input": {
            "diagnosticReportPath": str(input_path),
            "diagnosticReportSchema": schema,
            "parsedRoot": str(parsed_root_path),
            "paperIds": requested,
        },
        "counts": counts,
        "gate": {
            "bridgeAuditReady": bool(rows) and not schema_violations,
            "sourceSpanCreationReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "equation_normalization_bridge_audited" if rows and not schema_violations else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "tex_equation_canonical_text_normalizer_design",
        },
        "policy": {
            "reportOnly": True,
            "bridgeAuditOnly": True,
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
            "normalization_bridge_windows_do_not_create_source_spans",
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


def render_tex_equation_normalization_bridge_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# TeX Equation Normalization Bridge Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Equation diagnostic rows: `{int(counts.get('equationDiagnosticRows') or 0)}`",
        f"- Text-bearing equation rows: `{int(counts.get('textBearingEquationRows') or 0)}`",
        f"- Bridge window rows: `{int(counts.get('bridgeWindowRows') or 0)}`",
        f"- Unique bridge window rows: `{int(counts.get('uniqueBridgeWindowRows') or 0)}`",
        f"- Ambiguous bridge window rows: `{int(counts.get('ambiguousBridgeWindowRows') or 0)}`",
        f"- Failed bridge window rows: `{int(counts.get('failedBridgeWindowRows') or 0)}`",
        f"- Source spans created: `{int(counts.get('sourceSpanCreatedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Policy",
        "",
        "This audit is a normalization bridge only. Ordered token windows are diagnostic context, not source spans or evidence.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By bridge status: `{json.dumps(counts.get('byBridgeStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_tex_equation_normalization_bridge_audit_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "tex-equation-normalization-bridge-report.json"
    summary_path = root / "tex-equation-normalization-bridge-summary.json"
    markdown_path = root / "tex-equation-normalization-bridge-audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_tex_equation_normalization_bridge_audit_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only TeX equation normalization bridge audit.")
    parser.add_argument("--diagnostic-report", default=str(DEFAULT_TEX_EQUATION_DIAGNOSTIC_REPORT))
    parser.add_argument("--parsed-root", default=str(DEFAULT_PARSED_ROOT))
    parser.add_argument("--paper-id", action="append", default=[], help="Filter to a paper id; can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_tex_equation_normalization_bridge_audit(
        diagnostic_report=args.diagnostic_report,
        parsed_root=args.parsed_root,
        paper_ids=args.paper_id or None,
    )
    paths = write_tex_equation_normalization_bridge_audit_reports(report, args.output_dir)
    summary = _summary_payload(report)
    summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEX_EQUATION_NORMALIZATION_BRIDGE_AUDIT_SCHEMA_ID",
    "build_tex_equation_normalization_bridge_audit",
    "render_tex_equation_normalization_bridge_audit_markdown",
    "write_tex_equation_normalization_bridge_audit_reports",
]
