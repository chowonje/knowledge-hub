"""Report-only EquationQuoteCandidate audit helpers.

This module projects MinerU/PyMuPDF source-alignment rows into a formal
candidate layer for equation quotes.  It does not interpret equations and does
not create runtime evidence.  Equation text must align to canonical generated
markdown before it can even become a useful quote candidate; layout/bbox alone
remains non-strict.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.equation-quote-candidate-report.v1"

_TAG_RE = re.compile(r"\\tag\s*\{\s*([^}]+?)\s*\}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    coords: list[float] = []
    for item in value[:4]:
        number = _safe_float(item)
        if number is None:
            return None
        coords.append(number)
    return coords


def _equation_label(text: str) -> str:
    match = _TAG_RE.search(str(text or ""))
    if not match:
        return ""
    return f"tag:{_clean_text(match.group(1))}"


def _equation_rows(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in list(source_report.get("candidates") or []):
        if isinstance(item, dict) and str(item.get("candidate_type") or "") == "equation_candidate":
            rows.append(dict(item))
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int(item.get("page")) or 0,
            _safe_int(item.get("chars_start")) or 0,
            str(item.get("candidate_id") or ""),
        )
    )
    return rows


def _strict_blockers(row: dict[str, Any], *, has_layout_region: bool) -> list[str]:
    blockers = list(dict.fromkeys(str(item) for item in list(row.get("strict_blockers") or [])))
    required = [
        "runtime_promotion_disabled_for_tranche",
        "equation_alignment_missing",
        "equation_semantics_not_interpreted",
        "markdown_offsets_are_generated_not_original_pdf_offsets",
    ]
    alignment_status = str(row.get("alignment_status") or "")
    alignment_method = str(row.get("alignment_method") or "")
    if alignment_status != "aligned":
        required.append("equation_text_alignment_not_available")
    if alignment_method not in {"exact"}:
        required.append("equation_alignment_not_exact")
    if row.get("chars_start") is None or row.get("chars_end") is None:
        required.append("missing_chars_start_end")
    if row.get("page") is None:
        required.append("missing_page")
    if not str(row.get("sourceContentHash") or "").strip():
        required.append("missing_source_content_hash")
    if not has_layout_region:
        required.append("missing_layout_region_candidate")
    return list(dict.fromkeys([*blockers, *required]))


def _readiness(row: dict[str, Any], blockers: list[str], *, has_layout_region: bool) -> str:
    if str(row.get("alignment_status") or "") != "aligned":
        return "blocked_alignment_incomplete"
    if row.get("chars_start") is None or row.get("chars_end") is None:
        return "blocked_missing_source_span"
    if row.get("page") is None:
        return "blocked_missing_page"
    if not str(row.get("sourceContentHash") or "").strip():
        return "blocked_missing_source_hash"
    if not has_layout_region:
        return "quote_span_aligned_region_missing_non_strict"
    if "equation_alignment_not_exact" in blockers:
        return "quote_span_aligned_fuzzy_non_strict"
    return "quote_span_aligned_region_candidate_non_strict"


def _candidate(index: int, row: dict[str, Any]) -> dict[str, Any]:
    mineru = dict(row.get("mineruCandidate") or {})
    layout_ids = [str(item) for item in list(mineru.get("layout_element_ids") or []) if str(item)]
    bbox = _bbox(mineru.get("bbox"))
    has_layout_region = bool(layout_ids and bbox)
    blockers = _strict_blockers(row, has_layout_region=has_layout_region)
    chars_start = _safe_int(row.get("chars_start"))
    chars_end = _safe_int(row.get("chars_end"))
    page = _safe_int(row.get("page"))
    text = _clean_text(row.get("candidate_text"))
    return {
        "candidate_id": f"equationquote:{row.get('paper_id')}:{index:04d}",
        "candidate_type": "equation_quote_candidate",
        "source_candidate_id": str(row.get("candidate_id") or ""),
        "source_candidate_type": "equation_candidate",
        "paper_id": str(row.get("paper_id") or ""),
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": text,
        "equation_text": text,
        "equation_label": _equation_label(text),
        "canonical_alignment_status": str(row.get("alignment_status") or ""),
        "alignment_method": str(row.get("alignment_method") or ""),
        "alignment_reason": str(row.get("alignment_reason") or ""),
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "sourceContentHash": str(row.get("sourceContentHash") or ""),
        "sourceContentHashSource": str(row.get("sourceContentHashSource") or ""),
        "confidence": float(row.get("confidence") or 0.0),
        "source_span_locator": dict(row.get("source_span_locator") or {}),
        "layout_element_ids": layout_ids,
        "bbox": bbox,
        "layout_link_reason": str(mineru.get("link_reason") or ""),
        "layout_region_candidate_present": has_layout_region,
        "equation_region_link_verified": False,
        "equation_semantics_interpreted": False,
        "equation_region_link_status": (
            "layout_region_candidate_only_unverified"
            if has_layout_region
            else "missing_layout_region_candidate"
        ),
        "readiness": _readiness(row, blockers, has_layout_region=has_layout_region),
        "source_classification": str(row.get("classification") or ""),
        "evidence_tier": "equation_quote_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": blockers,
        "non_strict_reason": blockers,
    }


def _counts(candidates: list[dict[str, Any]], *, input_candidate_count: int) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in candidates:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    return {
        "inputCandidateCount": input_candidate_count,
        "equationQuoteCandidates": len(candidates),
        "alignedEquationQuoteCandidates": sum(
            1
            for item in candidates
            if item.get("canonical_alignment_status") == "aligned"
            and item.get("chars_start") is not None
            and item.get("chars_end") is not None
            and item.get("page") is not None
            and bool(item.get("sourceContentHash"))
        ),
        "layoutRegionCandidateCount": sum(1 for item in candidates if item.get("layout_region_candidate_present")),
        "equationRegionVerifiedCandidates": 0,
        "equationSemanticsInterpretedCandidates": 0,
        "strictEligibleCandidates": 0,
        "citationGradeCandidates": 0,
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in candidates)),
        "byAlignmentStatus": dict(Counter(str(item.get("canonical_alignment_status") or "") for item in candidates)),
        "byAlignmentMethod": dict(Counter(str(item.get("alignment_method") or "") for item in candidates)),
        "byReadiness": dict(Counter(str(item.get("readiness") or "") for item in candidates)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_equation_quote_candidate_report(source_alignment_report_path: str | Path) -> dict[str, Any]:
    """Build a report-only EquationQuoteCandidate payload from source alignment rows."""

    report_path = Path(str(source_alignment_report_path)).expanduser()
    source_report = _read_json(report_path)
    source_rows = [dict(item) for item in list(source_report.get("candidates") or []) if isinstance(item, dict)]
    equation_rows = _equation_rows(source_report)
    candidates = [_candidate(index + 1, row) for index, row in enumerate(equation_rows)]
    counts = _counts(candidates, input_candidate_count=len(source_rows))
    return {
        "schema": EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID,
        "status": "ok" if candidates else "empty",
        "generatedAt": _now(),
        "input": {
            "sourceAlignmentReportPath": str(report_path),
            "sourceAlignmentSchema": str(source_report.get("schema") or ""),
        },
        "counts": counts,
        "policy": {
            "allCandidatesNonStrict": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "equationInterpretationAllowed": False,
        },
        "promotionRules": [
            "emit_only_equation_quote_candidates",
            "preserve_canonical_equation_quote_span_when_available",
            "preserve_mineru_layout_bbox_as_candidate_only",
            "do_not_interpret_equations",
            "keep_equation_quote_candidates_non_strict",
        ],
        "warnings": [
            "equation_quote_candidates_are_not_runtime_evidence",
            "equation_layout_bbox_does_not_create_a_source_quote_span",
            "equation_semantics_are_not_interpreted",
            "generated_markdown_offsets_are_not_original_pdf_byte_offsets",
        ],
        "candidates": candidates,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "policy", "promotionRules", "warnings")
        if key in report
    }


def render_equation_quote_candidate_report_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# EquationQuoteCandidate Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Equation quote candidates: `{int(counts.get('equationQuoteCandidates') or 0)}`",
        f"- Aligned equation quote spans: `{int(counts.get('alignedEquationQuoteCandidates') or 0)}`",
        f"- Layout region candidates: `{int(counts.get('layoutRegionCandidateCount') or 0)}`",
        f"- Verified equation regions: `{int(counts.get('equationRegionVerifiedCandidates') or 0)}`",
        f"- Equation semantics interpreted: `{int(counts.get('equationSemanticsInterpretedCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Citation-grade: `{int(counts.get('citationGradeCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All rows are `equation_quote_candidate_only`. They are not strict evidence and are not runtime citations.",
        "No equation is interpreted; the report only tracks quote/location feasibility.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By alignment status: `{json.dumps(counts.get('byAlignmentStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By readiness: `{json.dumps(counts.get('byReadiness') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Candidates",
        "",
    ]
    for item in list(report.get("candidates") or []):
        lines.append(
            f"- `{item.get('paper_id')}` page `{item.get('page')}` `{item.get('equation_label')}` "
            f"{item.get('equation_text')} -> `{item.get('readiness')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_equation_quote_candidate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "equation-quote-candidates.json"
    summary_path = root / "equation-quote-candidate-summary.json"
    markdown_path = root / "equation-quote-candidate-audit.md"
    candidates_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_equation_quote_candidate_report_markdown(report), encoding="utf-8")
    return {
        "candidates": str(candidates_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only EquationQuoteCandidate audit.")
    parser.add_argument("--source-alignment-report", required=True, help="Path to mineru-source-alignment-report.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_equation_quote_candidate_report(args.source_alignment_report)
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_equation_quote_candidate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EQUATION_QUOTE_CANDIDATE_REPORT_SCHEMA_ID",
    "build_equation_quote_candidate_report",
    "render_equation_quote_candidate_report_markdown",
    "write_equation_quote_candidate_reports",
]
