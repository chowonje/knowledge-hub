"""Report-only SectionSpanCandidate audit helpers.

This module converts human-reviewed MinerU/PyMuPDF section review cards into a
formal candidate layer.  It is deliberately not a runtime evidence artifact:
the output remains non-strict, is not citation-grade, and is not consumed by
answer generation or parser routing.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID = "knowledge-hub.paper.sectionspan-candidate-report.v1"

_APPROVED_CLASSES = {"numbered_section", "abstract", "backmatter", "named_section"}
_HELD_OUT_CLASSES = {"paper_title", "toc"}
_SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.)?\s+(.+?)\s*$")


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


def _section_parts(title: str, section_type: str) -> tuple[str, str, int]:
    text = _clean_text(title)
    match = _SECTION_NUMBER_RE.match(text)
    if match:
        label = match.group(1)
        section_title = _clean_text(match.group(2))
        level = len(label.split("."))
        return label, section_title, level
    if section_type == "abstract":
        return "", text or "Abstract", 0
    if section_type == "backmatter":
        return "", text, 0
    return "", text, 1


def _hold_reason(card: dict[str, Any]) -> str | None:
    review_class = str(card.get("reviewClass") or "")
    if review_class in _HELD_OUT_CLASSES:
        return f"held_out_{review_class}"
    if review_class not in _APPROVED_CLASSES:
        return "review_class_not_approved"
    if str(card.get("alignmentMethod") or "") != "exact":
        return "non_exact_alignment"
    if card.get("page") is None:
        return "missing_page"
    if card.get("charsStart") is None or card.get("charsEnd") is None:
        return "missing_chars_start_end"
    if not str(card.get("sourceContentHash") or "").strip():
        return "missing_source_content_hash"
    blockers = set(str(item) for item in list(card.get("strictBlockers") or []))
    if "runtime_promotion_disabled_for_tranche" not in blockers:
        return "runtime_promotion_blocker_missing"
    return None


def _candidate(index: int, card: dict[str, Any]) -> dict[str, Any]:
    section_type = str(card.get("reviewClass") or "")
    label, title, level = _section_parts(str(card.get("candidateText") or ""), section_type)
    strict_blockers = list(dict.fromkeys(
        [
            "runtime_promotion_disabled_for_tranche",
            "sectionspan_candidate_layer_not_runtime_evidence",
            "markdown_offsets_are_generated_not_original_pdf_offsets",
            *[str(item) for item in list(card.get("strictBlockers") or [])],
        ]
    ))
    chars_start = _safe_int(card.get("charsStart"))
    chars_end = _safe_int(card.get("charsEnd"))
    page = _safe_int(card.get("page"))
    return {
        "candidate_id": f"sectionspan:{card.get('paperId')}:{index:04d}",
        "candidate_type": "section_span_candidate",
        "source_candidate_id": str(card.get("candidateId") or ""),
        "paper_id": str(card.get("paperId") or ""),
        "source_parser": "mineru+pymupdf_alignment",
        "candidate_text": _clean_text(card.get("candidateText")),
        "section_label": label,
        "section_title": title,
        "section_type": section_type,
        "section_level": level,
        "canonical_alignment_status": "aligned",
        "alignment_method": str(card.get("alignmentMethod") or ""),
        "chars_start": chars_start,
        "chars_end": chars_end,
        "page": page,
        "sourceContentHash": str(card.get("sourceContentHash") or ""),
        "confidence": 0.99 if str(card.get("alignmentMethod") or "") == "exact" else 0.0,
        "source_span_locator": {
            "path": "document.md",
            "locatorKind": "canonical_generated_markdown",
            "chars": {"start": chars_start, "end": chars_end},
        },
        "review": {
            "sourceReviewCardId": str(card.get("cardId") or ""),
            "recommendedDecision": str(card.get("recommendedDecision") or ""),
            "humanReviewQuestions": list(card.get("humanReviewQuestions") or []),
        },
        "evidence_tier": "sectionspan_candidate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": strict_blockers,
    }


def _held_out_card(card: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "sourceReviewCardId": str(card.get("cardId") or ""),
        "sourceCandidateId": str(card.get("candidateId") or ""),
        "paperId": str(card.get("paperId") or ""),
        "candidateText": _clean_text(card.get("candidateText")),
        "reviewClass": str(card.get("reviewClass") or ""),
        "reason": reason,
        "strictEligible": False,
        "citationGrade": False,
    }


def _counts(candidates: list[dict[str, Any]], held_out: list[dict[str, Any]], *, input_cards: int) -> dict[str, Any]:
    return {
        "inputReviewCards": input_cards,
        "sectionSpanCandidates": len(candidates),
        "heldOutCandidates": len(held_out),
        "strictEligibleCandidates": sum(1 for item in candidates if item.get("strict_eligible")),
        "citationGradeCandidates": sum(1 for item in candidates if item.get("citation_grade")),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in candidates)),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in candidates)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_sectionspan_candidate_report(
    decision_review_path: str | Path,
    *,
    schema_design_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only SectionSpanCandidate payload from local decision cards."""

    review_path = Path(str(decision_review_path)).expanduser()
    review = _read_json(review_path)
    design_path = Path(str(schema_design_path)).expanduser() if schema_design_path else None
    design = _read_json(design_path) if design_path is not None else {}
    raw_cards = [dict(item) for item in list(review.get("cards") or []) if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    held_out: list[dict[str, Any]] = []
    for card in raw_cards:
        reason = _hold_reason(card)
        if reason:
            held_out.append(_held_out_card(card, reason))
            continue
        candidates.append(_candidate(len(candidates) + 1, card))

    counts = _counts(candidates, held_out, input_cards=len(raw_cards))
    return {
        "schema": SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID,
        "status": "ok" if candidates else "empty",
        "generatedAt": _now(),
        "input": {
            "decisionReviewPath": str(review_path),
            "schemaDesignPath": str(design_path) if design_path is not None else "",
            "decisionReviewSchema": (review.get("summary") or {}).get("schema") or review.get("schema"),
            "schemaDesignStatus": str(design.get("status") or ""),
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
        },
        "promotionRules": [
            "emit_only_review_approved_section_boundaries",
            "exclude_paper_title_and_toc",
            "require_exact_alignment_page_chars_and_source_hash",
            "keep_sectionspan_candidates_non_strict",
        ],
        "warnings": [
            "sectionspan_candidates_are_not_runtime_evidence",
            "canonical_generated_markdown_offsets_are_not_original_pdf_byte_offsets",
            "source_hash_page_and_chars_do_not_imply_strict_eligibility",
        ],
        "candidates": candidates,
        "heldOut": held_out,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "input", "counts", "policy", "promotionRules", "warnings", "heldOut")
        if key in report
    }


def render_sectionspan_candidate_report_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# SectionSpanCandidate Audit",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Input review cards: `{int(counts.get('inputReviewCards') or 0)}`",
        f"- SectionSpan candidates: `{int(counts.get('sectionSpanCandidates') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutCandidates') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleCandidates') or 0)}`",
        f"- Citation-grade: `{int(counts.get('citationGradeCandidates') or 0)}`",
        "",
        "## Evidence Tier",
        "",
        "All rows are `sectionspan_candidate_only`. They are not strict evidence and are not runtime citations.",
        "",
        "## Counts",
        "",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Held out by reason: `{json.dumps(counts.get('heldOutByReason') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Candidates",
        "",
    ]
    for item in list(report.get("candidates") or []):
        lines.append(
            f"- `{item.get('paper_id')}` page `{item.get('page')}` `{item.get('section_type')}` "
            f"`{item.get('section_label')}` {item.get('section_title')}"
        )
    lines.extend(["", "## Held Out", ""])
    for item in list(report.get("heldOut") or []):
        lines.append(
            f"- `{item.get('paperId')}` `{item.get('reviewClass')}` {item.get('candidateText')} -> `{item.get('reason')}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_sectionspan_candidate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    candidates_path = root / "sectionspan-candidates.json"
    summary_path = root / "sectionspan-candidate-summary.json"
    markdown_path = root / "sectionspan-candidate-audit.md"
    candidates_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_candidate_report_markdown(report), encoding="utf-8")
    return {
        "candidates": str(candidates_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpanCandidate audit.")
    parser.add_argument("--decision-review", required=True, help="Path to sectionspan-decision-review.json.")
    parser.add_argument("--schema-design", default="", help="Optional path to sectionspan-schema-design.json.")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_candidate_report(
        args.decision_review,
        schema_design_path=args.schema_design or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_candidate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_CANDIDATE_REPORT_SCHEMA_ID",
    "build_sectionspan_candidate_report",
    "render_sectionspan_candidate_report_markdown",
    "write_sectionspan_candidate_reports",
]
