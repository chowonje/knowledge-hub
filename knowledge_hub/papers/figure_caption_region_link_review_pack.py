"""Report-only FigureCaption region-link review pack helpers.

This module joins recovered original-PDF caption offsets with unverified MinerU
layout-region candidates.  The output is a human/operator review pack only:
caption source spans do not verify the figure/image region, and no row becomes
strict evidence or runtime citation material in this tranche.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.figure-caption-region-link-review-pack.v1"
)
FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.figure-caption-pdf-offset-feasibility.v1"
)
FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.figure-region-link-feasibility-audit.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _pdf_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in list(report.get("feasibilityRows") or []) if isinstance(item, dict)]
    rows.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int((item.get("original_pdf_span") or {}).get("page") or item.get("page")),
            str(item.get("source_figure_caption_candidate_id") or ""),
        )
    )
    return rows


def _region_rows_by_candidate_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in list(report.get("rows") or []):
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("figure_caption_candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = dict(item)
    return rows


def _schema_violations(pdf_report: dict[str, Any], region_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if pdf_report.get("schema") != FIGURE_CAPTION_PDF_OFFSET_FEASIBILITY_SCHEMA_ID:
        violations.append("figure_caption_pdf_offset_feasibility_schema_mismatch")
    if pdf_report.get("status") != "feasibility_complete":
        violations.append("figure_caption_pdf_offset_feasibility_not_complete")
    if region_report.get("schema") != FIGURE_REGION_LINK_FEASIBILITY_AUDIT_SCHEMA_ID:
        violations.append("figure_region_link_feasibility_schema_mismatch")
    if region_report.get("status") != "ok":
        violations.append("figure_region_link_feasibility_not_ok")
    return violations


def _review_status(pdf_row: dict[str, Any], region_row: dict[str, Any] | None) -> str:
    if not pdf_row.get("original_pdf_offset_recovered"):
        return "held_out_original_pdf_offset_missing"
    if region_row is None:
        return "held_out_region_candidate_missing"
    if not bool(region_row.get("layout_region_candidate_present")):
        return "held_out_layout_region_missing"
    return "ready_for_region_link_review"


def _strict_blockers(pdf_row: dict[str, Any], region_row: dict[str, Any] | None, status: str) -> list[str]:
    blockers = [
        *[str(value) for value in list(pdf_row.get("strict_blockers") or []) if str(value)],
        *[str(value) for value in list((region_row or {}).get("strict_blockers") or []) if str(value)],
        "region_link_review_pack_only",
        "figure_region_link_incomplete",
        "figure_region_type_unverified",
        "figure_region_page_missing",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_explicit_later_tranche",
    ]
    if status != "ready_for_region_link_review":
        blockers.append(status)
    return list(dict.fromkeys(blockers))


def _card(index: int, pdf_row: dict[str, Any], region_row: dict[str, Any] | None) -> dict[str, Any]:
    status = _review_status(pdf_row, region_row)
    original_pdf_span = dict(pdf_row.get("original_pdf_span") or {})
    blockers = _strict_blockers(pdf_row, region_row, status)
    return {
        "review_card_id": f"figure-caption-region-link-review:{index:04d}",
        "source_figure_caption_candidate_id": str(pdf_row.get("source_figure_caption_candidate_id") or ""),
        "source_region_audit_id": str((region_row or {}).get("audit_id") or ""),
        "paper_id": str(pdf_row.get("paper_id") or ""),
        "candidate_type": "figure_caption_region_link_review_card",
        "source_parser": "mineru+pymupdf_alignment",
        "figure_label": _clean_text(pdf_row.get("figure_label")),
        "candidate_text": _clean_text(pdf_row.get("candidate_text")),
        "caption_text": _clean_text(pdf_row.get("caption_text")),
        "original_pdf_offset_recovered": bool(pdf_row.get("original_pdf_offset_recovered")),
        "original_pdf_span": {
            "originalPdfCharsStart": original_pdf_span.get("originalPdfCharsStart"),
            "originalPdfCharsEnd": original_pdf_span.get("originalPdfCharsEnd"),
            "page": original_pdf_span.get("page"),
            "sourceContentHash": str(original_pdf_span.get("sourceContentHash") or ""),
            "matchMethod": str(original_pdf_span.get("matchMethod") or ""),
            "matchConfidence": original_pdf_span.get("matchConfidence"),
        },
        "page_agrees_with_canonical": bool(pdf_row.get("page_agrees_with_canonical")),
        "source_hash_agrees_with_canonical": bool(pdf_row.get("source_hash_agrees_with_canonical")),
        "layout_region_candidate_present": bool((region_row or {}).get("layout_region_candidate_present")),
        "layout_element_ids": [str(value) for value in list((region_row or {}).get("layout_element_ids") or [])],
        "layout_element_count": _safe_int((region_row or {}).get("layout_element_count")),
        "bbox": (region_row or {}).get("bbox"),
        "layout_link_reason": str((region_row or {}).get("layout_link_reason") or ""),
        "normalizer_candidate_id": str((region_row or {}).get("normalizer_candidate_id") or ""),
        "normalizer_region_page": (region_row or {}).get("normalizer_region_page"),
        "region_page_recovered": bool((region_row or {}).get("region_page_recovered")),
        "caption_page_matches_region_page": bool((region_row or {}).get("caption_page_matches_region_page")),
        "figure_region_type_verified": False,
        "figure_region_link_verified": False,
        "review_status": status,
        "recommended_review_action": (
            "review_caption_source_span_against_layout_bbox_and_recover_region_page"
            if status == "ready_for_region_link_review"
            else "hold_until_caption_offset_and_layout_region_are_available"
        ),
        "evidence_tier": "figure_caption_region_link_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "review_cards_are_not_evidence",
            "caption_source_span_does_not_verify_figure_image_region",
            "figure_region_page_and_type_remain_unverified",
            "no_runtime_or_strict_evidence_created",
        ],
    }


def _counts(cards: list[dict[str, Any]], schema_violations: list[str]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    for item in cards:
        blocker_counts.update(str(blocker) for blocker in list(item.get("strict_blockers") or []))
    ready = [item for item in cards if item.get("review_status") == "ready_for_region_link_review"]
    return {
        "reviewCardRows": len(cards),
        "readyForRegionReviewRows": len(ready),
        "heldOutRows": len(cards) - len(ready),
        "originalPdfOffsetRecoveredRows": sum(1 for item in cards if item.get("original_pdf_offset_recovered")),
        "layoutRegionCandidateRows": sum(1 for item in cards if item.get("layout_region_candidate_present")),
        "regionPageRecoveredRows": sum(1 for item in cards if item.get("region_page_recovered")),
        "figureRegionTypeVerifiedRows": 0,
        "figureRegionLinkVerifiedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len(schema_violations),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "byReviewStatus": dict(Counter(str(item.get("review_status") or "") for item in cards)),
        "byMatchMethod": dict(Counter(str((item.get("original_pdf_span") or {}).get("matchMethod") or "") for item in cards)),
        "strictBlockerSummary": dict(blocker_counts),
    }


def build_figure_caption_region_link_review_pack(
    *,
    figure_caption_pdf_offset_feasibility_report: str | Path,
    figure_region_link_feasibility_report: str | Path,
) -> dict[str, Any]:
    """Build report-only FigureCaption region-link review cards."""

    pdf_path = Path(str(figure_caption_pdf_offset_feasibility_report)).expanduser()
    region_path = Path(str(figure_region_link_feasibility_report)).expanduser()
    pdf_report = _read_json(pdf_path)
    region_report = _read_json(region_path)
    schema_violations = _schema_violations(pdf_report, region_report)
    region_by_id = _region_rows_by_candidate_id(region_report)
    cards = [
        _card(index + 1, row, region_by_id.get(str(row.get("source_figure_caption_candidate_id") or "")))
        for index, row in enumerate(_pdf_rows(pdf_report))
    ]
    counts = _counts(cards, schema_violations)
    blocked = bool(schema_violations) or not cards
    return {
        "schema": FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID,
        "status": "blocked" if blocked else "review_pack_ready",
        "generatedAt": _now(),
        "inputs": {
            "figureCaptionPdfOffsetFeasibilityReport": str(pdf_path),
            "figureRegionLinkFeasibilityReport": str(region_path),
            "figureCaptionPdfOffsetFeasibilitySchema": str(pdf_report.get("schema") or ""),
            "figureRegionLinkFeasibilitySchema": str(region_report.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "reviewPackReady": not blocked,
            "figureRegionCitationGradeReady": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "ready_for_region_link_human_review" if not blocked else "blocked",
            "schemaViolations": schema_violations,
            "recommendedNextTranche": "figure_caption_region_page_and_type_review",
        },
        "policy": {
            "reportOnly": True,
            "reviewOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "review_cards_are_not_runtime_evidence",
            "caption_source_spans_do_not_verify_figure_image_regions",
            "figure_region_page_and_type_require_human_or_layout_authority_review",
            "no_strict_evidence_or_parser_routing_is_created",
        ],
        "reviewCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings", "reviewCards")
        if key in report
    }


def render_figure_caption_region_link_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# FigureCaption Region Link Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardRows') or 0)}`",
        f"- Ready for region review: `{int(counts.get('readyForRegionReviewRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Figure-region links verified: `{int(counts.get('figureRegionLinkVerifiedRows') or 0)}`",
        f"- Strict eligible: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These cards are for human/operator review only. A recovered caption source span does not verify the visual figure/image region, and no row is strict evidence or runtime citation material.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By review status: `{json.dumps(counts.get('byReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By match method: `{json.dumps(counts.get('byMatchMethod') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_figure_caption_region_link_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "figure-caption-region-link-review-cards.json"
    summary_path = root / "figure-caption-region-link-review-summary.json"
    markdown_path = root / "figure-caption-region-link-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_figure_caption_region_link_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only FigureCaption region-link review cards.")
    parser.add_argument("--figure-caption-pdf-offset-feasibility-report", required=True)
    parser.add_argument("--figure-region-link-feasibility-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_figure_caption_region_link_review_pack(
        figure_caption_pdf_offset_feasibility_report=args.figure_caption_pdf_offset_feasibility_report,
        figure_region_link_feasibility_report=args.figure_region_link_feasibility_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_figure_caption_region_link_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIGURE_CAPTION_REGION_LINK_REVIEW_PACK_SCHEMA_ID",
    "build_figure_caption_region_link_review_pack",
    "render_figure_caption_region_link_review_pack_markdown",
    "write_figure_caption_region_link_review_pack_reports",
]
