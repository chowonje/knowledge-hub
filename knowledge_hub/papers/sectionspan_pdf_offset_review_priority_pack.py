"""Report-only priority pack for SectionSpan PDF offset review decisions."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-priority-pack.v1"
)
SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_RECORD_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-decision-record.v1"
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


def _unsafe_flags(record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(record.get("counts") or {})
    gate = dict(record.get("gate") or {})
    policy = dict(record.get("policy") or {})
    if record.get("schema") != SECTIONSPAN_PDF_OFFSET_REVIEW_DECISION_RECORD_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_review_decision_record_schema_mismatch")
    if record.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_review_decision_record_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"decisionRecord_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"decisionRecord_{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"decisionRecord_{key}_true")
    return list(dict.fromkeys(flags))


def _priority(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    section_type = str(row.get("section_type") or "")
    section_level = _safe_int(row.get("section_level"))
    match_method = str((row.get("original_pdf_span") or {}).get("matchMethod") or "")
    if section_type == "abstract":
        reasons.append("abstract_boundary_is_high_value")
        return "P0", reasons
    if section_type == "numbered_section" and section_level <= 1:
        reasons.append("top_level_numbered_section_boundary")
        if match_method == "exact":
            reasons.append("exact_original_pdf_offset_match")
        return "P0", reasons
    if section_type == "numbered_section":
        reasons.append("numbered_section_boundary")
        if match_method != "exact":
            reasons.append("normalized_match_needs_spot_check")
        return "P1", reasons
    if section_type == "backmatter":
        reasons.append("backmatter_boundary_lower_runtime_value")
        return "P2", reasons
    reasons.append("unknown_section_type")
    return "P2", reasons


def _sort_key(row: dict[str, Any]) -> tuple[int, str, int, str]:
    priority, _ = _priority(row)
    rank = {"P0": 0, "P1": 1, "P2": 2}.get(priority, 3)
    return (rank, str(row.get("paper_id") or ""), _safe_int(row.get("section_level")), str(row.get("candidate_text") or ""))


def _select_initial_review_rows(rows: list[dict[str, Any]], max_initial_cards: int) -> set[str]:
    if max_initial_cards <= 0:
        return set()
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        row_id = str(row.get("record_row_id") or "")
        if row_id and row_id not in seen and len(selected) < max_initial_cards:
            selected.append(row)
            seen.add(row_id)

    for row in sorted(rows, key=_sort_key):
        if str(row.get("section_type") or "") == "abstract":
            add(row)

    for paper_id in sorted(set(str(row.get("paper_id") or "") for row in rows)):
        for row in sorted([item for item in rows if str(item.get("paper_id") or "") == paper_id], key=_sort_key):
            if str(row.get("section_type") or "") == "numbered_section" and _safe_int(row.get("section_level")) <= 1:
                add(row)
                break

    for paper_id in sorted(set(str(row.get("paper_id") or "") for row in rows)):
        for row in sorted([item for item in rows if str(item.get("paper_id") or "") == paper_id], key=_sort_key):
            if str((row.get("original_pdf_span") or {}).get("matchMethod") or "") != "exact":
                add(row)
                break

    for row in sorted(rows, key=_sort_key):
        add(row)
    return seen


def _card(index: int, row: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    priority, reasons = _priority(row)
    row_id = str(row.get("record_row_id") or "")
    return {
        "priority_card_id": f"sectionspan-pdf-offset-review-priority-card:{index:04d}",
        "source_decision_record_row_id": row_id,
        "source_decision_row_id": str(row.get("source_decision_row_id") or ""),
        "source_gate_row_id": str(row.get("source_gate_row_id") or ""),
        "source_review_card_id": str(row.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "recorded_decision": str(row.get("recorded_decision") or ""),
        "review_priority": priority,
        "selected_for_initial_review": row_id in selected_ids,
        "priority_reasons": reasons,
        "canonical_span": dict(row.get("canonical_span") or {}),
        "original_pdf_span": dict(row.get("original_pdf_span") or {}),
        "page_agreement": bool(row.get("page_agreement")),
        "source_hash_agreement": bool(row.get("source_hash_agreement")),
        "review_checklist": [
            "verify_heading_text_on_original_pdf_page",
            "verify_original_pdf_offset_points_to_the_heading_text",
            "verify_boundary_is_a_real_section_boundary_not_toc_or_running_header",
            "record_reject_if_heading_or_boundary_is_ambiguous",
        ],
        "evidence_tier": "sectionspan_pdf_offset_review_priority_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "priority_pack_only",
            "manual_review_decision_still_required",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "priority_cards_are_review_work_items_not_evidence",
            "priority_cards_do_not_authorize_runtime_use",
            "priority_cards_do_not_record_approval_or_rejection",
        ],
    }


def _counts(cards: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "inputDecisionRecordRows": len(cards),
        "priorityCardRows": len(cards),
        "selectedInitialReviewRows": sum(1 for item in cards if item.get("selected_for_initial_review")),
        "needsReviewRows": sum(1 for item in cards if item.get("recorded_decision") == "needs_review"),
        "approvedRows": 0,
        "rejectedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in cards)),
        "byReviewPriority": dict(Counter(str(item.get("review_priority") or "") for item in cards)),
        "bySelection": dict(Counter("selected" if item.get("selected_for_initial_review") else "not_selected" for item in cards)),
    }


def build_sectionspan_pdf_offset_review_priority_pack(
    *,
    sectionspan_pdf_offset_review_decision_record_report: str | Path,
    max_initial_cards: int = 12,
) -> dict[str, Any]:
    """Build a report-only priority pack from SectionSpan review decision records."""

    record_path = Path(str(sectionspan_pdf_offset_review_decision_record_report)).expanduser()
    record = _read_json(record_path)
    unsafe_flags = _unsafe_flags(record)
    rows = [
        dict(item)
        for item in list(record.get("decisionRecords") or [])
        if isinstance(item, dict) and item.get("recorded_decision") == "needs_review"
    ]
    selected_ids = _select_initial_review_rows(rows, max_initial_cards=max_initial_cards)
    cards = [_card(index, row, selected_ids) for index, row in enumerate(sorted(rows, key=_sort_key), start=1)]
    counts = _counts(cards, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif cards:
        status = "priority_pack_ready"
        decision = "initial_manual_review_priority_pack_ready"
    else:
        status = "no_pending_review_rows"
        decision = "no_pending_sectionspan_review_rows"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetReviewDecisionRecordReport": str(record_path),
            "sectionspanPdfOffsetReviewDecisionRecordSchema": str(record.get("schema") or ""),
            "maxInitialCards": max_initial_cards,
        },
        "counts": counts,
        "gate": {
            "priorityPackReady": bool(cards) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_review_selected_sectionspan_priority_cards"
            if cards
            else "sectionspan_review_decision_record_refresh",
        },
        "policy": {
            "reportOnly": True,
            "priorityPackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "priority_cards_are_not_review_decisions",
            "priority_cards_do_not_authorize_strict_or_runtime_evidence",
            "manual_review_and_later_apply_tranche_required_before_promotion",
        ],
        "priorityCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_review_priority_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Review Priority Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Priority cards: `{int(counts.get('priorityCardRows') or 0)}`",
        f"- Selected initial review rows: `{int(counts.get('selectedInitialReviewRows') or 0)}`",
        f"- Needs review rows: `{int(counts.get('needsReviewRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This pack is a review-prioritization report only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By selection: `{json.dumps(counts.get('bySelection') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_review_priority_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "sectionspan-pdf-offset-review-priority-pack.json"
    summary_path = root / "sectionspan-pdf-offset-review-priority-pack-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-review-priority-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_review_priority_pack_markdown(report), encoding="utf-8")
    return {"priorityPack": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset review priority pack.")
    parser.add_argument("--sectionspan-pdf-offset-review-decision-record-report", required=True)
    parser.add_argument("--max-initial-cards", type=int, default=12)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_review_priority_pack(
        sectionspan_pdf_offset_review_decision_record_report=args.sectionspan_pdf_offset_review_decision_record_report,
        max_initial_cards=args.max_initial_cards,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_review_priority_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID",
    "build_sectionspan_pdf_offset_review_priority_pack",
    "render_sectionspan_pdf_offset_review_priority_pack_markdown",
    "write_sectionspan_pdf_offset_review_priority_pack_reports",
]
