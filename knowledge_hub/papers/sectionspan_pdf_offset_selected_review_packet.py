"""Report-only selected review packet for SectionSpan PDF offset priority cards."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-packet.v1"
)
SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-review-priority-pack.v1"
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


def _unsafe_flags(priority_pack: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(priority_pack.get("counts") or {})
    gate = dict(priority_pack.get("gate") or {})
    policy = dict(priority_pack.get("policy") or {})
    if priority_pack.get("schema") != SECTIONSPAN_PDF_OFFSET_REVIEW_PRIORITY_PACK_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_review_priority_pack_schema_mismatch")
    if priority_pack.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_review_priority_pack_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"priorityPack_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"priorityPack_{key}_true")
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
            flags.append(f"priorityPack_{key}_true")
    return list(dict.fromkeys(flags))


def _card(index: int, source: dict[str, Any]) -> dict[str, Any]:
    source_decision_row_id = str(source.get("source_decision_row_id") or "")
    return {
        "selected_review_card_id": f"sectionspan-pdf-offset-selected-review-card:{index:04d}",
        "source_priority_card_id": str(source.get("priority_card_id") or ""),
        "source_decision_record_row_id": str(source.get("source_decision_record_row_id") or ""),
        "source_decision_row_id": source_decision_row_id,
        "source_gate_row_id": str(source.get("source_gate_row_id") or ""),
        "source_review_card_id": str(source.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(source.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(source.get("paper_id") or ""),
        "candidate_text": str(source.get("candidate_text") or ""),
        "section_type": str(source.get("section_type") or ""),
        "section_level": _safe_int(source.get("section_level")),
        "review_priority": str(source.get("review_priority") or ""),
        "priority_reasons": list(source.get("priority_reasons") or []),
        "canonical_span": dict(source.get("canonical_span") or {}),
        "original_pdf_span": dict(source.get("original_pdf_span") or {}),
        "page_agreement": bool(source.get("page_agreement")),
        "source_hash_agreement": bool(source.get("source_hash_agreement")),
        "review_checklist": list(source.get("review_checklist") or []),
        "allowed_decisions": [
            "approve_for_later_promotion_design",
            "reject_keep_candidate_only",
        ],
        "decision_record_input_hint": {
            "source_decision_row_id": source_decision_row_id,
            "allowedDecisionValues": [
                "approve_for_later_promotion_design",
                "reject_keep_candidate_only",
            ],
            "reviewer": "",
            "notes": "",
        },
        "evidence_tier": "sectionspan_pdf_offset_selected_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": [
            "selected_review_packet_only",
            "manual_review_decision_not_recorded",
            "strict_promotion_requires_later_explicit_apply_tranche",
            "runtime_promotion_disabled_for_tranche",
        ],
        "non_strict_reason": [
            "selected_review_cards_are_not_decisions",
            "selected_review_cards_do_not_authorize_runtime_use",
            "selected_review_cards_do_not_create_strict_evidence",
        ],
    }


def _counts(cards: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "selectedReviewCards": len(cards),
        "approvedRows": 0,
        "rejectedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in cards)),
        "byReviewPriority": dict(Counter(str(item.get("review_priority") or "") for item in cards)),
    }


def build_sectionspan_pdf_offset_selected_review_packet(
    *,
    sectionspan_pdf_offset_review_priority_pack_report: str | Path,
) -> dict[str, Any]:
    """Build a selected-card review packet from a SectionSpan priority pack."""

    priority_pack_path = Path(str(sectionspan_pdf_offset_review_priority_pack_report)).expanduser()
    priority_pack = _read_json(priority_pack_path)
    unsafe_flags = _unsafe_flags(priority_pack)
    selected = [
        dict(item)
        for item in list(priority_pack.get("priorityCards") or [])
        if isinstance(item, dict) and item.get("selected_for_initial_review")
    ]
    cards = [_card(index, item) for index, item in enumerate(selected, start=1)]
    counts = _counts(cards, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif cards:
        status = "selected_review_packet_ready"
        decision = "selected_manual_review_packet_ready"
    else:
        status = "no_selected_review_cards"
        decision = "no_selected_sectionspan_review_cards"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetReviewPriorityPackReport": str(priority_pack_path),
            "sectionspanPdfOffsetReviewPriorityPackSchema": str(priority_pack.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "selectedReviewPacketReady": bool(cards) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_review_selected_sectionspan_cards"
            if cards
            else "sectionspan_review_priority_pack_refresh",
        },
        "policy": {
            "reportOnly": True,
            "selectedReviewPacketOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "selected_review_cards_are_not_review_decisions",
            "selected_review_cards_do_not_authorize_strict_or_runtime_evidence",
            "manual_review_and_later_apply_tranche_required_before_promotion",
        ],
        "selectedReviewCards": cards,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_packet_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Packet",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Selected review cards: `{int(counts.get('selectedReviewCards') or 0)}`",
        f"- Approved rows: `{int(counts.get('approvedRows') or 0)}`",
        f"- Rejected rows: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This packet is a selected manual-review input only. It does not record approvals, create strict evidence, allow runtime citations, route parsers, write canonical parsed artifacts, mutate DB state, reindex, reembed, or change answer behavior.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_packet_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    packet_path = root / "sectionspan-pdf-offset-selected-review-packet.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-packet-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-packet.md"
    packet_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_selected_review_packet_markdown(report), encoding="utf-8")
    return {"packet": str(packet_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset selected review packet.")
    parser.add_argument("--sectionspan-pdf-offset-review-priority-pack-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_packet(
        sectionspan_pdf_offset_review_priority_pack_report=args.sectionspan_pdf_offset_review_priority_pack_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_packet_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_PACKET_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_packet",
    "render_sectionspan_pdf_offset_selected_review_packet_markdown",
    "write_sectionspan_pdf_offset_selected_review_packet_reports",
]
