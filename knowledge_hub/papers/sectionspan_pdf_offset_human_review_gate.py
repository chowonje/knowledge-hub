"""Report-only human review gate for recovered SectionSpan PDF offsets."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-human-review-gate.v1"
)
SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1"
)


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


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _decision_sets(decisions: dict[str, Any]) -> tuple[set[str], set[str]]:
    approved = set(str(item) for item in list(decisions.get("approvedReviewCardIds") or []) if str(item))
    rejected = set(str(item) for item in list(decisions.get("rejectedReviewCardIds") or []) if str(item))
    return approved, rejected


def _unsafe_flags(review_pack: dict[str, Any], decisions: dict[str, Any], approved: set[str], rejected: set[str]) -> list[str]:
    flags: list[str] = []
    counts = dict(review_pack.get("counts") or {})
    gate = dict(review_pack.get("gate") or {})
    policy = dict(review_pack.get("policy") or {})
    if review_pack.get("schema") != SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_recovery_review_pack_schema_mismatch")
    if review_pack.get("status") != "review_pack_ready":
        flags.append("sectionspan_pdf_offset_recovery_review_pack_not_ready")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"reviewPack_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"reviewPack_{key}_true")
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
            flags.append(f"reviewPack_{key}_true")
    overlap = approved.intersection(rejected)
    if overlap:
        flags.append("review_decisions_approved_and_rejected_overlap")
    if decisions and not approved and not rejected:
        flags.append("review_decisions_file_has_no_approved_or_rejected_ids")
    return list(dict.fromkeys(flags))


def _row(index: int, card: dict[str, Any], approved: set[str], rejected: set[str]) -> dict[str, Any]:
    card_id = str(card.get("review_card_id") or "")
    review_status = str(card.get("review_status") or "")
    upstream_ready = review_status == "ready_for_human_review"
    if card_id in approved and upstream_ready:
        human_status = "approved_for_later_promotion_design"
    elif card_id in rejected and upstream_ready:
        human_status = "rejected_keep_candidate_only"
    elif upstream_ready:
        human_status = "pending_human_review"
    else:
        human_status = "held_out_upstream_blocked"

    strict_blockers = [
        "human_review_gate_only",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_later_explicit_apply_tranche",
    ]
    if human_status == "pending_human_review":
        strict_blockers.append("human_review_not_completed")
    elif human_status == "rejected_keep_candidate_only":
        strict_blockers.append("human_review_rejected")
    elif human_status == "held_out_upstream_blocked":
        strict_blockers.append("upstream_review_card_not_ready")
    elif human_status == "approved_for_later_promotion_design":
        strict_blockers.append("approval_is_for_later_design_only_not_runtime_evidence")

    return {
        "gate_row_id": f"sectionspan-pdf-offset-human-review-gate:{index:04d}",
        "source_review_card_id": card_id,
        "source_sectionspan_candidate_id": str(card.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(card.get("paper_id") or ""),
        "candidate_text": str(card.get("candidate_text") or ""),
        "section_type": str(card.get("section_type") or ""),
        "section_level": _safe_int(card.get("section_level")),
        "canonical_span": dict(card.get("canonical_span") or {}),
        "original_pdf_span": dict(card.get("original_pdf_span") or {}),
        "page_agreement": bool(card.get("page_agreement")),
        "source_hash_agreement": bool(card.get("source_hash_agreement")),
        "upstream_review_status": review_status,
        "human_review_status": human_status,
        "review_decision_scope": (
            "later_sectionspan_promotion_design_only"
            if human_status == "approved_for_later_promotion_design"
            else "not_approved_for_promotion"
        ),
        "evidence_tier": "sectionspan_pdf_offset_human_review_gate_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": strict_blockers,
        "non_strict_reason": [
            "human_review_gate_report_only",
            "no_strict_or_runtime_evidence_created",
            "review_decision_does_not_modify_answer_runtime",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "inputReviewCards": len(rows),
        "gateRows": len(rows),
        "pendingHumanReviewRows": sum(1 for item in rows if item.get("human_review_status") == "pending_human_review"),
        "approvedForLaterPromotionDesignRows": sum(
            1 for item in rows if item.get("human_review_status") == "approved_for_later_promotion_design"
        ),
        "rejectedRows": sum(1 for item in rows if item.get("human_review_status") == "rejected_keep_candidate_only"),
        "heldOutRows": sum(1 for item in rows if item.get("human_review_status") == "held_out_upstream_blocked"),
        "pageAgreementRows": sum(1 for item in rows if item.get("page_agreement")),
        "sourceHashAgreementRows": sum(1 for item in rows if item.get("source_hash_agreement")),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byHumanReviewStatus": dict(Counter(str(item.get("human_review_status") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_human_review_gate(
    *,
    sectionspan_pdf_offset_recovery_review_pack_report: str | Path,
    review_decisions_report: str | Path | None = None,
) -> dict[str, Any]:
    """Build a report-only human review gate over recovered SectionSpan PDF offsets."""

    review_pack_path = Path(str(sectionspan_pdf_offset_recovery_review_pack_report)).expanduser()
    decisions_path = Path(str(review_decisions_report)).expanduser() if review_decisions_report else None
    review_pack = _read_json(review_pack_path)
    decisions = _read_json(decisions_path) if decisions_path else {}
    approved, rejected = _decision_sets(decisions)
    unsafe_flags = _unsafe_flags(review_pack, decisions, approved, rejected)
    cards = [dict(item) for item in list(review_pack.get("reviewCards") or []) if isinstance(item, dict)]
    rows = [_row(index, card, approved, rejected) for index, card in enumerate(cards, start=1)]
    counts = _counts(rows, unsafe_flags)
    blocked = bool(unsafe_flags)
    pending = _safe_int(counts.get("pendingHumanReviewRows"))
    status = "blocked" if blocked else ("review_required" if pending else "review_recorded")
    decision = "blocked" if blocked else (
        "human_review_required_before_any_promotion" if pending else "human_review_recorded_non_strict"
    )
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetRecoveryReviewPackReport": str(review_pack_path),
            "sectionspanPdfOffsetRecoveryReviewPackSchema": str(review_pack.get("schema") or ""),
            "reviewDecisionsReport": str(decisions_path or ""),
            "reviewDecisionApprovedCount": len(approved),
            "reviewDecisionRejectedCount": len(rejected),
        },
        "counts": counts,
        "gate": {
            "humanReviewGateReady": bool(rows) and not blocked,
            "humanReviewComplete": bool(rows) and not blocked and pending == 0,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": (
                "sectionspan_pdf_offset_human_review_execution"
                if pending
                else "sectionspan_promotion_apply_design_requires_explicit_approval"
            ),
        },
        "policy": {
            "reportOnly": True,
            "humanReviewGateOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "human_review_gate_rows_are_not_strict_evidence",
            "approved_rows_only_authorize_later_design_review_not_runtime_promotion",
            "strict_promotion_requires_a_separate_explicit_apply_tranche",
        ],
        "gateRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_human_review_gate_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Human Review Gate",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Gate rows: `{int(counts.get('gateRows') or 0)}`",
        f"- Pending human review: `{int(counts.get('pendingHumanReviewRows') or 0)}`",
        f"- Approved for later promotion design: `{int(counts.get('approvedForLaterPromotionDesignRows') or 0)}`",
        f"- Rejected: `{int(counts.get('rejectedRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This gate records review state only. It does not create strict evidence, runtime citations, parser routing, canonical parsed artifacts, DB mutations, reindex, reembed, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By human review status: `{json.dumps(counts.get('byHumanReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_human_review_gate_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / "sectionspan-pdf-offset-human-review-gate.json"
    summary_path = root / "sectionspan-pdf-offset-human-review-gate-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-human-review-gate.md"
    gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_human_review_gate_markdown(report), encoding="utf-8")
    return {"gate": str(gate_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset human review gate.")
    parser.add_argument("--sectionspan-pdf-offset-recovery-review-pack-report", required=True)
    parser.add_argument("--review-decisions-report", default="")
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_human_review_gate(
        sectionspan_pdf_offset_recovery_review_pack_report=args.sectionspan_pdf_offset_recovery_review_pack_report,
        review_decisions_report=args.review_decisions_report or None,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_human_review_gate_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_HUMAN_REVIEW_GATE_SCHEMA_ID",
    "build_sectionspan_pdf_offset_human_review_gate",
    "render_sectionspan_pdf_offset_human_review_gate_markdown",
    "write_sectionspan_pdf_offset_human_review_gate_reports",
]
