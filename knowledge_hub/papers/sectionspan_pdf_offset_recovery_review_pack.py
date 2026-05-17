"""Report-only review pack for recovered SectionSpan original PDF offsets."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-review-pack.v1"
)
SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-dry-run.v1"
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _unsafe_flags(dry_run: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    counts = dict(dry_run.get("counts") or {})
    gate = dict(dry_run.get("gate") or {})
    policy = dict(dry_run.get("policy") or {})
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(counts.get(key)) > 0:
            flags.append(f"dryRun_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(gate.get(key)):
            flags.append(f"dryRun_{key}_true")
    for key in (
        "applyExecuted",
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(policy.get(key)):
            flags.append(f"dryRun_{key}_true")
    if dry_run.get("status") != "dry_run_complete":
        flags.append("sectionspan_pdf_offset_recovery_dry_run_not_complete")
    return flags


def _card(index: int, row: dict[str, Any]) -> dict[str, Any]:
    canonical_span = dict(row.get("canonical_span") or {})
    original_span = dict(row.get("original_pdf_span") or {})
    recovered = bool(row.get("original_pdf_offset_recovered"))
    canonical_hash = str(canonical_span.get("sourceContentHash") or "")
    original_hash = str(original_span.get("sourceContentHash") or "")
    page_agreement = _safe_int(canonical_span.get("page")) == _safe_int(original_span.get("page"))
    source_hash_agreement = bool(canonical_hash and original_hash and canonical_hash == original_hash)
    ready = recovered and page_agreement and source_hash_agreement
    review_status = "ready_for_human_review" if ready else "held_out_recovery_blocked"
    if recovered and not page_agreement:
        review_status = "held_out_page_conflict"
    elif recovered and not source_hash_agreement:
        review_status = "held_out_source_hash_conflict"
    blockers = [
        "review_pack_only",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_later_explicit_approval",
    ]
    if not recovered:
        blockers.append("original_pdf_offset_not_recovered")
    if recovered and not page_agreement:
        blockers.append("canonical_page_original_pdf_page_conflict")
    if recovered and not source_hash_agreement:
        blockers.append("source_hash_conflict")
    return {
        "review_card_id": f"sectionspan-pdf-offset-review-card:{index:04d}",
        "source_recovery_plan_id": str(row.get("recovery_plan_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": str(row.get("candidate_text") or ""),
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "canonical_span": {
            "chars_start": _safe_int(canonical_span.get("chars_start")),
            "chars_end": _safe_int(canonical_span.get("chars_end")),
            "page": _safe_int(canonical_span.get("page")),
            "sourceContentHash": canonical_hash,
            "locatorKind": str(canonical_span.get("locatorKind") or ""),
        },
        "original_pdf_span": {
            "originalPdfCharsStart": original_span.get("originalPdfCharsStart"),
            "originalPdfCharsEnd": original_span.get("originalPdfCharsEnd"),
            "page": original_span.get("page"),
            "sourceContentHash": original_hash,
            "matchMethod": str(original_span.get("matchMethod") or ""),
            "matchConfidence": _safe_float(original_span.get("matchConfidence")),
        },
        "page_agreement": page_agreement,
        "source_hash_agreement": source_hash_agreement,
        "review_status": review_status,
        "review_recommendation": (
            "review_for_later_sectionspan_strict_promotion_candidate" if ready else "keep_candidate_only"
        ),
        "evidence_tier": "sectionspan_pdf_offset_recovery_review_card_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "review_pack_report_only",
            "no_runtime_or_strict_evidence_created",
            "later_explicit_apply_and_promotion_tranches_required",
        ],
    }


def _counts(cards: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    ready = [item for item in cards if item.get("review_status") == "ready_for_human_review"]
    return {
        "inputRecoveryRows": len(cards),
        "reviewCardRows": len(cards),
        "readyForHumanReviewRows": len(ready),
        "heldOutRows": len(cards) - len(ready),
        "pageAgreementRows": sum(1 for item in cards if item.get("page_agreement")),
        "sourceHashAgreementRows": sum(1 for item in cards if item.get("source_hash_agreement")),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in cards)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in cards)),
        "byReviewStatus": dict(Counter(str(item.get("review_status") or "") for item in cards)),
        "byMatchMethod": dict(Counter(str((item.get("original_pdf_span") or {}).get("matchMethod") or "") for item in cards)),
    }


def build_sectionspan_pdf_offset_recovery_review_pack(
    *,
    sectionspan_pdf_offset_recovery_dry_run_report: str | Path,
) -> dict[str, Any]:
    """Build report-only review cards from an original-PDF-offset dry-run."""

    path = Path(str(sectionspan_pdf_offset_recovery_dry_run_report)).expanduser()
    dry_run = _read_json(path)
    unsafe_flags = []
    if dry_run.get("schema") != SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID:
        unsafe_flags.append("sectionspan_pdf_offset_recovery_dry_run_schema_mismatch")
    unsafe_flags.extend(_unsafe_flags(dry_run))
    rows = [dict(item) for item in list(dry_run.get("recoveryRows") or []) if isinstance(item, dict)]
    cards = [_card(index, row) for index, row in enumerate(rows, start=1)]
    ready = bool(cards) and not unsafe_flags
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID,
        "status": "review_pack_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetRecoveryDryRunReport": str(path),
            "sectionspanPdfOffsetRecoveryDryRunSchema": str(dry_run.get("schema") or ""),
        },
        "counts": _counts(cards, unsafe_flags),
        "gate": {
            "reviewPackReady": ready,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "review_pack_ready_non_strict" if ready else "blocked",
            "unsafeUpstreamFlags": list(dict.fromkeys(unsafe_flags)),
            "recommendedNextTranche": "sectionspan_strict_promotion_gate_design_requires_explicit_approval",
        },
        "policy": {
            "reportOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "review_cards_are_not_runtime_or_strict_evidence",
            "human_review_and_later_explicit_promotion_gate_required",
        ],
        "reviewCards": cards,
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
            "reviewCards",
        )
        if key in report
    }


def render_sectionspan_pdf_offset_recovery_review_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Original PDF Offset Recovery Review Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Review cards: `{int(counts.get('reviewCardRows') or 0)}`",
        f"- Ready for human review: `{int(counts.get('readyForHumanReviewRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "These cards are review inputs only. They do not create strict evidence, allow runtime citations, change parser routing, write canonical parsed artifacts, mutate SQLite, reindex, or reembed.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By review status: `{json.dumps(counts.get('byReviewStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By match method: `{json.dumps(counts.get('byMatchMethod') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_recovery_review_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    cards_path = root / "sectionspan-pdf-offset-recovery-review-cards.json"
    summary_path = root / "sectionspan-pdf-offset-recovery-review-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-recovery-review-pack.md"
    cards_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_recovery_review_pack_markdown(report), encoding="utf-8")
    return {"cards": str(cards_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only SectionSpan original PDF offset recovery review pack.")
    parser.add_argument("--sectionspan-pdf-offset-recovery-dry-run-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_recovery_review_pack(
        sectionspan_pdf_offset_recovery_dry_run_report=args.sectionspan_pdf_offset_recovery_dry_run_report
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_recovery_review_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_RECOVERY_REVIEW_PACK_SCHEMA_ID",
    "build_sectionspan_pdf_offset_recovery_review_pack",
    "render_sectionspan_pdf_offset_recovery_review_pack_markdown",
    "write_sectionspan_pdf_offset_recovery_review_pack_reports",
]
