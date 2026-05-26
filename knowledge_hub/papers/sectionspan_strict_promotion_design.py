"""Report-only SectionSpan strict-promotion readiness design.

This helper documents what would be required before SectionSpan candidates can
be considered for strict evidence.  It deliberately does not implement strict
promotion, runtime citations, parser routing, answer integration, DB writes,
index updates, embedding updates, or canonical parsed artifact writes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-strict-promotion-design.v1"
)
SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID = "knowledge-hub.paper.sectionspan-contract-review-pack.v1"
SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID = "knowledge-hub.paper.source-span-offset-authority-audit.v1"


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


def _schema_violations(review_pack: dict[str, Any], source_audit: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if review_pack.get("schema") != SECTIONSPAN_CONTRACT_REVIEW_PACK_SCHEMA_ID:
        violations.append("sectionspan_contract_review_pack_schema_mismatch")
    if source_audit.get("schema") != SOURCE_SPAN_OFFSET_AUTHORITY_AUDIT_SCHEMA_ID:
        violations.append("source_span_offset_authority_audit_schema_mismatch")
    return violations


def _unsafe_flags(review_pack: dict[str, Any], source_audit: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    for name, payload in (("reviewPack", review_pack), ("sourceAuthority", source_audit)):
        counts = dict(payload.get("counts") or {})
        gate = dict(payload.get("gate") or {})
        policy = dict(payload.get("policy") or {})
        for key in ("strictEligibleCards", "citationGradeCards", "runtimeEvidenceCards", "strictEligibleCandidates", "citationGradeCandidates", "runtimeEvidenceCandidates"):
            if _safe_int(counts.get(key)) > 0:
                unsafe.append(f"{name}_{key}_nonzero")
        for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
            if bool(gate.get(key)):
                unsafe.append(f"{name}_{key}_true")
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
                unsafe.append(f"{name}_{key}_true")
    if review_pack.get("status") != "review_pack_ready":
        unsafe.append("sectionspan_contract_review_pack_not_ready")
    if source_audit.get("status") != "ok":
        unsafe.append("source_span_offset_authority_audit_not_ok")
    return list(dict.fromkeys(unsafe))


def _source_rows_by_candidate_id(source_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in list(source_audit.get("rows") or []):
        if not isinstance(item, dict) or item.get("layer") != "sectionspan":
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            rows[candidate_id] = item
    return rows


def _design_status(card: dict[str, Any], source_row: dict[str, Any] | None) -> tuple[str, list[str]]:
    blockers = [
        "sectionspan_strict_promotion_design_only",
        "runtime_promotion_disabled_for_tranche",
        "strict_promotion_requires_explicit_later_tranche",
        "parser_routing_requires_explicit_later_tranche",
        "answer_integration_requires_explicit_later_tranche",
    ]
    if source_row is None:
        blockers.append("source_span_authority_row_missing")
        return "blocked_source_authority_row_missing", blockers
    if not bool(source_row.get("canonical_parsed_text_span_available")):
        blockers.append("canonical_generated_text_span_missing")
    if not bool(source_row.get("original_pdf_offset_available")):
        blockers.append("original_pdf_offset_not_available")
    if source_row.get("locatorKind") == "canonical_generated_markdown":
        blockers.append("canonical_generated_markdown_offset_authority_decision_required")
    if not str(source_row.get("sourceContentHash") or "").strip():
        blockers.append("source_content_hash_missing")
    if source_row.get("page") is None:
        blockers.append("page_missing")
    if source_row.get("chars_start") is None or source_row.get("chars_end") is None:
        blockers.append("canonical_chars_start_end_missing")
    if "original_pdf_offset_not_available" in blockers:
        return "blocked_original_pdf_offset_or_authority_decision_required", list(dict.fromkeys(blockers))
    if "canonical_generated_markdown_offset_authority_decision_required" in blockers:
        return "blocked_canonical_markdown_authority_decision_required", list(dict.fromkeys(blockers))
    if any(item.endswith("_missing") for item in blockers):
        return "blocked_missing_source_authority", list(dict.fromkeys(blockers))
    return "design_candidate_ready_for_later_approval_only", list(dict.fromkeys(blockers))


def _row(index: int, card: dict[str, Any], source_row: dict[str, Any] | None) -> dict[str, Any]:
    canonical_span = dict(card.get("canonical_span") or {})
    status, blockers = _design_status(card, source_row)
    return {
        "design_id": f"sectionspan-strict-promotion-design:{index:04d}",
        "source_review_card_id": str(card.get("card_id") or ""),
        "source_contract_candidate_id": str(card.get("source_contract_candidate_id") or ""),
        "source_sectionspan_candidate_id": str(card.get("source_sectionspan_candidate_id") or ""),
        "candidate_type": "sectionspan_strict_promotion_design",
        "paper_id": str(card.get("paper_id") or ""),
        "candidate_text": _clean_text(card.get("candidate_text")),
        "section_label": _clean_text(card.get("section_label")),
        "section_title": _clean_text(card.get("section_title")),
        "section_type": str(card.get("section_type") or ""),
        "section_level": _safe_int(card.get("section_level")),
        "canonical_span": {
            "chars_start": _safe_int(canonical_span.get("chars_start")),
            "chars_end": _safe_int(canonical_span.get("chars_end")),
            "page": _safe_int(canonical_span.get("page")),
            "sourceContentHash": str(canonical_span.get("sourceContentHash") or ""),
            "alignmentMethod": str(canonical_span.get("alignmentMethod") or ""),
            "alignmentStatus": str(canonical_span.get("alignmentStatus") or ""),
            "locatorKind": str(canonical_span.get("locatorKind") or ""),
        },
        "source_span_authority": {
            "audit_id": str((source_row or {}).get("audit_id") or ""),
            "authorityStatus": str((source_row or {}).get("source_span_authority_status") or ""),
            "locatorKind": str((source_row or {}).get("locatorKind") or ""),
            "canonicalParsedTextSpanAvailable": bool((source_row or {}).get("canonical_parsed_text_span_available")),
            "originalPdfOffsetAvailable": bool((source_row or {}).get("original_pdf_offset_available")),
            "layoutOrBboxOnly": bool((source_row or {}).get("layout_or_bbox_only")),
            "markdownOffsetOnly": bool((source_row or {}).get("markdown_offset_only")),
        },
        "promotion_design_status": status,
        "candidate_formalization_ready": bool(card.get("review_status") == "needs_human_review"),
        "strict_promotion_ready": False,
        "runtime_promotion_allowed": False,
        "requirements_before_strict_promotion": [
            "explicit_human_approval_of_sectionspan_contract",
            "explicit_authority_decision_for_canonical_generated_markdown_offsets_or_original_pdf_offset_recovery",
            "runtime_answer_citation_policy_update_in_later_tranche",
            "protecting_tests_for_strict_sectionspan_runtime_use",
        ],
        "allowed_next_actions": [
            "human_operator_review",
            "report_only_strict_promotion_design_refinement",
            "original_pdf_offset_recovery_design",
        ],
        "disallowed_actions": [
            "strict_evidence_promotion",
            "runtime_answer_citation",
            "parser_routing",
            "canonical_parsed_artifact_write",
            "database_mutation",
            "reindex_or_reembed",
        ],
        "evidence_tier": "sectionspan_strict_promotion_design_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "strict_promotion_design_is_report_only",
            "sectionspan_rows_remain_candidate_layer_only",
            "later_explicit_approval_and_policy_tranche_required",
        ],
    }


def _held_out_rows(review_pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(review_pack.get("heldOut") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "sourceCandidateId": str(item.get("sourceCandidateId") or ""),
                "paperId": str(item.get("paperId") or ""),
                "candidateText": _clean_text(item.get("candidateText")),
                "reviewClass": str(item.get("reviewClass") or ""),
                "reason": str(item.get("reason") or "held_out_upstream"),
                "promotionDesignStatus": "held_out",
                "strictEligible": False,
                "citationGrade": False,
                "runtimeEvidence": False,
            }
        )
    return rows


def _counts(rows: list[dict[str, Any]], held_out: list[dict[str, Any]], violations: list[str], input_cards: int) -> dict[str, Any]:
    return {
        "inputReviewCards": input_cards,
        "designRowCount": len(rows),
        "candidateFormalizationReadyRows": sum(1 for item in rows if item.get("candidate_formalization_ready")),
        "strictPromotionReadyRows": 0,
        "runtimePromotionAllowedRows": 0,
        "heldOutCandidates": len(held_out),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byPromotionDesignStatus": dict(Counter(str(item.get("promotion_design_status") or "") for item in rows)),
        "heldOutByReason": dict(Counter(str(item.get("reason") or "") for item in held_out)),
    }


def build_sectionspan_strict_promotion_design(
    *,
    sectionspan_contract_review_pack_report: str | Path,
    source_span_offset_authority_audit_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only SectionSpan strict-promotion readiness design."""

    paths = {
        "sectionspanContractReviewPackReport": Path(str(sectionspan_contract_review_pack_report)).expanduser(),
        "sourceSpanOffsetAuthorityAuditReport": Path(str(source_span_offset_authority_audit_report)).expanduser(),
    }
    review_pack = _read_json(paths["sectionspanContractReviewPackReport"])
    source_audit = _read_json(paths["sourceSpanOffsetAuthorityAuditReport"])
    violations = [*_schema_violations(review_pack, source_audit), *_unsafe_flags(review_pack, source_audit)]
    source_rows = _source_rows_by_candidate_id(source_audit)
    cards = [dict(item) for item in list(review_pack.get("reviewCards") or []) if isinstance(item, dict)]
    cards.sort(
        key=lambda item: (
            str(item.get("paper_id") or ""),
            _safe_int((item.get("canonical_span") or {}).get("page")),
            _safe_int((item.get("canonical_span") or {}).get("chars_start")),
            str(item.get("source_sectionspan_candidate_id") or ""),
        )
    )
    rows = [
        _row(index, card, source_rows.get(str(card.get("source_sectionspan_candidate_id") or "")))
        for index, card in enumerate(cards, start=1)
    ]
    held_out = _held_out_rows(review_pack)
    counts = _counts(rows, held_out, violations, input_cards=len(cards))
    design_ready = not violations and bool(rows)
    return {
        "schema": SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID,
        "status": "design_ready" if design_ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanContractReviewPackReport": str(paths["sectionspanContractReviewPackReport"]),
            "sourceSpanOffsetAuthorityAuditReport": str(paths["sourceSpanOffsetAuthorityAuditReport"]),
            "sectionspanContractReviewPackSchema": str(review_pack.get("schema") or ""),
            "sourceSpanOffsetAuthorityAuditSchema": str(source_audit.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "strictPromotionDesignReady": design_ready,
            "candidateFormalizationReady": counts["candidateFormalizationReadyRows"] > 0 and not violations,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "sectionspan_strict_promotion_design_ready_but_blocked_for_runtime" if design_ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "original_pdf_offset_or_canonical_markdown_authority_decision",
        },
        "policy": {
            "reportOnly": True,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "promotionPrinciples": [
            "sectionspan_strict_promotion_requires_explicit_later_approval",
            "source_hash_page_and_chars_are_candidate_layer_inputs_not_runtime_citations",
            "canonical_generated_markdown_offsets_need_authority_decision_or_original_pdf_offset_recovery",
            "answer_runtime_integration_requires_separate_policy_and_tests",
        ],
        "warnings": [
            "this_design_does_not_create_strict_evidence",
            "no_sectionspan_row_is_runtime_evidence",
            "parser_routing_and_answer_integration_remain_blocked",
        ],
        "designRows": rows,
        "heldOut": held_out,
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
            "promotionPrinciples",
            "warnings",
            "heldOut",
        )
        if key in report
    }


def render_sectionspan_strict_promotion_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Strict-Promotion Readiness Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Design rows: `{int(counts.get('designRowCount') or 0)}`",
        f"- Candidate formalization ready rows: `{int(counts.get('candidateFormalizationReadyRows') or 0)}`",
        f"- Strict promotion ready rows: `{int(counts.get('strictPromotionReadyRows') or 0)}`",
        f"- Runtime promotion allowed rows: `{int(counts.get('runtimePromotionAllowedRows') or 0)}`",
        f"- Held out: `{int(counts.get('heldOutCandidates') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a design-only report. It does not create strict evidence, runtime citations, parser routing, answer integration, canonical parsed artifacts, SQLite writes, indexes, or embeddings.",
        "",
        "## Counts",
        "",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By design status: `{json.dumps(counts.get('byPromotionDesignStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_strict_promotion_design_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "sectionspan-strict-promotion-design.json"
    summary_path = root / "sectionspan-strict-promotion-summary.json"
    markdown_path = root / "sectionspan-strict-promotion-design.md"
    design_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_strict_promotion_design_markdown(report), encoding="utf-8")
    return {"design": str(design_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan strict-promotion readiness design.")
    parser.add_argument("--sectionspan-contract-review-pack-report", required=True)
    parser.add_argument("--source-span-offset-authority-audit-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_strict_promotion_design(
        sectionspan_contract_review_pack_report=args.sectionspan_contract_review_pack_report,
        source_span_offset_authority_audit_report=args.source_span_offset_authority_audit_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_strict_promotion_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID",
    "build_sectionspan_strict_promotion_design",
    "render_sectionspan_strict_promotion_design_markdown",
    "write_sectionspan_strict_promotion_design_reports",
]
