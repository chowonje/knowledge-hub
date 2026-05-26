"""Report-only SectionSpan original PDF offset recovery design.

This helper plans how SectionSpan candidates could later be mapped from
canonical generated Markdown spans back to original PDF/source spans. It does
not execute recovery, create strict evidence, change source authority, route
parsers, wire answer citations, mutate SQLite, reindex, reembed, or write
canonical parsed artifacts.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-recovery-design.v1"
)
SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-source-authority-options.v1"
)
SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-strict-promotion-design.v1"
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


def _schema_violations(options: dict[str, Any], design: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if options.get("schema") != SECTIONSPAN_SOURCE_AUTHORITY_OPTIONS_SCHEMA_ID:
        violations.append("sectionspan_source_authority_options_schema_mismatch")
    if design.get("schema") != SECTIONSPAN_STRICT_PROMOTION_DESIGN_SCHEMA_ID:
        violations.append("sectionspan_strict_promotion_design_schema_mismatch")
    return violations


def _unsafe_flags(options: dict[str, Any], design: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    for name, payload in (("sourceAuthorityOptions", options), ("strictPromotionDesign", design)):
        counts = dict(payload.get("counts") or {})
        gate = dict(payload.get("gate") or {})
        policy = dict(payload.get("policy") or {})
        for key in (
            "authorityDecisionMadeOptions",
            "strictPromotionReadyOptions",
            "runtimePromotionAllowedOptions",
            "strictPromotionReadyRows",
            "runtimePromotionAllowedRows",
            "strictEligibleRows",
            "citationGradeRows",
            "runtimeEvidenceRows",
        ):
            if _safe_int(counts.get(key)) > 0:
                unsafe.append(f"{name}_{key}_nonzero")
        for key in (
            "authorityDecisionMade",
            "strictEvidenceReady",
            "parserRoutingReady",
            "answerIntegrationReady",
            "runtimePromotionAllowed",
        ):
            if bool(gate.get(key)):
                unsafe.append(f"{name}_{key}_true")
        for key in (
            "authorityDecisionMade",
            "strictPromotionImplemented",
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
    if options.get("status") != "options_ready":
        unsafe.append("sectionspan_source_authority_options_not_ready")
    if design.get("status") != "design_ready":
        unsafe.append("sectionspan_strict_promotion_design_not_ready")
    return list(dict.fromkeys(unsafe))


def _recommended_option_present(options: dict[str, Any]) -> bool:
    for option in list(options.get("options") or []):
        if isinstance(option, dict) and option.get("option_id") == "recover_original_pdf_offsets_first":
            return True
    return False


def _plan_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    canonical_span = dict(row.get("canonical_span") or {})
    authority = dict(row.get("source_span_authority") or {})
    return {
        "recovery_plan_id": f"sectionspan-pdf-offset-recovery:{index:04d}",
        "source_design_id": str(row.get("design_id") or ""),
        "source_review_card_id": str(row.get("source_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": _clean_text(row.get("candidate_text")),
        "section_label": _clean_text(row.get("section_label")),
        "section_title": _clean_text(row.get("section_title")),
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "canonical_span": {
            "chars_start": _safe_int(canonical_span.get("chars_start")),
            "chars_end": _safe_int(canonical_span.get("chars_end")),
            "page": _safe_int(canonical_span.get("page")),
            "sourceContentHash": str(canonical_span.get("sourceContentHash") or ""),
            "alignmentMethod": str(canonical_span.get("alignmentMethod") or ""),
            "alignmentStatus": str(canonical_span.get("alignmentStatus") or ""),
            "locatorKind": str(canonical_span.get("locatorKind") or ""),
        },
        "current_authority_status": {
            "authorityStatus": str(authority.get("authorityStatus") or ""),
            "locatorKind": str(authority.get("locatorKind") or ""),
            "canonicalParsedTextSpanAvailable": bool(authority.get("canonicalParsedTextSpanAvailable")),
            "originalPdfOffsetAvailable": bool(authority.get("originalPdfOffsetAvailable")),
        },
        "recovery_status": "planned_not_executed",
        "recovery_strategy": "map_canonical_generated_markdown_heading_to_original_pdf_text_span",
        "required_inputs": [
            "registered_local_source_pdf",
            "deterministic_pdf_text_extractor_output",
            "canonical_generated_markdown_span",
            "sourceContentHash_for_original_source",
        ],
        "matching_requirements": [
            "exact_or_normalized_unique_match_against_original_pdf_text",
            "page_must_match_or_be_recovered_from_pdf_text_boundaries",
            "ambiguous_multi_match_must_remain_non_strict",
            "summary_or_paraphrase_matches_must_be_rejected",
        ],
        "stop_conditions": [
            "source_pdf_missing",
            "pdf_text_extraction_unavailable",
            "match_ambiguous",
            "match_requires_fuzzy_paraphrase",
            "page_conflict",
            "source_hash_missing_or_mismatch",
        ],
        "planned_output_contract": {
            "originalPdfCharsStart": None,
            "originalPdfCharsEnd": None,
            "page": _safe_int(canonical_span.get("page")),
            "sourceContentHash": str(canonical_span.get("sourceContentHash") or ""),
            "matchMethod": "",
            "matchConfidence": 0.0,
        },
        "evidence_tier": "sectionspan_pdf_offset_recovery_design_only",
        "original_pdf_offset_recovered": False,
        "strict_promotion_ready": False,
        "runtime_promotion_allowed": False,
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "strict_blockers": [
            "pdf_offset_recovery_design_only",
            "recovery_not_executed",
            "original_pdf_offset_not_available",
            "runtime_promotion_disabled_for_tranche",
            "strict_promotion_requires_explicit_later_tranche",
        ],
        "non_strict_reason": [
            "this_is_a_recovery_design_not_a_recovered_source_span",
            "no_original_pdf_offset_has_been_created",
            "later_explicit_recovery_and_promotion_tranches_required",
        ],
    }


def _counts(rows: list[dict[str, Any]], violations: list[str], *, recommended_option_present: bool) -> dict[str, Any]:
    return {
        "inputDesignRows": len(rows),
        "recoveryPlanRows": len(rows),
        "plannedRows": len(rows),
        "executedRows": 0,
        "originalPdfOffsetRecoveredRows": 0,
        "strictPromotionReadyRows": 0,
        "runtimePromotionAllowedRows": 0,
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "recommendedRecoveryOptionPresent": 1 if recommended_option_present else 0,
        "schemaViolationCount": len([item for item in violations if item.endswith("_mismatch")]),
        "unsafeUpstreamFlagCount": len([item for item in violations if not item.endswith("_mismatch")]),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "byRecoveryStatus": dict(Counter(str(item.get("recovery_status") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_recovery_design(
    *,
    sectionspan_source_authority_options_report: str | Path,
    sectionspan_strict_promotion_design_report: str | Path,
) -> dict[str, Any]:
    """Build a report-only original PDF offset recovery design."""

    paths = {
        "sectionspanSourceAuthorityOptionsReport": Path(str(sectionspan_source_authority_options_report)).expanduser(),
        "sectionspanStrictPromotionDesignReport": Path(str(sectionspan_strict_promotion_design_report)).expanduser(),
    }
    options = _read_json(paths["sectionspanSourceAuthorityOptionsReport"])
    design = _read_json(paths["sectionspanStrictPromotionDesignReport"])
    recommended_option_present = _recommended_option_present(options)
    violations = [
        *_schema_violations(options, design),
        *_unsafe_flags(options, design),
        *([] if recommended_option_present else ["recommended_recovery_option_missing"]),
    ]
    source_rows = [dict(item) for item in list(design.get("designRows") or []) if isinstance(item, dict)]
    rows = [_plan_row(index, row) for index, row in enumerate(source_rows, start=1)]
    counts = _counts(rows, violations, recommended_option_present=recommended_option_present)
    ready = not violations and bool(rows)
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID,
        "status": "design_ready" if ready else "blocked",
        "generatedAt": _now(),
        "inputs": {
            "sectionspanSourceAuthorityOptionsReport": str(paths["sectionspanSourceAuthorityOptionsReport"]),
            "sectionspanStrictPromotionDesignReport": str(paths["sectionspanStrictPromotionDesignReport"]),
            "sectionspanSourceAuthorityOptionsSchema": str(options.get("schema") or ""),
            "sectionspanStrictPromotionDesignSchema": str(design.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "pdfOffsetRecoveryDesignReady": ready,
            "pdfOffsetRecoveryImplemented": False,
            "originalPdfOffsetRecovered": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": "pdf_offset_recovery_design_ready_not_executed" if ready else "blocked",
            "schemaViolations": [item for item in violations if item.endswith("_mismatch")],
            "unsafeUpstreamFlags": [item for item in violations if not item.endswith("_mismatch")],
            "recommendedNextTranche": "bounded_original_pdf_offset_recovery_dry_run",
        },
        "policy": {
            "reportOnly": True,
            "authorityDecisionMade": False,
            "pdfOffsetRecoveryImplemented": False,
            "originalPdfOffsetRecovered": False,
            "strictPromotionImplemented": False,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "designPrinciples": [
            "recover_original_pdf_offsets_before_strict_runtime_use",
            "reject_ambiguous_or_fuzzy_paraphrase_matches",
            "never_promote_generated_markdown_offsets_without_explicit_authority_decision",
            "run_recovery_as_a_later_dry_run_before_any_apply_behavior",
        ],
        "warnings": [
            "this_report_does_not_execute_pdf_offset_recovery",
            "no_original_pdf_offsets_are_created",
            "no_sectionspan_row_is_strict_or_runtime_evidence",
        ],
        "recoveryPlanRows": rows,
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
            "designPrinciples",
            "warnings",
            "recoveryPlanRows",
        )
        if key in report
    }


def render_sectionspan_pdf_offset_recovery_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan Original PDF Offset Recovery Design",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Recovery plan rows: `{int(counts.get('recoveryPlanRows') or 0)}`",
        f"- Executed rows: `{int(counts.get('executedRows') or 0)}`",
        f"- Original PDF offsets recovered: `{int(counts.get('originalPdfOffsetRecoveredRows') or 0)}`",
        f"- Strict promotion ready rows: `{int(counts.get('strictPromotionReadyRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This is a recovery design only. It does not execute PDF text extraction, create original PDF offsets, create strict evidence, allow runtime citations, change parser routing, write canonical parsed artifacts, mutate SQLite, reindex, or reembed.",
        "",
        "## Counts",
        "",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By recovery status: `{json.dumps(counts.get('byRecoveryStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_recovery_design_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "sectionspan-pdf-offset-recovery-design.json"
    summary_path = root / "sectionspan-pdf-offset-recovery-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-recovery-design.md"
    design_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_recovery_design_markdown(report), encoding="utf-8")
    return {"design": str(design_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate report-only SectionSpan original PDF offset recovery design.")
    parser.add_argument("--sectionspan-source-authority-options-report", required=True)
    parser.add_argument("--sectionspan-strict-promotion-design-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_recovery_design(
        sectionspan_source_authority_options_report=args.sectionspan_source_authority_options_report,
        sectionspan_strict_promotion_design_report=args.sectionspan_strict_promotion_design_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_recovery_design_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_RECOVERY_DESIGN_SCHEMA_ID",
    "build_sectionspan_pdf_offset_recovery_design",
    "render_sectionspan_pdf_offset_recovery_design_markdown",
    "write_sectionspan_pdf_offset_recovery_design_reports",
]
