"""Report-only original PDF context pack for selected SectionSpan review rows."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
import unicodedata

from knowledge_hub.papers.source_text import source_hash_for_path


SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-evidence-pack.v1"
)
SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID = (
    "knowledge-hub.paper.sectionspan-pdf-offset-selected-review-decision-template.v1"
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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize(value: Any) -> str:
    parts: list[str] = []
    for char in str(value or ""):
        folded = unicodedata.normalize("NFKC", char).casefold()
        for item in folded:
            if item.isspace():
                if parts and parts[-1] != " ":
                    parts.append(" ")
                continue
            parts.append(item)
    return "".join(parts).strip()


def _extract_pdf_pages(source_pdf: str | Path) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    path = Path(str(source_pdf)).expanduser()
    try:
        document = fitz.open(str(path))
    except Exception:
        return []
    pages: list[dict[str, Any]] = []
    try:
        page_total = int(getattr(document, "page_count", 0) or 0)
        for page_index in range(page_total):
            try:
                page = document.load_page(page_index)
                text = str(page.get_text("text") or "")
            except Exception:
                text = ""
            pages.append({"page": page_index + 1, "text": text})
    finally:
        try:
            document.close()
        except Exception:
            pass
    return pages


def _with_offsets(raw_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    cursor = 0
    for item in raw_pages:
        text = str(item.get("text") or "")
        start = cursor
        end = start + len(text)
        pages.append({"page": _safe_int(item.get("page")), "text": text, "chars_start": start, "chars_end": end})
        cursor = end + 2
    return pages


def _unsafe_flags(template: dict[str, Any], recovery: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    template_counts = dict(template.get("counts") or {})
    template_gate = dict(template.get("gate") or {})
    template_policy = dict(template.get("policy") or {})
    if template.get("schema") != SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_DECISION_TEMPLATE_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_selected_review_decision_template_schema_mismatch")
    if template.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_selected_review_decision_template_blocked")
    for key in ("approvedRows", "rejectedRows", "strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(template_counts.get(key)) > 0:
            flags.append(f"selectedDecisionTemplate_{key}_nonzero")
    for key in ("humanReviewComplete", "strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(template_gate.get(key)):
            flags.append(f"selectedDecisionTemplate_{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(template_policy.get(key)):
            flags.append(f"selectedDecisionTemplate_{key}_true")

    recovery_counts = dict(recovery.get("counts") or {})
    recovery_gate = dict(recovery.get("gate") or {})
    recovery_policy = dict(recovery.get("policy") or {})
    if recovery.get("schema") != SECTIONSPAN_PDF_OFFSET_RECOVERY_DRY_RUN_SCHEMA_ID:
        flags.append("sectionspan_pdf_offset_recovery_dry_run_schema_mismatch")
    if recovery.get("status") == "blocked":
        flags.append("sectionspan_pdf_offset_recovery_dry_run_blocked")
    for key in ("strictEligibleRows", "citationGradeRows", "runtimeEvidenceRows"):
        if _safe_int(recovery_counts.get(key)) > 0:
            flags.append(f"recoveryDryRun_{key}_nonzero")
    for key in ("strictEvidenceReady", "parserRoutingReady", "answerIntegrationReady", "runtimePromotionAllowed"):
        if bool(recovery_gate.get(key)):
            flags.append(f"recoveryDryRun_{key}_true")
    for key in (
        "strictEvidenceCreated",
        "runtimePromotionAllowed",
        "parserRoutingChanged",
        "canonicalParsedArtifactsWritten",
        "databaseMutation",
        "reindexOrReembed",
        "answerIntegrationChanged",
    ):
        if bool(recovery_policy.get(key)):
            flags.append(f"recoveryDryRun_{key}_true")
    return list(dict.fromkeys(flags))


def _page_for_span(pages: list[dict[str, Any]], page_number: int, start: int, end: int) -> dict[str, Any] | None:
    for page in pages:
        if _safe_int(page.get("page")) != page_number:
            continue
        page_start = _safe_int(page.get("chars_start"))
        page_end = _safe_int(page.get("chars_end"))
        if page_start <= start <= end <= page_end:
            return page
    return None


def _context_row(
    *,
    index: int,
    row: dict[str, Any],
    paper_context: dict[str, Any],
    pages: list[dict[str, Any]],
    source_hash: str,
) -> dict[str, Any]:
    candidate_text = _clean_text(row.get("candidate_text"))
    original_span = dict(row.get("original_pdf_span") or {})
    canonical_span = dict(row.get("canonical_span") or {})
    span_start = _safe_int(original_span.get("originalPdfCharsStart"))
    span_end = _safe_int(original_span.get("originalPdfCharsEnd"))
    page_number = _safe_int(original_span.get("page"))
    source_pdf_path = str(paper_context.get("sourcePdfPath") or "")
    expected_hash = str(original_span.get("sourceContentHash") or "")
    base = {
        "review_evidence_row_id": f"sectionspan-pdf-offset-selected-review-evidence:{index:04d}",
        "source_decision_row_id": str(row.get("decision_row_id") or ""),
        "source_selected_review_card_id": str(row.get("source_selected_review_card_id") or ""),
        "source_sectionspan_candidate_id": str(row.get("source_sectionspan_candidate_id") or ""),
        "paper_id": str(row.get("paper_id") or ""),
        "candidate_text": candidate_text,
        "section_type": str(row.get("section_type") or ""),
        "section_level": _safe_int(row.get("section_level")),
        "review_priority": str(row.get("review_priority") or ""),
        "canonical_span": canonical_span,
        "original_pdf_span": original_span,
        "source_pdf_path": source_pdf_path,
        "sourceContentHash": expected_hash,
        "computedSourceContentHash": source_hash,
        "evidence_tier": "sectionspan_pdf_offset_selected_review_evidence_pack_only",
        "strict_eligible": False,
        "citation_grade": False,
        "runtime_evidence": False,
        "runtime_promotion_allowed": False,
    }
    blockers = [
        "review_evidence_pack_only",
        "manual_review_decision_not_recorded",
        "strict_promotion_requires_later_explicit_apply_tranche",
        "runtime_promotion_disabled_for_tranche",
    ]
    if not source_pdf_path:
        return {
            **base,
            "review_context_status": "blocked_source_pdf_missing",
            "page_text_match": False,
            "context_match_method": "",
            "matched_text": "",
            "context_before": "",
            "context_after": "",
            "review_suggestion": "needs_review",
            "review_suggestion_reason": "source_pdf_missing",
            "strict_blockers": blockers + ["source_pdf_context_missing"],
            "non_strict_reason": ["source_pdf_context_missing", "selected_review_evidence_pack_is_not_a_decision"],
        }
    if expected_hash and source_hash and expected_hash != source_hash:
        return {
            **base,
            "review_context_status": "blocked_source_hash_mismatch",
            "page_text_match": False,
            "context_match_method": "",
            "matched_text": "",
            "context_before": "",
            "context_after": "",
            "review_suggestion": "needs_review",
            "review_suggestion_reason": "source_hash_mismatch",
            "strict_blockers": blockers + ["source_hash_mismatch"],
            "non_strict_reason": ["source_hash_mismatch", "selected_review_evidence_pack_is_not_a_decision"],
        }
    page = _page_for_span(pages, page_number, span_start, span_end)
    if page is None:
        return {
            **base,
            "review_context_status": "blocked_original_pdf_span_out_of_range",
            "page_text_match": False,
            "context_match_method": "",
            "matched_text": "",
            "context_before": "",
            "context_after": "",
            "review_suggestion": "needs_review",
            "review_suggestion_reason": "original_pdf_span_out_of_range",
            "strict_blockers": blockers + ["original_pdf_context_not_verified"],
            "non_strict_reason": ["original_pdf_context_not_verified", "selected_review_evidence_pack_is_not_a_decision"],
        }
    text = str(page.get("text") or "")
    local_start = span_start - _safe_int(page.get("chars_start"))
    local_end = span_end - _safe_int(page.get("chars_start"))
    matched = text[local_start:local_end]
    exact_match = matched == candidate_text
    normalized_match = _normalize(matched) == _normalize(candidate_text)
    if exact_match:
        status = "review_context_ready"
        match_method = "exact"
        suggestion = "approve_for_later_promotion_design"
        suggestion_reason = "candidate_text_exactly_matches_original_pdf_page_offset"
    elif normalized_match:
        status = "review_context_ready"
        match_method = "normalized_whitespace_case"
        suggestion = "approve_for_later_promotion_design"
        suggestion_reason = "candidate_text_matches_original_pdf_page_offset_after_normalization"
    else:
        status = "blocked_page_text_mismatch"
        match_method = ""
        suggestion = "needs_review"
        suggestion_reason = "candidate_text_does_not_match_original_pdf_page_offset"
    return {
        **base,
        "review_context_status": status,
        "page_text_match": exact_match or normalized_match,
        "context_match_method": match_method,
        "matched_text": matched,
        "context_before": text[max(0, local_start - 180) : local_start],
        "context_after": text[local_end : min(len(text), local_end + 220)],
        "review_suggestion": suggestion,
        "review_suggestion_reason": suggestion_reason,
        "strict_blockers": blockers,
        "non_strict_reason": [
            "review_suggestions_are_not_human_review_decisions",
            "selected_review_evidence_pack_does_not_authorize_runtime_use",
            "selected_review_evidence_pack_does_not_create_strict_evidence",
        ],
    }


def _counts(rows: list[dict[str, Any]], unsafe_flags: list[str]) -> dict[str, Any]:
    return {
        "evidenceRows": len(rows),
        "reviewContextReadyRows": sum(1 for item in rows if item.get("review_context_status") == "review_context_ready"),
        "blockedRows": sum(1 for item in rows if str(item.get("review_context_status") or "").startswith("blocked_")),
        "pageTextMatchRows": sum(1 for item in rows if item.get("page_text_match")),
        "exactTextRows": sum(1 for item in rows if item.get("context_match_method") == "exact"),
        "normalizedTextRows": sum(1 for item in rows if item.get("context_match_method") == "normalized_whitespace_case"),
        "suggestedApproveForLaterPromotionDesignRows": sum(
            1 for item in rows if item.get("review_suggestion") == "approve_for_later_promotion_design"
        ),
        "suggestedNeedsReviewRows": sum(1 for item in rows if item.get("review_suggestion") == "needs_review"),
        "strictEligibleRows": 0,
        "citationGradeRows": 0,
        "runtimeEvidenceRows": 0,
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "byPaper": dict(Counter(str(item.get("paper_id") or "") for item in rows)),
        "bySectionType": dict(Counter(str(item.get("section_type") or "") for item in rows)),
        "byReviewPriority": dict(Counter(str(item.get("review_priority") or "") for item in rows)),
        "byContextStatus": dict(Counter(str(item.get("review_context_status") or "") for item in rows)),
        "bySuggestion": dict(Counter(str(item.get("review_suggestion") or "") for item in rows)),
    }


def build_sectionspan_pdf_offset_selected_review_evidence_pack(
    *,
    sectionspan_pdf_offset_selected_review_decision_template_report: str | Path,
    sectionspan_pdf_offset_recovery_dry_run_report: str | Path,
    pdf_page_text_loader: Callable[[str | Path], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build a report-only original PDF context pack for selected review rows."""

    template_path = Path(str(sectionspan_pdf_offset_selected_review_decision_template_report)).expanduser()
    recovery_path = Path(str(sectionspan_pdf_offset_recovery_dry_run_report)).expanduser()
    template = _read_json(template_path)
    recovery = _read_json(recovery_path)
    unsafe_flags = _unsafe_flags(template, recovery)
    paper_contexts = dict(recovery.get("paperContexts") or {})
    page_loader = pdf_page_text_loader or _extract_pdf_pages
    page_cache: dict[str, list[dict[str, Any]]] = {}
    hash_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate([dict(item) for item in list(template.get("decisionRows") or []) if isinstance(item, dict)], start=1):
        paper_id = str(row.get("paper_id") or "")
        context = dict(paper_contexts.get(paper_id) or {})
        source_pdf = str(context.get("sourcePdfPath") or "")
        if source_pdf and source_pdf not in page_cache:
            page_cache[source_pdf] = _with_offsets(page_loader(source_pdf))
            hash_cache[source_pdf] = source_hash_for_path(source_pdf)
        rows.append(
            _context_row(
                index=index,
                row=row,
                paper_context=context,
                pages=page_cache.get(source_pdf, []),
                source_hash=hash_cache.get(source_pdf, ""),
            )
        )
    counts = _counts(rows, unsafe_flags)
    if unsafe_flags:
        status = "blocked"
        decision = "blocked"
    elif rows:
        status = "selected_review_evidence_pack_ready"
        decision = "selected_review_context_ready_non_strict"
    else:
        status = "no_selected_review_evidence_rows"
        decision = "no_selected_review_rows_for_context_pack"
    return {
        "schema": SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID,
        "status": status,
        "generatedAt": _now(),
        "inputs": {
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateReport": str(template_path),
            "sectionspanPdfOffsetSelectedReviewDecisionTemplateSchema": str(template.get("schema") or ""),
            "sectionspanPdfOffsetRecoveryDryRunReport": str(recovery_path),
            "sectionspanPdfOffsetRecoveryDryRunSchema": str(recovery.get("schema") or ""),
        },
        "counts": counts,
        "gate": {
            "selectedReviewEvidencePackReady": bool(rows) and not unsafe_flags,
            "humanReviewComplete": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimePromotionAllowed": False,
            "decision": decision,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": "manual_record_selected_sectionspan_review_decisions",
        },
        "policy": {
            "reportOnly": True,
            "selectedReviewEvidencePackOnly": True,
            "strictEvidenceCreated": False,
            "runtimePromotionAllowed": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "databaseMutation": False,
            "reindexOrReembed": False,
            "answerIntegrationChanged": False,
        },
        "warnings": [
            "review_suggestions_are_not_human_review_decisions",
            "selected_review_evidence_pack_rows_do_not_authorize_strict_or_runtime_evidence",
            "approval_requires_a_separate_review_decision_file_and_later_apply_tranche",
        ],
        "evidenceRows": rows,
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in ("schema", "status", "generatedAt", "inputs", "counts", "gate", "policy", "warnings")
        if key in report
    }


def render_sectionspan_pdf_offset_selected_review_evidence_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# SectionSpan PDF Offset Selected Review Evidence Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Decision: `{gate.get('decision', '')}`",
        f"- Evidence rows: `{int(counts.get('evidenceRows') or 0)}`",
        f"- Context-ready rows: `{int(counts.get('reviewContextReadyRows') or 0)}`",
        f"- Blocked rows: `{int(counts.get('blockedRows') or 0)}`",
        f"- Suggested approvals for later promotion design: `{int(counts.get('suggestedApproveForLaterPromotionDesignRows') or 0)}`",
        f"- Strict eligible rows: `{int(counts.get('strictEligibleRows') or 0)}`",
        "",
        "## Boundary",
        "",
        "This pack provides original-PDF context for manual review only. Suggestions are not decisions. It does not create strict evidence, runtime citations, parser routing, canonical parsed artifacts, DB mutations, reindex, reembed, or answer integration.",
        "",
        "## Counts",
        "",
        f"- By paper: `{json.dumps(counts.get('byPaper') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By section type: `{json.dumps(counts.get('bySectionType') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By priority: `{json.dumps(counts.get('byReviewPriority') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By context status: `{json.dumps(counts.get('byContextStatus') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By suggestion: `{json.dumps(counts.get('bySuggestion') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def write_sectionspan_pdf_offset_selected_review_evidence_pack_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "sectionspan-pdf-offset-selected-review-evidence-pack.json"
    summary_path = root / "sectionspan-pdf-offset-selected-review-evidence-pack-summary.json"
    markdown_path = root / "sectionspan-pdf-offset-selected-review-evidence-pack.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_sectionspan_pdf_offset_selected_review_evidence_pack_markdown(report), encoding="utf-8")
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only SectionSpan PDF offset selected review evidence pack.")
    parser.add_argument("--sectionspan-pdf-offset-selected-review-decision-template-report", required=True)
    parser.add_argument("--sectionspan-pdf-offset-recovery-dry-run-report", required=True)
    parser.add_argument("--output-dir", default="", help="Directory for local JSON/Markdown reports.")
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_sectionspan_pdf_offset_selected_review_evidence_pack(
        sectionspan_pdf_offset_selected_review_decision_template_report=(
            args.sectionspan_pdf_offset_selected_review_decision_template_report
        ),
        sectionspan_pdf_offset_recovery_dry_run_report=args.sectionspan_pdf_offset_recovery_dry_run_report,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_sectionspan_pdf_offset_selected_review_evidence_pack_reports(report, args.output_dir)
    summary = _summary_payload(report)
    if paths:
        summary["reportPaths"] = paths
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SECTIONSPAN_PDF_OFFSET_SELECTED_REVIEW_EVIDENCE_PACK_SCHEMA_ID",
    "build_sectionspan_pdf_offset_selected_review_evidence_pack",
    "render_sectionspan_pdf_offset_selected_review_evidence_pack_markdown",
    "write_sectionspan_pdf_offset_selected_review_evidence_pack_reports",
]
