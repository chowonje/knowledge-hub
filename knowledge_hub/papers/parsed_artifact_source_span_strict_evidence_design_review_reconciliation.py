"""Report-only reconciliation of strict-evidence design-review rows.

Combines design-review-ready rows with disambiguation design candidates into a
final design-review packet and preserves still-blocked ambiguous rows for manual
follow-up. Does not create StrictEvidence, mutate SourceSpan JSONL, or change
runtime/parser/answer/DB state.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_strict_evidence_design_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
    REVIEW_STATUS_READY,
)
from knowledge_hub.papers.parsed_artifact_source_span_text_match_disambiguation_design import (
    DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE,
    DISAMBIGUATION_STATUS_CANDIDATE,
    PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
)


PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-span-strict-evidence-design-review-reconciliation.v1"
)

SOURCE_ORIGINAL_DESIGN_REVIEW = "original_design_review"
SOURCE_DISAMBIGUATION_DESIGN = "disambiguation_design"

FINAL_STATUS_READY = "ready_for_strict_evidence_design_review"
FINAL_STATUS_BLOCKED_AMBIGUOUS = "blocked_still_ambiguous_after_disambiguation"
FINAL_STATUS_BLOCKED_INPUT_SCHEMA = "blocked_input_schema_violation"

RECOMMENDED_ACTION_READY = "queue_for_strict_evidence_design_packet_review"
RECOMMENDED_ACTION_BLOCKED = "queue_for_manual_or_later_extractor_disambiguation"
RECOMMENDED_ACTION_REPAIR = "repair_input_reports_before_reconciliation"

DEFAULT_DESIGN_REVIEW_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-review"
    / "01-parsed-artifact-source-span-strict-evidence-design-review"
    / "parsed-artifact-source-span-strict-evidence-design-review.json"
)

DEFAULT_DISAMBIGUATION_REPORT_PATH = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-text-match-disambiguation-design"
    / "01-parsed-artifact-source-span-text-match-disambiguation-design"
    / "parsed-artifact-source-span-text-match-disambiguation-design.json"
)

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / ".khub"
    / "reports"
    / "layout-parser-pilot"
    / "2026-05-19"
    / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation"
    / "01-parsed-artifact-source-span-strict-evidence-design-review-reconciliation"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = _safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _proposed_chars_dict(row: dict[str, Any]) -> dict[str, Any]:
    proposed = row.get("proposed_chars")
    return proposed if isinstance(proposed, dict) else {}


def _chars_summary(proposed: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": proposed.get("start"),
        "end": proposed.get("end"),
        "basis": _safe_text(proposed.get("basis")),
        "normalization": _safe_text(proposed.get("normalization")),
        "expectedSubstringSha256": _safe_text(proposed.get("expectedSubstringSha256")),
        "sourceContentHash": _safe_text(proposed.get("sourceContentHash")),
    }


def _final_ready_from_review(review_row: dict[str, Any], *, index: int) -> dict[str, Any]:
    proposed = _proposed_chars_dict(review_row)
    return {
        "reconciliation_row_id": f"parsed-artifact-source-span-strict-evidence-design-review-reconciliation:ready:{index:04d}",
        "source": SOURCE_ORIGINAL_DESIGN_REVIEW,
        "review_row_id": _safe_text(review_row.get("review_row_id")),
        "design_row_id": _safe_text(review_row.get("design_row_id")),
        "disambiguation_row_id": "",
        "policy_gate_row_id": _safe_text(review_row.get("policy_gate_row_id")),
        "sourceSpanId": _safe_text(review_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(review_row.get("candidateRecordId")),
        "paper_id": _safe_text(review_row.get("paper_id")),
        "artifact_type": _safe_text(review_row.get("artifact_type")),
        "source_candidate_id": _safe_text(review_row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(review_row.get("sourceContentHash")),
        "source_file": _safe_text(review_row.get("source_file")),
        "text_surface": _safe_text(review_row.get("text_surface")),
        "proposed_chars": proposed,
        "chars": _chars_summary(proposed),
        "locator": review_row.get("locator") if isinstance(review_row.get("locator"), dict) else {},
        "final_status": FINAL_STATUS_READY,
        "final_blockers": _dedupe(
            [
                "strict_evidence_design_review_reconciliation_only",
                "source_span_remains_non_strict",
                "strict_evidence_creation_disabled_for_tranche",
                *[
                    _safe_text(item)
                    for item in (review_row.get("review_blockers") or [])
                    if _safe_text(item)
                ],
            ]
        ),
        "recommended_action": RECOMMENDED_ACTION_READY,
        "readyForStrictEvidenceDesignReview": True,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _final_ready_from_disambiguation(dis_row: dict[str, Any], *, index: int) -> dict[str, Any]:
    proposed = _proposed_chars_dict(dis_row)
    return {
        "reconciliation_row_id": f"parsed-artifact-source-span-strict-evidence-design-review-reconciliation:ready:{index:04d}",
        "source": SOURCE_DISAMBIGUATION_DESIGN,
        "review_row_id": _safe_text(dis_row.get("review_row_id")),
        "design_row_id": _safe_text(dis_row.get("design_row_id")),
        "disambiguation_row_id": _safe_text(dis_row.get("disambiguation_row_id")),
        "policy_gate_row_id": "",
        "sourceSpanId": _safe_text(dis_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(dis_row.get("candidateRecordId")),
        "paper_id": _safe_text(dis_row.get("paper_id")),
        "artifact_type": _safe_text(dis_row.get("artifact_type")),
        "source_candidate_id": _safe_text(dis_row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(dis_row.get("sourceContentHash")),
        "source_file": _safe_text(dis_row.get("source_file")),
        "text_surface": _safe_text(dis_row.get("text_surface")),
        "proposed_chars": proposed,
        "chars": _chars_summary(proposed),
        "locator": dis_row.get("locator") if isinstance(dis_row.get("locator"), dict) else {},
        "disambiguation_method": _safe_text(dis_row.get("disambiguation_method")),
        "final_status": FINAL_STATUS_READY,
        "final_blockers": _dedupe(
            [
                "strict_evidence_design_review_reconciliation_only",
                "disambiguation_design_only_not_applied_to_source_span_store",
                "source_span_remains_non_strict",
                "strict_evidence_creation_disabled_for_tranche",
                *[
                    _safe_text(item)
                    for item in (dis_row.get("disambiguation_blockers") or [])
                    if _safe_text(item)
                ],
            ]
        ),
        "recommended_action": RECOMMENDED_ACTION_READY,
        "readyForStrictEvidenceDesignReview": True,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _follow_up_from_disambiguation(dis_row: dict[str, Any], *, index: int) -> dict[str, Any]:
    proposed = _proposed_chars_dict(dis_row)
    disambiguation_status = _safe_text(dis_row.get("disambiguation_status"))
    return {
        "reconciliation_row_id": (
            f"parsed-artifact-source-span-strict-evidence-design-review-reconciliation:blocked:{index:04d}"
        ),
        "source": SOURCE_DISAMBIGUATION_DESIGN,
        "review_row_id": _safe_text(dis_row.get("review_row_id")),
        "design_row_id": _safe_text(dis_row.get("design_row_id")),
        "disambiguation_row_id": _safe_text(dis_row.get("disambiguation_row_id")),
        "sourceSpanId": _safe_text(dis_row.get("sourceSpanId")),
        "candidateRecordId": _safe_text(dis_row.get("candidateRecordId")),
        "paper_id": _safe_text(dis_row.get("paper_id")),
        "artifact_type": _safe_text(dis_row.get("artifact_type")),
        "source_candidate_id": _safe_text(dis_row.get("source_candidate_id")),
        "sourceContentHash": _safe_text(dis_row.get("sourceContentHash")),
        "text_surface": _safe_text(dis_row.get("text_surface")),
        "proposed_chars": proposed,
        "chars": _chars_summary(proposed),
        "locator": dis_row.get("locator") if isinstance(dis_row.get("locator"), dict) else {},
        "disambiguation_status": disambiguation_status,
        "final_status": FINAL_STATUS_BLOCKED_AMBIGUOUS,
        "final_blockers": _dedupe(
            [
                "still_ambiguous_after_text_match_disambiguation_design",
                *[
                    _safe_text(item)
                    for item in (dis_row.get("disambiguation_blockers") or [])
                    if _safe_text(item)
                ],
            ]
        ),
        "recommended_action": RECOMMENDED_ACTION_BLOCKED,
        "readyForStrictEvidenceDesignReview": False,
        "strictEligible": False,
        "strictEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "sourceSpanUpdatedRows": 0,
    }


def _count_rows(
    *,
    final_ready_rows: list[dict[str, Any]],
    follow_up_rows: list[dict[str, Any]],
    input_design_review_rows: int,
    input_disambiguation_rows: int,
    ready_from_original: int,
    ready_from_disambiguation: int,
    input_schema_violations: list[str],
) -> dict[str, Any]:
    all_rows = [*final_ready_rows, *follow_up_rows]
    return {
        "inputDesignReviewRows": input_design_review_rows,
        "inputDisambiguationRows": input_disambiguation_rows,
        "readyFromOriginalDesignReviewRows": ready_from_original,
        "readyFromDisambiguationRows": ready_from_disambiguation,
        "finalReadyForStrictEvidenceDesignReviewRows": len(final_ready_rows),
        "stillBlockedAmbiguousRows": len(follow_up_rows),
        "blockedInputSchemaViolationRows": int(bool(input_schema_violations)),
        "sourceSpanUpdatedRows": 0,
        "strictEvidenceCreatedRows": 0,
        "citationGradeEvidenceCreatedRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "parserRoutingChangedRows": 0,
        "answerIntegrationChangedRows": 0,
        "databaseMutationRows": 0,
        "canonicalParsedArtifactWriteRows": 0,
        "schemaViolationCount": len(input_schema_violations),
        "byPaperId": dict(Counter(str(row.get("paper_id") or "") for row in all_rows)),
        "byArtifactType": dict(Counter(str(row.get("artifact_type") or "") for row in all_rows)),
        "bySource": dict(Counter(str(row.get("source") or "") for row in all_rows)),
        "byFinalStatus": dict(Counter(str(row.get("final_status") or "") for row in all_rows)),
        "byRecommendedAction": dict(Counter(str(row.get("recommended_action") or "") for row in all_rows)),
    }


def build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation(
    *,
    design_review_report_path: str | Path = DEFAULT_DESIGN_REVIEW_REPORT_PATH,
    disambiguation_report_path: str | Path = DEFAULT_DISAMBIGUATION_REPORT_PATH,
) -> dict[str, Any]:
    review_path = Path(str(design_review_report_path)).expanduser()
    disambiguation_path = Path(str(disambiguation_report_path)).expanduser()

    warnings: list[str] = []
    input_schema_violations: list[str] = []

    design_review_report = _read_json(review_path)
    disambiguation_report = _read_json(disambiguation_path)

    if not design_review_report:
        warnings.append("design_review_report_missing_or_unreadable")
    if not disambiguation_report:
        warnings.append("disambiguation_report_missing_or_unreadable")

    review_validation = validate_payload(
        design_review_report,
        PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not review_validation.ok:
        input_schema_violations.extend(str(error) for error in review_validation.errors)
        if not design_review_report:
            input_schema_violations.append("design_review_report_missing_or_unreadable")

    disambiguation_validation = validate_payload(
        disambiguation_report,
        PARSED_ARTIFACT_SOURCE_SPAN_TEXT_MATCH_DISAMBIGUATION_DESIGN_SCHEMA_ID,
        strict=True,
    )
    if not disambiguation_validation.ok:
        input_schema_violations.extend(str(error) for error in disambiguation_validation.errors)
        if not disambiguation_report:
            input_schema_violations.append("disambiguation_report_missing_or_unreadable")

    input_design_review_rows = int((design_review_report.get("counts") or {}).get("inputRows") or 0)
    input_disambiguation_rows = int((disambiguation_report.get("counts") or {}).get("targetRows") or 0)

    original_ready_rows = [
        row
        for row in design_review_report.get("readyDesignRows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("review_status")) == REVIEW_STATUS_READY
    ] if isinstance(design_review_report, dict) else []

    if not original_ready_rows and isinstance(design_review_report, dict):
        original_ready_rows = [
            row
            for row in design_review_report.get("rows", [])
            if isinstance(row, dict)
            and _safe_text(row.get("review_status")) == REVIEW_STATUS_READY
        ]

    disambiguation_candidates = [
        row
        for row in disambiguation_report.get("disambiguationDesignRows", [])
        if isinstance(row, dict)
        and _safe_text(row.get("disambiguation_status")) == DISAMBIGUATION_STATUS_CANDIDATE
    ] if isinstance(disambiguation_report, dict) else []

    still_ambiguous_rows = [
        row
        for row in disambiguation_report.get("stillAmbiguousRows", [])
        if isinstance(row, dict)
    ] if isinstance(disambiguation_report, dict) else []

    if not still_ambiguous_rows and isinstance(disambiguation_report, dict):
        still_ambiguous_rows = [
            row
            for row in disambiguation_report.get("rows", [])
            if isinstance(row, dict)
            and _safe_text(row.get("disambiguation_status")) == DISAMBIGUATION_STATUS_BLOCKED_STILL_NON_UNIQUE
        ]

    ready_from_original = len(original_ready_rows)
    ready_from_disambiguation = len(disambiguation_candidates)
    still_blocked = len(still_ambiguous_rows)

    final_ready_rows = [
        _final_ready_from_review(row, index=index)
        for index, row in enumerate(original_ready_rows)
    ]
    final_ready_rows.extend(
        _final_ready_from_disambiguation(row, index=ready_from_original + index)
        for index, row in enumerate(disambiguation_candidates)
    )
    follow_up_rows = [
        _follow_up_from_disambiguation(row, index=index)
        for index, row in enumerate(still_ambiguous_rows)
    ]

    ready_span_ids = {_safe_text(row.get("sourceSpanId")) for row in final_ready_rows if _safe_text(row.get("sourceSpanId"))}
    if len(ready_span_ids) != len(final_ready_rows):
        warnings.append("duplicate_source_span_id_in_final_ready_rows")

    original_span_ids = {
        _safe_text(row.get("sourceSpanId")) for row in original_ready_rows if _safe_text(row.get("sourceSpanId"))
    }
    disambig_span_ids = {
        _safe_text(row.get("sourceSpanId")) for row in disambiguation_candidates if _safe_text(row.get("sourceSpanId"))
    }
    if original_span_ids & disambig_span_ids:
        warnings.append("source_span_overlap_between_original_ready_and_disambiguation_candidates")

    expected_total = input_design_review_rows or (ready_from_original + input_disambiguation_rows)
    reconciled_total = len(final_ready_rows) + len(follow_up_rows)
    if expected_total and reconciled_total != expected_total:
        warnings.append(
            f"reconciled_row_count_mismatch expected={expected_total} actual={reconciled_total}"
        )

    if input_disambiguation_rows and input_disambiguation_rows != ready_from_disambiguation + still_blocked:
        warnings.append("disambiguation_row_count_mismatch_with_input_target_rows")

    counts = _count_rows(
        final_ready_rows=final_ready_rows,
        follow_up_rows=follow_up_rows,
        input_design_review_rows=input_design_review_rows,
        input_disambiguation_rows=input_disambiguation_rows,
        ready_from_original=ready_from_original,
        ready_from_disambiguation=ready_from_disambiguation,
        input_schema_violations=_dedupe(input_schema_violations),
    )

    status = "ok"
    if input_schema_violations:
        status = "blocked"
        for row in final_ready_rows:
            row["final_status"] = FINAL_STATUS_BLOCKED_INPUT_SCHEMA
            row["recommended_action"] = RECOMMENDED_ACTION_REPAIR
            row["readyForStrictEvidenceDesignReview"] = False
        for row in follow_up_rows:
            row["final_status"] = FINAL_STATUS_BLOCKED_INPUT_SCHEMA
            row["recommended_action"] = RECOMMENDED_ACTION_REPAIR

    final_ready_count = int(counts.get("finalReadyForStrictEvidenceDesignReviewRows") or 0)
    packet_ready = (
        status == "ok"
        and final_ready_count > 0
        and not input_schema_violations
        and reconciled_total == (input_design_review_rows or reconciled_total)
    )

    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "input": {
            "designReviewReportPath": str(review_path),
            "designReviewSchema": _safe_text(design_review_report.get("schema")) if design_review_report else "",
            "disambiguationReportPath": str(disambiguation_path),
            "disambiguationSchema": _safe_text(disambiguation_report.get("schema")) if disambiguation_report else "",
        },
        "counts": counts,
        "gate": {
            "designReviewPacketReconciled": packet_ready,
            "strictEvidenceCreated": False,
            "strictEvidenceReady": False,
            "parserRoutingReady": False,
            "answerIntegrationReady": False,
            "runtimeMutationAllowed": False,
            "schemaViolations": _dedupe(input_schema_violations),
            "decision": (
                "parsed_artifact_source_span_strict_evidence_design_review_reconciliation_ready"
                if packet_ready
                else "blocked"
            ),
            "recommendedNextTranche": (
                "parsed_artifact_source_span_strict_evidence_design_packet_review"
                if packet_ready
                else "parsed_artifact_source_span_strict_evidence_design_review_reconciliation_repair"
            ),
        },
        "policy": {
            "reportOnly": True,
            "sourceSpanStoreWrite": False,
            "strictEvidenceCreated": False,
            "strictEligibleMutation": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "warnings": _dedupe(warnings),
        "finalReadyRows": final_ready_rows,
        "manualOrExtractorFollowUpRows": follow_up_rows,
        "rows": [*final_ready_rows, *follow_up_rows],
    }


def _summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "schema",
            "status",
            "generatedAt",
            "input",
            "counts",
            "gate",
            "policy",
            "warnings",
            "finalReadyRows",
            "manualOrExtractorFollowUpRows",
            "rows",
        )
        if key in report
    }


def render_parsed_artifact_source_span_strict_evidence_design_review_reconciliation_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    by_status = [
        f"{status}: {count}"
        for status, count in sorted((dict(counts.get("byFinalStatus") or {})).items())
    ]
    by_source = [
        f"{source}: {count}"
        for source, count in sorted((dict(counts.get("bySource") or {})).items())
    ]
    return "\n".join(
        [
            "# Parsed Artifact SourceSpan Strict Evidence Design Review Reconciliation",
            "",
            f"- status: {report.get('status', '')}",
            f"- report-only: {json.dumps(report.get('policy', {}).get('reportOnly'))}",
            f"- input design review rows: {int(counts.get('inputDesignReviewRows') or 0)}",
            f"- input disambiguation rows: {int(counts.get('inputDisambiguationRows') or 0)}",
            f"- final ready rows: {int(counts.get('finalReadyForStrictEvidenceDesignReviewRows') or 0)}",
            f"- still blocked ambiguous rows: {int(counts.get('stillBlockedAmbiguousRows') or 0)}",
            f"- strict evidence created: {int(counts.get('strictEvidenceCreatedRows') or 0)}",
            "",
            "## Final status breakdown",
            *[f"- {item}" for item in by_status],
            "",
            "## Source breakdown",
            *[f"- {item}" for item in by_source],
        ]
    )


def write_parsed_artifact_source_span_strict_evidence_design_review_reconciliation_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation.json"
    summary_path = (
        root / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation-summary.json"
    )
    markdown_path = (
        root / "parsed-artifact-source-span-strict-evidence-design-review-reconciliation.md"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(_summary_payload(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_parsed_artifact_source_span_strict_evidence_design_review_reconciliation_markdown(
            report
        ),
        encoding="utf-8",
    )
    return {"report": str(report_path), "summary": str(summary_path), "markdown": str(markdown_path)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = ArgumentParser(
        description=(
            "Reconcile strict-evidence design-review ready rows with disambiguation "
            "candidates into a final design-review packet without mutating SourceSpan "
            "records or creating StrictEvidence."
        )
    )
    parser.add_argument(
        "--design-review-report",
        default=str(DEFAULT_DESIGN_REVIEW_REPORT_PATH),
        help="Strict-evidence design review JSON report.",
    )
    parser.add_argument(
        "--disambiguation-report",
        default=str(DEFAULT_DISAMBIGUATION_REPORT_PATH),
        help="Text-match disambiguation design JSON report.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true", help="Print summary payload as JSON.")
    args = parser.parse_args(argv)

    report = build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation(
        design_review_report_path=args.design_review_report,
        disambiguation_report_path=args.disambiguation_report,
    )

    if args.output_dir:
        paths = write_parsed_artifact_source_span_strict_evidence_design_review_reconciliation_reports(
            report,
            args.output_dir,
        )
        print(f"wrote report: {paths['report']}")
        print(f"wrote summary: {paths['summary']}")
        print(f"wrote markdown: {paths['markdown']}")

    if args.json or not args.output_dir:
        print(json.dumps(_summary_payload(report), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_SOURCE_SPAN_STRICT_EVIDENCE_DESIGN_REVIEW_RECONCILIATION_SCHEMA_ID",
    "SOURCE_ORIGINAL_DESIGN_REVIEW",
    "SOURCE_DISAMBIGUATION_DESIGN",
    "FINAL_STATUS_READY",
    "build_parsed_artifact_source_span_strict_evidence_design_review_reconciliation",
    "write_parsed_artifact_source_span_strict_evidence_design_review_reconciliation_reports",
]
