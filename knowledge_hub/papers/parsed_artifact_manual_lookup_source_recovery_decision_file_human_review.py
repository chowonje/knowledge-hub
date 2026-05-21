"""Report-only manual review sheet for manual lookup source recovery decisions.

Joins the validated decision file with review context into an operator-readable
sheet. It does not record human decisions, perform lookup, download PDFs, mutate
source registrations, materialize parsed artifacts, create evidence, scan the
vault, or change answer behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_review_pack import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-decision-file-human-review.v1"
)

BLOCKER_MANUAL_LOOKUP_REQUIRED = "manual_lookup_required"
BLOCKER_TEXT_SOURCE_UNSUPPORTED = "text_source_unsupported"


def _default_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


DEFAULT_DECISION_FILE_VALIDATION_REPORT_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-validation"
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation.json"
)
DEFAULT_DECISION_FILE_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-draft"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft"
    / "manual-lookup-source-recovery-decisions.draft.json"
)
DEFAULT_REVIEW_PACK_REPORT_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-review-pack"
    / "01-parsed-artifact-manual-lookup-source-recovery-review-pack"
    / "parsed-artifact-manual-lookup-source-recovery-review-pack.json"
)
DEFAULT_SOURCE_RECOVERY_FEASIBILITY_REPORT_PATH = (
    _default_report_root()
    / "parsed-artifact-source-recovery-feasibility-post-oversized-apply"
    / "parsed-artifact-source-recovery-feasibility.json"
)
DEFAULT_OUTPUT_DIR = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-human-review"
)

EXPECTED_MANUAL_LOOKUP_ROWS = 15
EXPECTED_TEXT_SOURCE_HOLDOUT_ROWS = 1
ALLOWED_REMAINING_BLOCKERS = frozenset({BLOCKER_MANUAL_LOOKUP_REQUIRED, BLOCKER_TEXT_SOURCE_UNSUPPORTED})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _mutation_policy() -> dict[str, Any]:
    return {
        "externalLookup": False,
        "sourceDownload": False,
        "sourcePathRewrite": False,
        "sourceRegistrationMutation": False,
        "parsedArtifactWrite": False,
        "parserRouting": False,
        "sourceSpanCreated": False,
        "strictEvidence": False,
        "citationEvidence": False,
        "runtimeEvidence": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "vaultWrite": False,
        "answerIntegration": False,
        "manualBlockerResolution": False,
        "runManifestWrite": False,
        "decisionFileMutation": False,
        "humanDecisionRecording": False,
    }


def _mutation_counters() -> dict[str, int]:
    return {
        "externalLookupRows": 0,
        "sourceDownloadRows": 0,
        "sourcePathRewriteRows": 0,
        "sourceRegistrationMutationRows": 0,
        "parsedArtifactWriteRows": 0,
        "parserRoutingRows": 0,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationEvidenceRows": 0,
        "runtimeEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reindexOrReembedRows": 0,
        "vaultReadRows": 0,
        "vaultWriteRows": 0,
        "answerIntegrationRows": 0,
        "manualBlockerResolutionRows": 0,
        "runManifestWriteRows": 0,
        "decisionFileMutationRows": 0,
        "humanDecisionRecordingRows": 0,
    }


def _row_key(paper_id: str, source_review_card_id: str) -> str:
    return f"{paper_id}::{source_review_card_id}"


def _unsafe_validation_flags(validation_report: dict[str, Any], validation_ok: bool) -> list[str]:
    flags: list[str] = []
    gate = dict(validation_report.get("gate") or {})
    counts = dict(validation_report.get("counts") or {})
    if not validation_ok:
        flags.append("decision_file_validation_report_schema_violation")
    if validation_report.get("schema") != PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID:
        flags.append("decision_file_validation_report_schema_mismatch")
    if _clean_text(validation_report.get("status")) != "decision_file_validation_ready":
        flags.append(f"decision_file_validation_status={_clean_text(validation_report.get('status')) or 'unknown'}")
    if not bool(gate.get("decisionFileValidationReady")):
        flags.append("decision_file_validation_not_ready")
    if list(gate.get("schemaViolations") or []):
        flags.extend(str(item) for item in gate.get("schemaViolations") or [])
    if list(gate.get("unsafeUpstreamFlags") or []):
        flags.extend(str(item) for item in gate.get("unsafeUpstreamFlags") or [])
    if int(counts.get("invalidRows") or 0) > 0:
        flags.append("decision_file_validation_invalid_rows_present")
    if int(counts.get("applyReadyRows") or 0) > 0:
        flags.append("decision_file_validation_contains_apply_ready_rows")
    return list(dict.fromkeys(flags))


def _review_pack_index(review_pack_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not review_pack_report:
        return {}
    return {
        _clean_text(card.get("reviewCardId")): dict(card)
        for card in list(review_pack_report.get("reviewCards") or [])
        if isinstance(card, dict) and _clean_text(card.get("reviewCardId"))
    }


def _decision_index(decision_file: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _row_key(_clean_text(row.get("paperId")), _clean_text(row.get("sourceReviewCardId"))): dict(row)
        for row in list(decision_file.get("decisions") or [])
        if isinstance(row, dict)
    }


def _text_holdout_rows(
    review_pack_report: dict[str, Any] | None,
    *,
    holdout_paper_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if review_pack_report:
        for row in list(review_pack_report.get("textSourceHoldouts") or []):
            if isinstance(row, dict):
                rows.append(dict(row))
    if rows:
        return rows
    return [
        {
            "paperId": paper_id,
            "textSourceHoldoutStatus": "text_source_contract_holdout",
            "recommendedAction": "parsed_artifact_text_source_materialization_contract",
        }
        for paper_id in sorted(holdout_paper_ids)
    ]


def _remaining_blocker_taxonomy(
    *,
    manual_lookup_rows: int,
    text_source_rows: int,
    feasibility_report: dict[str, Any] | None,
) -> dict[str, Any]:
    taxonomy = [
        {
            "reason": BLOCKER_MANUAL_LOOKUP_REQUIRED,
            "count": manual_lookup_rows,
            "workflow": "human_decision_file_review",
        },
        {
            "reason": BLOCKER_TEXT_SOURCE_UNSUPPORTED,
            "count": text_source_rows,
            "workflow": "separate_text_source_contract_holdout",
        },
    ]
    unexpected: list[str] = []
    missing_parsed = int(dict((feasibility_report or {}).get("sourceBlockerSnapshot") or {}).get("missingParsedArtifacts") or 0)
    if feasibility_report:
        for item in list(dict((feasibility_report.get("sourceBlockerSnapshot") or {})).get("blockerTaxonomy") or []):
            if not isinstance(item, dict):
                continue
            reason = _clean_text(item.get("reason"))
            if reason in {"source_pdf_missing", BLOCKER_MANUAL_LOOKUP_REQUIRED}:
                continue
            if reason == "text_source_unsupported":
                continue
            if reason == "source_pdf_oversized":
                unexpected.append(f"unexpected_oversized_blocker:{int(item.get('count') or 0)}")
            elif reason:
                unexpected.append(f"unexpected_blocker:{reason}:{int(item.get('count') or 0)}")
        for item in list(dict((feasibility_report.get("sourceMissingRecoverySummary") or {})).get("reacquisitionTaxonomy") or []):
            if not isinstance(item, dict):
                continue
            reason = _clean_text(item.get("reason"))
            if reason and reason not in ALLOWED_REMAINING_BLOCKERS:
                unexpected.append(f"unexpected_reacquisition:{reason}:{int(item.get('count') or 0)}")
    return {
        "automatedRecoveryTranchesClosed": not unexpected,
        "missingParsedArtifactsRemaining": missing_parsed or manual_lookup_rows + text_source_rows,
        "blockerTaxonomy": taxonomy,
        "unexpectedBlockerReasons": list(dict.fromkeys(unexpected)),
        "unexpectedBlockerCount": len(unexpected),
    }


def _manual_review_row(
    index: int,
    validation_row: dict[str, Any],
    *,
    decision_row: dict[str, Any] | None,
    review_card: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = _clean_text((decision_row or validation_row).get("decision")) or "needs_review"
    lookup_queries = dict((decision_row or {}).get("lookupQueries") or (review_card or {}).get("lookupQueries") or {})
    return {
        "humanReviewRowId": f"manual-lookup-source-recovery-decision-file-human-review:{index:04d}",
        "sourceReviewCardId": _clean_text(validation_row.get("sourceReviewCardId")),
        "paperId": _clean_text(validation_row.get("paperId")),
        "paperTitle": _clean_text(validation_row.get("paperTitle")),
        "manualLookupMode": _clean_text(validation_row.get("manualLookupMode")),
        "lookupPriority": _clean_text(validation_row.get("lookupPriority")),
        "lookupQueries": lookup_queries,
        "preferredReviewOrder": list((review_card or {}).get("preferredReviewOrder") or []),
        "requiredFutureApproval": _clean_text((review_card or {}).get("requiredFutureApproval")),
        "futureApplyTarget": _clean_text((review_card or {}).get("futureApplyTarget")),
        "currentDecision": decision,
        "allowedDecisions": list(validation_row.get("allowedDecisions") or []),
        "validationStatus": _clean_text(validation_row.get("validationStatus")),
        "validationBlockers": list(validation_row.get("validationBlockers") or []),
        "reviewPrompt": (
            "Inspect lookup queries and source context, then edit the decision draft file. "
            "Leave needs_review when unsure; approved URL/local PDF decisions require reviewer fields."
        ),
        "decisionEditTarget": "manual-lookup-source-recovery-decisions.draft.json",
        "decisionScope": "manual_lookup_source_recovery_decision_file_human_review_only_no_apply",
        "evidenceTier": "manual_lookup_source_recovery_decision_file_human_review_only",
        "reportOnly": True,
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "strictEligible": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerIntegration": False,
    }


def build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
    *,
    decision_file_validation_report: dict[str, Any],
    decision_file: dict[str, Any],
    decision_file_validation_report_path: str | Path | None = None,
    decision_file_path: str | Path | None = None,
    review_pack_report: dict[str, Any] | None = None,
    review_pack_report_path: str | Path | None = None,
    source_recovery_feasibility_report: dict[str, Any] | None = None,
    source_recovery_feasibility_report_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review",
    generated_at: str | None = None,
    expected_manual_lookup_rows: int = EXPECTED_MANUAL_LOOKUP_ROWS,
    expected_text_source_holdout_rows: int = EXPECTED_TEXT_SOURCE_HOLDOUT_ROWS,
) -> dict[str, Any]:
    """Build a report-only manual review sheet from a validated decision file."""

    validation_result = validate_payload(
        decision_file_validation_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    unsafe_flags = _unsafe_validation_flags(decision_file_validation_report, bool(validation_result.ok))

    review_pack_validation = (
        validate_payload(
            review_pack_report,
            PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_REVIEW_PACK_SCHEMA_ID,
            strict=True,
        )
        if review_pack_report
        else None
    )
    if review_pack_report and review_pack_validation and not review_pack_validation.ok:
        unsafe_flags.extend(
            str(error) for error in review_pack_validation.errors[:3]
        )
        unsafe_flags.append("review_pack_report_schema_violation")

    feasibility_validation = (
        validate_payload(
            source_recovery_feasibility_report,
            PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
            strict=True,
        )
        if source_recovery_feasibility_report
        else None
    )
    if source_recovery_feasibility_report and feasibility_validation and not feasibility_validation.ok:
        unsafe_flags.append("source_recovery_feasibility_report_schema_violation")

    validation_rows = [
        dict(row)
        for row in list(decision_file_validation_report.get("validationRows") or [])
        if isinstance(row, dict)
    ] if validation_result.ok else []

    holdout_paper_ids = {
        _clean_text(item) for item in list(decision_file_validation_report.get("textSourceHoldoutPaperIds") or [])
    }
    holdout_paper_ids.discard("")

    decision_index = _decision_index(decision_file)
    review_card_index = _review_pack_index(review_pack_report)

    manual_rows: list[dict[str, Any]] = []
    for index, validation_row in enumerate(validation_rows, start=1):
        key = _row_key(
            _clean_text(validation_row.get("paperId")),
            _clean_text(validation_row.get("sourceReviewCardId")),
        )
        manual_rows.append(
            _manual_review_row(
                index,
                validation_row,
                decision_row=decision_index.get(key),
                review_card=review_card_index.get(_clean_text(validation_row.get("sourceReviewCardId"))),
            )
        )

    text_holdouts = _text_holdout_rows(review_pack_report, holdout_paper_ids=holdout_paper_ids)
    remaining_blockers = _remaining_blocker_taxonomy(
        manual_lookup_rows=len(manual_rows),
        text_source_rows=len(text_holdouts),
        feasibility_report=source_recovery_feasibility_report,
    )
    if remaining_blockers["unexpectedBlockerCount"] > 0:
        unsafe_flags.extend(remaining_blockers["unexpectedBlockerReasons"])

    decision_counter = Counter(_clean_text(row.get("currentDecision")) for row in manual_rows)
    validation_counter = Counter(_clean_text(row.get("validationStatus")) for row in manual_rows)
    counts = {
        "manualReviewRows": len(manual_rows),
        "needsReviewRows": decision_counter.get("needs_review", 0),
        "nonNeedsReviewRows": len(manual_rows) - decision_counter.get("needs_review", 0),
        "validValidationRows": sum(
            1 for row in manual_rows if _clean_text(row.get("validationStatus")).startswith("valid_")
        ),
        "textSourceHoldoutRows": len(text_holdouts),
        "unexpectedBlockerCount": remaining_blockers["unexpectedBlockerCount"],
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "decisionTaxonomy": _counter_items(decision_counter),
        "validationStatusTaxonomy": _counter_items(validation_counter),
    }
    counts.update(_mutation_counters())

    taxonomy_closed = (
        not unsafe_flags
        and len(manual_rows) == expected_manual_lookup_rows
        and len(text_holdouts) == expected_text_source_holdout_rows
        and remaining_blockers["unexpectedBlockerCount"] == 0
    )
    all_needs_review = (
        taxonomy_closed
        and counts["needsReviewRows"] == expected_manual_lookup_rows
        and counts["nonNeedsReviewRows"] == 0
    )
    status = (
        "decision_file_human_review_ready"
        if all_needs_review
        else "blocked"
    )
    gate_decision = (
        "manual_lookup_source_recovery_automated_blocker_tranche_closed_pending_human_decisions"
        if all_needs_review
        else "manual_lookup_source_recovery_decision_file_human_review_blocked"
    )

    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputDecisionFileValidationReportPath": str(Path(str(decision_file_validation_report_path)).expanduser())
            if decision_file_validation_report_path
            else "",
            "inputDecisionFilePath": str(Path(str(decision_file_path)).expanduser()) if decision_file_path else "",
            "inputReviewPackReportPath": str(Path(str(review_pack_report_path)).expanduser())
            if review_pack_report_path
            else "",
            "inputSourceRecoveryFeasibilityReportPath": str(
                Path(str(source_recovery_feasibility_report_path)).expanduser()
            )
            if source_recovery_feasibility_report_path
            else "",
            "selectionRule": (
                "join validated decision rows with review context for operator human review; "
                "text-source holdouts remain excluded from editable decisions"
            ),
            "nonScope": [
                "external_lookup",
                "source_download",
                "source_path_rewrite",
                "source_registration_mutation",
                "parsed_artifact_write",
                "parser_routing",
                "source_span_creation",
                "strict_or_citation_or_runtime_evidence",
                "database_or_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
                "manual_blocker_resolution",
                "decision_file_mutation",
                "human_decision_recording",
                "approval_application",
                "run_manifest_write",
            ],
        },
        "inputDecisionFileValidation": {
            "schemaValidation": {"ok": bool(validation_result.ok), "errors": list(validation_result.errors)},
            "schema": _clean_text(decision_file_validation_report.get("schema")),
            "status": _clean_text(decision_file_validation_report.get("status")),
            "validRows": int(dict(decision_file_validation_report.get("counts") or {}).get("validRows") or 0)
            if validation_result.ok
            else 0,
        },
        "inputDecisionFile": {
            "draftOnly": bool(decision_file.get("draftOnly")),
            "decisionRows": len(list(decision_file.get("decisions") or [])),
        },
        "counts": counts,
        "coverageTarget": dict(decision_file_validation_report.get("coverageTarget") or {}),
        "remainingBlockerClosure": remaining_blockers,
        "textSourceHoldoutPaperIds": sorted(holdout_paper_ids),
        "textSourceHoldouts": text_holdouts,
        "humanReviewRows": manual_rows,
        "gate": {
            "humanReviewSheetReady": all_needs_review,
            "automatedBlockerTrancheClosed": taxonomy_closed,
            "containsOnlyNeedsReviewDecisions": counts["needsReviewRows"] == len(manual_rows) and bool(manual_rows),
            "containsNonNeedsReviewDecisions": counts["nonNeedsReviewRows"] > 0,
            "applyReady": False,
            "sourceRegistrationMutationReady": False,
            "parsedArtifactMaterializationReady": False,
            "parserRoutingReady": False,
            "strictEvidenceReady": False,
            "answerIntegrationReady": False,
            "decision": gate_decision,
            "schemaViolations": list(dict.fromkeys(unsafe_flags)),
            "unsafeUpstreamFlags": list(dict.fromkeys(unsafe_flags)),
            "recommendedNextTranche": (
                "parsed_artifact_manual_lookup_source_recovery_operator_decision_packet"
                if all_needs_review
                else "parsed_artifact_manual_lookup_source_recovery_decision_file_validation_repair"
            ),
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "operatorInstructions": [
            "Edit manual-lookup-source-recovery-decisions.draft.json locally after reviewing each row.",
            "Leave decision=needs_review when unsure.",
            "Approved URL/local PDF decisions require approvedSourceType, exactly one source locator, approvedSourceContentHash, approvedBy, approvedAt, and notes.",
            "Reject/hold decisions require reviewer and notes.",
            "Re-run decision-file validation after edits; do not treat needs_review as approval.",
            "Text-source holdout rows remain outside this decision file until a separate text-source contract tranche.",
        ],
        "warnings": [
            "human_review_sheet_rows_are_not_recorded_decisions",
            "human_review_does_not_apply_approved_sources",
            "text_source_holdout_rows_remain_excluded",
            "automated_source_recovery_tranches_are_closed_for_pdf_backed_blockers",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file human review schema failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write human review JSON, summary JSON, and Markdown."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file human review schema failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review.json"
    summary_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review-summary.json"
    markdown_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-human-review.md"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "coverageTarget": report.get("coverageTarget"),
        "remainingBlockerClosure": report.get("remainingBlockerClosure"),
        "gate": report.get("gate"),
        "mutationCounters": report.get("mutationCounters"),
        "reportFiles": {"reportJsonPath": str(report_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "summaryJsonPath": str(summary_path),
        "reportMarkdownPath": str(markdown_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    closure = dict(report.get("remainingBlockerClosure") or {})
    lines = [
        "# Parsed Artifact Manual Lookup Source Recovery Decision File Human Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- manual review rows: {counts.get('manualReviewRows', 0)}",
        f"- needs_review rows: {counts.get('needsReviewRows', 0)}",
        f"- text-source holdout rows: {counts.get('textSourceHoldoutRows', 0)}",
        f"- unexpected blockers: {counts.get('unexpectedBlockerCount', 0)}",
        f"- automated blocker tranche closed: `{gate.get('automatedBlockerTrancheClosed')}`",
        f"- recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
        "",
        "## Remaining Blocker Taxonomy",
        "",
    ]
    for item in list(closure.get("blockerTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')} ({item.get('workflow')})")
    lines.extend(["", "## Operator Instructions", ""])
    for item in list(report.get("operatorInstructions") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-file-validation-report",
        default=str(DEFAULT_DECISION_FILE_VALIDATION_REPORT_PATH),
    )
    parser.add_argument("--decision-file", default=str(DEFAULT_DECISION_FILE_PATH))
    parser.add_argument("--review-pack-report", default=str(DEFAULT_REVIEW_PACK_REPORT_PATH))
    parser.add_argument(
        "--source-recovery-feasibility-report",
        default=str(DEFAULT_SOURCE_RECOVERY_FEASIBILITY_REPORT_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write local JSON/Markdown review reports. Default is dry-run summary only.",
    )
    parser.add_argument(
        "--report-name",
        default="parsed-artifact-manual-lookup-source-recovery-decision-file-human-review",
    )
    args = parser.parse_args(argv)

    validation_report_path = Path(args.decision_file_validation_report).expanduser()
    decision_file_path = Path(args.decision_file).expanduser()
    review_pack_path = Path(args.review_pack_report).expanduser()
    feasibility_path = Path(args.source_recovery_feasibility_report).expanduser()

    review_pack = _load_json(review_pack_path) if review_pack_path.exists() else None
    feasibility = _load_json(feasibility_path) if feasibility_path.exists() else None

    report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
        decision_file_validation_report=_load_json(validation_report_path),
        decision_file=_load_json(decision_file_path),
        decision_file_validation_report_path=validation_report_path,
        decision_file_path=decision_file_path,
        review_pack_report=review_pack,
        review_pack_report_path=review_pack_path if review_pack else None,
        source_recovery_feasibility_report=feasibility,
        source_recovery_feasibility_report_path=feasibility_path if feasibility else None,
        report_name=args.report_name,
    )

    payload: dict[str, Any] = {
        "schema": report["schema"],
        "status": report["status"],
        "counts": report["counts"],
        "remainingBlockerClosure": report["remainingBlockerClosure"],
        "gate": report["gate"],
        "mutationCounters": report["mutationCounters"],
    }
    if args.apply:
        paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review(
            report,
            args.output_dir,
        )
        payload["paths"] = paths
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_HUMAN_REVIEW_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review",
    "write_parsed_artifact_manual_lookup_source_recovery_decision_file_human_review",
]
