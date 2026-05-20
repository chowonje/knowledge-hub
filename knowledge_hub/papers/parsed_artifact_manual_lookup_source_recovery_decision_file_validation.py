"""Report-only validator for manual parsed-artifact lookup source recovery decisions.

Strict-validates the decision-file draft report and an editable decision JSON,
matches rows back to draft metadata, and classifies decision semantics without
performing lookup, downloads, source registration mutation, or parsed-artifact
writes.
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
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_draft import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-decision-file-validation.v1"
)

DECISION_NEEDS_REVIEW = "needs_review"
DECISION_APPROVE_SOURCE_URL = "approve_source_url_for_later_apply"
DECISION_APPROVE_LOCAL_PDF = "approve_local_pdf_path_for_later_apply"
DECISION_REJECT = "reject_candidate_keep_missing"
DECISION_HOLD = "hold_for_manual_lookup"

APPROVED_DECISIONS = frozenset({DECISION_APPROVE_SOURCE_URL, DECISION_APPROVE_LOCAL_PDF})
REJECTED_DECISIONS = frozenset({DECISION_REJECT})
HOLD_DECISIONS = frozenset({DECISION_HOLD})

ROW_STATUS_VALID_NEEDS_REVIEW = "valid_needs_review"
ROW_STATUS_VALID_APPLY_READY = "valid_apply_ready_for_later_apply"
ROW_STATUS_VALID_REJECTED = "valid_rejected_decision"
ROW_STATUS_VALID_HOLD = "valid_hold_for_manual_lookup"
ROW_STATUS_INVALID_MISSING_DRAFT_MATCH = "invalid_missing_draft_row_match"
ROW_STATUS_INVALID_DISALLOWED_DECISION = "invalid_disallowed_decision"
ROW_STATUS_INVALID_APPROVAL_FIELDS = "invalid_approval_fields"
ROW_STATUS_INVALID_REJECT_HOLD_FIELDS = "invalid_reject_or_hold_fields"
ROW_STATUS_INVALID_NEEDS_REVIEW_FIELDS = "invalid_needs_review_fields"
ROW_STATUS_INVALID_TEXT_SOURCE_HOLDOUT = "invalid_text_source_holdout_row"
ROW_STATUS_INVALID_DUPLICATE_KEY = "invalid_duplicate_paper_or_review_card"

DEFAULT_DECISION_FILE_DRAFT_REPORT_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
    "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
    "parsed-artifact-manual-lookup-source-recovery-decision-file-draft.json"
).expanduser()
DEFAULT_DECISION_FILE_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
    "01-parsed-artifact-manual-lookup-source-recovery-decision-file-draft/"
    "manual-lookup-source-recovery-decisions.draft.json"
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-decision-file-validation/"
    "01-parsed-artifact-manual-lookup-source-recovery-decision-file-validation"
).expanduser()

EXPECTED_INPUT_ROWS = 15
EXPECTED_TEXT_SOURCE_HOLDOUT_ROWS = 1


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
    }


def _draft_row_key(paper_id: str, source_review_card_id: str) -> str:
    return f"{paper_id}::{source_review_card_id}"


def _unsafe_draft_flags(draft_report: dict[str, Any], draft_validation_ok: bool) -> list[str]:
    flags: list[str] = []
    gate = dict(draft_report.get("gate") or {})
    counts = dict(draft_report.get("counts") or {})
    if not draft_validation_ok:
        flags.append("decision_file_draft_report_schema_violation")
    if draft_report.get("schema") != PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID:
        flags.append("decision_file_draft_report_schema_mismatch")
    if _clean_text(draft_report.get("status")) != "decision_file_draft_ready":
        flags.append(f"decision_file_draft_status={_clean_text(draft_report.get('status')) or 'unknown'}")
    if not bool(gate.get("decisionFileDraftReady")):
        flags.append("decision_file_draft_not_ready")
    if list(gate.get("unsafeUpstreamFlags") or []):
        flags.extend(str(item) for item in gate.get("unsafeUpstreamFlags") or [])
    if bool(gate.get("containsAcceptedSourceApprovals")):
        flags.append("draft_report_contains_accepted_source_approvals")
    if bool(gate.get("applyReady")):
        flags.append("draft_report_apply_ready_true")
    for key in ("approvedDecisionRows", "applyReadyRows"):
        if int(counts.get(key) or 0) > 0:
            flags.append(f"draft_report_{key}_nonzero")
    return list(dict.fromkeys(flags))


def _validate_decision_file_shape(decision_file: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "decisions" not in decision_file:
        errors.append("decision_file_missing_decisions_array")
        return errors
    decisions = decision_file.get("decisions")
    if not isinstance(decisions, list):
        errors.append("decision_file_decisions_not_array")
        return errors
    required_fields = (
        "sourceReviewCardId",
        "paperId",
        "paperTitle",
        "manualLookupMode",
        "lookupPriority",
        "lookupQueries",
        "allowedDecisions",
        "decision",
        "approvedSourceType",
        "approvedSourceUrl",
        "approvedLocalPdfPath",
        "approvedSourceContentHash",
        "approvedBy",
        "approvedAt",
        "reviewer",
        "notes",
    )
    for index, row in enumerate(decisions):
        if not isinstance(row, dict):
            errors.append(f"decision_row_{index}_not_object")
            continue
        for field_name in required_fields:
            if field_name not in row:
                errors.append(f"decision_row_{index}_missing_{field_name}")
    return errors


def _approval_fields_present(row: dict[str, Any]) -> bool:
    return any(
        _clean_text(row.get(field_name))
        for field_name in (
            "approvedSourceType",
            "approvedSourceUrl",
            "approvedLocalPdfPath",
            "approvedSourceContentHash",
            "approvedBy",
            "approvedAt",
        )
    )


def _validate_decision_semantics(
    row: dict[str, Any],
    *,
    draft_row: dict[str, Any] | None,
    holdout_paper_ids: set[str],
) -> tuple[str, list[str], bool]:
    blockers: list[str] = []
    paper_id = _clean_text(row.get("paperId"))
    card_id = _clean_text(row.get("sourceReviewCardId"))
    decision = _clean_text(row.get("decision"))

    if paper_id in holdout_paper_ids:
        return ROW_STATUS_INVALID_TEXT_SOURCE_HOLDOUT, ["text_source_holdout_row_excluded"], False

    if draft_row is None:
        return ROW_STATUS_INVALID_MISSING_DRAFT_MATCH, ["draft_row_not_found_for_paper_and_review_card"], False

    allowed = {_clean_text(item) for item in (draft_row.get("allowedDecisions") or row.get("allowedDecisions") or [])}
    allowed.discard("")
    if decision not in allowed:
        blockers.append(f"decision_not_in_allowedDecisions:{decision}")
        return ROW_STATUS_INVALID_DISALLOWED_DECISION, blockers, False

    if _clean_text(draft_row.get("paperId")) != paper_id:
        blockers.append("paperId_does_not_match_draft_row")
    if _clean_text(draft_row.get("sourceReviewCardId")) != card_id:
        blockers.append("sourceReviewCardId_does_not_match_draft_row")
    if blockers:
        return ROW_STATUS_INVALID_MISSING_DRAFT_MATCH, blockers, False

    if decision == DECISION_NEEDS_REVIEW:
        if _approval_fields_present(row):
            blockers.append("needs_review_must_not_include_approval_fields")
        if _clean_text(row.get("reviewer")) or _clean_text(row.get("notes")):
            blockers.append("needs_review_must_not_include_reviewer_or_notes")
        if blockers:
            return ROW_STATUS_INVALID_NEEDS_REVIEW_FIELDS, blockers, False
        return ROW_STATUS_VALID_NEEDS_REVIEW, [], False

    if decision in APPROVED_DECISIONS:
        if not _clean_text(row.get("approvedSourceType")):
            blockers.append("approvedSourceType_required")
        if not _clean_text(row.get("approvedBy")):
            blockers.append("approvedBy_required")
        if not _clean_text(row.get("approvedAt")):
            blockers.append("approvedAt_required")
        if not _clean_text(row.get("notes")):
            blockers.append("notes_required")
        has_url = bool(_clean_text(row.get("approvedSourceUrl")))
        has_local = bool(_clean_text(row.get("approvedLocalPdfPath")))
        if has_url and has_local:
            blockers.append("approved_source_must_be_url_xor_local_pdf_not_both")
        if not has_url and not has_local:
            blockers.append("approved_source_requires_exactly_one_url_or_local_pdf")
        if decision == DECISION_APPROVE_SOURCE_URL:
            if not has_url:
                blockers.append("approve_source_url_requires_approvedSourceUrl")
            if has_local:
                blockers.append("approve_source_url_must_not_set_approvedLocalPdfPath")
        if decision == DECISION_APPROVE_LOCAL_PDF:
            if not has_local:
                blockers.append("approve_local_pdf_requires_approvedLocalPdfPath")
            if has_url:
                blockers.append("approve_local_pdf_must_not_set_approvedSourceUrl")
        if blockers:
            return ROW_STATUS_INVALID_APPROVAL_FIELDS, blockers, False
        return ROW_STATUS_VALID_APPLY_READY, [], True

    if decision in REJECTED_DECISIONS | HOLD_DECISIONS:
        if not _clean_text(row.get("reviewer")):
            blockers.append("reviewer_required")
        if not _clean_text(row.get("notes")):
            blockers.append("notes_required")
        if _approval_fields_present(row):
            blockers.append("reject_or_hold_must_not_include_approval_fields")
        if blockers:
            return ROW_STATUS_INVALID_REJECT_HOLD_FIELDS, blockers, False
        if decision in REJECTED_DECISIONS:
            return ROW_STATUS_VALID_REJECTED, [], False
        return ROW_STATUS_VALID_HOLD, [], False

    blockers.append(f"unknown_decision:{decision}")
    return ROW_STATUS_INVALID_DISALLOWED_DECISION, blockers, False


def _validation_row(
    index: int,
    row: dict[str, Any],
    *,
    draft_row: dict[str, Any] | None,
    holdout_paper_ids: set[str],
) -> dict[str, Any]:
    status, blockers, apply_ready = _validate_decision_semantics(
        row,
        draft_row=draft_row,
        holdout_paper_ids=holdout_paper_ids,
    )
    decision = _clean_text(row.get("decision"))
    valid = status.startswith("valid_")
    return {
        "validationRowId": f"manual-lookup-source-recovery-decision-file-validation:{index:04d}",
        "sourceReviewCardId": _clean_text(row.get("sourceReviewCardId")),
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "manualLookupMode": _clean_text(row.get("manualLookupMode")),
        "lookupPriority": _clean_text(row.get("lookupPriority")),
        "decision": decision,
        "allowedDecisions": list(row.get("allowedDecisions") or []),
        "validationStatus": status,
        "validationBlockers": blockers,
        "validDecisionRow": valid,
        "applyReadyForLaterApply": apply_ready,
        "needsReviewDecisionRow": decision == DECISION_NEEDS_REVIEW and valid,
        "approvedDecisionRow": decision in APPROVED_DECISIONS and valid,
        "rejectedDecisionRow": decision in REJECTED_DECISIONS and valid,
        "holdForManualLookupRow": decision in HOLD_DECISIONS and valid,
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


def build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
    *,
    decision_file_draft_report: dict[str, Any],
    decision_file: dict[str, Any],
    decision_file_draft_report_path: str | Path | None = None,
    decision_file_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-decision-file-validation",
    generated_at: str | None = None,
    expected_input_rows: int = EXPECTED_INPUT_ROWS,
    expected_text_source_holdout_rows: int = EXPECTED_TEXT_SOURCE_HOLDOUT_ROWS,
) -> dict[str, Any]:
    """Validate an editable manual lookup source recovery decision file."""

    draft_validation = validate_payload(
        decision_file_draft_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_DRAFT_SCHEMA_ID,
        strict=True,
    )
    unsafe_flags = _unsafe_draft_flags(decision_file_draft_report, bool(draft_validation.ok))
    decision_shape_errors = _validate_decision_file_shape(decision_file)

    draft_rows = [
        dict(row)
        for row in list(decision_file_draft_report.get("draftRows") or [])
        if isinstance(row, dict)
    ] if draft_validation.ok else []
    holdout_paper_ids = {
        _clean_text(item) for item in list(decision_file_draft_report.get("textSourceHoldoutPaperIds") or [])
    }
    holdout_paper_ids.discard("")

    draft_index = {
        _draft_row_key(_clean_text(row.get("paperId")), _clean_text(row.get("sourceReviewCardId"))): row
        for row in draft_rows
    }

    decision_rows = [
        dict(row)
        for row in list(decision_file.get("decisions") or [])
        if isinstance(row, dict)
    ] if isinstance(decision_file.get("decisions"), list) else []

    seen_keys: set[str] = set()
    validation_rows: list[dict[str, Any]] = []
    duplicate_blockers: list[str] = []
    for index, row in enumerate(decision_rows, start=1):
        key = _draft_row_key(_clean_text(row.get("paperId")), _clean_text(row.get("sourceReviewCardId")))
        if key in seen_keys:
            duplicate_blockers.append(f"duplicate_decision_key:{key}")
        seen_keys.add(key)
        validation_rows.append(
            _validation_row(
                index,
                row,
                draft_row=draft_index.get(key),
                holdout_paper_ids=holdout_paper_ids,
            )
        )

    if duplicate_blockers:
        for validation_row in validation_rows:
            key = _draft_row_key(validation_row["paperId"], validation_row["sourceReviewCardId"])
            if sum(
                1
                for item in validation_rows
                if _draft_row_key(item["paperId"], item["sourceReviewCardId"]) == key
            ) > 1:
                validation_row["validationStatus"] = ROW_STATUS_INVALID_DUPLICATE_KEY
                validation_row["validationBlockers"] = list(
                    dict.fromkeys(list(validation_row.get("validationBlockers") or []) + duplicate_blockers)
                )
                validation_row["validDecisionRow"] = False
                validation_row["applyReadyForLaterApply"] = False
                validation_row["needsReviewDecisionRow"] = False
                validation_row["approvedDecisionRow"] = False
                validation_row["rejectedDecisionRow"] = False
                validation_row["holdForManualLookupRow"] = False

    expected_draft_keys = set(draft_index)
    actual_keys = {_draft_row_key(row["paperId"], row["sourceReviewCardId"]) for row in validation_rows}
    missing_keys = expected_draft_keys - actual_keys
    extra_keys = actual_keys - expected_draft_keys - {
        _draft_row_key(paper_id, "") for paper_id in holdout_paper_ids
    }

    input_schema_violations = list(dict.fromkeys(
        (["decision_file_draft_report_missing_or_unreadable"] if not decision_file_draft_report else [])
        + ([] if draft_validation.ok else [str(error) for error in draft_validation.errors])
        + unsafe_flags
        + decision_shape_errors
        + ([f"missing_decision_rows:{len(missing_keys)}"] if missing_keys else [])
        + ([f"extra_decision_rows:{len(extra_keys)}"] if extra_keys else [])
    ))

    status_counter = Counter(_clean_text(row.get("validationStatus")) for row in validation_rows)
    decision_counter = Counter(_clean_text(row.get("decision")) for row in validation_rows)
    valid_rows = [row for row in validation_rows if bool(row.get("validDecisionRow"))]
    counts = {
        "inputRows": len(validation_rows),
        "validRows": len(valid_rows),
        "invalidRows": len(validation_rows) - len(valid_rows),
        "needsReviewRows": sum(1 for row in validation_rows if bool(row.get("needsReviewDecisionRow"))),
        "approvedDecisionRows": sum(1 for row in validation_rows if bool(row.get("approvedDecisionRow"))),
        "rejectedDecisionRows": sum(1 for row in validation_rows if bool(row.get("rejectedDecisionRow"))),
        "holdForManualLookupRows": sum(1 for row in validation_rows if bool(row.get("holdForManualLookupRow"))),
        "applyReadyRows": sum(1 for row in validation_rows if bool(row.get("applyReadyForLaterApply"))),
        "textSourceHoldoutRows": expected_text_source_holdout_rows,
        "draftDecisionRows": len(draft_rows),
        "missingDraftDecisionRows": len(missing_keys),
        "extraDecisionRows": len(extra_keys),
        "unsafeUpstreamFlagCount": len(unsafe_flags),
        "inputSchemaViolationCount": len(input_schema_violations),
        "validationStatusTaxonomy": _counter_items(status_counter),
        "decisionTaxonomy": _counter_items(decision_counter),
    }
    counts.update(_mutation_counters())

    all_valid = (
        not input_schema_violations
        and len(validation_rows) == expected_input_rows
        and len(valid_rows) == expected_input_rows
        and counts["invalidRows"] == 0
        and len(missing_keys) == 0
        and len(extra_keys) == 0
    )
    status = "decision_file_validation_ready" if all_valid else "blocked"
    gate_decision = (
        "manual_lookup_source_recovery_decision_file_validated_needs_review_only"
        if all_valid and counts["needsReviewRows"] == expected_input_rows
        else "manual_lookup_source_recovery_decision_file_validation_blocked"
        if not all_valid
        else "manual_lookup_source_recovery_decision_file_validated_with_apply_ready_rows"
    )

    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputDecisionFileDraftReportPath": str(Path(str(decision_file_draft_report_path)).expanduser())
            if decision_file_draft_report_path
            else "",
            "inputDecisionFilePath": str(Path(str(decision_file_path)).expanduser())
            if decision_file_path
            else "",
            "selectionRule": (
                "validate editable decision rows matched to draftRows by paperId and sourceReviewCardId; "
                "exclude textSourceHoldouts"
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
                "approval_application",
                "run_manifest_write",
            ],
        },
        "inputDecisionFileDraft": {
            "schemaValidation": {"ok": bool(draft_validation.ok), "errors": list(draft_validation.errors)},
            "schema": _clean_text(decision_file_draft_report.get("schema")),
            "status": _clean_text(decision_file_draft_report.get("status")),
            "draftDecisionRows": int(dict(decision_file_draft_report.get("counts") or {}).get("draftDecisionRows") or 0)
            if draft_validation.ok
            else 0,
            "textSourceHoldoutRows": int(
                dict(decision_file_draft_report.get("counts") or {}).get("textSourceHoldoutRows") or 0
            )
            if draft_validation.ok
            else 0,
        },
        "inputDecisionFile": {
            "draftOnly": bool(decision_file.get("draftOnly")),
            "decisionRows": len(decision_rows),
            "shapeValidation": {"ok": not decision_shape_errors, "errors": decision_shape_errors},
        },
        "counts": counts,
        "coverageTarget": dict(decision_file_draft_report.get("coverageTarget") or {}),
        "textSourceHoldoutPaperIds": sorted(holdout_paper_ids),
        "validationRows": validation_rows,
        "gate": {
            "decisionFileValidationReady": all_valid,
            "containsOnlyValidatedNeedsReviewRows": all_valid and counts["needsReviewRows"] == expected_input_rows,
            "containsApplyReadyRows": counts["applyReadyRows"] > 0,
            "containsApprovedDecisionRows": counts["approvedDecisionRows"] > 0,
            "containsRejectedDecisionRows": counts["rejectedDecisionRows"] > 0,
            "applyReady": False,
            "sourceRegistrationMutationReady": False,
            "parsedArtifactMaterializationReady": False,
            "parserRoutingReady": False,
            "strictEvidenceReady": False,
            "answerIntegrationReady": False,
            "decision": gate_decision,
            "schemaViolations": input_schema_violations,
            "unsafeUpstreamFlags": unsafe_flags,
            "recommendedNextTranche": (
                "parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run"
                if all_valid and counts["applyReadyRows"] > 0
                else "parsed_artifact_manual_lookup_source_recovery_decision_file_human_review"
                if all_valid and counts["needsReviewRows"] == expected_input_rows
                else "parsed_artifact_manual_lookup_source_recovery_decision_file_validation_repair"
            ),
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "warnings": [
            "validation_does_not_apply_approved_sources",
            "needs_review_is_valid_but_not_apply_ready",
            "text_source_holdout_rows_remain_excluded",
            "source_recovery_and_parsed_artifact_materialization_require_later_explicit_apply_tranches",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file validation schema failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write validation JSON, summary JSON, and Markdown."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery decision file validation schema failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation.json"
    summary_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation-summary.json"
    markdown_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-validation.md"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "coverageTarget": report.get("coverageTarget"),
        "gate": report.get("gate"),
        "mutationCounters": report.get("mutationCounters"),
        "reportFiles": {
            "reportJsonPath": str(report_path),
        },
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
    lines = [
        "# Parsed Artifact Manual Lookup Source Recovery Decision File Validation",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- input rows: {counts.get('inputRows', 0)}",
        f"- valid rows: {counts.get('validRows', 0)}",
        f"- invalid rows: {counts.get('invalidRows', 0)}",
        f"- needs_review rows: {counts.get('needsReviewRows', 0)}",
        f"- approved decision rows: {counts.get('approvedDecisionRows', 0)}",
        f"- apply-ready rows: {counts.get('applyReadyRows', 0)}",
        f"- recommended next tranche: `{gate.get('recommendedNextTranche', '')}`",
        "",
        "## Boundary",
        "",
        "This validator checks decision-file semantics only. It does not perform lookup, download files, mutate source registrations, materialize parsed artifacts, create evidence, scan the vault, or change answer behavior.",
        "",
        "## Validation Status Taxonomy",
        "",
    ]
    for item in list(counts.get("validationStatusTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-file-draft-report", default=str(DEFAULT_DECISION_FILE_DRAFT_REPORT_PATH))
    parser.add_argument("--decision-file", default=str(DEFAULT_DECISION_FILE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--report-name",
        default="parsed-artifact-manual-lookup-source-recovery-decision-file-validation",
    )
    args = parser.parse_args(argv)

    draft_report_path = Path(args.decision_file_draft_report).expanduser()
    decision_file_path = Path(args.decision_file).expanduser()
    report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(
        decision_file_draft_report=_load_json(draft_report_path),
        decision_file=_load_json(decision_file_path),
        decision_file_draft_report_path=draft_report_path,
        decision_file_path=decision_file_path,
        report_name=args.report_name,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_validation(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "counts": report["counts"],
                "gate": report["gate"],
                "mutationCounters": report["mutationCounters"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_decision_file_validation",
    "write_parsed_artifact_manual_lookup_source_recovery_decision_file_validation",
]
