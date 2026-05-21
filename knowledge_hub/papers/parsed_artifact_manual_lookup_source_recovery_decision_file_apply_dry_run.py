"""Dry-run planner for manual lookup source recovery decision apply.

Consumes a validated decision file and plans bounded source recovery plus parsed
artifact materialization for apply-ready rows only. Default behavior is
write-free; report writes require explicit ``--apply``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.extraction_diagnostics import build_extraction_report
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items, _safe_relative
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_LOCAL_PDF,
    DECISION_APPROVE_SOURCE_URL,
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
    _normalize_source_content_hash,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run.v1"
)

TARGET_MISSING_PARSED_ARTIFACTS = 10
DEFAULT_RECOVERY_LIMIT = 15
DEFAULT_MIN_RECOVERIES_FOR_TARGET = 6

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _default_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


DEFAULT_VALIDATION_REPORT_PATH = (
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
DEFAULT_OUTPUT_DIR = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def _row_key(paper_id: str, source_review_card_id: str) -> str:
    return f"{paper_id}::{source_review_card_id}"


def _safe_paper_filename(paper_id: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("_", _clean_text(paper_id)).strip("._")
    return cleaned or "paper"


def _target_recovered_path(*, papers_dir: str | Path, paper_id: str) -> Path:
    return (
        Path(str(papers_dir)).expanduser()
        / "recovered_sources"
        / "manual_lookup"
        / f"{_safe_paper_filename(paper_id)}.pdf"
    )


def _decision_index(decision_file: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _row_key(_clean_text(row.get("paperId")), _clean_text(row.get("sourceReviewCardId"))): dict(row)
        for row in list(decision_file.get("decisions") or [])
        if isinstance(row, dict)
    }


def _select_apply_ready_rows(
    validation_report: dict[str, Any],
    decision_file: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    decision_index = _decision_index(decision_file)
    selected: list[dict[str, Any]] = []
    for vrow in list(validation_report.get("validationRows") or []):
        if not isinstance(vrow, dict) or not bool(vrow.get("applyReadyForLaterApply")):
            continue
        key = _row_key(_clean_text(vrow.get("paperId")), _clean_text(vrow.get("sourceReviewCardId")))
        drow = decision_index.get(key) or {}
        selected.append({**vrow, **drow})
        if len(selected) >= max(0, limit):
            break
    return selected


def _missing_parsed_artifact_ids(extraction_report: dict[str, Any]) -> set[str]:
    missing_ids: set[str] = set()
    for paper in list(extraction_report.get("papers") or []):
        if not isinstance(paper, dict):
            continue
        diagnostic = dict(paper.get("diagnostic") or {})
        reasons = {
            _clean_text(reason)
            for reason in list(paper.get("warnings") or []) + list(diagnostic.get("degradationReasons") or [])
        }
        if "parsed_artifact_missing" in reasons or "parsed_document_missing" in reasons:
            paper_id = _clean_text(paper.get("paperId") or diagnostic.get("paperId"))
            if paper_id:
                missing_ids.add(paper_id)
    return missing_ids


def _dry_run_item(
    row: dict[str, Any],
    *,
    papers_dir: str | Path,
) -> dict[str, Any]:
    paper_id = _clean_text(row.get("paperId"))
    decision = _clean_text(row.get("decision"))
    approved_url = _clean_text(row.get("approvedSourceUrl"))
    approved_local = _clean_text(row.get("approvedLocalPdfPath"))
    approved_hash = _normalize_source_content_hash(row.get("approvedSourceContentHash"))
    if decision == DECISION_APPROVE_SOURCE_URL:
        source_action = "download_approved_source_url_to_recovered_sources"
        source_target = _target_recovered_path(papers_dir=papers_dir, paper_id=paper_id)
        source_input = approved_url
    else:
        source_action = "register_approved_local_pdf_path"
        source_target = Path(approved_local).expanduser()
        source_input = approved_local
    return {
        "paperId": paper_id,
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceReviewCardId": _clean_text(row.get("sourceReviewCardId")),
        "decision": decision,
        "approvedSourceType": _clean_text(row.get("approvedSourceType")),
        "approvedSourceUrl": approved_url,
        "approvedLocalPdfPath": approved_local,
        "approvedSourceContentHash": approved_hash,
        "approvedBy": _clean_text(row.get("approvedBy")),
        "sourceIntegrity": {
            "algorithm": "sha256",
            "expectedSourceContentHash": approved_hash,
            "hashCheckRequiredAtApply": True,
        },
        "sourceRecoveryAction": source_action,
        "sourceRecoveryInput": source_input,
        "targetSourceArtifactPath": _safe_relative(source_target, root=papers_dir, prefix="papers_dir"),
        "materializationAction": "materialize_parsed_artifacts",
        "status": "planned",
        "reason": "apply_required",
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "strictEvidenceAttempted": False,
        "runtimeEvidenceAttempted": False,
    }


def build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
    *,
    validation_report: dict[str, Any],
    decision_file: dict[str, Any],
    sqlite_db: Any,
    papers_dir: str | Path,
    validation_report_path: str | Path | None = None,
    decision_file_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run",
    generated_at: str | None = None,
    limit: int = DEFAULT_RECOVERY_LIMIT,
    target_missing_parsed_artifacts: int = TARGET_MISSING_PARSED_ARTIFACTS,
    min_recoveries_for_target: int = DEFAULT_MIN_RECOVERIES_FOR_TARGET,
) -> dict[str, Any]:
    validation_result = validate_payload(
        validation_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID,
        strict=True,
    )
    unsafe_flags: list[str] = []
    if not validation_result.ok:
        unsafe_flags.append("decision_file_validation_report_schema_violation")
    if validation_report.get("schema") != PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_VALIDATION_SCHEMA_ID:
        unsafe_flags.append("decision_file_validation_report_schema_mismatch")
    if _clean_text(validation_report.get("status")) != "decision_file_validation_ready":
        unsafe_flags.append(f"decision_file_validation_status={_clean_text(validation_report.get('status')) or 'unknown'}")

    counts_validation = dict(validation_report.get("counts") or {})
    apply_ready_total = int(counts_validation.get("applyReadyRows") or 0)
    if apply_ready_total == 0:
        unsafe_flags.append("no_apply_ready_decision_rows")

    baseline = build_extraction_report(sqlite_db=sqlite_db, papers_dir=papers_dir)
    baseline_counts = dict(baseline.get("counts") or {})
    missing_before = int(baseline_counts.get("missingParsedArtifacts") or 0)
    missing_ids_before = _missing_parsed_artifact_ids(baseline)

    selected_rows = _select_apply_ready_rows(validation_report, decision_file, limit=limit) if validation_result.ok else []
    items = [_dry_run_item(row, papers_dir=papers_dir) for row in selected_rows]
    status_counter = Counter(_clean_text(item.get("status")) for item in items)
    ready = bool(validation_result.ok and not unsafe_flags and items and status_counter.get("planned", 0) == len(items))
    selected_missing_ids = [
        _clean_text(item.get("paperId")) for item in items if _clean_text(item.get("paperId")) in missing_ids_before
    ]
    expected_reduction = len(selected_missing_ids)
    expected_after = max(0, missing_before - expected_reduction)
    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
        "status": "ready" if ready else "blocked",
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputValidationReportPath": str(Path(str(validation_report_path)).expanduser())
            if validation_report_path
            else "",
            "inputDecisionFilePath": str(Path(str(decision_file_path)).expanduser()) if decision_file_path else "",
            "selectionRule": "apply-ready validation rows only, bounded by limit; text-source holdout excluded",
            "targetRoot": "papers_dir/recovered_sources/manual_lookup",
            "writeRoot": "papers_dir/parsed/<paper_id>",
            "nonScope": [
                "invented_source_urls",
                "needs_review_without_explicit_approval",
                "text_source_unsupported",
                "strict_or_citation_or_runtime_evidence",
                "parser_routing",
                "database_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
            ],
        },
        "inputValidation": {
            "schemaValidation": {"ok": bool(validation_result.ok), "errors": list(validation_result.errors)},
            "status": _clean_text(validation_report.get("status")),
            "applyReadyRows": apply_ready_total,
            "needsReviewRows": int(counts_validation.get("needsReviewRows") or 0),
        },
        "baselineBefore": {
            "missingParsedArtifacts": missing_before,
            "scannedPapers": int(baseline_counts.get("scannedPapers") or 0),
        },
        "coverageTarget": {
            "targetMissingParsedArtifacts": target_missing_parsed_artifacts,
            "minRecoveriesForTarget": min_recoveries_for_target,
            "selectedApplyReadyRows": len(items),
            "selectedCurrentlyMissingRows": expected_reduction,
            "selectedAlreadyMaterializedRows": max(0, len(items) - expected_reduction),
            "recoveriesNeededForTarget": max(0, missing_before - target_missing_parsed_artifacts),
            "targetReachableIfSelectedApplied": expected_after <= target_missing_parsed_artifacts,
        },
        "selectedCandidatePaperIds": [_clean_text(item.get("paperId")) for item in items],
        "items": items,
        "counts": {
            "selectedRows": len(items),
            "plannedRows": status_counter.get("planned", 0),
            "blockedRows": status_counter.get("blocked", 0),
            "applyReadyRowsAvailable": apply_ready_total,
            "statusTaxonomy": _counter_items(status_counter),
        },
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": missing_before,
            "missingParsedArtifactsAfterIfAllSelectedMaterialize": expected_after,
            "expectedReductionIfAllSelectedMaterialize": expected_reduction,
        },
        "dryRunMaterializationReadiness": {
            "ready": ready,
            "unsafeUpstreamFlags": list(dict.fromkeys(unsafe_flags)),
            "recommendedNextTranche": (
                "parsed_artifact_manual_lookup_source_recovery_decision_file_apply"
                if ready
                else "parsed_artifact_manual_lookup_source_recovery_decision_file_validation_after_human_edit"
            ),
        },
        "mutationPolicy": {
            "applyRequested": False,
            "sourceDownload": False,
            "sourceRegistrationMutation": False,
            "parsedArtifactWrite": False,
            "strictEvidence": False,
            "runtimeEvidence": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "vaultScan": False,
            "vaultWrite": False,
            "answerIntegration": False,
        },
        "mutationCounters": {
            "sourceDownloadRows": 0,
            "sourceRegistrationMutationRows": 0,
            "parsedArtifactWriteRows": 0,
            "strictEvidenceRows": 0,
            "runtimeEvidenceRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultReadRows": 0,
            "vaultWriteRows": 0,
            "answerIntegrationRows": 0,
        },
        "warnings": [
            "dry_run_does_not_download_or_register_sources",
            "dry_run_does_not_materialize_parsed_artifacts",
            "only_explicitly_approved_decision_rows_are_selected",
            "text_source_holdout_remains_out_of_scope",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "manual lookup decision file apply dry-run schema failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError("manual lookup decision file apply dry-run schema failed")
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run.json"
    summary_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run-summary.json"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "coverageTarget": report.get("coverageTarget"),
        "expectedCoverageChangeIfApplied": report.get("expectedCoverageChangeIfApplied"),
        "dryRunMaterializationReadiness": report.get("dryRunMaterializationReadiness"),
        "reportFiles": {"reportJsonPath": str(report_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"reportJsonPath": str(report_path), "summaryJsonPath": str(summary_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--validation-report", default=str(DEFAULT_VALIDATION_REPORT_PATH))
    parser.add_argument("--decision-file", default=str(DEFAULT_DECISION_FILE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=DEFAULT_RECOVERY_LIMIT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write local dry-run JSON reports. Default prints summary only.",
    )
    args = parser.parse_args(argv)

    from knowledge_hub.infrastructure.config import Config
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    config = Config(args.config)
    sqlite_db = SQLiteDatabase(config.sqlite_path, read_only=True, enable_event_store=False)
    try:
        report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
            validation_report=_load_json(args.validation_report),
            decision_file=_load_json(args.decision_file),
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            validation_report_path=args.validation_report,
            decision_file_path=args.decision_file,
            limit=args.limit,
        )
    finally:
        sqlite_db.close()

    payload: dict[str, Any] = {
        "schema": report["schema"],
        "status": report["status"],
        "counts": report["counts"],
        "coverageTarget": report["coverageTarget"],
        "expectedCoverageChangeIfApplied": report["expectedCoverageChangeIfApplied"],
        "dryRunMaterializationReadiness": report["dryRunMaterializationReadiness"],
    }
    if args.apply:
        payload["paths"] = write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run(
            report,
            args.output_dir,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run",
    "write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run",
]
