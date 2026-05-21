"""Apply-gated executor for manual lookup source recovery decisions.

Consumes a ready apply dry-run report, downloads or registers approved sources,
and materializes parsed artifacts only when invoked with ``--apply``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.extraction_diagnostics import build_extraction_report
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items, _safe_relative
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_validation import (
    DECISION_APPROVE_LOCAL_PDF,
    DECISION_APPROVE_SOURCE_URL,
    _normalize_source_content_hash,
)
from knowledge_hub.papers.parsed_artifact_manual_lookup_source_recovery_decision_file_apply_dry_run import (
    PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
    _target_recovered_path,
)
from knowledge_hub.papers.parsed_artifact_url_source_recovery import _download_pdf, _looks_like_pdf
from knowledge_hub.papers.parsed_materialization import PARSED_MATERIALIZATION_SCHEMA_ID, materialize_parsed_artifacts
from knowledge_hub.papers.source_text import source_hash_for_path


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-decision-file-apply.v1"
)


def _default_report_root() -> Path:
    return Path.home() / ("." + "khub") / "reports" / "parsed-artifact-coverage" / "2026-05-21"


DEFAULT_DRY_RUN_REPORT_PATH = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run"
    / "01-parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run"
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-dry-run.json"
)
DEFAULT_PLAN_OUTPUT_DIR = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply"
    / "01-plan"
)
DEFAULT_APPLY_OUTPUT_DIR = (
    _default_report_root()
    / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply"
    / "02-apply"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))


def _extraction_counts(*, sqlite_db: Any, papers_dir: str | Path) -> dict[str, int]:
    report = build_extraction_report(sqlite_db=sqlite_db, papers_dir=papers_dir)
    counts = dict(report.get("counts") or {})
    return {
        "scannedPapers": int(counts.get("scannedPapers") or 0),
        "missingParsedArtifacts": int(counts.get("missingParsedArtifacts") or 0),
    }


def _source_integrity(
    *,
    source_path: Path,
    expected_hash: str,
) -> dict[str, Any]:
    observed_hash = source_hash_for_path(str(source_path))
    return {
        "algorithm": "sha256",
        "expectedSourceContentHash": expected_hash,
        "observedSourceContentHash": observed_hash,
        "sourceContentHashMatched": bool(expected_hash and observed_hash and expected_hash == observed_hash),
    }


def _recover_source(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    item: dict[str, Any],
    apply: bool,
    allow_network: bool,
    overwrite: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    paper_id = _clean_text(item.get("paperId"))
    decision = _clean_text(item.get("decision"))
    expected_hash = _normalize_source_content_hash(item.get("approvedSourceContentHash"))
    row = sqlite_db.get_paper(paper_id) if hasattr(sqlite_db, "get_paper") else None
    base = dict(item)
    base.update(
        {
            "approvedSourceContentHash": expected_hash,
            "sourceDownloadAttempted": False,
            "sourceDownloadPerformed": False,
            "sourceContentHashComputed": False,
            "sourceContentHashMatched": False,
            "sourceRegistrationMutationAttempted": False,
            "sourceRegistrationMutationPerformed": False,
            "databaseMutationAttempted": False,
            "databaseMutationPerformed": False,
        }
    )
    if not isinstance(row, dict) or not row:
        return {**base, "status": "blocked", "reason": "paper_not_registered", "sourceRecoveryStatus": "blocked"}
    if not expected_hash:
        return {
            **base,
            "status": "blocked",
            "reason": "approved_source_content_hash_required",
            "sourceRecoveryStatus": "blocked",
        }

    if decision == DECISION_APPROVE_SOURCE_URL:
        target = _target_recovered_path(papers_dir=papers_dir, paper_id=paper_id)
        url = _clean_text(item.get("approvedSourceUrl"))
        if not apply:
            return {
                **base,
                "status": "planned",
                "reason": "apply_required",
                "sourceRecoveryStatus": "planned",
                "targetSourceArtifactPath": _safe_relative(target, root=papers_dir, prefix="papers_dir"),
                "sourceIntegrity": {
                    "algorithm": "sha256",
                    "expectedSourceContentHash": expected_hash,
                    "hashCheckRequiredAtApply": True,
                },
            }
        if not allow_network:
            return {
                **base,
                "status": "blocked",
                "reason": "network_not_allowed_for_approved_source_url",
                "sourceRecoveryStatus": "blocked",
                "targetSourceArtifactPath": _safe_relative(target, root=papers_dir, prefix="papers_dir"),
                "sourceDownloadAttempted": False,
                "sourceDownloadPerformed": False,
                "sourceIntegrity": {
                    "algorithm": "sha256",
                    "expectedSourceContentHash": expected_hash,
                    "hashCheckRequiredAtApply": True,
                },
            }
        try:
            downloaded = False
            if target.exists() and not overwrite and _looks_like_pdf(target):
                download_result = {"reusedExistingTarget": True, "sizeBytes": target.stat().st_size}
            else:
                _download_pdf(url, target, timeout_seconds)
                downloaded = True
                download_result = {"sizeBytes": target.stat().st_size}
            integrity = _source_integrity(source_path=target, expected_hash=expected_hash)
            if not integrity["sourceContentHashMatched"]:
                if downloaded:
                    try:
                        target.unlink()
                    except OSError:
                        pass
                    download_result["mismatchedDownloadRemoved"] = True
                return {
                    **base,
                    "status": "blocked",
                    "reason": "source_content_hash_mismatch",
                    "sourceRecoveryStatus": "blocked",
                    "targetSourceArtifactPath": _safe_relative(target, root=papers_dir, prefix="papers_dir"),
                    "sourceDownloadAttempted": downloaded,
                    "sourceDownloadPerformed": downloaded,
                    "sourceContentHashComputed": bool(integrity["observedSourceContentHash"]),
                    "sourceContentHashMatched": False,
                    "sourceIntegrity": integrity,
                    "downloadResult": download_result,
                }
            updated = dict(row)
            updated["pdf_path"] = str(target)
            updated["source_content_hash"] = expected_hash
            sqlite_db.upsert_paper(updated)
            return {
                **base,
                "status": "source_recovered",
                "reason": "ok",
                "sourceRecoveryStatus": "recovered",
                "targetSourceArtifactPath": _safe_relative(target, root=papers_dir, prefix="papers_dir"),
                "sourceDownloadAttempted": downloaded,
                "sourceDownloadPerformed": downloaded,
                "sourceContentHashComputed": True,
                "sourceContentHashMatched": True,
                "sourceIntegrity": integrity,
                "sourceRegistrationMutationAttempted": True,
                "sourceRegistrationMutationPerformed": True,
                "databaseMutationAttempted": True,
                "databaseMutationPerformed": True,
                "downloadResult": download_result,
            }
        except Exception as error:
            return {
                **base,
                "status": "failed",
                "reason": f"source_download_failed:{type(error).__name__}",
                "sourceRecoveryStatus": "failed",
                "sourceDownloadAttempted": True,
            }

    if decision == DECISION_APPROVE_LOCAL_PDF:
        local_path = Path(_clean_text(item.get("approvedLocalPdfPath"))).expanduser()
        if not local_path.is_file() or not _looks_like_pdf(local_path):
            return {
                **base,
                "status": "blocked",
                "reason": "approved_local_pdf_missing_or_not_pdf",
                "sourceRecoveryStatus": "blocked",
            }
        if not apply:
            return {
                **base,
                "status": "planned",
                "reason": "apply_required",
                "sourceRecoveryStatus": "planned",
                "targetSourceArtifactPath": _safe_relative(local_path, root=papers_dir, prefix="papers_dir"),
                "sourceIntegrity": {
                    "algorithm": "sha256",
                    "expectedSourceContentHash": expected_hash,
                    "hashCheckRequiredAtApply": True,
                },
            }
        integrity = _source_integrity(source_path=local_path, expected_hash=expected_hash)
        if not integrity["sourceContentHashMatched"]:
            return {
                **base,
                "status": "blocked",
                "reason": "source_content_hash_mismatch",
                "sourceRecoveryStatus": "blocked",
                "targetSourceArtifactPath": _safe_relative(local_path, root=papers_dir, prefix="papers_dir"),
                "sourceContentHashComputed": bool(integrity["observedSourceContentHash"]),
                "sourceContentHashMatched": False,
                "sourceIntegrity": integrity,
            }
        updated = dict(row)
        updated["pdf_path"] = str(local_path)
        updated["source_content_hash"] = expected_hash
        sqlite_db.upsert_paper(updated)
        return {
            **base,
            "status": "source_recovered",
            "reason": "ok",
            "sourceRecoveryStatus": "recovered",
            "targetSourceArtifactPath": _safe_relative(local_path, root=papers_dir, prefix="papers_dir"),
            "sourceContentHashComputed": True,
            "sourceContentHashMatched": True,
            "sourceIntegrity": integrity,
            "sourceRegistrationMutationAttempted": True,
            "sourceRegistrationMutationPerformed": True,
            "databaseMutationAttempted": True,
            "databaseMutationPerformed": True,
        }

    return {**base, "status": "blocked", "reason": "unsupported_decision", "sourceRecoveryStatus": "blocked"}


def build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    dry_run_report: dict[str, Any],
    dry_run_report_path: str | Path = "",
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-decision-file-apply",
    apply: bool = False,
    allow_network: bool = True,
    overwrite: bool = False,
    timeout_seconds: float = 30.0,
    delay_seconds: float = 0.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    validation = validate_payload(
        dry_run_report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    readiness = dict(dry_run_report.get("dryRunMaterializationReadiness") or {})
    selected_ids = [
        _clean_text(paper_id)
        for paper_id in list(dry_run_report.get("selectedCandidatePaperIds") or [])
        if _clean_text(paper_id)
    ]
    dry_run_ready = bool(
        validation.ok
        and dry_run_report.get("status") == "ready"
        and readiness.get("ready") is True
        and selected_ids
    )
    before_counts = _extraction_counts(sqlite_db=sqlite_db, papers_dir=papers_dir)
    dry_items = {
        _clean_text(item.get("paperId")): dict(item)
        for item in list(dry_run_report.get("items") or [])
        if isinstance(item, dict)
    }

    source_items: list[dict[str, Any]] = []
    for paper_id in selected_ids:
        item = _recover_source(
            sqlite_db=sqlite_db,
            papers_dir=papers_dir,
            item=dry_items.get(paper_id) or {"paperId": paper_id},
            apply=apply,
            allow_network=allow_network,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
        )
        source_items.append(item)
        if apply and delay_seconds and item.get("sourceDownloadPerformed"):
            time.sleep(delay_seconds)

    recovered_ids = [
        _clean_text(item.get("paperId"))
        for item in source_items
        if _clean_text(item.get("sourceRecoveryStatus")) == "recovered"
        or (not apply and _clean_text(item.get("sourceRecoveryStatus")) == "planned")
    ]
    materialized_ids: list[str] = []
    materialization_items: list[dict[str, Any]] = []
    if dry_run_ready and recovered_ids and apply:
        materialization = materialize_parsed_artifacts(
            sqlite_db=sqlite_db,
            papers_dir=papers_dir,
            paper_ids=recovered_ids,
            apply=True,
            overwrite=overwrite,
        )
        materialization_items = [dict(item) for item in list(materialization.get("items") or [])]
        materialized_ids = [
            _clean_text(item.get("paperId"))
            for item in materialization_items
            if _clean_text(item.get("status")) in {"materialized", "skipped_existing"}
        ]
    elif dry_run_ready and recovered_ids and not apply:
        materialization = {
            "schema": PARSED_MATERIALIZATION_SCHEMA_ID,
            "status": "ready",
            "counts": {"planned": len(recovered_ids), "materialized": 0, "blocked": 0, "failed": 0, "skippedExisting": 0},
            "items": [],
        }
    else:
        materialization = {
            "schema": PARSED_MATERIALIZATION_SCHEMA_ID,
            "status": "blocked",
            "counts": {"planned": 0, "materialized": 0, "blocked": len(selected_ids), "failed": 0, "skippedExisting": 0},
            "items": [],
        }

    combined_items: list[dict[str, Any]] = []
    mat_by_id = {_clean_text(item.get("paperId")): item for item in materialization_items}
    for source_item in source_items:
        paper_id = _clean_text(source_item.get("paperId"))
        mat_item = mat_by_id.get(paper_id) or {}
        if apply:
            if _clean_text(source_item.get("sourceRecoveryStatus")) != "recovered":
                final_status = _clean_text(source_item.get("status")) or "blocked"
            else:
                final_status = _clean_text(mat_item.get("status")) or "blocked"
        else:
            final_status = "planned"
        combined_items.append(
            {
                **source_item,
                "materializationStatus": _clean_text(mat_item.get("status")) or ("planned" if not apply else ""),
                "parsedArtifactWriteAttempted": bool(apply and mat_item),
                "parsedArtifactWritePerformed": _clean_text(mat_item.get("status")) == "materialized",
                "status": final_status,
            }
        )

    after_counts = _extraction_counts(sqlite_db=sqlite_db, papers_dir=papers_dir) if apply else dict(before_counts)
    status_counter = Counter(_clean_text(item.get("status")) for item in combined_items)
    if not dry_run_ready:
        overall = "blocked"
    elif status_counter.get("failed"):
        overall = "partial" if status_counter.get("materialized") or status_counter.get("planned") else "failed"
    elif apply:
        overall = "applied" if materialized_ids else "blocked"
    else:
        overall = "ready" if status_counter.get("planned") == len(combined_items) else "blocked"

    source_download_rows = sum(1 for item in combined_items if item.get("sourceDownloadPerformed"))
    source_registration_rows = sum(1 for item in combined_items if item.get("sourceRegistrationMutationPerformed"))
    parsed_write_rows = sum(1 for item in combined_items if item.get("parsedArtifactWritePerformed"))

    report = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID,
        "status": overall,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "applyRequested": bool(apply),
            "dryRunReportPath": str(Path(str(dry_run_report_path)).expanduser()) if dry_run_report_path else "",
            "selectionRule": "consume selectedCandidatePaperIds from ready manual lookup apply dry-run report",
            "nonScope": [
                "invented_source_urls",
                "text_source_unsupported",
                "strict_or_citation_or_runtime_evidence",
                "parser_routing",
                "index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
            ],
        },
        "inputDryRun": {
            "schemaValidation": {"ok": bool(validation.ok), "errors": list(validation.errors)},
            "status": dry_run_report.get("status"),
            "ready": bool(readiness.get("ready")),
            "selectedRows": len(selected_ids),
        },
        "baselineBefore": before_counts,
        "baselineAfter": after_counts,
        "selectedCandidatePaperIds": selected_ids,
        "recoveredPaperIds": [
            _clean_text(item.get("paperId"))
            for item in combined_items
            if _clean_text(item.get("sourceRecoveryStatus")) in {"recovered", "planned"}
        ],
        "materializedPaperIds": materialized_ids,
        "items": combined_items,
        "counts": {
            "selectedRows": len(combined_items),
            "planned": status_counter.get("planned", 0),
            "sourceRecovered": sum(
                1 for item in combined_items if _clean_text(item.get("sourceRecoveryStatus")) == "recovered"
            ),
            "sourceContentHashMatchedRows": sum(
                1 for item in combined_items if bool(item.get("sourceContentHashMatched"))
            ),
            "sourceContentHashMismatchRows": sum(
                1 for item in combined_items if _clean_text(item.get("reason")) == "source_content_hash_mismatch"
            ),
            "materialized": status_counter.get("materialized", 0),
            "blocked": status_counter.get("blocked", 0),
            "failed": status_counter.get("failed", 0),
            "statusTaxonomy": _counter_items(status_counter),
        },
        "observedCoverageChange": {
            "missingParsedArtifactsBefore": int(before_counts.get("missingParsedArtifacts") or 0),
            "missingParsedArtifactsAfter": int(after_counts.get("missingParsedArtifacts") or 0),
            "reduction": max(
                0,
                int(before_counts.get("missingParsedArtifacts") or 0) - int(after_counts.get("missingParsedArtifacts") or 0),
            ),
        },
        "mutationPolicy": {
            "applyRequested": bool(apply),
            "networkAllowed": bool(allow_network),
            "sourceDownload": bool(apply and allow_network),
            "sourceRegistrationMutation": bool(apply),
            "parsedArtifactWrite": bool(apply),
            "strictEvidence": False,
            "runtimeEvidence": False,
            "databaseMutation": bool(apply),
            "indexMutation": False,
            "reindexOrReembed": False,
            "vaultScan": False,
            "vaultWrite": False,
            "answerIntegration": False,
        },
        "mutationCounters": {
            "sourceDownloadRows": source_download_rows,
            "sourceRegistrationMutationRows": source_registration_rows,
            "parsedArtifactWriteRows": parsed_write_rows,
            "strictEvidenceRows": 0,
            "runtimeEvidenceRows": 0,
            "databaseMutationRows": source_registration_rows,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultReadRows": 0,
            "vaultWriteRows": 0,
            "answerIntegrationRows": 0,
        },
        "warnings": [
            "apply_executes_only_selected_apply_ready_rows",
            "text_source_holdout_remains_out_of_scope",
            "no_strict_or_runtime_evidence_created",
        ],
    }
    output_validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID,
        strict=True,
    )
    if not output_validation.ok:
        raise ValueError(
            "manual lookup decision file apply schema failed: "
            + "; ".join(output_validation.errors[:5])
        )
    return report


def write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError("manual lookup decision file apply schema failed")
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply.json"
    summary_path = root / "parsed-artifact-manual-lookup-source-recovery-decision-file-apply-summary.json"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "counts": report.get("counts"),
        "observedCoverageChange": report.get("observedCoverageChange"),
        "mutationCounters": report.get("mutationCounters"),
        "reportFiles": {"reportJsonPath": str(report_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"reportJsonPath": str(report_path), "summaryJsonPath": str(summary_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run-report", default=str(DEFAULT_DRY_RUN_REPORT_PATH))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)

    from knowledge_hub.infrastructure.config import Config
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    config = Config(args.config)
    output_dir = args.output_dir or (
        str(DEFAULT_APPLY_OUTPUT_DIR) if args.apply else str(DEFAULT_PLAN_OUTPUT_DIR)
    )
    sqlite_db = SQLiteDatabase(
        config.sqlite_path,
        enable_event_store=False,
        bootstrap=bool(args.apply),
        read_only=not bool(args.apply),
    )
    try:
        report = build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            dry_run_report=_load_json(args.dry_run_report),
            dry_run_report_path=args.dry_run_report,
            apply=args.apply,
            allow_network=args.allow_network,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
            delay_seconds=args.delay_seconds if args.apply else 0.0,
        )
    finally:
        sqlite_db.close()

    paths = write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply(report, output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "counts": report["counts"],
                "observedCoverageChange": report["observedCoverageChange"],
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
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_DECISION_FILE_APPLY_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_decision_file_apply",
    "write_parsed_artifact_manual_lookup_source_recovery_decision_file_apply",
]
