"""Apply-gated materialization for oversized parsed artifact PDF blockers.

This helper consumes the oversized-PDF dry-run report and materializes only the
selected local PDF rows when explicitly invoked with ``--apply``. It writes
only canonical parsed artifacts under ``papers_dir/parsed/<paper_id>/`` and
keeps source registration, parser routing, evidence stores, indexes, vaults,
and answer behavior out of scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.extraction_diagnostics import build_extraction_report
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items
from knowledge_hub.papers.parsed_artifact_oversized_pdf_materialization_dry_run import (
    PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_materialization import (
    PARSED_MATERIALIZATION_SCHEMA_ID,
    materialize_parsed_artifacts,
)


PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-oversized-pdf-materialization-apply.v1"
)

DEFAULT_DRY_RUN_REPORT_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-oversized-pdf-materialization-dry-run/"
    "parsed-artifact-oversized-pdf-materialization-dry-run.json"
).expanduser()
DEFAULT_PLAN_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-oversized-pdf-materialization-apply/01-plan"
).expanduser()
DEFAULT_APPLY_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-oversized-pdf-materialization-apply/02-apply"
).expanduser()


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
        "reportedPapers": int(counts.get("reportedPapers") or 0),
        "degradedPapers": int(counts.get("degradedPapers") or 0),
        "missingParsedArtifacts": int(counts.get("missingParsedArtifacts") or 0),
    }


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [_clean_text(item.get("status")) for item in items]
    return {
        "planned": statuses.count("planned"),
        "materialized": statuses.count("materialized"),
        "blocked": statuses.count("blocked"),
        "failed": statuses.count("failed"),
        "skippedExisting": statuses.count("skipped_existing"),
    }


def _overall_status(*, apply: bool, counts: dict[str, int], selected_count: int, dry_run_ready: bool) -> str:
    if not dry_run_ready:
        return "blocked"
    if counts.get("failed", 0):
        return "partial" if counts.get("materialized", 0) or counts.get("planned", 0) else "failed"
    if counts.get("blocked", 0):
        return "partial" if counts.get("materialized", 0) or counts.get("planned", 0) else "blocked"
    if apply:
        return "applied" if counts.get("materialized", 0) or counts.get("skippedExisting", 0) else "blocked"
    return "ready" if selected_count and counts.get("planned", 0) == selected_count else "blocked"


def _mutation_policy(*, apply: bool) -> dict[str, Any]:
    return {
        "applyRequested": bool(apply),
        "applyPerformed": bool(apply),
        "parsedArtifactWrite": bool(apply),
        "sourceDownload": False,
        "sourcePathRewrite": False,
        "sourceRegistrationMutation": False,
        "parserRouting": False,
        "strictEvidence": False,
        "citationEvidence": False,
        "runtimeEvidence": False,
        "sourceSpanCreated": False,
        "databaseMutation": False,
        "indexMutation": False,
        "reindexOrReembed": False,
        "vaultScan": False,
        "vaultWrite": False,
        "answerIntegration": False,
        "manualBlockerResolution": False,
    }


def _mutation_counters(items: list[dict[str, Any]]) -> dict[str, int]:
    materialized = sum(1 for item in items if _clean_text(item.get("status")) == "materialized")
    return {
        "parsedArtifactWriteRows": materialized,
        "sourceSpanCreatedRows": 0,
        "strictEvidenceRows": 0,
        "citationEvidenceRows": 0,
        "runtimeEvidenceRows": 0,
        "databaseMutationRows": 0,
        "indexMutationRows": 0,
        "reembedRows": 0,
        "vaultReadRows": 0,
        "vaultWriteRows": 0,
        "answerIntegrationRows": 0,
        "parserRoutingRows": 0,
        "manualBlockerResolutionRows": 0,
        "sourceDownloadRows": 0,
        "sourcePathRewriteRows": 0,
        "sourceRegistrationMutationRows": 0,
    }


def build_parsed_artifact_oversized_pdf_materialization_apply(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    dry_run_report: dict[str, Any],
    dry_run_report_path: str | Path = "",
    report_name: str = "parsed-artifact-oversized-pdf-materialization-apply",
    apply: bool = False,
    overwrite: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Plan or apply the oversized-PDF parsed artifact materialization tranche."""

    validation = validate_payload(
        dry_run_report,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    selected_ids = [_clean_text(paper_id) for paper_id in list(dry_run_report.get("selectedCandidatePaperIds") or []) if _clean_text(paper_id)]
    readiness = dict(dry_run_report.get("dryRunMaterializationReadiness") or {})
    dry_run_ready = bool(
        validation.ok
        and dry_run_report.get("status") == "ready"
        and readiness.get("ready") is True
        and selected_ids
    )
    before_counts = _extraction_counts(sqlite_db=sqlite_db, papers_dir=papers_dir)
    if dry_run_ready:
        materialization = materialize_parsed_artifacts(
            sqlite_db=sqlite_db,
            papers_dir=papers_dir,
            paper_ids=selected_ids,
            apply=bool(apply),
            overwrite=bool(overwrite),
        )
    else:
        materialization = {
            "schema": PARSED_MATERIALIZATION_SCHEMA_ID,
            "status": "blocked",
            "generatedAt": generated_at or _utc_now(),
            "request": {
                "paperIds": selected_ids,
                "parser": "pymupdf",
                "apply": bool(apply),
                "overwrite": bool(overwrite),
            },
            "counts": {"planned": 0, "materialized": 0, "blocked": len(selected_ids), "failed": 0, "skippedExisting": 0},
            "items": [],
            "warnings": ["input_dry_run_report_not_ready"],
        }
    materialization_validation = validate_payload(materialization, PARSED_MATERIALIZATION_SCHEMA_ID, strict=True)
    items = [dict(item) for item in list(materialization.get("items") or [])]
    counts = _status_counts(items)
    materialized_ids = [
        _clean_text(item.get("paperId"))
        for item in items
        if _clean_text(item.get("status")) in {"materialized", "skipped_existing"}
    ]
    after_counts = _extraction_counts(sqlite_db=sqlite_db, papers_dir=papers_dir) if apply else dict(before_counts)
    expected = dict(dry_run_report.get("expectedCoverageChangeIfApplied") or {})
    actual_reduction = max(
        0,
        int(before_counts.get("missingParsedArtifacts") or 0) - int(after_counts.get("missingParsedArtifacts") or 0),
    )

    payload = {
        "schema": PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID,
        "status": _overall_status(
            apply=bool(apply),
            counts=counts,
            selected_count=len(selected_ids),
            dry_run_ready=dry_run_ready and materialization_validation.ok,
        ),
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "applyRequested": bool(apply),
            "overwrite": bool(overwrite),
            "dryRunReportPath": str(Path(str(dry_run_report_path)).expanduser()) if dry_run_report_path else "",
            "selectionRule": "consume selectedCandidatePaperIds from a ready oversized PDF materialization dry-run report",
            "writeRoot": "papers_dir/parsed/<paper_id>",
            "nonScope": [
                "source_download",
                "source_path_rewrite",
                "source_registration_mutation",
                "parser_routing",
                "strict_or_citation_or_runtime_evidence",
                "source_span_creation",
                "database_or_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
                "manual_blocker_resolution",
            ],
        },
        "inputDryRun": {
            "schemaValidation": {
                "ok": bool(validation.ok),
                "errors": list(validation.errors),
            },
            "status": dry_run_report.get("status"),
            "ready": bool(readiness.get("ready")),
            "selectedDryRunRows": int((dry_run_report.get("candidatePool") or {}).get("selectedDryRunRows") or 0),
            "expectedCoverageChangeIfApplied": expected,
        },
        "baselineBefore": before_counts,
        "baselineAfter": after_counts,
        "selectedCandidatePaperIds": selected_ids,
        "materializedPaperIds": materialized_ids,
        "items": items,
        "counts": counts,
        "statusTaxonomy": _counter_items(Counter(_clean_text(item.get("status")) for item in items)),
        "materializationValidation": {
            "ok": bool(materialization_validation.ok),
            "errors": list(materialization_validation.errors),
        },
        "observedCoverageChange": {
            "missingParsedArtifactsBefore": int(before_counts.get("missingParsedArtifacts") or 0),
            "expectedMissingParsedArtifactsReduction": int(expected.get("expectedMissingParsedArtifactsReduction") or 0),
            "expectedMissingParsedArtifactsAfter": int(expected.get("expectedMissingParsedArtifactsAfter") or 0),
            "actualMissingParsedArtifactsReduction": actual_reduction,
            "missingParsedArtifactsAfter": int(after_counts.get("missingParsedArtifacts") or 0),
        },
        "nextRecommendedTranche": {
            "name": "parsed_artifact_manual_lookup_source_recovery_plan",
            "candidateCount": max(0, int(after_counts.get("missingParsedArtifacts") or 0)),
            "rationale": "after oversized local-PDF materialization, remaining rows require manual source lookup or text-source policy",
        },
        "mutationPolicy": _mutation_policy(apply=bool(apply)),
        "mutationCounters": _mutation_counters(items),
        "materialization": materialization,
        "warnings": list(materialization.get("warnings") or []),
    }
    payload_validation = validate_payload(
        payload,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID,
        strict=True,
    )
    if not payload_validation.ok:
        raise ValueError(
            "parsed artifact oversized PDF materialization apply schema validation failed: "
            + "; ".join(payload_validation.errors[:5])
        )
    return payload


def write_parsed_artifact_oversized_pdf_materialization_apply(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, summary JSON, Markdown, and materialized paper IDs."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact oversized PDF materialization apply schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-oversized-pdf-materialization-apply.json"
    summary_json_path = root / "parsed-artifact-oversized-pdf-materialization-apply-summary.json"
    summary_path = root / "parsed-artifact-oversized-pdf-materialization-apply.md"
    materialized_ids_path = root / "materialized-oversized-pdf-paper-ids.txt"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "report": report.get("report"),
        "baselineBefore": report.get("baselineBefore"),
        "baselineAfter": report.get("baselineAfter"),
        "counts": report.get("counts"),
        "observedCoverageChange": report.get("observedCoverageChange"),
        "nextRecommendedTranche": report.get("nextRecommendedTranche"),
        "mutationCounters": report.get("mutationCounters"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialized_ids_path.write_text("\n".join(report.get("materializedPaperIds") or []) + "\n", encoding="utf-8")
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "summaryJsonPath": str(summary_json_path),
        "reportMarkdownPath": str(summary_path),
        "materializedPaperIdsPath": str(materialized_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    before = dict(report.get("baselineBefore") or {})
    after = dict(report.get("baselineAfter") or {})
    counts = dict(report.get("counts") or {})
    observed = dict(report.get("observedCoverageChange") or {})
    lines = [
        "# Parsed Artifact Oversized PDF Materialization Apply",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- apply requested: {dict(report.get('report') or {}).get('applyRequested')}",
        f"- missingParsedArtifacts before: {before.get('missingParsedArtifacts', 0)}",
        f"- missingParsedArtifacts after: {after.get('missingParsedArtifacts', 0)}",
        f"- planned: {counts.get('planned', 0)}",
        f"- materialized: {counts.get('materialized', 0)}",
        f"- blocked: {counts.get('blocked', 0)}",
        f"- failed: {counts.get('failed', 0)}",
        f"- actual missingParsedArtifacts reduction: {observed.get('actualMissingParsedArtifactsReduction', 0)}",
        "",
        "## Materialized Paper IDs",
        "",
    ]
    lines.extend(f"- `{paper_id}`" for paper_id in list(report.get("materializedPaperIds") or []))
    lines.extend(["", "## Status Taxonomy", ""])
    for item in list(report.get("statusTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional khub config path.")
    parser.add_argument("--dry-run-report", default=str(DEFAULT_DRY_RUN_REPORT_PATH))
    parser.add_argument("--report-name", default="parsed-artifact-oversized-pdf-materialization-apply")
    parser.add_argument("--apply", action="store_true", help="Write parsed artifacts for selected oversized local PDFs.")
    parser.add_argument("--overwrite", action="store_true", help="Refresh existing parsed artifacts for selected IDs.")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (
        DEFAULT_APPLY_OUTPUT_DIR if args.apply else DEFAULT_PLAN_OUTPUT_DIR
    )
    config = Config(args.config)
    dry_run_report_path = Path(args.dry_run_report).expanduser()
    dry_run_report = _load_json(dry_run_report_path)
    sqlite_db = SQLiteDatabase(
        config.sqlite_path,
        enable_event_store=False,
        bootstrap=False,
        read_only=not bool(args.apply),
    )
    try:
        report = build_parsed_artifact_oversized_pdf_materialization_apply(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            dry_run_report=dry_run_report,
            dry_run_report_path=dry_run_report_path,
            report_name=args.report_name,
            apply=bool(args.apply),
            overwrite=bool(args.overwrite),
        )
    finally:
        sqlite_db.close()

    paths = write_parsed_artifact_oversized_pdf_materialization_apply(report, output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "baselineBefore": report["baselineBefore"],
                "baselineAfter": report["baselineAfter"],
                "counts": report["counts"],
                "observedCoverageChange": report["observedCoverageChange"],
                "nextRecommendedTranche": report["nextRecommendedTranche"],
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
    "PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_APPLY_SCHEMA_ID",
    "build_parsed_artifact_oversized_pdf_materialization_apply",
    "write_parsed_artifact_oversized_pdf_materialization_apply",
]
