"""Report-only dry-run for oversized parsed artifact PDF materialization.

This helper handles the post-source-recovery bucket where source PDFs are
registered and local, but were held out by the default parsed-artifact coverage
resource limit. It plans an explicit-resource materialization dry-run only. It
does not write parsed artifacts, rewrite source registrations, scan the vault,
route parsers, create evidence, mutate DB/index state, or change answers.
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
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import (
    DEFAULT_MAX_SOURCE_PDF_BYTES,
    _counter_items,
    _mutation_counters,
    _mutation_policy,
)
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    build_parsed_artifact_source_recovery_feasibility,
)
from knowledge_hub.papers.parsed_materialization import (
    PARSED_MATERIALIZATION_SCHEMA_ID,
    materialize_parsed_artifacts,
)


PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-oversized-pdf-materialization-dry-run.v1"
)

DEFAULT_POLICY_MAX_SOURCE_PDF_BYTES = 50 * 1024 * 1024
DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-oversized-pdf-materialization-dry-run"
).expanduser()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _blocked_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "sizeBytes": int(row.get("sizeBytes") or 0),
        "oversizedPolicyStatus": _clean_text(row.get("oversizedPolicyStatus")),
        "blockedReason": reason,
        "recommendedAction": "keep_in_manual_resource_review",
    }


def _selected_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "sizeBytes": int(row.get("sizeBytes") or 0),
        "oversizedPolicyStatus": _clean_text(row.get("oversizedPolicyStatus")),
        "dryRunStatus": "oversized_pdf_materialization_dry_run_candidate_only",
        "plannedAction": "materialize_parsed_artifacts_apply_false",
    }


def build_parsed_artifact_oversized_pdf_materialization_dry_run(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    report_name: str = "parsed-artifact-oversized-pdf-materialization-dry-run",
    base_max_source_pdf_bytes: int = DEFAULT_MAX_SOURCE_PDF_BYTES,
    policy_max_source_pdf_bytes: int = DEFAULT_POLICY_MAX_SOURCE_PDF_BYTES,
    limit: int = 0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an explicit-resource dry-run for oversized local PDF blockers."""

    effective_base_max = max(0, int(base_max_source_pdf_bytes or 0))
    effective_policy_max = max(0, int(policy_max_source_pdf_bytes or 0))
    effective_limit = max(0, int(limit or 0))
    feasibility = build_parsed_artifact_source_recovery_feasibility(
        sqlite_db=sqlite_db,
        papers_dir=papers_dir,
        report_name=f"{report_name}-input-source-recovery-feasibility",
        max_source_pdf_bytes=effective_base_max,
        generated_at=generated_at,
    )
    oversized_rows = [dict(row) for row in list(feasibility.get("oversizedPolicyRows") or [])]
    selected_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for row in oversized_rows:
        status = _clean_text(row.get("oversizedPolicyStatus"))
        size_bytes = int(row.get("sizeBytes") or 0)
        if status != "oversized_small_policy_candidate":
            blocked_rows.append(_blocked_row(row, reason="blocked_oversized_policy_status_not_small"))
            continue
        if effective_policy_max and size_bytes > effective_policy_max:
            blocked_rows.append(_blocked_row(row, reason="blocked_policy_max_source_pdf_bytes_exceeded"))
            continue
        selected_rows.append(_selected_row(row))

    if effective_limit:
        blocked_rows.extend(
            _blocked_row(row, reason="blocked_explicit_batch_limit")
            for row in selected_rows[effective_limit:]
        )
        selected_rows = selected_rows[:effective_limit]

    selected_ids = [_clean_text(row.get("paperId")) for row in selected_rows]
    dry_run_payload = materialize_parsed_artifacts(
        sqlite_db=sqlite_db,
        papers_dir=papers_dir,
        paper_ids=selected_ids,
        apply=False,
        overwrite=False,
    )
    materialization_validation = validate_payload(dry_run_payload, PARSED_MATERIALIZATION_SCHEMA_ID, strict=True)
    dry_run_counts = dict(dry_run_payload.get("counts") or {})
    planned_count = int(dry_run_counts.get("planned") or 0)
    blocked_count = int(dry_run_counts.get("blocked") or 0)
    failed_count = int(dry_run_counts.get("failed") or 0)
    skipped_count = int(dry_run_counts.get("skippedExisting") or 0)
    ready = bool(
        selected_ids
        and materialization_validation.ok
        and planned_count == len(selected_ids)
        and blocked_count == 0
        and failed_count == 0
        and skipped_count == 0
    )
    blocker_counter = Counter(_clean_text(row.get("blockedReason")) for row in blocked_rows)
    baseline = dict(feasibility.get("baseline") or {})
    missing_before = int(baseline.get("missingParsedArtifacts") or 0)

    payload = {
        "schema": PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        "status": "ready" if ready else "blocked",
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "baseMaxSourcePdfBytes": effective_base_max,
            "policyMaxSourcePdfBytes": effective_policy_max,
            "limit": effective_limit,
            "selectionRule": (
                "source recovery feasibility oversizedPolicyRows where "
                "oversizedPolicyStatus == oversized_small_policy_candidate and "
                "sizeBytes <= policyMaxSourcePdfBytes"
            ),
            "nonScope": [
                "parsed_artifact_write",
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
        "baseline": baseline,
        "inputSourceRecoveryFeasibility": {
            "status": feasibility.get("status"),
            "sourceBlockerSnapshot": dict(feasibility.get("sourceBlockerSnapshot") or {}),
            "oversizedPolicySummary": dict(feasibility.get("oversizedPolicySummary") or {}),
            "sourceMissingRecoverySummary": dict(feasibility.get("sourceMissingRecoverySummary") or {}),
            "textSourcePolicySummary": dict(feasibility.get("textSourcePolicySummary") or {}),
        },
        "candidatePool": {
            "inputRows": len(oversized_rows),
            "sourcePdfOversizedRows": len(oversized_rows),
            "oversizedSmallPolicyCandidateRows": sum(
                1
                for row in oversized_rows
                if _clean_text(row.get("oversizedPolicyStatus")) == "oversized_small_policy_candidate"
            ),
            "selectedDryRunRows": len(selected_rows),
            "blockedRows": len(blocked_rows),
            "blockerTaxonomy": _counter_items(blocker_counter),
        },
        "selectedCandidatePaperIds": selected_ids,
        "selectedCandidates": selected_rows,
        "blockedCandidates": blocked_rows,
        "dryRunMaterializationReadiness": {
            "ready": ready,
            "status": dry_run_payload.get("status"),
            "counts": dry_run_counts,
            "allSelectedPlanned": bool(planned_count == len(selected_ids)),
            "schemaValidation": {
                "ok": bool(materialization_validation.ok),
                "errors": list(materialization_validation.errors),
            },
            "applySkippedReason": "report_only_oversized_pdf_materialization_requires_separate_apply_tranche",
        },
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": missing_before,
            "expectedMissingParsedArtifactsReduction": planned_count,
            "expectedMissingParsedArtifactsAfter": max(0, missing_before - planned_count),
        },
        "nextRecommendedTranche": {
            "name": "parsed_artifact_oversized_pdf_materialization_apply",
            "candidateCount": planned_count,
            "paperIds": selected_ids,
            "rationale": (
                "apply only these explicit-resource oversized local PDFs before manual lookup "
                "or text-source policy work"
            ),
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "dryRunMaterialization": dry_run_payload,
        "warnings": [],
    }
    validation = validate_payload(
        payload,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact oversized PDF materialization dry-run schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_oversized_pdf_materialization_dry_run(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, summary JSON, Markdown, and selected-paper IDs."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact oversized PDF materialization dry-run schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-oversized-pdf-materialization-dry-run.json"
    summary_json_path = root / "parsed-artifact-oversized-pdf-materialization-dry-run-summary.json"
    summary_path = root / "parsed-artifact-oversized-pdf-materialization-dry-run.md"
    selected_ids_path = root / "selected-oversized-pdf-materialization-paper-ids.txt"
    summary = {
        "schema": report.get("schema"),
        "status": report.get("status"),
        "baseline": report.get("baseline"),
        "candidatePool": report.get("candidatePool"),
        "dryRunMaterializationReadiness": report.get("dryRunMaterializationReadiness"),
        "expectedCoverageChangeIfApplied": report.get("expectedCoverageChangeIfApplied"),
        "nextRecommendedTranche": report.get("nextRecommendedTranche"),
        "mutationCounters": report.get("mutationCounters"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_ids_path.write_text("\n".join(report.get("selectedCandidatePaperIds") or []) + "\n", encoding="utf-8")
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "summaryJsonPath": str(summary_json_path),
        "reportMarkdownPath": str(summary_path),
        "selectedCandidateIdsPath": str(selected_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    pool = dict(report.get("candidatePool") or {})
    readiness = dict(report.get("dryRunMaterializationReadiness") or {})
    expected = dict(report.get("expectedCoverageChangeIfApplied") or {})
    lines = [
        "# Parsed Artifact Oversized PDF Materialization Dry-Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- source PDF oversized rows: {pool.get('sourcePdfOversizedRows', 0)}",
        f"- selected dry-run rows: {pool.get('selectedDryRunRows', 0)}",
        f"- dry-run ready: {readiness.get('ready')}",
        f"- expected missingParsedArtifacts reduction if applied: {expected.get('expectedMissingParsedArtifactsReduction', 0)}",
        f"- expected missingParsedArtifacts after apply: {expected.get('expectedMissingParsedArtifactsAfter', 0)}",
        f"- apply: skipped (`{readiness.get('applySkippedReason')}`)",
        "",
        "## Selected Candidate Paper IDs",
        "",
    ]
    lines.extend(f"- `{paper_id}`" for paper_id in list(report.get("selectedCandidatePaperIds") or []))
    lines.extend(["", "## Blocker Taxonomy", ""])
    for item in list(pool.get("blockerTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional khub config path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Report output directory.")
    parser.add_argument("--report-name", default="parsed-artifact-oversized-pdf-materialization-dry-run")
    parser.add_argument("--base-max-pdf-bytes", type=int, default=DEFAULT_MAX_SOURCE_PDF_BYTES)
    parser.add_argument("--policy-max-pdf-bytes", type=int, default=DEFAULT_POLICY_MAX_SOURCE_PDF_BYTES)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    config = Config(args.config)
    sqlite_db = SQLiteDatabase(
        config.sqlite_path,
        enable_event_store=False,
        bootstrap=False,
        read_only=True,
    )
    try:
        report = build_parsed_artifact_oversized_pdf_materialization_dry_run(
            sqlite_db=sqlite_db,
            papers_dir=config.papers_dir,
            report_name=args.report_name,
            base_max_source_pdf_bytes=args.base_max_pdf_bytes,
            policy_max_source_pdf_bytes=args.policy_max_pdf_bytes,
            limit=args.limit,
        )
    finally:
        sqlite_db.close()

    paths = write_parsed_artifact_oversized_pdf_materialization_dry_run(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "baseline": report["baseline"],
                "candidatePool": report["candidatePool"],
                "dryRunMaterializationReadiness": report["dryRunMaterializationReadiness"],
                "expectedCoverageChangeIfApplied": report["expectedCoverageChangeIfApplied"],
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
    "DEFAULT_POLICY_MAX_SOURCE_PDF_BYTES",
    "PARSED_ARTIFACT_OVERSIZED_PDF_MATERIALIZATION_DRY_RUN_SCHEMA_ID",
    "build_parsed_artifact_oversized_pdf_materialization_dry_run",
    "write_parsed_artifact_oversized_pdf_materialization_dry_run",
]
