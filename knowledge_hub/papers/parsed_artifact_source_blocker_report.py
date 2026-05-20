"""Report-only source blocker audit for missing parsed artifacts.

This helper runs after existing local-PDF coverage has been exhausted. It
classifies the remaining missing parsed artifacts by registered source state
without downloading sources, changing parser routing, writing parsed artifacts,
touching indexes, scanning the vault, or resolving manual blockers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.extraction_diagnostics import build_extraction_report
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import (
    DEFAULT_MAX_SOURCE_PDF_BYTES,
    _baseline_command_snapshot,
    _classify_source_artifact,
    _clean_text,
    _counter_items,
    _diagnostic_reasons,
    _missing_parsed_items,
    _mutation_counters,
    _source_artifact,
)


PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-blocker-report.v1"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_blocker_policy() -> dict[str, Any]:
    return {
        "applyRequested": False,
        "applyPerformed": False,
        "sourceDownload": False,
        "sourcePathRewrite": False,
        "sourceRegistrationMutation": False,
        "parsedArtifactWrite": False,
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


def _source_blocker_counters() -> dict[str, int]:
    counters = _mutation_counters()
    counters.update(
        {
            "sourceDownloadRows": 0,
            "sourcePathRewriteRows": 0,
            "sourceRegistrationMutationRows": 0,
        }
    )
    return counters


def _action_bucket(source_status: str) -> str:
    if source_status == "source_pdf_missing":
        return "manual_source_recovery_or_registration_required"
    if source_status == "source_pdf_oversized":
        return "separate_oversized_pdf_materialization_policy_required"
    if source_status == "text_source_unsupported":
        return "separate_text_source_materialization_policy_required"
    if source_status == "source_pdf_zero_byte":
        return "manual_source_integrity_review_required"
    if source_status == "source_pdf_not_file":
        return "manual_source_path_review_required"
    if source_status == "paper_not_registered":
        return "registry_integrity_review_required"
    return "manual_source_blocker_review_required"


def _source_status_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = defaultdict(list)
    for item in items:
        by_status[str(item.get("sourceStatus") or "")].append(str(item.get("paperId") or ""))
    return {
        status: {
            "count": len(paper_ids),
            "paperIds": paper_ids,
            "recommendedAction": _action_bucket(status),
        }
        for status, paper_ids in sorted(by_status.items())
    }


def build_parsed_artifact_source_blocker_report(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    report_name: str = "parsed-artifact-source-blocker-report",
    max_source_pdf_bytes: int = DEFAULT_MAX_SOURCE_PDF_BYTES,
    generated_at: str | None = None,
    baseline_command_report: dict[str, Any] | None = None,
    baseline_command_report_path: str | Path | None = None,
    sqlite_hash_before_cli: str = "",
    sqlite_hash_after_cli: str = "",
) -> dict[str, Any]:
    """Build a source-blocker-only report for missing parsed artifacts."""

    effective_max_bytes = max(0, int(max_source_pdf_bytes or 0))
    baseline_report = build_extraction_report(sqlite_db=sqlite_db, papers_dir=papers_dir)
    baseline_counts = dict(baseline_report.get("counts") or {})
    missing_items = _missing_parsed_items(baseline_report)

    source_artifact_counter: Counter[str] = Counter()
    blocker_counter: Counter[str] = Counter()
    blocked_rows: list[dict[str, Any]] = []
    eligible_existing_pdf_rows: list[dict[str, Any]] = []

    for item in missing_items:
        paper_id = _clean_text(item.get("paperId"))
        row = sqlite_db.get_paper(paper_id) if hasattr(sqlite_db, "get_paper") else None
        source_artifact = (
            _source_artifact(dict(row), papers_dir=papers_dir)
            if isinstance(row, dict) and row
            else {"kind": "", "exists": False, "isFile": False, "sizeBytes": 0, "path": ""}
        )
        source_status = (
            _classify_source_artifact(source_artifact, max_source_pdf_bytes=effective_max_bytes)
            if isinstance(row, dict) and row
            else "paper_not_registered"
        )
        source_artifact_counter[source_status] += 1
        title = _clean_text(item.get("paperTitle") or (row.get("title") if isinstance(row, dict) else ""))
        blocker = {
            "paperId": paper_id,
            "paperTitle": title,
            "degradationReasons": _diagnostic_reasons(item),
            "sourceArtifact": source_artifact,
            "sourceStatus": source_status,
            "recommendedAction": _action_bucket(source_status),
            "resolutionMode": "separate_manual_or_policy_tranche_required",
        }
        if source_status == "eligible_existing_pdf":
            eligible_existing_pdf_rows.append(blocker)
            continue
        blocker_counter[source_status] += 1
        blocked_rows.append(blocker)

    payload = {
        "schema": PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID,
        "status": "blocked_source_only" if blocked_rows else "no_source_blockers",
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "maxSourcePdfBytes": effective_max_bytes,
            "selectionRule": "missing parsed artifact + sourceStatus != eligible_existing_pdf",
            "nonScope": [
                "source_download",
                "source_path_rewrite",
                "source_registration_mutation",
                "parsed_artifact_write",
                "parser_routing",
                "strict_or_citation_or_runtime_evidence",
                "database_or_index_or_reembed",
                "vault_scan_or_write",
                "answer_integration",
                "manual_blocker_resolution",
            ],
        },
        "baseline": {
            "scannedPapers": int(baseline_counts.get("scannedPapers") or 0),
            "reportedPapers": int(baseline_counts.get("reportedPapers") or 0),
            "degradedPapers": int(baseline_counts.get("degradedPapers") or 0),
            "missingParsedArtifacts": int(baseline_counts.get("missingParsedArtifacts") or 0),
            "source": "read_only_build_extraction_report",
            "inspectedCommandReport": _baseline_command_snapshot(
                baseline_command_report,
                baseline_command_report_path=baseline_command_report_path,
                read_only_counts=baseline_counts,
            ),
            "sqliteHashBeforeCliInspection": sqlite_hash_before_cli,
            "sqliteHashAfterCliInspection": sqlite_hash_after_cli,
            "sqliteHashChangedAfterCliInspection": bool(
                sqlite_hash_before_cli
                and sqlite_hash_after_cli
                and sqlite_hash_before_cli != sqlite_hash_after_cli
            ),
        },
        "sourceBlockerPool": {
            "missingParsedArtifacts": len(missing_items),
            "sourceBlockerRows": len(blocked_rows),
            "eligibleExistingPdfRows": len(eligible_existing_pdf_rows),
            "sourceArtifactTaxonomy": _counter_items(source_artifact_counter),
            "blockerTaxonomy": _counter_items(blocker_counter),
        },
        "sourceBlockerPaperIdsByStatus": _source_status_summary(blocked_rows),
        "sourceBlockers": blocked_rows,
        "eligibleExistingPdfUnexpected": eligible_existing_pdf_rows,
        "nextRecommendedTranche": {
            "name": "parsed-artifact-source-blocker-resolution-plan",
            "candidateCount": len(blocked_rows),
            "rationale": "separate source-blocker planning is required before any further parsed-artifact coverage apply",
            "recommendedFirstBucket": "source_pdf_missing" if blocker_counter.get("source_pdf_missing") else "",
        },
        "mutationPolicy": _source_blocker_policy(),
        "mutationCounters": _source_blocker_counters(),
        "warnings": [
            "direct_cli_extraction_report_changed_sqlite_file_hash"
        ]
        if (
            sqlite_hash_before_cli
            and sqlite_hash_after_cli
            and sqlite_hash_before_cli != sqlite_hash_after_cli
        )
        else [],
    }
    validation = validate_payload(payload, PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact source blocker report schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_source_blocker_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and blocker-ID artifacts."""

    validation = validate_payload(report, PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact source blocker report schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-blocker-report.json"
    summary_path = root / "parsed-artifact-source-blocker-report.md"
    ids_by_status_path = root / "source-blocker-paper-ids-by-status.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ids_by_status_path.write_text(
        json.dumps(report.get("sourceBlockerPaperIdsByStatus") or {}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "reportMarkdownPath": str(summary_path),
        "sourceBlockerIdsByStatusPath": str(ids_by_status_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    pool = dict(report.get("sourceBlockerPool") or {})
    lines = [
        "# Parsed Artifact Source Blocker Report",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- source blocker rows: {pool.get('sourceBlockerRows', 0)}",
        f"- eligible existing PDF rows: {pool.get('eligibleExistingPdfRows', 0)}",
        "",
        "## Blocker Taxonomy",
        "",
    ]
    for item in list(pool.get("blockerTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Recommended Buckets", ""])
    for status, item in dict(report.get("sourceBlockerPaperIdsByStatus") or {}).items():
        lines.append(f"- `{status}`: {item.get('count', 0)} -> `{item.get('recommendedAction', '')}`")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PARSED_ARTIFACT_SOURCE_BLOCKER_REPORT_SCHEMA_ID",
    "build_parsed_artifact_source_blocker_report",
    "write_parsed_artifact_source_blocker_report",
]
