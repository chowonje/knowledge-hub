"""Report-only parsed artifact coverage batch planning.

This helper stays below parser routing, evidence, indexing, vault, and answer
integration surfaces. It selects a bounded dry-run batch from papers that are
already missing parsed artifacts and already have local source PDFs.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.extraction_diagnostics import build_extraction_report
from knowledge_hub.papers.parsed_materialization import (
    PARSED_MATERIALIZATION_SCHEMA_ID,
    materialize_parsed_artifacts,
)


PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-coverage-batch-report.v1"
)

DEFAULT_BATCH_LIMIT = 40
DEFAULT_MAX_SOURCE_PDF_BYTES = 25 * 1024 * 1024


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative(path: str | Path | None, *, root: str | Path, prefix: str = "papers_dir") -> str:
    if not path:
        return ""
    candidate = Path(str(path)).expanduser()
    root_path = Path(str(root)).expanduser()
    try:
        rel = candidate.resolve().relative_to(root_path.resolve())
        return str(Path(prefix) / rel)
    except Exception:
        return str(Path("external") / (candidate.name or "unknown"))


def _sha256_file(path: str | Path | None) -> str:
    if not path:
        return ""
    candidate = Path(str(path)).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact(row: dict[str, Any], *, papers_dir: str | Path) -> dict[str, Any]:
    raw_pdf = _clean_text(row.get("pdf_path") or row.get("pdfPath"))
    raw_text = _clean_text(row.get("text_path") or row.get("textPath"))
    kind = "pdf" if raw_pdf else ("text" if raw_text else "")
    raw_path = raw_pdf or raw_text
    path = Path(raw_path).expanduser() if raw_path else None
    exists = bool(path and path.exists())
    is_file = bool(path and path.is_file()) if exists else False
    size_bytes = path.stat().st_size if path and is_file else 0
    return {
        "kind": kind,
        "exists": exists,
        "isFile": is_file,
        "sizeBytes": size_bytes,
        "path": _safe_relative(path, root=papers_dir) if path else "",
    }


def _classify_source_artifact(
    source_artifact: dict[str, Any],
    *,
    max_source_pdf_bytes: int,
) -> str:
    if source_artifact.get("kind") != "pdf":
        return "source_pdf_missing" if not source_artifact.get("kind") else "text_source_unsupported"
    if not source_artifact.get("exists"):
        return "source_pdf_missing"
    if not source_artifact.get("isFile"):
        return "source_pdf_not_file"
    size_bytes = int(source_artifact.get("sizeBytes") or 0)
    if size_bytes <= 0:
        return "source_pdf_zero_byte"
    if size_bytes > max_source_pdf_bytes:
        return "source_pdf_oversized"
    return "eligible_existing_pdf"


def _diagnostic_reasons(item: dict[str, Any]) -> list[str]:
    diagnostic = dict(item.get("diagnostic") or {})
    return [_clean_text(reason) for reason in list(diagnostic.get("degradationReasons") or []) if _clean_text(reason)]


def _missing_parsed_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in list(report.get("papers") or [])
        if "parsed_artifact_missing" in _diagnostic_reasons(dict(item))
    ]


def _counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"reason": reason, "count": int(count)} for reason, count in sorted(counter.items())]


def _mutation_counters() -> dict[str, int]:
    return {
        "parsedArtifactWriteRows": 0,
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
    }


def _mutation_policy() -> dict[str, Any]:
    return {
        "applyRequested": False,
        "applyPerformed": False,
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


def _baseline_command_snapshot(
    baseline_command_report: dict[str, Any] | None,
    *,
    baseline_command_report_path: str | Path | None,
    read_only_counts: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_command_report:
        return {}
    path = Path(str(baseline_command_report_path)).expanduser() if baseline_command_report_path else None
    counts = dict(baseline_command_report.get("counts") or {})
    return {
        "path": str(path) if path else "",
        "sha256": _sha256_file(path),
        "counts": {
            "scannedPapers": int(counts.get("scannedPapers") or 0),
            "missingParsedArtifacts": int(counts.get("missingParsedArtifacts") or 0),
        },
        "matchesReadOnlyBaseline": {
            "scannedPapers": int(counts.get("scannedPapers") or 0)
            == int(read_only_counts.get("scannedPapers") or 0),
            "missingParsedArtifacts": int(counts.get("missingParsedArtifacts") or 0)
            == int(read_only_counts.get("missingParsedArtifacts") or 0),
        },
    }


def build_parsed_artifact_coverage_batch_report(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    batch_name: str = "parsed-artifact-coverage-batch-report",
    batch_limit: int = DEFAULT_BATCH_LIMIT,
    max_source_pdf_bytes: int = DEFAULT_MAX_SOURCE_PDF_BYTES,
    generated_at: str | None = None,
    baseline_command_report: dict[str, Any] | None = None,
    baseline_command_report_path: str | Path | None = None,
    sqlite_hash_before_cli: str = "",
    sqlite_hash_after_cli: str = "",
) -> dict[str, Any]:
    """Build a report-only parsed artifact coverage batch plan."""

    effective_limit = max(0, int(batch_limit or 0))
    effective_max_bytes = max(0, int(max_source_pdf_bytes or 0))
    baseline_report = build_extraction_report(sqlite_db=sqlite_db, papers_dir=papers_dir)
    baseline_counts = dict(baseline_report.get("counts") or {})
    missing_items = _missing_parsed_items(baseline_report)

    rows_by_id: dict[str, dict[str, Any]] = {}
    if hasattr(sqlite_db, "get_paper"):
        for item in missing_items:
            paper_id = _clean_text(item.get("paperId"))
            row = sqlite_db.get_paper(paper_id)
            if isinstance(row, dict) and row:
                rows_by_id[paper_id] = dict(row)

    all_missing_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    blocker_counter: Counter[str] = Counter()
    source_artifact_counter: Counter[str] = Counter()
    for item in missing_items:
        paper_id = _clean_text(item.get("paperId"))
        row = rows_by_id.get(paper_id, {})
        source_artifact = _source_artifact(row, papers_dir=papers_dir) if row else {
            "kind": "",
            "exists": False,
            "isFile": False,
            "sizeBytes": 0,
            "path": "",
        }
        source_status = (
            _classify_source_artifact(source_artifact, max_source_pdf_bytes=effective_max_bytes)
            if row
            else "paper_not_registered"
        )
        title = _clean_text(item.get("paperTitle") or (row.get("title") if row else ""))
        candidate = {
            "paperId": paper_id,
            "paperTitle": title,
            "degradationReasons": _diagnostic_reasons(item),
            "sourceArtifact": source_artifact,
            "sourceStatus": source_status,
        }
        all_missing_rows.append(candidate)
        source_artifact_counter[source_status] += 1
        if source_status == "eligible_existing_pdf":
            eligible_rows.append(candidate)
        else:
            blocker_counter[source_status] += 1

    selected = eligible_rows[:effective_limit] if effective_limit else []
    unselected_eligible = eligible_rows[len(selected) :]
    selected_ids = [str(item["paperId"]) for item in selected]
    dry_run_payload = materialize_parsed_artifacts(
        sqlite_db=sqlite_db,
        papers_dir=papers_dir,
        paper_ids=selected_ids,
        apply=False,
        overwrite=False,
    )
    dry_run_validation = validate_payload(dry_run_payload, PARSED_MATERIALIZATION_SCHEMA_ID, strict=True)
    dry_run_counts = dict(dry_run_payload.get("counts") or {})
    dry_run_ready = bool(
        selected_ids
        and dry_run_validation.ok
        and int(dry_run_counts.get("planned") or 0) == len(selected_ids)
        and int(dry_run_counts.get("blocked") or 0) == 0
        and int(dry_run_counts.get("failed") or 0) == 0
        and int(dry_run_counts.get("skippedExisting") or 0) == 0
    )
    expected_reduction = int(dry_run_counts.get("planned") or 0)
    expected_after = max(0, int(baseline_counts.get("missingParsedArtifacts") or 0) - expected_reduction)

    if unselected_eligible:
        blocker_counter["eligible_existing_pdf_not_selected_batch_cap"] += len(unselected_eligible)

    payload = {
        "schema": PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID,
        "status": "ready" if dry_run_ready else "blocked",
        "generatedAt": generated_at or _utc_now(),
        "batch": {
            "name": batch_name,
            "limit": effective_limit,
            "maxSourcePdfBytes": effective_max_bytes,
            "selectionRule": "missing parsed artifact + sourceArtifact.kind == pdf + sourceArtifact.exists == true + nonzero file <= maxSourcePdfBytes",
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
        "candidatePool": {
            "missingParsedArtifacts": len(missing_items),
            "eligibleExistingPdf": len(eligible_rows),
            "selectedCandidateCount": len(selected),
            "unselectedEligibleExistingPdf": max(0, len(eligible_rows) - len(selected)),
            "sourceArtifactTaxonomy": _counter_items(source_artifact_counter),
            "blockerTaxonomy": _counter_items(blocker_counter),
        },
        "selectedCandidatePaperIds": selected_ids,
        "unselectedEligibleCandidatePaperIds": [str(item["paperId"]) for item in unselected_eligible],
        "nextRecommendedCoverageTranche": {
            "candidateCount": min(effective_limit, len(unselected_eligible)) if effective_limit else 0,
            "paperIds": [str(item["paperId"]) for item in unselected_eligible[:effective_limit]],
            "rationale": "continue with the next existing-PDF missing parsed artifact batch before parser/evidence/reindex/vault work",
        },
        "candidates": selected,
        "blockedMissingParsedArtifacts": [item for item in all_missing_rows if item.get("sourceStatus") != "eligible_existing_pdf"],
        "dryRunMaterializationReadiness": {
            "ready": dry_run_ready,
            "status": dry_run_payload.get("status"),
            "counts": dry_run_counts,
            "allSelectedPlanned": bool(int(dry_run_counts.get("planned") or 0) == len(selected_ids)),
            "schemaValidation": {
                "ok": bool(dry_run_validation.ok),
                "errors": list(dry_run_validation.errors),
            },
            "applySkippedReason": "report_only_no_separate_apply_approval",
        },
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": int(baseline_counts.get("missingParsedArtifacts") or 0),
            "expectedMissingParsedArtifactsReduction": expected_reduction,
            "expectedMissingParsedArtifactsAfter": expected_after,
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "dryRunMaterialization": dry_run_payload,
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
    validation = validate_payload(payload, PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact coverage batch report schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_coverage_batch_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON and Markdown report artifacts."""

    validation = validate_payload(report, PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact coverage batch report schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-coverage-batch-report.json"
    summary_path = root / "parsed-artifact-coverage-batch-report.md"
    candidate_ids_path = root / "selected-candidate-paper-ids.txt"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate_ids_path.write_text("\n".join(report.get("selectedCandidatePaperIds") or []) + "\n", encoding="utf-8")
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "reportMarkdownPath": str(summary_path),
        "selectedCandidateIdsPath": str(candidate_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    pool = dict(report.get("candidatePool") or {})
    readiness = dict(report.get("dryRunMaterializationReadiness") or {})
    expected = dict(report.get("expectedCoverageChangeIfApplied") or {})
    lines = [
        "# Parsed Artifact Coverage Batch Report",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- selected candidate count: {pool.get('selectedCandidateCount', 0)}",
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


__all__ = [
    "DEFAULT_BATCH_LIMIT",
    "DEFAULT_MAX_SOURCE_PDF_BYTES",
    "PARSED_ARTIFACT_COVERAGE_BATCH_REPORT_SCHEMA_ID",
    "build_parsed_artifact_coverage_batch_report",
    "write_parsed_artifact_coverage_batch_report",
]
