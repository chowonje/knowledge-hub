"""Report-only resolution planning for missing parsed artifact source PDFs.

This helper consumes the source-blocker classifier and narrows the next tranche
to `source_pdf_missing` rows only. It does not download sources, rewrite source
paths, update source registrations, materialize parsed artifacts, scan the
vault, or resolve manual blockers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import _counter_items
from knowledge_hub.papers.parsed_artifact_source_blocker_report import (
    build_parsed_artifact_source_blocker_report,
    _source_blocker_counters,
    _source_blocker_policy,
)


PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-missing-resolution-plan.v1"
)

DEFAULT_SOURCE_MISSING_LIMIT = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _resolution_plan_status(blocker: dict[str, Any]) -> str:
    source_artifact = dict(blocker.get("sourceArtifact") or {})
    if _clean_text(blocker.get("sourceStatus")) != "source_pdf_missing":
        return "held_out_not_source_pdf_missing"
    kind = _clean_text(source_artifact.get("kind"))
    path = _clean_text(source_artifact.get("path"))
    exists = bool(source_artifact.get("exists"))
    if kind == "pdf" and path and not exists:
        return "registered_pdf_path_missing"
    if not kind and not path:
        return "no_registered_source_artifact"
    return "source_pdf_missing_unclassified"


def _recommended_action(resolution_plan_status: str) -> str:
    if resolution_plan_status == "registered_pdf_path_missing":
        return "separate_manual_source_recovery_or_path_drift_decision_required"
    if resolution_plan_status == "no_registered_source_artifact":
        return "separate_manual_source_identification_and_registration_decision_required"
    if resolution_plan_status == "source_pdf_missing_unclassified":
        return "separate_registry_shape_review_required"
    return "held_out_for_non_source_pdf_missing_tranche"


def _source_presence(source_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "registeredKind": _clean_text(source_artifact.get("kind")),
        "registeredPathPresent": bool(_clean_text(source_artifact.get("path"))),
        "localArtifactExists": bool(source_artifact.get("exists")),
        "localArtifactIsFile": bool(source_artifact.get("isFile")),
        "localArtifactSizeBytes": int(source_artifact.get("sizeBytes") or 0),
    }


def _planned_candidate(blocker: dict[str, Any]) -> dict[str, Any]:
    source_artifact = dict(blocker.get("sourceArtifact") or {})
    resolution_status = _resolution_plan_status(blocker)
    return {
        "paperId": _clean_text(blocker.get("paperId")),
        "paperTitle": _clean_text(blocker.get("paperTitle")),
        "degradationReasons": list(blocker.get("degradationReasons") or []),
        "sourceArtifact": source_artifact,
        "sourceArtifactPresence": _source_presence(source_artifact),
        "sourceStatus": _clean_text(blocker.get("sourceStatus")),
        "resolutionPlanStatus": resolution_status,
        "recommendedAction": _recommended_action(resolution_status),
        "resolutionMode": "report_only_manual_or_policy_decision_required",
    }


def _ids_by_status(rows: list[dict[str, Any]], *, status_key: str) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        status = _clean_text(row.get(status_key))
        grouped.setdefault(status, []).append(_clean_text(row.get("paperId")))
    return {
        status: {
            "count": len(paper_ids),
            "paperIds": paper_ids,
        }
        for status, paper_ids in sorted(grouped.items())
    }


def build_parsed_artifact_source_missing_resolution_plan(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    plan_name: str = "parsed-artifact-source-missing-resolution-plan",
    candidate_limit: int = DEFAULT_SOURCE_MISSING_LIMIT,
    max_source_pdf_bytes: int | None = None,
    generated_at: str | None = None,
    baseline_command_report: dict[str, Any] | None = None,
    baseline_command_report_path: str | Path | None = None,
    sqlite_hash_before_cli: str = "",
    sqlite_hash_after_cli: str = "",
) -> dict[str, Any]:
    """Build a report-only plan for the next `source_pdf_missing` tranche."""

    effective_limit = max(0, int(candidate_limit or 0))
    kwargs: dict[str, Any] = {
        "sqlite_db": sqlite_db,
        "papers_dir": papers_dir,
        "report_name": f"{plan_name}-input-source-blocker-snapshot",
        "generated_at": generated_at,
        "baseline_command_report": baseline_command_report,
        "baseline_command_report_path": baseline_command_report_path,
        "sqlite_hash_before_cli": sqlite_hash_before_cli,
        "sqlite_hash_after_cli": sqlite_hash_after_cli,
    }
    if max_source_pdf_bytes is not None:
        kwargs["max_source_pdf_bytes"] = max_source_pdf_bytes
    source_blocker_report = build_parsed_artifact_source_blocker_report(**kwargs)
    source_blockers = list(source_blocker_report.get("sourceBlockers") or [])
    source_pdf_missing_rows = [
        _planned_candidate(dict(row))
        for row in source_blockers
        if _clean_text(dict(row).get("sourceStatus")) == "source_pdf_missing"
    ]
    selected = source_pdf_missing_rows[:effective_limit] if effective_limit else []
    unselected = source_pdf_missing_rows[len(selected) :]
    held_out_rows = [
        dict(row)
        for row in source_blockers
        if _clean_text(dict(row).get("sourceStatus")) != "source_pdf_missing"
    ]

    source_presence_counter: Counter[str] = Counter(
        _clean_text(row.get("resolutionPlanStatus")) for row in source_pdf_missing_rows
    )
    selected_presence_counter: Counter[str] = Counter(
        _clean_text(row.get("resolutionPlanStatus")) for row in selected
    )
    held_out_counter: Counter[str] = Counter(_clean_text(row.get("sourceStatus")) for row in held_out_rows)
    baseline = dict(source_blocker_report.get("baseline") or {})

    immediate_reduction = 0
    potential_unlocked = len(selected)
    missing_before = int(baseline.get("missingParsedArtifacts") or 0)
    payload = {
        "schema": PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID,
        "status": "planned_report_only" if selected else "no_source_pdf_missing_candidates",
        "generatedAt": generated_at or _utc_now(),
        "plan": {
            "name": plan_name,
            "limit": effective_limit,
            "sourceBucket": "source_pdf_missing",
            "selectionRule": "sourceBlocker.sourceStatus == source_pdf_missing, preserving source blocker report order, bounded by limit",
            "nonScope": [
                "source_download",
                "source_path_rewrite",
                "source_registration_mutation",
                "parsed_artifact_write",
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
        "sourceBlockerSnapshot": {
            "missingParsedArtifacts": int(source_blocker_report.get("sourceBlockerPool", {}).get("missingParsedArtifacts") or 0),
            "sourceBlockerRows": int(source_blocker_report.get("sourceBlockerPool", {}).get("sourceBlockerRows") or 0),
            "eligibleExistingPdfRows": int(source_blocker_report.get("sourceBlockerPool", {}).get("eligibleExistingPdfRows") or 0),
            "sourcePdfMissingRows": len(source_pdf_missing_rows),
            "heldOutSourceBlockerRows": len(held_out_rows),
            "sourceArtifactTaxonomy": list(source_blocker_report.get("sourceBlockerPool", {}).get("sourceArtifactTaxonomy") or []),
            "blockerTaxonomy": list(source_blocker_report.get("sourceBlockerPool", {}).get("blockerTaxonomy") or []),
            "heldOutTaxonomy": _counter_items(held_out_counter),
        },
        "candidatePool": {
            "totalSourcePdfMissingRows": len(source_pdf_missing_rows),
            "selectedCandidateCount": len(selected),
            "unselectedSourcePdfMissingRows": len(unselected),
            "sourceArtifactPresenceTaxonomy": _counter_items(source_presence_counter),
            "selectedSourceArtifactPresenceTaxonomy": _counter_items(selected_presence_counter),
        },
        "selectedCandidatePaperIds": [_clean_text(row.get("paperId")) for row in selected],
        "unselectedSourcePdfMissingPaperIds": [_clean_text(row.get("paperId")) for row in unselected],
        "sourcePdfMissingCandidates": selected,
        "heldOutSourceBlockerPaperIdsByStatus": _ids_by_status(held_out_rows, status_key="sourceStatus"),
        "sourcePdfMissingPaperIdsByResolutionPlanStatus": _ids_by_status(
            source_pdf_missing_rows,
            status_key="resolutionPlanStatus",
        ),
        "dryRunMaterializationReadiness": {
            "ready": False,
            "dryRunAttempted": False,
            "status": "blocked_before_materialization",
            "blockedReason": "source_pdf_missing_local_artifact_absent",
            "selectedCandidateCount": len(selected),
            "applySkippedReason": "report_only_source_missing_resolution_plan_no_apply_path",
        },
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": missing_before,
            "expectedMissingParsedArtifactsReduction": immediate_reduction,
            "expectedMissingParsedArtifactsAfter": max(0, missing_before - immediate_reduction),
            "potentialMissingParsedArtifactsUnlockedAfterSeparateSourceRecovery": potential_unlocked,
        },
        "nextRecommendedCoverageTranche": {
            "name": "parsed-artifact-source-missing-resolution-plan-next",
            "candidateCount": len(unselected),
            "paperIds": [_clean_text(row.get("paperId")) for row in unselected],
            "rationale": "finish the remaining source_pdf_missing planning bucket before oversized-PDF or text-source policy work",
        },
        "mutationPolicy": _source_blocker_policy(),
        "mutationCounters": _source_blocker_counters(),
        "warnings": list(source_blocker_report.get("warnings") or []),
    }
    validation = validate_payload(
        payload,
        PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact source missing resolution plan schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_source_missing_resolution_plan(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, Markdown, and candidate-ID artifacts."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact source missing resolution plan schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-missing-resolution-plan.json"
    summary_path = root / "parsed-artifact-source-missing-resolution-plan.md"
    selected_ids_path = root / "selected-source-pdf-missing-paper-ids.txt"
    ids_by_status_path = root / "source-pdf-missing-paper-ids-by-resolution-plan-status.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_ids_path.write_text("\n".join(report.get("selectedCandidatePaperIds") or []) + "\n", encoding="utf-8")
    ids_by_status_path.write_text(
        json.dumps(
            report.get("sourcePdfMissingPaperIdsByResolutionPlanStatus") or {},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "reportMarkdownPath": str(summary_path),
        "selectedCandidateIdsPath": str(selected_ids_path),
        "sourcePdfMissingIdsByResolutionPlanStatusPath": str(ids_by_status_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    snapshot = dict(report.get("sourceBlockerSnapshot") or {})
    pool = dict(report.get("candidatePool") or {})
    readiness = dict(report.get("dryRunMaterializationReadiness") or {})
    expected = dict(report.get("expectedCoverageChangeIfApplied") or {})
    lines = [
        "# Parsed Artifact Source Missing Resolution Plan",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- source blocker rows: {snapshot.get('sourceBlockerRows', 0)}",
        f"- source_pdf_missing rows: {snapshot.get('sourcePdfMissingRows', 0)}",
        f"- selected candidate count: {pool.get('selectedCandidateCount', 0)}",
        f"- dry-run materialization ready: {readiness.get('ready')}",
        f"- expected missingParsedArtifacts reduction if applied: {expected.get('expectedMissingParsedArtifactsReduction', 0)}",
        f"- potential missingParsedArtifacts unlocked after separate source recovery: {expected.get('potentialMissingParsedArtifactsUnlockedAfterSeparateSourceRecovery', 0)}",
        f"- apply: skipped (`{readiness.get('applySkippedReason')}`)",
        "",
        "## Selected Candidate Paper IDs",
        "",
    ]
    lines.extend(f"- `{paper_id}`" for paper_id in list(report.get("selectedCandidatePaperIds") or []))
    lines.extend(["", "## Source Artifact Presence Taxonomy", ""])
    for item in list(pool.get("sourceArtifactPresenceTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Held Out Blocker Taxonomy", ""])
    for item in list(snapshot.get("heldOutTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SOURCE_MISSING_LIMIT",
    "PARSED_ARTIFACT_SOURCE_MISSING_RESOLUTION_PLAN_SCHEMA_ID",
    "build_parsed_artifact_source_missing_resolution_plan",
    "write_parsed_artifact_source_missing_resolution_plan",
]
