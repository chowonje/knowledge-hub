"""Report-only source recovery feasibility for parsed artifact blockers.

This helper answers the next operational question after source-missing planning:
whether the missing PDFs appear to be local path drift, identifier-based
reacquisition candidates, or manual lookup cases. It also records policy-only
recommendations for oversized PDFs and text-only sources.

It never downloads sources, rewrites paths, mutates paper registrations,
materializes parsed artifacts, changes parser routing, scans the vault, touches
DB/index state, or resolves manual blockers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_coverage_batch_report import (
    DEFAULT_MAX_SOURCE_PDF_BYTES,
    _counter_items,
    _safe_relative,
)
from knowledge_hub.papers.parsed_artifact_source_blocker_report import (
    build_parsed_artifact_source_blocker_report,
    _source_blocker_counters,
    _source_blocker_policy,
)


PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-source-recovery-feasibility.v1"
)

_ARXIV_ID_RE = re.compile(r"^(?P<year_month>\d{4})\.\d{4,5}(?:v\d+)?$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _is_plausible_arxiv_id(value: str) -> bool:
    match = _ARXIV_ID_RE.match(_clean_text(value))
    if not match:
        return False
    year_month = match.group("year_month")
    yy = int(year_month[:2])
    month = int(year_month[2:])
    if month < 1 or month > 12:
        return False
    # New-style arXiv IDs started in 2007. Allow older YY values only if they
    # wrap into the 2000s; classic subject IDs are not represented in this DB key.
    return 7 <= yy <= 99


def _pdf_index(papers_dir: str | Path) -> list[Path]:
    root = Path(str(papers_dir)).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def _paper_id_variants(paper_id: str) -> list[str]:
    token = _clean_text(paper_id).lower()
    variants = {
        token,
        token.replace(".", "_"),
        re.sub(r"[^a-z0-9]+", "_", token).strip("_"),
    }
    return sorted(item for item in variants if item)


def _candidate_path_item(path: Path, *, papers_dir: str | Path, match_type: str) -> dict[str, Any]:
    return {
        "path": _safe_relative(path, root=papers_dir),
        "fileName": path.name,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "matchType": match_type,
    }


def _local_pdf_candidates(
    *,
    paper_id: str,
    source_artifact: dict[str, Any],
    papers_dir: str | Path,
    pdf_paths: list[Path],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    registered_path = _clean_text(source_artifact.get("path"))
    registered_basename = Path(registered_path).name.lower() if registered_path else ""
    for path in pdf_paths:
        if registered_basename and path.name.lower() == registered_basename:
            key = str(path)
            if key not in seen:
                seen.add(key)
                matches.append(_candidate_path_item(path, papers_dir=papers_dir, match_type="same_registered_basename"))

    variants = _paper_id_variants(paper_id)
    for path in pdf_paths:
        lower_name = path.name.lower()
        if any(variant in lower_name for variant in variants):
            key = str(path)
            if key not in seen:
                seen.add(key)
                matches.append(_candidate_path_item(path, papers_dir=papers_dir, match_type="paper_id_filename_match"))
    return matches


def _local_presence_status(row: dict[str, Any], *, local_candidates: list[dict[str, Any]]) -> str:
    if local_candidates:
        first_type = _clean_text(local_candidates[0].get("matchType"))
        if first_type == "same_registered_basename":
            return "local_pdf_same_basename_found_elsewhere"
        return "local_pdf_paper_id_match_found"
    source_artifact = dict(row.get("sourceArtifact") or {})
    if (
        _clean_text(row.get("resolutionPlanStatus")) == "no_registered_source_artifact"
        or (not _clean_text(source_artifact.get("kind")) and not _clean_text(source_artifact.get("path")))
    ):
        return "no_registered_source_artifact_no_local_pdf_candidate_found"
    return "registered_pdf_path_missing_no_local_pdf_candidate_found"


def _reacquisition_status(paper_id: str, *, local_presence_status: str) -> str:
    if local_presence_status.startswith("local_pdf_"):
        return "local_path_repair_candidate_no_download_needed"
    token = _clean_text(paper_id)
    if _is_plausible_arxiv_id(token):
        return "arxiv_pdf_reacquisition_candidate"
    if token.startswith("http") or token.startswith("https___") or token.startswith("http___"):
        return "url_identifier_reacquisition_candidate"
    return "manual_lookup_required"


def _reacquisition_hint(paper_id: str, status: str) -> str:
    token = _clean_text(paper_id)
    if status == "arxiv_pdf_reacquisition_candidate":
        return f"https://arxiv.org/pdf/{token}.pdf"
    if status == "url_identifier_reacquisition_candidate":
        return "recover_original_source_url_from_registered_identifier_or_import_manifest"
    if status == "local_path_repair_candidate_no_download_needed":
        return "review local candidate path and approve a separate source registration/path repair tranche"
    return "manual bibliographic/source lookup required"


def _source_missing_row(
    blocker: dict[str, Any],
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    pdf_paths: list[Path],
) -> dict[str, Any]:
    paper_id = _clean_text(blocker.get("paperId"))
    row = sqlite_db.get_paper(paper_id) if hasattr(sqlite_db, "get_paper") else None
    source_artifact = dict(blocker.get("sourceArtifact") or {})
    local_candidates = _local_pdf_candidates(
        paper_id=paper_id,
        source_artifact=source_artifact,
        papers_dir=papers_dir,
        pdf_paths=pdf_paths,
    )
    local_status = _local_presence_status(blocker, local_candidates=local_candidates)
    reacquisition = _reacquisition_status(paper_id, local_presence_status=local_status)
    return {
        "paperId": paper_id,
        "paperTitle": _clean_text(blocker.get("paperTitle") or (row.get("title") if isinstance(row, dict) else "")),
        "degradationReasons": list(blocker.get("degradationReasons") or []),
        "sourceArtifact": source_artifact,
        "resolutionPlanStatus": _clean_text(blocker.get("resolutionPlanStatus") or ""),
        "localPdfCandidates": local_candidates,
        "localPdfCandidateCount": len(local_candidates),
        "localPresenceStatus": local_status,
        "reacquisitionStatus": reacquisition,
        "reacquisitionHint": _reacquisition_hint(paper_id, reacquisition),
        "downloadAttempted": False,
        "externalLookupAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "recommendedAction": (
            "separate_source_path_repair_review"
            if local_status.startswith("local_pdf_")
            else "separate_source_reacquisition_or_manual_lookup_review"
        ),
    }


def _oversized_policy_status(size_bytes: int, max_source_pdf_bytes: int) -> str:
    if size_bytes <= max_source_pdf_bytes:
        return "not_oversized_under_current_threshold"
    if size_bytes <= 50 * 1024 * 1024:
        return "oversized_small_policy_candidate"
    if size_bytes <= 100 * 1024 * 1024:
        return "oversized_medium_resource_policy_candidate"
    return "oversized_large_manual_resource_review_required"


def _oversized_policy_row(blocker: dict[str, Any], *, max_source_pdf_bytes: int) -> dict[str, Any]:
    source_artifact = dict(blocker.get("sourceArtifact") or {})
    size_bytes = int(source_artifact.get("sizeBytes") or 0)
    policy_status = _oversized_policy_status(size_bytes, max_source_pdf_bytes)
    return {
        "paperId": _clean_text(blocker.get("paperId")),
        "paperTitle": _clean_text(blocker.get("paperTitle")),
        "sourceArtifact": source_artifact,
        "sizeBytes": size_bytes,
        "maxSourcePdfBytes": max_source_pdf_bytes,
        "oversizedPolicyStatus": policy_status,
        "policyRecommendation": "separate_oversized_pdf_materialization_dry_run_with_explicit_resource_budget",
        "applyReadyNow": False,
        "parserRoutingChangeAllowed": False,
        "recommendedAction": "create_separate_oversized_pdf_policy_tranche",
    }


def _text_source_policy_row(blocker: dict[str, Any]) -> dict[str, Any]:
    source_artifact = dict(blocker.get("sourceArtifact") or {})
    return {
        "paperId": _clean_text(blocker.get("paperId")),
        "paperTitle": _clean_text(blocker.get("paperTitle")),
        "sourceArtifact": source_artifact,
        "textSourceExists": bool(source_artifact.get("exists") and source_artifact.get("isFile")),
        "decisionRecommendation": "do_not_count_as_current_pdf_backed_parsed_artifact",
        "futurePolicyOption": "allow_only_under_separate_text_source_parsed_artifact_contract",
        "applyReadyNow": False,
        "parserRoutingChangeAllowed": False,
        "recommendedAction": "create_separate_text_source_contract_decision_before_materialization",
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


def build_parsed_artifact_source_recovery_feasibility(
    *,
    sqlite_db: Any,
    papers_dir: str | Path,
    report_name: str = "parsed-artifact-source-recovery-feasibility",
    max_source_pdf_bytes: int = DEFAULT_MAX_SOURCE_PDF_BYTES,
    generated_at: str | None = None,
    baseline_command_report: dict[str, Any] | None = None,
    baseline_command_report_path: str | Path | None = None,
    sqlite_hash_before_cli: str = "",
    sqlite_hash_after_cli: str = "",
) -> dict[str, Any]:
    """Build a report-only source recovery feasibility payload."""

    effective_max_bytes = max(0, int(max_source_pdf_bytes or 0))
    source_blocker_report = build_parsed_artifact_source_blocker_report(
        sqlite_db=sqlite_db,
        papers_dir=papers_dir,
        report_name=f"{report_name}-input-source-blocker-snapshot",
        max_source_pdf_bytes=effective_max_bytes,
        generated_at=generated_at,
        baseline_command_report=baseline_command_report,
        baseline_command_report_path=baseline_command_report_path,
        sqlite_hash_before_cli=sqlite_hash_before_cli,
        sqlite_hash_after_cli=sqlite_hash_after_cli,
    )
    pdf_paths = _pdf_index(papers_dir)
    source_blockers = list(source_blocker_report.get("sourceBlockers") or [])
    source_missing_rows: list[dict[str, Any]] = []
    oversized_rows: list[dict[str, Any]] = []
    text_source_rows: list[dict[str, Any]] = []
    for blocker in source_blockers:
        item = dict(blocker)
        status = _clean_text(item.get("sourceStatus"))
        if status == "source_pdf_missing":
            source_missing_rows.append(
                _source_missing_row(item, sqlite_db=sqlite_db, papers_dir=papers_dir, pdf_paths=pdf_paths)
            )
        elif status == "source_pdf_oversized":
            oversized_rows.append(_oversized_policy_row(item, max_source_pdf_bytes=effective_max_bytes))
        elif status == "text_source_unsupported":
            text_source_rows.append(_text_source_policy_row(item))

    local_presence_counter = Counter(_clean_text(row.get("localPresenceStatus")) for row in source_missing_rows)
    reacquisition_counter = Counter(_clean_text(row.get("reacquisitionStatus")) for row in source_missing_rows)
    oversized_counter = Counter(_clean_text(row.get("oversizedPolicyStatus")) for row in oversized_rows)
    text_counter = Counter(_clean_text(row.get("decisionRecommendation")) for row in text_source_rows)
    local_path_repair_rows = [
        row for row in source_missing_rows if _clean_text(row.get("localPresenceStatus")).startswith("local_pdf_")
    ]
    identifier_reacquisition_rows = [
        row
        for row in source_missing_rows
        if _clean_text(row.get("reacquisitionStatus"))
        in {"arxiv_pdf_reacquisition_candidate", "url_identifier_reacquisition_candidate"}
    ]
    manual_lookup_rows = [
        row for row in source_missing_rows if _clean_text(row.get("reacquisitionStatus")) == "manual_lookup_required"
    ]
    baseline = dict(source_blocker_report.get("baseline") or {})

    payload = {
        "schema": PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
        "status": "planned_report_only",
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "maxSourcePdfBytes": effective_max_bytes,
            "selectionRule": "all source blockers after existing local-PDF coverage is exhausted",
            "localScanRoot": "papers_dir",
            "localScanExtensions": [".pdf"],
            "externalLookupAttempted": False,
            "sourceDownloadAttempted": False,
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
            "sourceArtifactTaxonomy": list(source_blocker_report.get("sourceBlockerPool", {}).get("sourceArtifactTaxonomy") or []),
            "blockerTaxonomy": list(source_blocker_report.get("sourceBlockerPool", {}).get("blockerTaxonomy") or []),
        },
        "localScanSummary": {
            "pdfFilesScanned": len(pdf_paths),
            "pathRepairCandidateRows": len(local_path_repair_rows),
            "noLocalPdfCandidateRows": len(source_missing_rows) - len(local_path_repair_rows),
            "localPresenceTaxonomy": _counter_items(local_presence_counter),
        },
        "sourceMissingRecoverySummary": {
            "sourcePdfMissingRows": len(source_missing_rows),
            "identifierReacquisitionCandidateRows": len(identifier_reacquisition_rows),
            "manualLookupRequiredRows": len(manual_lookup_rows),
            "reacquisitionTaxonomy": _counter_items(reacquisition_counter),
        },
        "oversizedPolicySummary": {
            "sourcePdfOversizedRows": len(oversized_rows),
            "oversizedPolicyTaxonomy": _counter_items(oversized_counter),
            "policyRecommendation": "do_not_mix_with_source_pdf_missing_recovery; run a separate oversized PDF dry-run with explicit max bytes, timeout, and resource budget",
        },
        "textSourcePolicySummary": {
            "textSourceUnsupportedRows": len(text_source_rows),
            "decisionTaxonomy": _counter_items(text_counter),
            "decisionRecommendation": "do_not_count_text_source_as_current_pdf_backed_parsed_artifact; require a separate text-source parsed artifact contract before any materialization",
        },
        "sourceMissingRows": source_missing_rows,
        "pathRepairCandidatePaperIds": [_clean_text(row.get("paperId")) for row in local_path_repair_rows],
        "reacquisitionCandidatePaperIdsByStatus": _ids_by_status(source_missing_rows, status_key="reacquisitionStatus"),
        "localPresencePaperIdsByStatus": _ids_by_status(source_missing_rows, status_key="localPresenceStatus"),
        "oversizedPolicyRows": oversized_rows,
        "oversizedPolicyPaperIdsByStatus": _ids_by_status(oversized_rows, status_key="oversizedPolicyStatus"),
        "textSourcePolicyRows": text_source_rows,
        "textSourcePolicyPaperIdsByDecision": _ids_by_status(text_source_rows, status_key="decisionRecommendation"),
        "expectedCoverageChangeIfApplied": {
            "missingParsedArtifactsBefore": int(baseline.get("missingParsedArtifacts") or 0),
            "expectedImmediateMissingParsedArtifactsReduction": 0,
            "expectedMissingParsedArtifactsAfterImmediateApply": int(baseline.get("missingParsedArtifacts") or 0),
            "potentialRowsUnlockedAfterPathRepairOnly": len(local_path_repair_rows),
            "potentialRowsRequiringSourceReacquisition": len(identifier_reacquisition_rows),
            "oversizedRowsRequiringSeparatePolicy": len(oversized_rows),
            "textSourceRowsRequiringSeparateContractDecision": len(text_source_rows),
        },
        "nextRecommendedTranche": {
            "name": "parsed-artifact-source-recovery-operator-review",
            "candidateCount": len(local_path_repair_rows) + len(identifier_reacquisition_rows),
            "paperIds": [_clean_text(row.get("paperId")) for row in local_path_repair_rows + identifier_reacquisition_rows],
            "rationale": "review local path repair candidates first, then identifier-based source reacquisition candidates, before any source registration or materialization apply",
        },
        "mutationPolicy": _source_blocker_policy(),
        "mutationCounters": _source_blocker_counters(),
        "warnings": list(source_blocker_report.get("warnings") or []),
    }
    validation = validate_payload(payload, PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact source recovery feasibility schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    return payload


def write_parsed_artifact_source_recovery_feasibility(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON, Markdown, and operator ID artifacts."""

    validation = validate_payload(report, PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID, strict=True)
    if not validation.ok:
        raise ValueError(
            "parsed artifact source recovery feasibility schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-source-recovery-feasibility.json"
    summary_path = root / "parsed-artifact-source-recovery-feasibility.md"
    path_repair_ids_path = root / "path-repair-candidate-paper-ids.txt"
    reacquisition_ids_path = root / "reacquisition-candidate-paper-ids-by-status.json"
    oversized_ids_path = root / "oversized-policy-paper-ids-by-status.json"
    text_ids_path = root / "text-source-policy-paper-ids-by-decision.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path_repair_ids_path.write_text("\n".join(report.get("pathRepairCandidatePaperIds") or []) + "\n", encoding="utf-8")
    reacquisition_ids_path.write_text(
        json.dumps(report.get("reacquisitionCandidatePaperIdsByStatus") or {}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    oversized_ids_path.write_text(
        json.dumps(report.get("oversizedPolicyPaperIdsByStatus") or {}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    text_ids_path.write_text(
        json.dumps(report.get("textSourcePolicyPaperIdsByDecision") or {}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "reportMarkdownPath": str(summary_path),
        "pathRepairCandidateIdsPath": str(path_repair_ids_path),
        "reacquisitionCandidateIdsByStatusPath": str(reacquisition_ids_path),
        "oversizedPolicyIdsByStatusPath": str(oversized_ids_path),
        "textSourcePolicyIdsByDecisionPath": str(text_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    blocker_snapshot = dict(report.get("sourceBlockerSnapshot") or {})
    local_scan = dict(report.get("localScanSummary") or {})
    recovery = dict(report.get("sourceMissingRecoverySummary") or {})
    oversized = dict(report.get("oversizedPolicySummary") or {})
    text_policy = dict(report.get("textSourcePolicySummary") or {})
    expected = dict(report.get("expectedCoverageChangeIfApplied") or {})
    lines = [
        "# Parsed Artifact Source Recovery Feasibility",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- source blocker rows: {blocker_snapshot.get('sourceBlockerRows', 0)}",
        f"- source_pdf_missing rows: {recovery.get('sourcePdfMissingRows', 0)}",
        f"- local PDF files scanned under papers_dir: {local_scan.get('pdfFilesScanned', 0)}",
        f"- path repair candidate rows: {local_scan.get('pathRepairCandidateRows', 0)}",
        f"- identifier reacquisition candidate rows: {recovery.get('identifierReacquisitionCandidateRows', 0)}",
        f"- manual lookup required rows: {recovery.get('manualLookupRequiredRows', 0)}",
        f"- oversized PDF policy rows: {oversized.get('sourcePdfOversizedRows', 0)}",
        f"- text source policy rows: {text_policy.get('textSourceUnsupportedRows', 0)}",
        f"- immediate expected missingParsedArtifacts reduction: {expected.get('expectedImmediateMissingParsedArtifactsReduction', 0)}",
        "",
        "## Reacquisition Taxonomy",
        "",
    ]
    for item in list(recovery.get("reacquisitionTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Local Presence Taxonomy", ""])
    for item in list(local_scan.get("localPresenceTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Oversized Policy Taxonomy", ""])
    for item in list(oversized.get("oversizedPolicyTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Text Source Decision", ""])
    lines.append(f"- `{text_policy.get('decisionRecommendation', '')}`")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID",
    "build_parsed_artifact_source_recovery_feasibility",
    "write_parsed_artifact_source_recovery_feasibility",
]
