"""Report-only manual lookup plan for remaining parsed-artifact source blockers.

This helper consumes the source recovery feasibility report after automated
source recovery and oversized-PDF materialization are exhausted. It classifies
manual lookup rows and text-source holdouts only. It does not perform external
lookup, download PDFs, rewrite source registrations, materialize parsed
artifacts, touch evidence/index/runtime stores, scan the vault, or change
answer behavior.
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
from knowledge_hub.papers.parsed_artifact_source_recovery_feasibility import (
    PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
)


PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-manual-lookup-source-recovery-plan.v1"
)

DEFAULT_FEASIBILITY_REPORT_PATH = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-source-recovery-feasibility-post-oversized-apply/"
    "parsed-artifact-source-recovery-feasibility.json"
).expanduser()
DEFAULT_OUTPUT_DIR = Path(
    "~/.khub/reports/parsed-artifact-coverage/2026-05-21/"
    "parsed-artifact-manual-lookup-source-recovery-plan/"
    "01-parsed-artifact-manual-lookup-source-recovery-plan"
).expanduser()
TARGET_MISSING_PARSED_ARTIFACTS = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))


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
    }


def _manual_lookup_mode(row: dict[str, Any]) -> str:
    source_artifact = dict(row.get("sourceArtifact") or {})
    kind = _clean_text(source_artifact.get("kind"))
    path = _clean_text(source_artifact.get("path"))
    if kind == "pdf" and path:
        return "registered_pdf_path_missing_manual_lookup"
    return "no_registered_source_artifact_manual_lookup"


def _lookup_queries(row: dict[str, Any]) -> dict[str, str]:
    paper_id = _clean_text(row.get("paperId"))
    title = _clean_text(row.get("paperTitle"))
    source_artifact = dict(row.get("sourceArtifact") or {})
    registered_path = _clean_text(source_artifact.get("path"))
    registered_name = Path(registered_path).name if registered_path else ""
    return {
        "exactTitle": title,
        "titlePdf": f'"{title}" pdf' if title else "",
        "paperId": paper_id,
        "registeredFileName": registered_name,
    }


def _manual_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    mode = _manual_lookup_mode(row)
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "manualLookupPlanStatus": "manual_lookup_source_recovery_plan_candidate_only",
        "manualLookupMode": mode,
        "lookupPriority": "high" if mode == "registered_pdf_path_missing_manual_lookup" else "normal",
        "lookupQueries": _lookup_queries(row),
        "preferredReviewOrder": [
            "existing_local_source_candidate",
            "official_publisher_or_project_pdf",
            "open_access_pdf_or_author_copy",
            "doi_crossref_openalex_metadata",
            "manual_registration_decision",
        ],
        "requiredFutureApproval": "explicit_source_url_or_local_pdf_path_allowlist_before_apply",
        "futureApplyTarget": "source_registration_then_parsed_artifact_materialization",
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "recommendedAction": "manual_lookup_review_pack",
    }


def _text_source_holdout_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": _clean_text(row.get("paperId")),
        "paperTitle": _clean_text(row.get("paperTitle")),
        "sourceArtifact": dict(row.get("sourceArtifact") or {}),
        "textSourcePlanStatus": "held_out_text_source_contract_required",
        "futurePolicyOption": _clean_text(row.get("futurePolicyOption")),
        "decisionRecommendation": _clean_text(row.get("decisionRecommendation")),
        "requiredFutureApproval": "separate_text_source_parsed_artifact_contract",
        "externalLookupAttempted": False,
        "sourceDownloadAttempted": False,
        "sourceRegistrationMutationAttempted": False,
        "parsedArtifactWriteAttempted": False,
        "recommendedAction": "parsed_artifact_text_source_materialization_contract",
    }


def build_parsed_artifact_manual_lookup_source_recovery_plan(
    *,
    feasibility_report: dict[str, Any],
    feasibility_report_path: str | Path | None = None,
    report_name: str = "parsed-artifact-manual-lookup-source-recovery-plan",
    target_missing_parsed_artifacts: int = TARGET_MISSING_PARSED_ARTIFACTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a report-only plan for manual source lookup candidates."""

    validation = validate_payload(
        feasibility_report,
        PARSED_ARTIFACT_SOURCE_RECOVERY_FEASIBILITY_SCHEMA_ID,
        strict=True,
    )
    input_validation = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    effective_target = max(0, int(target_missing_parsed_artifacts))

    if not validation.ok:
        manual_rows: list[dict[str, Any]] = []
        text_rows: list[dict[str, Any]] = []
        baseline: dict[str, Any] = {}
        warnings = ["blocked_input_schema_violation"]
        status = "blocked_input_schema_violation"
    else:
        source_missing_rows = [dict(row) for row in list(feasibility_report.get("sourceMissingRows") or [])]
        manual_rows = [
            _manual_candidate_row(row)
            for row in source_missing_rows
            if _clean_text(row.get("reacquisitionStatus")) == "manual_lookup_required"
        ]
        text_rows = [
            _text_source_holdout_row(dict(row))
            for row in list(feasibility_report.get("textSourcePolicyRows") or [])
        ]
        baseline = dict(feasibility_report.get("baseline") or {})
        warnings = list(feasibility_report.get("warnings") or [])
        status = (
            "manual_lookup_source_recovery_plan_candidate_only"
            if manual_rows
            else "no_manual_lookup_source_recovery_candidates"
        )

    missing_before = int(baseline.get("missingParsedArtifacts") or 0)
    manual_count = len(manual_rows)
    text_count = len(text_rows)
    needed_for_target = max(0, missing_before - effective_target)
    missing_after_all_manual = max(0, missing_before - manual_count)
    mode_counter = Counter(_clean_text(row.get("manualLookupMode")) for row in manual_rows)
    priority_counter = Counter(_clean_text(row.get("lookupPriority")) for row in manual_rows)
    text_counter = Counter(_clean_text(row.get("textSourcePlanStatus")) for row in text_rows)

    payload = {
        "schema": PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or _utc_now(),
        "report": {
            "name": report_name,
            "inputFeasibilityReportPath": str(Path(str(feasibility_report_path)).expanduser())
            if feasibility_report_path
            else "",
            "selectionRule": (
                "source recovery feasibility sourceMissingRows where "
                "reacquisitionStatus == manual_lookup_required; textSourcePolicyRows held out"
            ),
            "targetMissingParsedArtifacts": effective_target,
            "externalLookupAttempted": False,
            "sourceDownloadAttempted": False,
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
                "text_source_contract_decision",
            ],
        },
        "inputFeasibility": {
            "schemaValidation": input_validation,
            "status": _clean_text(feasibility_report.get("status")),
            "baselineMissingParsedArtifacts": missing_before,
            "sourcePdfMissingRows": int(
                dict(feasibility_report.get("sourceMissingRecoverySummary") or {}).get("sourcePdfMissingRows") or 0
            )
            if validation.ok
            else 0,
            "manualLookupRequiredRows": int(
                dict(feasibility_report.get("sourceMissingRecoverySummary") or {}).get("manualLookupRequiredRows") or 0
            )
            if validation.ok
            else 0,
            "textSourceUnsupportedRows": int(
                dict(feasibility_report.get("textSourcePolicySummary") or {}).get("textSourceUnsupportedRows") or 0
            )
            if validation.ok
            else 0,
        },
        "baseline": baseline,
        "candidatePool": {
            "inputRows": manual_count + text_count,
            "manualLookupCandidateOnlyRows": manual_count,
            "textSourceContractHoldRows": text_count,
            "automatedRecoveryCandidateRows": 0,
            "manualLookupModeTaxonomy": _counter_items(mode_counter),
            "lookupPriorityTaxonomy": _counter_items(priority_counter),
            "textSourceHoldoutTaxonomy": _counter_items(text_counter),
        },
        "manualLookupCandidatePaperIds": [_clean_text(row.get("paperId")) for row in manual_rows],
        "manualLookupCandidates": manual_rows,
        "textSourceHoldoutPaperIds": [_clean_text(row.get("paperId")) for row in text_rows],
        "textSourceHoldouts": text_rows,
        "coverageTarget": {
            "missingParsedArtifactsBefore": missing_before,
            "targetMissingParsedArtifacts": effective_target,
            "manualRecoveriesNeededForTarget": needed_for_target,
            "manualLookupCandidateRows": manual_count,
            "targetReachableIfEnoughManualSourcesResolved": bool(manual_count >= needed_for_target),
            "potentialMissingParsedArtifactsAfterAllManualLookupRecovered": missing_after_all_manual,
            "textSourceRowsRemainAfterManualLookup": text_count,
        },
        "expectedCoverageChangeIfApplied": {
            "expectedImmediateMissingParsedArtifactsReduction": 0,
            "expectedMissingParsedArtifactsAfterImmediateApply": missing_before,
            "potentialManualLookupMissingParsedArtifactsReduction": manual_count,
            "potentialMissingParsedArtifactsAfterManualLookupApply": missing_after_all_manual,
        },
        "nextRecommendedTranche": {
            "name": "parsed_artifact_manual_lookup_source_recovery_review_pack"
            if manual_count
            else "parsed_artifact_text_source_materialization_contract",
            "candidateCount": manual_count if manual_count else text_count,
            "paperIds": [_clean_text(row.get("paperId")) for row in (manual_rows if manual_count else text_rows)],
            "rationale": (
                "review and supply approved source URLs or local PDF paths for manual lookup candidates before any apply"
                if manual_count
                else "define text-source parsed artifact materialization semantics before using text sources"
            ),
        },
        "mutationPolicy": _mutation_policy(),
        "mutationCounters": _mutation_counters(),
        "warnings": warnings,
    }
    payload_validation = validate_payload(
        payload,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
        strict=True,
    )
    if not payload_validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery plan schema validation failed: "
            + "; ".join(payload_validation.errors[:5])
        )
    return payload


def write_parsed_artifact_manual_lookup_source_recovery_plan(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, summary JSON, Markdown, and candidate ID artifacts."""

    validation = validate_payload(
        report,
        PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID,
        strict=True,
    )
    if not validation.ok:
        raise ValueError(
            "parsed artifact manual lookup source recovery plan schema validation failed: "
            + "; ".join(validation.errors[:5])
        )
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "parsed-artifact-manual-lookup-source-recovery-plan.json"
    summary_json_path = root / "parsed-artifact-manual-lookup-source-recovery-plan-summary.json"
    markdown_path = root / "parsed-artifact-manual-lookup-source-recovery-plan.md"
    manual_ids_path = root / "manual-lookup-source-recovery-paper-ids.txt"
    text_ids_path = root / "text-source-contract-holdout-paper-ids.txt"
    summary = {
        "reportSchema": report.get("schema"),
        "status": report.get("status"),
        "baseline": report.get("baseline"),
        "candidatePool": report.get("candidatePool"),
        "coverageTarget": report.get("coverageTarget"),
        "expectedCoverageChangeIfApplied": report.get("expectedCoverageChangeIfApplied"),
        "nextRecommendedTranche": report.get("nextRecommendedTranche"),
        "mutationCounters": report.get("mutationCounters"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manual_ids_path.write_text("\n".join(report.get("manualLookupCandidatePaperIds") or []) + "\n", encoding="utf-8")
    text_ids_path.write_text("\n".join(report.get("textSourceHoldoutPaperIds") or []) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown_summary(report), encoding="utf-8")
    return {
        "reportJsonPath": str(report_path),
        "summaryJsonPath": str(summary_json_path),
        "reportMarkdownPath": str(markdown_path),
        "manualLookupCandidateIdsPath": str(manual_ids_path),
        "textSourceHoldoutIdsPath": str(text_ids_path),
    }


def _render_markdown_summary(report: dict[str, Any]) -> str:
    baseline = dict(report.get("baseline") or {})
    pool = dict(report.get("candidatePool") or {})
    target = dict(report.get("coverageTarget") or {})
    expected = dict(report.get("expectedCoverageChangeIfApplied") or {})
    lines = [
        "# Parsed Artifact Manual Lookup Source Recovery Plan",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- generatedAt: `{report.get('generatedAt')}`",
        f"- baseline scannedPapers: {baseline.get('scannedPapers', 0)}",
        f"- baseline missingParsedArtifacts: {baseline.get('missingParsedArtifacts', 0)}",
        f"- manual lookup candidate rows: {pool.get('manualLookupCandidateOnlyRows', 0)}",
        f"- text-source holdout rows: {pool.get('textSourceContractHoldRows', 0)}",
        f"- manual recoveries needed for target: {target.get('manualRecoveriesNeededForTarget', 0)}",
        f"- target missingParsedArtifacts: {target.get('targetMissingParsedArtifacts', 0)}",
        f"- target reachable if enough manual sources resolved: {target.get('targetReachableIfEnoughManualSourcesResolved')}",
        f"- immediate expected missingParsedArtifacts reduction: {expected.get('expectedImmediateMissingParsedArtifactsReduction', 0)}",
        f"- potential missingParsedArtifacts after all manual lookup recovered: {expected.get('potentialMissingParsedArtifactsAfterManualLookupApply', 0)}",
        "",
        "## Manual Lookup Mode Taxonomy",
        "",
    ]
    for item in list(pool.get("manualLookupModeTaxonomy") or []):
        lines.append(f"- `{item.get('reason')}`: {item.get('count')}")
    lines.extend(["", "## Manual Lookup Candidates", ""])
    for row in list(report.get("manualLookupCandidates") or []):
        lines.append(
            f"- `{row.get('paperId')}`: {row.get('paperTitle')} "
            f"(`{row.get('manualLookupMode')}`, priority `{row.get('lookupPriority')}`)"
        )
    lines.extend(["", "## Text Source Holdouts", ""])
    for row in list(report.get("textSourceHoldouts") or []):
        lines.append(f"- `{row.get('paperId')}`: {row.get('paperTitle')}")
    lines.extend(["", "## Mutation Counters", ""])
    for key, value in sorted(dict(report.get("mutationCounters") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility-report", default=str(DEFAULT_FEASIBILITY_REPORT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-name", default="parsed-artifact-manual-lookup-source-recovery-plan")
    parser.add_argument("--target-missing-parsed-artifacts", type=int, default=TARGET_MISSING_PARSED_ARTIFACTS)
    args = parser.parse_args(argv)

    feasibility_report_path = Path(args.feasibility_report).expanduser()
    feasibility_report = _load_json(feasibility_report_path)
    report = build_parsed_artifact_manual_lookup_source_recovery_plan(
        feasibility_report=feasibility_report,
        feasibility_report_path=feasibility_report_path,
        report_name=args.report_name,
        target_missing_parsed_artifacts=args.target_missing_parsed_artifacts,
    )
    paths = write_parsed_artifact_manual_lookup_source_recovery_plan(report, args.output_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "paths": paths,
                "baseline": report["baseline"],
                "candidatePool": report["candidatePool"],
                "coverageTarget": report["coverageTarget"],
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
    "PARSED_ARTIFACT_MANUAL_LOOKUP_SOURCE_RECOVERY_PLAN_SCHEMA_ID",
    "build_parsed_artifact_manual_lookup_source_recovery_plan",
    "write_parsed_artifact_manual_lookup_source_recovery_plan",
]
