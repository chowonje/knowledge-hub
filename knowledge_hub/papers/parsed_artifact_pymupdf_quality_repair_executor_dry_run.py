"""Report-only PyMuPDF parser-quality repair executor dry-run."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import (
    PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
)


PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-executor-dry-run.v1"
)

_READY_PARENT_GATE = "ready_for_pymupdf_quality_repair_executor_dry_run"

_PRIVATE_PATH_PATTERNS = (
    "/" + "Users/" + r"[^\\s\"']+",
    "/" + "private/var/" + r"[^\\s\"']+",
    "/" + "Volumes/" + r"[^\\s\"']+",
    "Mobile" + " Documents",
    "i" + "Cloud",
)
_PRIVATE_PATH_RE = re.compile("|".join(_PRIVATE_PATH_PATTERNS))

_COMPONENT_ACTIONS = {
    "page_block_segmentation": {
        "action": "plan_page_block_segmentation_candidate_diff",
        "target": "section_and_paragraph_boundary_candidates",
        "oracleRequirement": "section_candidate_oracle",
        "candidateParsers": ["grobid_local", "pymupdf4llm_local"],
    },
    "multi_column_reading_order_probe": {
        "action": "plan_multi_column_reading_order_candidate_diff",
        "target": "page_reading_order_candidates",
        "oracleRequirement": "reading_order_oracle",
        "candidateParsers": ["pymupdf4llm_local", "layout_block_clustering_local"],
    },
    "table_numeric_region_locator": {
        "action": "plan_table_region_and_cell_locator_candidates",
        "target": "table_bbox_row_column_cell_candidates",
        "oracleRequirement": "table_body_cell_oracle",
        "candidateParsers": ["camelot_local", "pdfplumber_local", "marker_or_docling_local_if_installed"],
    },
    "figure_caption_region_locator": {
        "action": "plan_figure_caption_region_and_identity_candidates",
        "target": "figure_caption_page_bbox_identity_candidates",
        "oracleRequirement": "figure_caption_conflict_detector",
        "candidateParsers": ["grobid_local", "pymupdf_image_or_vector_region_local"],
    },
    "equation_region_locator": {
        "action": "plan_equation_region_locator_candidates",
        "target": "equation_page_bbox_or_tex_alignment_candidates",
        "oracleRequirement": "equation_locator_oracle",
        "candidateParsers": ["grobid_local", "tex_alignment_local", "pymupdf_formula_region_local"],
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _zero_counters(private_path_leak_rows: int = 0, schema_violation_count: int = 0) -> dict[str, int]:
    counters = {key: 0 for key in ZERO_COUNTER_KEYS}
    counters["privatePathLeakRows"] = max(0, int(private_path_leak_rows))
    counters["schemaViolationCount"] = max(0, int(schema_violation_count))
    return counters


def _private_path_leak_rows(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if _PRIVATE_PATH_RE.search(json.dumps(item, ensure_ascii=False)))


def _input_schema_violations(design_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if design_report.get("schema") != PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID:
        violations.append("pymupdf_quality_repair_design_schema_mismatch")
    if design_report.get("status") not in {"design_ready", "ready"}:
        violations.append("pymupdf_quality_repair_design_not_ready")
    request = dict(design_report.get("request") or {})
    if not bool(request.get("reportOnly")):
        violations.append("pymupdf_quality_repair_design_not_report_only")
    gate = dict(design_report.get("gate") or {})
    if _clean_text(gate.get("decision")) != _READY_PARENT_GATE:
        violations.append("pymupdf_quality_repair_design_gate_not_ready_for_executor")
    if _clean_text(gate.get("nextTranche")) != "parsed_artifact_pymupdf_quality_repair_executor_dry_run":
        violations.append("unexpected_next_tranche")
    return violations


def _unsafe_upstream_flags(design_report: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(design_report.get("counts") or {})
    for key in ZERO_COUNTER_KEYS:
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    safety = dict(design_report.get("safety") or {})
    for key, value in safety.items():
        if bool(value):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _authority_check_plan(row: dict[str, Any]) -> list[dict[str, Any]]:
    affected = set(_as_list(row.get("affectedEvidenceTypes")))
    checks: list[dict[str, Any]] = [
        {
            "check": "source_content_hash_authority",
            "required": True,
            "planned": True,
            "materializedByThisReport": False,
            "basis": "parent_design_sourceContentHash_requirement",
        },
        {
            "check": "page_locator_authority",
            "required": True,
            "planned": True,
            "materializedByThisReport": False,
            "basis": "parent_design_pageLocatorBasis_requirement",
        },
        {
            "check": "deterministic_identity_key",
            "required": True,
            "planned": True,
            "materializedByThisReport": False,
            "basis": "artifactId_sourceContentHash_parser_structureType_page_locator_normalizedTextHash",
        },
    ]
    if "table_numeric" in affected:
        checks.append(
            {
                "check": "table_cell_locator_authority",
                "required": True,
                "planned": True,
                "materializedByThisReport": False,
                "basis": "table_bbox_row_column_or_cell_locator_required_before_numeric_evidence",
            }
        )
    if "figure_caption" in affected:
        checks.append(
            {
                "check": "figure_caption_identity_and_region_authority",
                "required": True,
                "planned": True,
                "materializedByThisReport": False,
                "basis": "caption_text_identity_plus_page_bbox_or_region_link_required_before_evidence",
            }
        )
    if "equation_region" in affected:
        checks.append(
            {
                "check": "equation_region_authority",
                "required": True,
                "planned": True,
                "materializedByThisReport": False,
                "basis": "formula_region_bbox_or_tex_pdf_alignment_required_before_evidence",
            }
        )
    return checks


def _planned_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for component in _as_list(row.get("repairStrategyComponents")):
        action_spec = _COMPONENT_ACTIONS.get(
            component,
            {
                "action": f"plan_{component}_candidate_diff",
                "target": "pymupdf_quality_repair_candidate",
                "oracleRequirement": "local_operator_review",
                "candidateParsers": ["pymupdf_local"],
            },
        )
        actions.append(
            {
                "component": component,
                "action": action_spec["action"],
                "target": action_spec["target"],
                "dryRunOnly": True,
                "writesCanonicalParsedArtifact": False,
                "createsSourceSpan": False,
                "createsStrictEvidence": False,
                "createsCitationEvidence": False,
                "createsRuntimeEvidence": False,
                "invokesAnswerPath": False,
                "requiresSourceContentHash": True,
                "requiresStablePageLocator": True,
                "outputKind": "candidate_diff_only",
            }
        )
    return actions


def _sidecar_oracle_requirements(row: dict[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for component in _as_list(row.get("repairStrategyComponents")):
        action_spec = _COMPONENT_ACTIONS.get(component)
        if not action_spec:
            continue
        requirements.append(
            {
                "requirement": action_spec["oracleRequirement"],
                "component": component,
                "requiredBeforeApply": True,
                "localOnly": True,
                "invokedByThisReport": False,
                "candidateParsers": action_spec["candidateParsers"],
            }
        )
    return requirements


def _base_disposition(row: dict[str, Any]) -> tuple[str, list[str]]:
    missing: list[str] = []
    if not _clean_text(row.get("artifactId")):
        missing.append("missing_artifact_id")
    if not _clean_text(row.get("paperId")):
        missing.append("missing_paper_id")
    if not _as_list(row.get("sourceIds")):
        missing.append("missing_source_ids")
    if missing:
        return "unsafe", missing
    if _clean_text(row.get("parser")) != "pymupdf":
        return "blocked_conflict", ["parent_design_parser_is_not_pymupdf"]
    if _clean_text(row.get("dryRunExecutorReadiness")) != _READY_PARENT_GATE:
        return "blocked_needs_sidecar_oracle", ["parent_design_not_ready_for_executor_dry_run"]
    if not _planned_actions(row):
        return "blocked_needs_sidecar_oracle", ["no_planned_repair_actions"]
    return (
        "dry_run_ready",
        [
            "parent_design_ready",
            "identity_fields_present",
            "authority_checks_planned",
            "dry_run_creates_no_evidence_or_canonical_artifact",
        ],
    )


def _conflict_keys(rows: list[dict[str, Any]], key: str) -> set[str]:
    counts = Counter(_clean_text(row.get(key)) for row in rows)
    return {value for value, count in counts.items() if value and count > 1}


def _plan_row(
    index: int,
    row: dict[str, Any],
    *,
    duplicate_artifact_ids: set[str],
    duplicate_idempotency_keys: set[str],
) -> dict[str, Any]:
    artifact_id = _clean_text(row.get("artifactId"))
    paper_id = _clean_text(row.get("paperId"))
    strategy = _clean_text(row.get("primaryRepairStrategy"))
    idempotency_key = f"pymupdf-quality-repair-executor-dry-run:{artifact_id}:{paper_id}:{strategy}"
    disposition, reasons = _base_disposition(row)
    conflict_reasons: list[str] = []
    if artifact_id in duplicate_artifact_ids:
        conflict_reasons.append("duplicate_artifact_id")
    if idempotency_key in duplicate_idempotency_keys:
        conflict_reasons.append("duplicate_idempotency_key")
    if conflict_reasons:
        disposition = "blocked_conflict"
        reasons = [*reasons, *conflict_reasons]

    oracle_requirements = _sidecar_oracle_requirements(row)
    sidecar_required_before_apply = any(item.get("requiredBeforeApply") for item in oracle_requirements)
    safe_apply_candidate = disposition == "dry_run_ready" and not sidecar_required_before_apply
    safe_apply_blockers = [] if safe_apply_candidate else ["sidecar_oracle_comparison_required_before_apply"]
    safe_apply_blockers.extend(_as_list(row.get("unsafeOrInsufficientProvenanceBlockers")))

    return {
        "planId": f"pymupdf-quality-repair-executor-dry-run:{index:04d}",
        "parentDesignId": _clean_text(row.get("designId")),
        "artifactId": artifact_id,
        "sourceIds": _as_list(row.get("sourceIds")),
        "paperId": paper_id,
        "paperTitle": _clean_text(row.get("paperTitle")),
        "parser": _clean_text(row.get("parser")) or "pymupdf",
        "primaryRepairStrategy": strategy,
        "repairStrategyComponents": _as_list(row.get("repairStrategyComponents")),
        "affectedEvidenceTypes": _as_list(row.get("affectedEvidenceTypes")),
        "diagnosticMetrics": dict(row.get("diagnosticMetrics") or {}),
        "idempotencyKey": idempotency_key,
        "dryRunDisposition": disposition,
        "dryRunDispositionReasons": reasons,
        "authorityCheckPlan": _authority_check_plan(row),
        "plannedRepairActions": _planned_actions(row),
        "sidecarOracleRequirements": oracle_requirements,
        "sidecarOracleRequiredBeforeApply": sidecar_required_before_apply,
        "safeApplyCandidate": safe_apply_candidate,
        "safeApplyBlockers": list(dict.fromkeys(safe_apply_blockers)),
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def build_pymupdf_quality_repair_executor_dry_run(*, design_report: dict[str, Any]) -> dict[str, Any]:
    """Build a dry-run executor plan without repairing parsed artifacts."""

    schema_violations = _input_schema_violations(design_report)
    unsafe_flags = _unsafe_upstream_flags(design_report)
    design_rows = [dict(row) for row in list(design_report.get("designRows") or []) if isinstance(row, dict)]

    duplicate_artifact_ids = _conflict_keys(design_rows, "artifactId")
    temporary_rows = []
    for row in design_rows:
        artifact_id = _clean_text(row.get("artifactId"))
        paper_id = _clean_text(row.get("paperId"))
        strategy = _clean_text(row.get("primaryRepairStrategy"))
        temporary_rows.append(
            {
                **row,
                "_idempotencyKey": f"pymupdf-quality-repair-executor-dry-run:{artifact_id}:{paper_id}:{strategy}",
            }
        )
    duplicate_idempotency_keys = _conflict_keys(temporary_rows, "_idempotencyKey")

    plan_rows = [
        _plan_row(
            index,
            row,
            duplicate_artifact_ids=duplicate_artifact_ids,
            duplicate_idempotency_keys=duplicate_idempotency_keys,
        )
        for index, row in enumerate(design_rows, start=1)
    ]
    private_leak_rows = _private_path_leak_rows(plan_rows)

    by_disposition: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()
    by_evidence: Counter[str] = Counter()
    by_oracle: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    planned_action_rows = 0
    authority_check_rows = 0
    for row in plan_rows:
        by_disposition.update([row["dryRunDisposition"]])
        by_strategy.update([row["primaryRepairStrategy"]])
        by_evidence.update(row["affectedEvidenceTypes"])
        by_oracle.update(item["requirement"] for item in row["sidecarOracleRequirements"])
        planned_action_rows += len(row["plannedRepairActions"])
        authority_check_rows += len(row["authorityCheckPlan"])
        by_paper_id[row["paperId"]] = {
            "artifactId": row["artifactId"],
            "dryRunDisposition": row["dryRunDisposition"],
            "primaryRepairStrategy": row["primaryRepairStrategy"],
            "sidecarOracleRequiredBeforeApply": row["sidecarOracleRequiredBeforeApply"],
            "safeApplyCandidate": row["safeApplyCandidate"],
        }

    schema_violation_count = len(schema_violations)
    counts = {
        "inputCorpusRows": _safe_int(dict(design_report.get("counts") or {}).get("inputCorpusRows")),
        "coverageReadyRows": _safe_int(dict(design_report.get("counts") or {}).get("coverageReadyRows")),
        "repairDesignRows": len(design_rows),
        "dryRunPlanRows": len(plan_rows),
        "plannedRepairActionRows": planned_action_rows,
        "plannedAuthorityCheckRows": authority_check_rows,
        "dryRunReadyRows": by_disposition.get("dry_run_ready", 0),
        "blockedNeedsSidecarOracleRows": by_disposition.get("blocked_needs_sidecar_oracle", 0),
        "blockedConflictRows": by_disposition.get("blocked_conflict", 0),
        "unsafeRows": by_disposition.get("unsafe", 0),
        "sidecarOracleRequiredBeforeApplyRows": sum(
            1 for row in plan_rows if bool(row.get("sidecarOracleRequiredBeforeApply"))
        ),
        "safeApplyCandidateRows": sum(1 for row in plan_rows if bool(row.get("safeApplyCandidate"))),
        "tableNumericPlannedRows": by_evidence.get("table_numeric", 0),
        "figureCaptionPlannedRows": by_evidence.get("figure_caption", 0),
        "equationRegionPlannedRows": by_evidence.get("equation_region", 0),
        **_zero_counters(
            private_path_leak_rows=private_leak_rows,
            schema_violation_count=schema_violation_count,
        ),
    }

    blocked = bool(
        schema_violations
        or unsafe_flags
        or private_leak_rows
        or counts["blockedNeedsSidecarOracleRows"]
        or counts["blockedConflictRows"]
        or counts["unsafeRows"]
    )
    if blocked:
        gate_decision = "blocked_until_pymupdf_quality_repair_dry_run_gaps_resolved"
        next_tranche = "resolve_pymupdf_quality_repair_executor_dry_run_gaps"
        status = "blocked"
    elif counts["sidecarOracleRequiredBeforeApplyRows"]:
        gate_decision = "ready_for_pymupdf_quality_repair_sidecar_oracle_comparison"
        next_tranche = "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison"
        status = "executor_dry_run_complete"
    else:
        gate_decision = "ready_for_pymupdf_quality_repair_safe_apply_subset_review"
        next_tranche = "parsed_artifact_pymupdf_quality_repair_safe_apply_subset_review"
        status = "executor_dry_run_complete"

    return {
        "schema": PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_executor_dry_run",
        "request": {
            "inputSchema": _clean_text(design_report.get("schema")),
            "inputStatus": _clean_text(design_report.get("status")),
            "inputGateDecision": _clean_text(dict(design_report.get("gate") or {}).get("decision")),
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_design",
            "reportOnly": True,
            "dryRunOnly": True,
        },
        "safety": {
            "parserRerun": False,
            "canonicalParsedArtifactWrite": False,
            "parserRoutingChanged": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "vaultScan": False,
            "externalDownload": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerPathInvoked": False,
            "answerabilityPromoted": False,
            "privatePathLeakAllowed": False,
        },
        "counts": counts,
        "gate": {
            "decision": gate_decision,
            "schemaViolations": schema_violations,
            "unsafeUpstreamFlags": unsafe_flags,
            "duplicateArtifactIdRows": len(duplicate_artifact_ids),
            "duplicateIdempotencyKeyRows": len(duplicate_idempotency_keys),
            "nextTranche": next_tranche,
        },
        "byPaperId": by_paper_id,
        "byDryRunDisposition": dict(sorted(by_disposition.items())),
        "byPrimaryRepairStrategy": dict(sorted(by_strategy.items())),
        "byAffectedEvidenceType": dict(sorted(by_evidence.items())),
        "bySidecarOracleRequirement": dict(sorted(by_oracle.items())),
        "executorPlanRows": plan_rows,
        "warnings": [
            "sidecar_oracle_comparison_required_before_any_apply"
        ]
        if counts["sidecarOracleRequiredBeforeApplyRows"]
        else [],
    }


def build_blocked_pymupdf_quality_repair_executor_dry_run(*, reason: str) -> dict[str, Any]:
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_executor_dry_run",
        "request": {
            "inputSchema": "",
            "inputStatus": "",
            "inputGateDecision": "",
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_design",
            "reportOnly": True,
            "dryRunOnly": True,
        },
        "safety": {
            "parserRerun": False,
            "canonicalParsedArtifactWrite": False,
            "parserRoutingChanged": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "vaultScan": False,
            "externalDownload": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerPathInvoked": False,
            "answerabilityPromoted": False,
            "privatePathLeakAllowed": False,
        },
        "counts": {
            "inputCorpusRows": 0,
            "coverageReadyRows": 0,
            "repairDesignRows": 0,
            "dryRunPlanRows": 0,
            "plannedRepairActionRows": 0,
            "plannedAuthorityCheckRows": 0,
            "dryRunReadyRows": 0,
            "blockedNeedsSidecarOracleRows": 0,
            "blockedConflictRows": 0,
            "unsafeRows": 0,
            "sidecarOracleRequiredBeforeApplyRows": 0,
            "safeApplyCandidateRows": 0,
            "tableNumericPlannedRows": 0,
            "figureCaptionPlannedRows": 0,
            "equationRegionPlannedRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "gate": {
            "decision": "blocked_until_pymupdf_quality_repair_dry_run_gaps_resolved",
            "schemaViolations": [reason],
            "unsafeUpstreamFlags": [],
            "duplicateArtifactIdRows": 0,
            "duplicateIdempotencyKeyRows": 0,
            "nextTranche": "resolve_pymupdf_quality_repair_executor_dry_run_gaps",
        },
        "byPaperId": {},
        "byDryRunDisposition": {},
        "byPrimaryRepairStrategy": {},
        "byAffectedEvidenceType": {},
        "bySidecarOracleRequirement": {},
        "executorPlanRows": [],
        "warnings": [reason],
    }


def render_pymupdf_quality_repair_executor_dry_run_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact PyMuPDF Quality Repair Executor Dry Run",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Repair design rows: `{counts.get('repairDesignRows', 0)}`",
        f"- Dry-run plan rows: `{counts.get('dryRunPlanRows', 0)}`",
        f"- Dry-run ready rows: `{counts.get('dryRunReadyRows', 0)}`",
        f"- Sidecar oracle required before apply rows: `{counts.get('sidecarOracleRequiredBeforeApplyRows', 0)}`",
        f"- Safe apply candidate rows: `{counts.get('safeApplyCandidateRows', 0)}`",
        f"- Next tranche: `{gate.get('nextTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only dry-run. No parser rerun, parsed artifact overwrite, parser routing change, DB/index mutation, reindex, reembed, vault scan, external download, SourceSpan, StrictEvidence, citation/runtime evidence, answer path invocation, or answerability promotion.",
        "",
        "## Dry-run Disposition",
        "",
    ]
    for disposition, count in dict(report.get("byDryRunDisposition") or {}).items():
        lines.append(f"- `{disposition}`: `{count}`")
    if not report.get("byDryRunDisposition"):
        lines.append("- none")

    lines.extend(["", "## Sidecar Oracle Requirements", ""])
    for requirement, count in dict(report.get("bySidecarOracleRequirement") or {}).items():
        lines.append(f"- `{requirement}`: `{count}`")
    if not report.get("bySidecarOracleRequirement"):
        lines.append("- none")

    lines.extend(["", "## Executor Plan Rows", ""])
    for row in list(report.get("executorPlanRows") or []):
        affected = ", ".join(row.get("affectedEvidenceTypes") or []) or "-"
        actions = ", ".join(action.get("action", "") for action in row.get("plannedRepairActions") or []) or "-"
        lines.append(
            f"- `{row.get('paperId')}` disposition=`{row.get('dryRunDisposition')}` "
            f"strategy=`{row.get('primaryRepairStrategy')}` affected=`{affected}` "
            f"sidecarBeforeApply=`{row.get('sidecarOracleRequiredBeforeApply')}` "
            f"actions=`{actions}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_pymupdf_quality_repair_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-pymupdf-quality-repair-executor-dry-run.json"
    markdown_path = root / "parsed-artifact-pymupdf-quality-repair-executor-dry-run.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_pymupdf_quality_repair_executor_dry_run_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_pymupdf_quality_repair_executor_dry_run",
    "build_pymupdf_quality_repair_executor_dry_run",
    "render_pymupdf_quality_repair_executor_dry_run_markdown",
    "write_pymupdf_quality_repair_executor_dry_run_reports",
]
