"""Report-only sidecar oracle comparison for PyMuPDF quality repair plans."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import ZERO_COUNTER_KEYS
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_executor_dry_run import (
    PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID,
)


PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-sidecar-oracle-comparison.v1"
)
SIDECAR_ORACLE_PACK_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-sidecar-oracle-pack.v1"
)

_READY_PARENT_GATE = "ready_for_pymupdf_quality_repair_sidecar_oracle_comparison"

_PRIVATE_PATH_PATTERNS = (
    "/" + "Users/" + r"[^\\s\"']+",
    "/" + "private/var/" + r"[^\\s\"']+",
    "/" + "Volumes/" + r"[^\\s\"']+",
    "Mobile" + " Documents",
    "i" + "Cloud",
)
_PRIVATE_PATH_RE = re.compile("|".join(_PRIVATE_PATH_PATTERNS))

_AGREE_STATUSES = {"oracle_agrees", "agrees", "ready"}
_CONFLICT_STATUSES = {"oracle_conflict", "conflict"}
_MISSING_STATUSES = {"oracle_missing", "missing", "not_found", "not_provided"}
_UNSAFE_STATUSES = {"unsafe", "oracle_unsafe"}


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


def _input_schema_violations(executor_dry_run: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if executor_dry_run.get("schema") != PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID:
        violations.append("pymupdf_quality_repair_executor_dry_run_schema_mismatch")
    if executor_dry_run.get("status") not in {"executor_dry_run_complete", "ready"}:
        violations.append("pymupdf_quality_repair_executor_dry_run_not_ready")
    request = dict(executor_dry_run.get("request") or {})
    if not bool(request.get("reportOnly")):
        violations.append("pymupdf_quality_repair_executor_dry_run_not_report_only")
    if not bool(request.get("dryRunOnly")):
        violations.append("pymupdf_quality_repair_executor_dry_run_not_dry_run_only")
    gate = dict(executor_dry_run.get("gate") or {})
    if _clean_text(gate.get("decision")) != _READY_PARENT_GATE:
        violations.append("pymupdf_quality_repair_executor_dry_run_gate_not_ready_for_oracle_comparison")
    if _clean_text(gate.get("nextTranche")) != "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison":
        violations.append("unexpected_next_tranche")
    return violations


def _unsafe_upstream_flags(payload: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(payload.get("counts") or {})
    for key in ZERO_COUNTER_KEYS:
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    safety = dict(payload.get("safety") or {})
    for key, value in safety.items():
        if bool(value):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _sidecar_pack_state(sidecar_oracle_pack: dict[str, Any] | None) -> str:
    if sidecar_oracle_pack is None:
        return "not_provided"
    if sidecar_oracle_pack.get("schema") != SIDECAR_ORACLE_PACK_SCHEMA_ID:
        return "schema_mismatch"
    if sidecar_oracle_pack.get("status") not in {"ready", "oracle_pack_ready", "complete"}:
        return "not_ready"
    return "ready"


def _sidecar_pack_unsafe_flags(sidecar_oracle_pack: dict[str, Any] | None) -> list[str]:
    if sidecar_oracle_pack is None:
        return []
    unsafe: list[str] = []
    counts = dict(sidecar_oracle_pack.get("counts") or {})
    for key in ZERO_COUNTER_KEYS:
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"sidecar_pack_{key}_nonzero")
    safety = dict(sidecar_oracle_pack.get("safety") or {})
    for key, value in safety.items():
        if bool(value):
            unsafe.append(f"sidecar_pack_{key}_true")
    return list(dict.fromkeys(unsafe))


def _oracle_key(plan_id: str, requirement: str, component: str) -> str:
    return f"{plan_id}::{requirement}::{component}"


def _oracle_index(sidecar_oracle_pack: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not sidecar_oracle_pack:
        return index
    for row in list(sidecar_oracle_pack.get("oracleRows") or []):
        if not isinstance(row, dict):
            continue
        plan_id = _clean_text(row.get("planId"))
        requirement = _clean_text(row.get("requirement"))
        component = _clean_text(row.get("component"))
        if not plan_id or not requirement or not component:
            continue
        index[_oracle_key(plan_id, requirement, component)] = dict(row)
    return index


def _normalized_oracle_status(row: dict[str, Any] | None, *, pack_state: str) -> tuple[str, list[str]]:
    if pack_state != "ready":
        return "oracle_missing", [f"sidecar_oracle_pack_{pack_state}"]
    if row is None:
        return "oracle_missing", ["sidecar_oracle_row_missing"]
    raw_status = _clean_text(row.get("oracleStatus") or row.get("status"))
    if raw_status in _AGREE_STATUSES:
        return "oracle_agrees", _as_list(row.get("reasons")) or ["sidecar_oracle_agrees"]
    if raw_status in _CONFLICT_STATUSES:
        return "oracle_conflict", _as_list(row.get("conflictReasons")) or ["sidecar_oracle_conflict"]
    if raw_status in _UNSAFE_STATUSES:
        return "unsafe", _as_list(row.get("unsafeReasons")) or ["sidecar_oracle_unsafe"]
    if raw_status in _MISSING_STATUSES:
        return "oracle_missing", _as_list(row.get("missingReasons")) or ["sidecar_oracle_missing"]
    return "unsafe", [f"unknown_sidecar_oracle_status:{raw_status or 'empty'}"]


def _signal_row(
    *,
    plan_row: dict[str, Any],
    requirement: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    pack_state: str,
) -> dict[str, Any]:
    requirement_name = _clean_text(requirement.get("requirement"))
    component = _clean_text(requirement.get("component"))
    status, reasons = _normalized_oracle_status(oracle_row, pack_state=pack_state)
    candidate_parsers = _as_list(requirement.get("candidateParsers"))
    observed_parsers = _as_list((oracle_row or {}).get("candidateParsers")) or candidate_parsers
    return {
        "planId": _clean_text(plan_row.get("planId")),
        "artifactId": _clean_text(plan_row.get("artifactId")),
        "paperId": _clean_text(plan_row.get("paperId")),
        "requirement": requirement_name,
        "component": component,
        "expectedCandidateParsers": candidate_parsers,
        "observedCandidateParsers": observed_parsers,
        "oracleStatus": status,
        "oracleStatusReasons": reasons,
        "sourceContentHashVerified": bool((oracle_row or {}).get("sourceContentHashVerified")),
        "pageLocatorVerified": bool((oracle_row or {}).get("pageLocatorVerified")),
        "bboxOrLocatorVerified": bool((oracle_row or {}).get("bboxOrLocatorVerified")),
        "deterministicIdentityVerified": bool((oracle_row or {}).get("deterministicIdentityVerified")),
        "oracleInvokedByThisReport": False,
        "localOnly": True,
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def _plan_disposition(signal_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    statuses = {row["oracleStatus"] for row in signal_rows}
    reasons: list[str] = []
    for row in signal_rows:
        reasons.extend(row.get("oracleStatusReasons") or [])
    reasons = list(dict.fromkeys(reasons))
    if "unsafe" in statuses:
        return "unsafe", reasons
    if "oracle_conflict" in statuses:
        return "oracle_conflict", reasons
    if "oracle_missing" in statuses:
        return "oracle_missing", reasons
    if statuses == {"oracle_agrees"}:
        return "oracle_agrees", ["all_required_sidecar_oracles_agree"]
    return "unsafe", reasons or ["unknown_oracle_comparison_state"]


def _comparison_row(plan_row: dict[str, Any], signal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    disposition, reasons = _plan_disposition(signal_rows)
    safe_apply_candidate = disposition == "oracle_agrees"
    blocker = disposition if disposition.startswith("oracle_") else f"sidecar_oracle_{disposition}"
    blockers = [] if safe_apply_candidate else [blocker, *_as_list(plan_row.get("safeApplyBlockers"))]
    return {
        "planId": _clean_text(plan_row.get("planId")),
        "artifactId": _clean_text(plan_row.get("artifactId")),
        "sourceIds": _as_list(plan_row.get("sourceIds")),
        "paperId": _clean_text(plan_row.get("paperId")),
        "paperTitle": _clean_text(plan_row.get("paperTitle")),
        "primaryRepairStrategy": _clean_text(plan_row.get("primaryRepairStrategy")),
        "affectedEvidenceTypes": _as_list(plan_row.get("affectedEvidenceTypes")),
        "requiredOracleRows": len(signal_rows),
        "oracleAgreementRows": sum(1 for row in signal_rows if row.get("oracleStatus") == "oracle_agrees"),
        "oracleMissingRows": sum(1 for row in signal_rows if row.get("oracleStatus") == "oracle_missing"),
        "oracleConflictRows": sum(1 for row in signal_rows if row.get("oracleStatus") == "oracle_conflict"),
        "unsafeOracleRows": sum(1 for row in signal_rows if row.get("oracleStatus") == "unsafe"),
        "oracleComparisonDisposition": disposition,
        "oracleComparisonReasons": reasons,
        "safeApplyCandidate": safe_apply_candidate,
        "safeApplyBlockers": list(dict.fromkeys(blockers)),
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def build_pymupdf_quality_repair_sidecar_oracle_comparison(
    *,
    executor_dry_run: dict[str, Any],
    sidecar_oracle_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare executor plan rows with already-materialized sidecar oracle rows."""

    schema_violations = _input_schema_violations(executor_dry_run)
    unsafe_flags = _unsafe_upstream_flags(executor_dry_run)
    pack_state = _sidecar_pack_state(sidecar_oracle_pack)
    if pack_state in {"schema_mismatch", "not_ready"}:
        schema_violations.append(f"sidecar_oracle_pack_{pack_state}")
    unsafe_flags.extend(_sidecar_pack_unsafe_flags(sidecar_oracle_pack))

    plan_rows = [
        dict(row) for row in list(executor_dry_run.get("executorPlanRows") or []) if isinstance(row, dict)
    ]
    oracle_rows_by_key = _oracle_index(sidecar_oracle_pack)

    signal_rows: list[dict[str, Any]] = []
    signals_by_plan: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for plan_row in plan_rows:
        plan_id = _clean_text(plan_row.get("planId"))
        for requirement in list(plan_row.get("sidecarOracleRequirements") or []):
            if not isinstance(requirement, dict):
                continue
            requirement_name = _clean_text(requirement.get("requirement"))
            component = _clean_text(requirement.get("component"))
            oracle_row = oracle_rows_by_key.get(_oracle_key(plan_id, requirement_name, component))
            signal = _signal_row(
                plan_row=plan_row,
                requirement=requirement,
                oracle_row=oracle_row,
                pack_state=pack_state,
            )
            signal_rows.append(signal)
            signals_by_plan[plan_id].append(signal)

    comparison_rows = [_comparison_row(row, signals_by_plan[_clean_text(row.get("planId"))]) for row in plan_rows]
    private_leak_rows = _private_path_leak_rows([*comparison_rows, *signal_rows])

    by_disposition: Counter[str] = Counter()
    by_signal: Counter[str] = Counter()
    by_requirement: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    for row in comparison_rows:
        by_disposition.update([row["oracleComparisonDisposition"]])
        by_strategy.update([row["primaryRepairStrategy"]])
        by_paper_id[row["paperId"]] = {
            "artifactId": row["artifactId"],
            "oracleComparisonDisposition": row["oracleComparisonDisposition"],
            "safeApplyCandidate": row["safeApplyCandidate"],
            "requiredOracleRows": row["requiredOracleRows"],
        }
    for row in signal_rows:
        by_signal.update([row["oracleStatus"]])
        by_requirement.update([row["requirement"]])

    schema_violation_count = len(schema_violations)
    counts = {
        "inputPlanRows": len(plan_rows),
        "oracleRequirementRows": len(signal_rows),
        "oracleAgreesPlanRows": by_disposition.get("oracle_agrees", 0),
        "oracleMissingPlanRows": by_disposition.get("oracle_missing", 0),
        "oracleConflictPlanRows": by_disposition.get("oracle_conflict", 0),
        "unsafePlanRows": by_disposition.get("unsafe", 0),
        "oracleAgreementSignalRows": by_signal.get("oracle_agrees", 0),
        "oracleMissingSignalRows": by_signal.get("oracle_missing", 0),
        "oracleConflictSignalRows": by_signal.get("oracle_conflict", 0),
        "unsafeSignalRows": by_signal.get("unsafe", 0),
        "safeApplyCandidateRows": sum(1 for row in comparison_rows if bool(row.get("safeApplyCandidate"))),
        "sidecarOraclePackRows": len(list((sidecar_oracle_pack or {}).get("oracleRows") or [])),
        **_zero_counters(
            private_path_leak_rows=private_leak_rows,
            schema_violation_count=schema_violation_count,
        ),
    }

    hard_blocked = bool(schema_violations or unsafe_flags or private_leak_rows or counts["unsafePlanRows"])
    if hard_blocked:
        gate_decision = "blocked_until_sidecar_oracle_comparison_gaps_resolved"
        next_tranche = "resolve_pymupdf_quality_repair_sidecar_oracle_comparison_gaps"
        status = "blocked"
    elif counts["oracleConflictPlanRows"]:
        gate_decision = "blocked_until_sidecar_oracle_conflicts_resolved"
        next_tranche = "parsed_artifact_pymupdf_quality_repair_oracle_conflict_review"
        status = "comparison_complete"
    elif counts["oracleMissingPlanRows"]:
        gate_decision = "blocked_until_sidecar_oracle_candidate_pack_available"
        next_tranche = "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design"
        status = "comparison_complete"
    else:
        gate_decision = "ready_for_pymupdf_quality_repair_safe_apply_subset_review"
        next_tranche = "parsed_artifact_pymupdf_quality_repair_safe_apply_subset_review"
        status = "comparison_complete"

    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_comparison",
        "request": {
            "inputSchema": _clean_text(executor_dry_run.get("schema")),
            "inputStatus": _clean_text(executor_dry_run.get("status")),
            "inputGateDecision": _clean_text(dict(executor_dry_run.get("gate") or {}).get("decision")),
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_executor_dry_run",
            "sidecarOraclePackSchema": _clean_text((sidecar_oracle_pack or {}).get("schema")),
            "sidecarOraclePackStatus": _clean_text((sidecar_oracle_pack or {}).get("status")),
            "sidecarOraclePackState": pack_state,
            "reportOnly": True,
            "comparisonOnly": True,
        },
        "safety": {
            "sidecarParserInvoked": False,
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
            "nextTranche": next_tranche,
        },
        "byPaperId": by_paper_id,
        "byOracleComparisonDisposition": dict(sorted(by_disposition.items())),
        "byOracleSignalStatus": dict(sorted(by_signal.items())),
        "bySidecarOracleRequirement": dict(sorted(by_requirement.items())),
        "byPrimaryRepairStrategy": dict(sorted(by_strategy.items())),
        "comparisonRows": comparison_rows,
        "oracleSignalRows": signal_rows,
        "warnings": ["sidecar_oracle_candidate_pack_missing_or_incomplete"]
        if counts["oracleMissingPlanRows"]
        else [],
    }


def build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison(*, reason: str) -> dict[str, Any]:
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_comparison",
        "request": {
            "inputSchema": "",
            "inputStatus": "",
            "inputGateDecision": "",
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_executor_dry_run",
            "sidecarOraclePackSchema": "",
            "sidecarOraclePackStatus": "",
            "sidecarOraclePackState": "not_provided",
            "reportOnly": True,
            "comparisonOnly": True,
        },
        "safety": {
            "sidecarParserInvoked": False,
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
            "inputPlanRows": 0,
            "oracleRequirementRows": 0,
            "oracleAgreesPlanRows": 0,
            "oracleMissingPlanRows": 0,
            "oracleConflictPlanRows": 0,
            "unsafePlanRows": 0,
            "oracleAgreementSignalRows": 0,
            "oracleMissingSignalRows": 0,
            "oracleConflictSignalRows": 0,
            "unsafeSignalRows": 0,
            "safeApplyCandidateRows": 0,
            "sidecarOraclePackRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "gate": {
            "decision": "blocked_until_sidecar_oracle_comparison_gaps_resolved",
            "schemaViolations": [reason],
            "unsafeUpstreamFlags": [],
            "nextTranche": "resolve_pymupdf_quality_repair_sidecar_oracle_comparison_gaps",
        },
        "byPaperId": {},
        "byOracleComparisonDisposition": {},
        "byOracleSignalStatus": {},
        "bySidecarOracleRequirement": {},
        "byPrimaryRepairStrategy": {},
        "comparisonRows": [],
        "oracleSignalRows": [],
        "warnings": [reason],
    }


def render_pymupdf_quality_repair_sidecar_oracle_comparison_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    request = dict(report.get("request") or {})
    lines = [
        "# Parsed Artifact PyMuPDF Quality Repair Sidecar Oracle Comparison",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Sidecar oracle pack state: `{request.get('sidecarOraclePackState')}`",
        f"- Input plan rows: `{counts.get('inputPlanRows', 0)}`",
        f"- Oracle requirement rows: `{counts.get('oracleRequirementRows', 0)}`",
        f"- Oracle-agrees plan rows: `{counts.get('oracleAgreesPlanRows', 0)}`",
        f"- Oracle-missing plan rows: `{counts.get('oracleMissingPlanRows', 0)}`",
        f"- Oracle-conflict plan rows: `{counts.get('oracleConflictPlanRows', 0)}`",
        f"- Safe apply candidate rows: `{counts.get('safeApplyCandidateRows', 0)}`",
        f"- Next tranche: `{gate.get('nextTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only comparison. No sidecar parser invocation, parser rerun, parsed artifact overwrite, parser routing change, DB/index mutation, reindex, reembed, vault scan, external download, SourceSpan, StrictEvidence, citation/runtime evidence, answer path invocation, or answerability promotion.",
        "",
        "## Disposition",
        "",
    ]
    for disposition, count in dict(report.get("byOracleComparisonDisposition") or {}).items():
        lines.append(f"- `{disposition}`: `{count}`")
    if not report.get("byOracleComparisonDisposition"):
        lines.append("- none")

    lines.extend(["", "## Sidecar Oracle Requirements", ""])
    for requirement, count in dict(report.get("bySidecarOracleRequirement") or {}).items():
        lines.append(f"- `{requirement}`: `{count}`")
    if not report.get("bySidecarOracleRequirement"):
        lines.append("- none")

    lines.extend(["", "## Comparison Rows", ""])
    for row in list(report.get("comparisonRows") or []):
        affected = ", ".join(row.get("affectedEvidenceTypes") or []) or "-"
        lines.append(
            f"- `{row.get('paperId')}` disposition=`{row.get('oracleComparisonDisposition')}` "
            f"strategy=`{row.get('primaryRepairStrategy')}` affected=`{affected}` "
            f"safeApply=`{row.get('safeApplyCandidate')}` requiredOracleRows=`{row.get('requiredOracleRows')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_pymupdf_quality_repair_sidecar_oracle_comparison_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-comparison.json"
    markdown_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_pymupdf_quality_repair_sidecar_oracle_comparison_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID",
    "PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "SIDECAR_ORACLE_PACK_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_pymupdf_quality_repair_sidecar_oracle_comparison",
    "build_pymupdf_quality_repair_sidecar_oracle_comparison",
    "render_pymupdf_quality_repair_sidecar_oracle_comparison_markdown",
    "write_pymupdf_quality_repair_sidecar_oracle_comparison_reports",
]
