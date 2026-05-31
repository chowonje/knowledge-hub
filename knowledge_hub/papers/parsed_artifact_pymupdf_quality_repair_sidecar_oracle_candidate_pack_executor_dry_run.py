"""Report-only executor dry-run for PyMuPDF repair sidecar oracle candidate packs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import ZERO_COUNTER_KEYS
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
    SIDECAR_ORACLE_PACK_SCHEMA_ID,
)


PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-executor-dry-run.v1"
)

_READY_PARENT_GATE = "ready_for_sidecar_oracle_candidate_pack_executor_dry_run"

_PRIVATE_PATH_PATTERNS = (
    "/" + "Users/" + r"[^\\s\"']+",
    "/" + "private/var/" + r"[^\\s\"']+",
    "/" + "Volumes/" + r"[^\\s\"']+",
    "Mobile" + " Documents",
    "i" + "Cloud",
)
_PRIVATE_PATH_RE = re.compile("|".join(_PRIVATE_PATH_PATTERNS))


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
    if design_report.get("schema") != PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID:
        violations.append("candidate_pack_design_schema_mismatch")
    if design_report.get("status") not in {"design_ready", "ready"}:
        violations.append("candidate_pack_design_not_ready")
    request = dict(design_report.get("request") or {})
    if not bool(request.get("reportOnly")):
        violations.append("candidate_pack_design_not_report_only")
    if not bool(request.get("designOnly")):
        violations.append("candidate_pack_design_not_design_only")
    gate = dict(design_report.get("gate") or {})
    if _clean_text(gate.get("decision")) != _READY_PARENT_GATE:
        violations.append("candidate_pack_design_gate_not_ready_for_executor_dry_run")
    if (
        _clean_text(gate.get("nextTranche"))
        != "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run"
    ):
        violations.append("unexpected_next_tranche")
    if _clean_text(request.get("designedPackSchemaId")) != SIDECAR_ORACLE_PACK_SCHEMA_ID:
        violations.append("candidate_pack_schema_mismatch")
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


def _dry_run_plan_row(design_row: dict[str, Any], index: int) -> dict[str, Any]:
    required_pack_fields = _as_list(design_row.get("requiredPackFields"))
    return {
        "dryRunPlanId": f"pymupdf-quality-repair-sidecar-oracle-candidate-pack-executor-dry-run:{index:04d}",
        "sourceDesignId": _clean_text(design_row.get("designId")),
        "planId": _clean_text(design_row.get("planId")),
        "artifactId": _clean_text(design_row.get("artifactId")),
        "paperId": _clean_text(design_row.get("paperId")),
        "requirement": _clean_text(design_row.get("requirement")),
        "component": _clean_text(design_row.get("component")),
        "candidateParsers": _as_list(design_row.get("candidateParsers")),
        "plannedProbeSignals": _as_list(design_row.get("requiredSignals")),
        "plannedAgreementChecks": _as_list(design_row.get("agreementCriteria")),
        "plannedOraclePackRowFields": required_pack_fields,
        "plannedOracleStatuses": _as_list(design_row.get("allowedOracleStatuses")),
        "defaultFailClosedOracleStatus": "oracle_missing",
        "plannedExecutionMode": "dry_run_only_no_sidecar_parser_invocation",
        "localParserDependencyProbeRequired": True,
        "sourceContentHashVerificationPlanned": bool(design_row.get("sourceContentHashRequiredForAgreement")),
        "pageLocatorVerificationPlanned": bool(design_row.get("pageLocatorRequiredForAgreement")),
        "bboxOrLocatorVerificationPlanned": bool(design_row.get("bboxOrLocatorRequiredForAgreement")),
        "deterministicIdentityVerificationPlanned": bool(
            design_row.get("deterministicIdentityRequiredForAgreement")
        ),
        "oraclePackRowWritePlanned": False,
        "sidecarParserInvoked": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "safeApplyCandidate": False,
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
    *,
    design_report: dict[str, Any],
) -> dict[str, Any]:
    """Build the dry-run plan that precedes local sidecar oracle pack generation."""

    schema_violations = _input_schema_violations(design_report)
    unsafe_flags = _unsafe_upstream_flags(design_report)
    design_rows = [
        dict(row)
        for row in list(design_report.get("candidatePackDesignRows") or [])
        if isinstance(row, dict)
    ]
    dry_run_rows = [_dry_run_plan_row(row, index) for index, row in enumerate(design_rows, start=1)]
    private_leak_rows = _private_path_leak_rows(dry_run_rows)

    by_requirement: Counter[str] = Counter()
    by_component: Counter[str] = Counter()
    by_parser: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    for row in dry_run_rows:
        by_requirement.update([row["requirement"]])
        by_component.update([row["component"]])
        by_parser.update(row["candidateParsers"])
        paper = by_paper_id.setdefault(
            row["paperId"],
            {
                "artifactId": row["artifactId"],
                "dryRunPlanRows": 0,
                "requirements": [],
            },
        )
        paper["dryRunPlanRows"] += 1
        paper["requirements"] = sorted(set([*paper["requirements"], row["requirement"]]))

    schema_violation_count = len(schema_violations)
    parser_reference_rows = sum(len(row["candidateParsers"]) for row in dry_run_rows)
    counts = {
        "inputDesignRows": len(design_rows),
        "dryRunPlanRows": len(dry_run_rows),
        "plannedOraclePackRows": len(dry_run_rows),
        "plannedLocalProbeRows": len(dry_run_rows),
        "plannedCandidateParserReferenceRows": parser_reference_rows,
        "localParserDependencyProbeRows": 0,
        "sidecarParserInvokedRows": 0,
        "oraclePackWriteRows": 0,
        "actualOracleRows": 0,
        "oracleAgreesRows": 0,
        "oracleMissingRows": 0,
        "oracleConflictRows": 0,
        "unsafeRows": 0,
        "safeApplyCandidateRows": 0,
        **_zero_counters(
            private_path_leak_rows=private_leak_rows,
            schema_violation_count=schema_violation_count,
        ),
    }

    blocked = bool(schema_violations or unsafe_flags or private_leak_rows or not dry_run_rows)
    gate_decision = (
        "blocked_until_sidecar_oracle_candidate_pack_executor_dry_run_gap_resolved"
        if blocked
        else "ready_for_sidecar_oracle_candidate_pack_local_dependency_probe"
    )
    status = "blocked" if blocked else "executor_dry_run_complete"
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run",
        "request": {
            "inputSchema": _clean_text(design_report.get("schema")),
            "inputStatus": _clean_text(design_report.get("status")),
            "inputGateDecision": _clean_text(dict(design_report.get("gate") or {}).get("decision")),
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
            "targetPackSchemaId": SIDECAR_ORACLE_PACK_SCHEMA_ID,
            "reportOnly": True,
            "dryRunOnly": True,
            "sidecarParserInvocationAllowed": False,
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
            "oraclePackWritten": False,
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
            "nextTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_local_dependency_probe"
            if gate_decision == "ready_for_sidecar_oracle_candidate_pack_local_dependency_probe"
            else "resolve_sidecar_oracle_candidate_pack_executor_dry_run_gap",
        },
        "byPaperId": by_paper_id,
        "bySidecarOracleRequirement": dict(sorted(by_requirement.items())),
        "byRepairComponent": dict(sorted(by_component.items())),
        "byCandidateParser": dict(sorted(by_parser.items())),
        "dryRunPlanRows": dry_run_rows,
        "warnings": [],
    }


def build_blocked_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run(
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run",
        "request": {
            "inputSchema": "",
            "inputStatus": "",
            "inputGateDecision": "",
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
            "targetPackSchemaId": SIDECAR_ORACLE_PACK_SCHEMA_ID,
            "reportOnly": True,
            "dryRunOnly": True,
            "sidecarParserInvocationAllowed": False,
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
            "oraclePackWritten": False,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "answerPathInvoked": False,
            "answerabilityPromoted": False,
            "privatePathLeakAllowed": False,
        },
        "counts": {
            "inputDesignRows": 0,
            "dryRunPlanRows": 0,
            "plannedOraclePackRows": 0,
            "plannedLocalProbeRows": 0,
            "plannedCandidateParserReferenceRows": 0,
            "localParserDependencyProbeRows": 0,
            "sidecarParserInvokedRows": 0,
            "oraclePackWriteRows": 0,
            "actualOracleRows": 0,
            "oracleAgreesRows": 0,
            "oracleMissingRows": 0,
            "oracleConflictRows": 0,
            "unsafeRows": 0,
            "safeApplyCandidateRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "gate": {
            "decision": "blocked_until_sidecar_oracle_candidate_pack_executor_dry_run_gap_resolved",
            "schemaViolations": [reason],
            "unsafeUpstreamFlags": [],
            "nextTranche": "resolve_sidecar_oracle_candidate_pack_executor_dry_run_gap",
        },
        "byPaperId": {},
        "bySidecarOracleRequirement": {},
        "byRepairComponent": {},
        "byCandidateParser": {},
        "dryRunPlanRows": [],
        "warnings": [reason],
    }


def render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact PyMuPDF Quality Repair Sidecar Oracle Candidate Pack Executor Dry Run",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Input design rows: `{counts.get('inputDesignRows', 0)}`",
        f"- Dry-run plan rows: `{counts.get('dryRunPlanRows', 0)}`",
        f"- Planned oracle pack rows: `{counts.get('plannedOraclePackRows', 0)}`",
        f"- Planned local probe rows: `{counts.get('plannedLocalProbeRows', 0)}`",
        f"- Candidate parser references: `{counts.get('plannedCandidateParserReferenceRows', 0)}`",
        f"- Sidecar parser invoked rows: `{counts.get('sidecarParserInvokedRows', 0)}`",
        f"- Oracle pack write rows: `{counts.get('oraclePackWriteRows', 0)}`",
        f"- Safe apply candidate rows: `{counts.get('safeApplyCandidateRows', 0)}`",
        f"- Next tranche: `{gate.get('nextTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only executor dry-run. No sidecar parser invocation, dependency probe, oracle-pack write, parser rerun, parsed artifact overwrite, parser routing change, DB/index mutation, reindex, reembed, vault scan, external download, SourceSpan, StrictEvidence, citation/runtime evidence, answer path invocation, or answerability promotion.",
        "",
        "## Sidecar Oracle Requirements",
        "",
    ]
    for requirement, count in dict(report.get("bySidecarOracleRequirement") or {}).items():
        lines.append(f"- `{requirement}`: `{count}`")
    if not report.get("bySidecarOracleRequirement"):
        lines.append("- none")
    lines.extend(["", "## Candidate Parsers", ""])
    for parser, count in dict(report.get("byCandidateParser") or {}).items():
        lines.append(f"- `{parser}`: `{count}`")
    if not report.get("byCandidateParser"):
        lines.append("- none")
    lines.extend(["", "## Dry-Run Plan Rows", ""])
    for row in list(report.get("dryRunPlanRows") or []):
        parsers = ", ".join(row.get("candidateParsers") or []) or "-"
        lines.append(
            f"- `{row.get('paperId')}` requirement=`{row.get('requirement')}` "
            f"component=`{row.get('component')}` parsers=`{parsers}` "
            f"mode=`{row.get('plannedExecutionMode')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-executor-dry-run.json"
    markdown_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-executor-dry-run.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID",
    "PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_EXECUTOR_DRY_RUN_SCHEMA_ID",
    "SIDECAR_ORACLE_PACK_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run",
    "build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run",
    "render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_markdown",
    "write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run_reports",
]
