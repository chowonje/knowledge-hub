"""Report-only design for PyMuPDF repair sidecar oracle candidate packs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import ZERO_COUNTER_KEYS
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID,
    SIDECAR_ORACLE_PACK_SCHEMA_ID,
)


PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-design.v1"
)

_READY_PARENT_GATE = "blocked_until_sidecar_oracle_candidate_pack_available"

_PRIVATE_PATH_PATTERNS = (
    "/" + "Users/" + r"[^\\s\"']+",
    "/" + "private/var/" + r"[^\\s\"']+",
    "/" + "Volumes/" + r"[^\\s\"']+",
    "Mobile" + " Documents",
    "i" + "Cloud",
)
_PRIVATE_PATH_RE = re.compile("|".join(_PRIVATE_PATH_PATTERNS))

_REQUIREMENT_PROFILES = {
    "section_candidate_oracle": {
        "candidateParsers": ["grobid_local", "pymupdf4llm_local"],
        "requiredSignals": [
            "section_title_or_boundary_candidate",
            "page_locator",
            "source_content_hash",
            "deterministic_identity_key",
        ],
        "agreementCriteria": [
            "candidate section boundary is page-local",
            "candidate text maps to source content hash",
            "no conflicting section order candidate is present",
        ],
    },
    "reading_order_oracle": {
        "candidateParsers": ["pymupdf4llm_local", "layout_block_clustering_local"],
        "requiredSignals": [
            "page_block_order_candidate",
            "column_count_candidate",
            "page_locator",
            "deterministic_identity_key",
        ],
        "agreementCriteria": [
            "reading order preserves page-local block order",
            "multi-column ordering is deterministic",
            "no cross-page block merge is introduced",
        ],
    },
    "table_body_cell_oracle": {
        "candidateParsers": ["camelot_local", "pdfplumber_local", "marker_or_docling_local_if_installed"],
        "requiredSignals": [
            "table_bbox_candidate",
            "row_column_grid_candidate",
            "cell_text_candidate",
            "cell_locator",
            "source_content_hash",
        ],
        "agreementCriteria": [
            "table body is linked to caption vicinity",
            "numeric cells keep row or column locator",
            "cell text can be traced to page-local bbox or locator",
        ],
    },
    "figure_caption_conflict_detector": {
        "candidateParsers": ["grobid_local", "pymupdf_image_or_vector_region_local"],
        "requiredSignals": [
            "caption_identity_candidate",
            "caption_bbox_candidate",
            "figure_region_candidate",
            "caption_figure_link_candidate",
        ],
        "agreementCriteria": [
            "caption identity is deterministic",
            "caption page and figure region do not conflict",
            "subfigure identity is not collapsed into parent figure identity",
        ],
    },
    "equation_locator_oracle": {
        "candidateParsers": ["grobid_local", "tex_alignment_local", "pymupdf_formula_region_local"],
        "requiredSignals": [
            "equation_region_candidate",
            "formula_bbox_or_tex_pdf_alignment",
            "equation_number_or_callout_context",
            "source_content_hash",
        ],
        "agreementCriteria": [
            "formula region is page-local",
            "equation identity does not depend on generated markdown text alone",
            "callout or equation number context is preserved when available",
        ],
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


def _input_schema_violations(comparison_report: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if comparison_report.get("schema") != PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID:
        violations.append("sidecar_oracle_comparison_schema_mismatch")
    if comparison_report.get("status") not in {"comparison_complete", "ready"}:
        violations.append("sidecar_oracle_comparison_not_complete")
    request = dict(comparison_report.get("request") or {})
    if not bool(request.get("reportOnly")):
        violations.append("sidecar_oracle_comparison_not_report_only")
    if not bool(request.get("comparisonOnly")):
        violations.append("sidecar_oracle_comparison_not_comparison_only")
    gate = dict(comparison_report.get("gate") or {})
    if _clean_text(gate.get("decision")) != _READY_PARENT_GATE:
        violations.append("sidecar_oracle_comparison_gate_not_waiting_for_candidate_pack")
    if _clean_text(gate.get("nextTranche")) != "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design":
        violations.append("unexpected_next_tranche")
    return violations


def _unsafe_upstream_flags(comparison_report: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(comparison_report.get("counts") or {})
    for key in ZERO_COUNTER_KEYS:
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    safety = dict(comparison_report.get("safety") or {})
    for key, value in safety.items():
        if bool(value):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _oracle_row_design(signal_row: dict[str, Any], index: int) -> dict[str, Any]:
    requirement = _clean_text(signal_row.get("requirement"))
    profile = _REQUIREMENT_PROFILES.get(requirement, {})
    candidate_parsers = _as_list(signal_row.get("expectedCandidateParsers")) or _as_list(profile.get("candidateParsers"))
    return {
        "designId": f"pymupdf-quality-repair-sidecar-oracle-pack-row-design:{index:04d}",
        "planId": _clean_text(signal_row.get("planId")),
        "artifactId": _clean_text(signal_row.get("artifactId")),
        "paperId": _clean_text(signal_row.get("paperId")),
        "requirement": requirement,
        "component": _clean_text(signal_row.get("component")),
        "candidateParsers": candidate_parsers,
        "requiredSignals": _as_list(profile.get("requiredSignals")),
        "agreementCriteria": _as_list(profile.get("agreementCriteria")),
        "allowedOracleStatuses": ["oracle_agrees", "oracle_missing", "oracle_conflict", "unsafe"],
        "requiredPackFields": [
            "planId",
            "artifactId",
            "paperId",
            "requirement",
            "component",
            "oracleStatus",
            "candidateParsers",
            "sourceContentHashVerified",
            "pageLocatorVerified",
            "bboxOrLocatorVerified",
            "deterministicIdentityVerified",
            "reportOnly",
            "mutationCounters",
        ],
        "mustRemainCandidateOnly": True,
        "sourceContentHashRequiredForAgreement": True,
        "pageLocatorRequiredForAgreement": True,
        "bboxOrLocatorRequiredForAgreement": requirement
        in {"table_body_cell_oracle", "figure_caption_conflict_detector", "equation_locator_oracle"},
        "deterministicIdentityRequiredForAgreement": True,
        "futureExecutorAction": "populate_local_oracle_candidate_row",
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def _pack_schema_summary() -> dict[str, Any]:
    return {
        "schemaId": SIDECAR_ORACLE_PACK_SCHEMA_ID,
        "statusValues": ["oracle_pack_ready", "ready", "complete", "blocked"],
        "rowStatusValues": ["oracle_agrees", "oracle_missing", "oracle_conflict", "unsafe"],
        "requiredTopLevelFields": [
            "schema",
            "status",
            "generatedAt",
            "scope",
            "request",
            "safety",
            "counts",
            "oracleRows",
            "warnings",
        ],
        "requiredRowFields": [
            "planId",
            "artifactId",
            "paperId",
            "requirement",
            "component",
            "oracleStatus",
            "candidateParsers",
            "sourceContentHashVerified",
            "pageLocatorVerified",
            "bboxOrLocatorVerified",
            "deterministicIdentityVerified",
            "reportOnly",
            "mutationCounters",
        ],
        "nonEvidenceGuarantees": [
            "sidecar_parser_output_is_candidate_only",
            "oracle_pack_rows_are_not_source_spans",
            "oracle_pack_rows_are_not_strict_or_citation_evidence",
            "oracle_pack_rows_do_not_change_parser_routing",
            "oracle_pack_rows_do_not_write_canonical_parsed_artifacts",
        ],
    }


def build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(
    *,
    comparison_report: dict[str, Any],
) -> dict[str, Any]:
    """Build a design-only report for the sidecar oracle pack contract."""

    schema_violations = _input_schema_violations(comparison_report)
    unsafe_flags = _unsafe_upstream_flags(comparison_report)
    signal_rows = [
        dict(row)
        for row in list(comparison_report.get("oracleSignalRows") or [])
        if isinstance(row, dict) and _clean_text(row.get("oracleStatus")) == "oracle_missing"
    ]
    design_rows = [_oracle_row_design(row, index) for index, row in enumerate(signal_rows, start=1)]
    private_leak_rows = _private_path_leak_rows(design_rows)

    by_requirement: Counter[str] = Counter()
    by_component: Counter[str] = Counter()
    by_parser: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    for row in design_rows:
        by_requirement.update([row["requirement"]])
        by_component.update([row["component"]])
        by_parser.update(row["candidateParsers"])
        paper = by_paper_id.setdefault(
            row["paperId"],
            {
                "artifactId": row["artifactId"],
                "candidatePackDesignRows": 0,
                "requirements": [],
            },
        )
        paper["candidatePackDesignRows"] += 1
        paper["requirements"] = sorted(set([*paper["requirements"], row["requirement"]]))

    schema_violation_count = len(schema_violations)
    counts = {
        "inputPlanRows": _safe_int(dict(comparison_report.get("counts") or {}).get("inputPlanRows")),
        "oracleRequirementRows": _safe_int(dict(comparison_report.get("counts") or {}).get("oracleRequirementRows")),
        "oracleMissingSignalRows": len(signal_rows),
        "candidatePackDesignRows": len(design_rows),
        "sidecarOraclePackSchemaDefinedRows": len(design_rows),
        "sidecarParserProfileRows": len(by_requirement),
        "futureExecutorDryRunReadyRows": len(design_rows),
        "futureParserInvocationRows": 0,
        "safeApplyCandidateRows": 0,
        **_zero_counters(
            private_path_leak_rows=private_leak_rows,
            schema_violation_count=schema_violation_count,
        ),
    }

    blocked = bool(schema_violations or unsafe_flags or private_leak_rows or not design_rows)
    gate_decision = (
        "blocked_until_sidecar_oracle_candidate_pack_design_gap_resolved"
        if blocked
        else "ready_for_sidecar_oracle_candidate_pack_executor_dry_run"
    )
    status = "blocked" if blocked else "design_ready"
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
        "request": {
            "inputSchema": _clean_text(comparison_report.get("schema")),
            "inputStatus": _clean_text(comparison_report.get("status")),
            "inputGateDecision": _clean_text(dict(comparison_report.get("gate") or {}).get("decision")),
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison",
            "designedPackSchemaId": SIDECAR_ORACLE_PACK_SCHEMA_ID,
            "reportOnly": True,
            "designOnly": True,
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
            "nextTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_candidate_pack_executor_dry_run"
            if gate_decision == "ready_for_sidecar_oracle_candidate_pack_executor_dry_run"
            else "resolve_sidecar_oracle_candidate_pack_design_gap",
        },
        "oraclePackSchema": _pack_schema_summary(),
        "byPaperId": by_paper_id,
        "bySidecarOracleRequirement": dict(sorted(by_requirement.items())),
        "byRepairComponent": dict(sorted(by_component.items())),
        "byCandidateParser": dict(sorted(by_parser.items())),
        "candidatePackDesignRows": design_rows,
        "warnings": [],
    }


def build_blocked_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design(*, reason: str) -> dict[str, Any]:
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
        "request": {
            "inputSchema": "",
            "inputStatus": "",
            "inputGateDecision": "",
            "requiredParentTranche": "parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison",
            "designedPackSchemaId": SIDECAR_ORACLE_PACK_SCHEMA_ID,
            "reportOnly": True,
            "designOnly": True,
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
            "oracleMissingSignalRows": 0,
            "candidatePackDesignRows": 0,
            "sidecarOraclePackSchemaDefinedRows": 0,
            "sidecarParserProfileRows": 0,
            "futureExecutorDryRunReadyRows": 0,
            "futureParserInvocationRows": 0,
            "safeApplyCandidateRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "gate": {
            "decision": "blocked_until_sidecar_oracle_candidate_pack_design_gap_resolved",
            "schemaViolations": [reason],
            "unsafeUpstreamFlags": [],
            "nextTranche": "resolve_sidecar_oracle_candidate_pack_design_gap",
        },
        "oraclePackSchema": _pack_schema_summary(),
        "byPaperId": {},
        "bySidecarOracleRequirement": {},
        "byRepairComponent": {},
        "byCandidateParser": {},
        "candidatePackDesignRows": [],
        "warnings": [reason],
    }


def render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact PyMuPDF Quality Repair Sidecar Oracle Candidate Pack Design",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Input plan rows: `{counts.get('inputPlanRows', 0)}`",
        f"- Oracle requirement rows: `{counts.get('oracleRequirementRows', 0)}`",
        f"- Candidate pack design rows: `{counts.get('candidatePackDesignRows', 0)}`",
        f"- Sidecar parser profile rows: `{counts.get('sidecarParserProfileRows', 0)}`",
        f"- Future parser invocation rows: `{counts.get('futureParserInvocationRows', 0)}`",
        f"- Safe apply candidate rows: `{counts.get('safeApplyCandidateRows', 0)}`",
        f"- Next tranche: `{gate.get('nextTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only design. No sidecar parser invocation, parser rerun, parsed artifact overwrite, parser routing change, DB/index mutation, reindex, reembed, vault scan, external download, SourceSpan, StrictEvidence, citation/runtime evidence, answer path invocation, or answerability promotion.",
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
    lines.extend(["", "## Design Rows", ""])
    for row in list(report.get("candidatePackDesignRows") or []):
        parsers = ", ".join(row.get("candidateParsers") or []) or "-"
        lines.append(
            f"- `{row.get('paperId')}` requirement=`{row.get('requirement')}` "
            f"component=`{row.get('component')}` parsers=`{parsers}` "
            f"futureAction=`{row.get('futureExecutorAction')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_reports(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-design.json"
    markdown_path = root / "parsed-artifact-pymupdf-quality-repair-sidecar-oracle-candidate-pack-design.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_CANDIDATE_PACK_DESIGN_SCHEMA_ID",
    "PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID",
    "SIDECAR_ORACLE_PACK_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
    "build_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design",
    "render_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_markdown",
    "write_pymupdf_quality_repair_sidecar_oracle_candidate_pack_design_reports",
]
