"""Report-only PyMuPDF parser-quality repair design for parsed artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-pymupdf-quality-repair-design.v1"
)
PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-parser-quality-degradation-audit.v1"
)

ZERO_COUNTER_KEYS = (
    "parserRerunRows",
    "canonicalParsedArtifactWriteRows",
    "parserRoutingChangedRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "vaultScanRows",
    "externalDownloadRows",
    "sourceSpanCreatedRows",
    "runtimeEvidenceCreatedRows",
    "citationEvidenceCreatedRows",
    "strictEvidenceCreatedRows",
    "answerPathInvokedRows",
    "answerabilityPromotionRows",
    "schemaViolationCount",
    "privatePathLeakRows",
)

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


def _input_schema_violations(degradation_audit: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if degradation_audit.get("schema") != PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID:
        violations.append("parser_quality_degradation_audit_schema_mismatch")
    if degradation_audit.get("status") not in {"ready", "degradation_audit_complete"}:
        violations.append("parser_quality_degradation_audit_not_ready")
    request = dict(degradation_audit.get("request") or {})
    if not bool(request.get("reportOnly")):
        violations.append("parser_quality_degradation_audit_not_report_only")
    if _clean_text(degradation_audit.get("nextRecommendedTranche")) not in {
        "",
        "parsed_artifact_pymupdf_quality_repair_design",
    }:
        violations.append("unexpected_next_recommended_tranche")
    return violations


def _unsafe_upstream_flags(degradation_audit: dict[str, Any]) -> list[str]:
    unsafe: list[str] = []
    counts = dict(degradation_audit.get("counts") or {})
    for key in (
        "canonicalParsedArtifactWriteRows",
        "parserRoutingChangedRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "vaultScanRows",
        "externalDownloadRows",
        "runtimeEvidenceCreatedRows",
        "citationEvidenceCreatedRows",
        "strictEvidenceCreatedRows",
        "answerPathInvokedRows",
        "answerabilityPromotionRows",
        "schemaViolationCount",
        "privatePathLeakRows",
    ):
        if _safe_int(counts.get(key)) > 0:
            unsafe.append(f"{key}_nonzero")
    safety = dict(degradation_audit.get("safety") or {})
    for key, value in safety.items():
        if bool(value):
            unsafe.append(f"{key}_true")
    return list(dict.fromkeys(unsafe))


def _affected_evidence_types(impacts: list[str]) -> list[str]:
    affected: list[str] = []
    if "table_numeric_blocked" in impacts:
        affected.append("table_numeric")
    if "figure_caption_blocked" in impacts:
        affected.append("figure_caption")
    if "equation_region_blocked" in impacts:
        affected.append("equation_region")
    if not affected:
        affected.append("unknown")
    return affected


def _usable_evidence_types(impacts: list[str]) -> list[str]:
    return ["section_span"] if "section_span_usable" in impacts else []


def _primary_repair_strategy(reasons: list[str], impacts: list[str]) -> str:
    reason_set = set(reasons)
    impact_set = set(impacts)
    if {
        "table_numeric_blocked",
        "figure_caption_blocked",
        "equation_region_blocked",
    }.issubset(impact_set):
        if {"multi_column_probe_only", "tables_caption_only"}.issubset(reason_set):
            return "pymupdf_multi_column_table_figure_equation_repair_design"
        if "tables_caption_only" in reason_set:
            return "pymupdf_table_figure_equation_locator_repair_design"
        if "multi_column_probe_only" in reason_set:
            return "pymupdf_multi_column_figure_equation_locator_repair_design"
        return "pymupdf_page_blob_figure_equation_locator_repair_design"
    if "table_numeric_blocked" in impact_set:
        return "pymupdf_table_numeric_locator_repair_design"
    if "figure_caption_blocked" in impact_set:
        return "pymupdf_figure_caption_locator_repair_design"
    if "equation_region_blocked" in impact_set:
        return "pymupdf_equation_region_locator_repair_design"
    return "no_action_coverage_ready"


def _strategy_components(impacts: list[str], reasons: list[str]) -> list[str]:
    components: list[str] = []
    if "page_blob_sections_only" in reasons:
        components.append("page_block_segmentation")
    if "multi_column_probe_only" in reasons:
        components.append("multi_column_reading_order_probe")
    if "table_numeric_blocked" in impacts:
        components.append("table_numeric_region_locator")
    if "figure_caption_blocked" in impacts:
        components.append("figure_caption_region_locator")
    if "equation_region_blocked" in impacts:
        components.append("equation_region_locator")
    return components or ["no_action"]


def _required_authority(affected: list[str]) -> dict[str, Any]:
    return {
        "sourceContentHash": {
            "required": True,
            "basis": "existing_canonical_source_pdf_hash_from_coverage_audit_or_corpus_manifest",
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": True,
        },
        "pageLocatorBasis": {
            "required": True,
            "basis": "pymupdf_page_index_plus_page_count_and_text_layer_diagnostics",
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": True,
        },
        "tableLocatorBasis": {
            "required": "table_numeric" in affected,
            "basis": "page_bbox_table_region_plus_row_column_or_cell_locator_before_numeric_evidence",
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": "table_numeric" in affected,
        },
        "captionLocatorBasis": {
            "required": "figure_caption" in affected,
            "basis": "caption_text_identity_plus_page_bbox_or_block_locator_and_figure_region_link",
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": "figure_caption" in affected,
        },
        "equationLocatorBasis": {
            "required": "equation_region" in affected,
            "basis": "formula_region_bbox_or_tex_pdf_alignment_with_page_and_source_hash",
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": "equation_region" in affected,
        },
        "deterministicIdentityKeyBasis": {
            "required": True,
            "basis": "artifactId_sourceContentHash_parser_structureType_page_locator_normalizedTextHash",
            "components": [
                "artifactId",
                "sourceContentHash",
                "parser",
                "structureType",
                "page",
                "locator",
                "normalizedTextHash",
            ],
            "materializedInThisDesign": False,
            "mustBeVerifiedByDryRun": True,
        },
    }


def _expected_repaired_artifact_shape(affected: list[str], components: list[str]) -> dict[str, Any]:
    fields = [
        "schema",
        "artifactId",
        "sourceContentHash",
        "parser",
        "pageLocators",
        "deterministicIdentityKeys",
        "repairDiagnostics",
        "promotionBlockers",
    ]
    if "table_numeric" in affected:
        fields.extend(["tableRegions", "tableNumericCellLocatorRequirements"])
    if "figure_caption" in affected:
        fields.extend(["figureCaptionRegions", "captionFigureLinkRequirements"])
    if "equation_region" in affected:
        fields.extend(["equationRegions", "formulaRegionLocatorRequirements"])
    return {
        "artifactKind": "pymupdf_quality_repair_dry_run_sidecar",
        "writeTarget": "future_ignored_report_or_noncanonical_sidecar_only",
        "canonicalParsedArtifactOverwrite": False,
        "parserRoutingChange": False,
        "sourceSpanCreated": False,
        "strictEvidenceCreated": False,
        "citationEvidenceCreated": False,
        "runtimeEvidenceCreated": False,
        "contains": sorted(dict.fromkeys(fields)),
        "repairComponents": components,
        "mustNotContain": [
            "SourceSpan",
            "StrictEvidence",
            "citation_grade_evidence",
            "runtime_answer_evidence",
            "canonical_parsed_artifact_overwrite",
        ],
    }


def _promotion_blockers(affected: list[str]) -> list[str]:
    blockers = [
        "source_content_hash_must_be_reverified_before_repair_output_is_trusted",
        "page_locator_must_be_stable_against_existing_source_pdf",
        "deterministic_identity_key_must_be_materialized_by_dry_run",
        "report_only_design_not_evidence",
        "strict_or_runtime_promotion_requires_later_explicit_tranche",
    ]
    if "table_numeric" in affected:
        blockers.extend(
            [
                "table_numeric_cells_need_page_bbox_row_column_or_cell_locator",
                "generated_markdown_table_text_is_not_table_cell_source_authority",
            ]
        )
    if "figure_caption" in affected:
        blockers.append("figure_caption_needs_caption_identity_and_region_link")
    if "equation_region" in affected:
        blockers.append("equation_region_needs_formula_bbox_or_tex_pdf_alignment")
    return list(dict.fromkeys(blockers))


def _dry_run_readiness(row: dict[str, Any], blockers: list[str]) -> tuple[str, list[str]]:
    missing: list[str] = []
    if not _clean_text(row.get("artifactId")):
        missing.append("missing_artifact_id")
    if not _clean_text(row.get("paperId")):
        missing.append("missing_paper_id")
    if not _as_list(row.get("sourceIds")):
        missing.append("missing_source_ids")
    if missing:
        return "blocked_until_repair_authority_design_gap_resolved", missing
    return (
        "ready_for_pymupdf_quality_repair_executor_dry_run",
        [
            "row_has_artifact_and_source_identity",
            "required_authority_is_explicitly_declared",
            "dry_run_must_resolve_provenance_blockers_before_any_later_promotion",
            *blockers,
        ],
    )


def _design_row(index: int, item: dict[str, Any]) -> dict[str, Any]:
    reasons = _as_list(item.get("degradationReasons"))
    impacts = _as_list(item.get("evidenceImpacts"))
    affected = _affected_evidence_types(impacts)
    components = _strategy_components(impacts, reasons)
    blockers = _promotion_blockers(affected)
    readiness, readiness_reasons = _dry_run_readiness(item, blockers)
    diagnostic = dict(item.get("diagnosticMetrics") or {})
    return {
        "designId": f"pymupdf-quality-repair-design:{index:04d}",
        "artifactId": _clean_text(item.get("artifactId")),
        "sourceIds": _as_list(item.get("sourceIds")),
        "paperId": _clean_text(item.get("paperId")),
        "paperTitle": _clean_text(item.get("paperTitle")),
        "parser": _clean_text(item.get("parser")) or "pymupdf",
        "coverageStatus": _clean_text(item.get("coverageStatus")),
        "parsedStatus": _clean_text(item.get("parsedStatus")),
        "parserQualityCategory": _clean_text(item.get("parserQualityCategory")),
        "degradationReasons": reasons,
        "affectedEvidenceTypes": affected,
        "usableEvidenceTypes": _usable_evidence_types(impacts),
        "primaryRepairStrategy": _primary_repair_strategy(reasons, impacts),
        "repairStrategyComponents": components,
        "diagnosticMetrics": {
            "pageCount": _safe_int(diagnostic.get("pageCount")),
            "pagesWithText": _safe_int(diagnostic.get("pagesWithText")),
            "textLayerDetected": bool(diagnostic.get("textLayerDetected")),
            "columnCountDetected": _safe_int(diagnostic.get("columnCountDetected")),
            "tablesDetected": _safe_int(diagnostic.get("tablesDetected")),
            "figuresDetected": _safe_int(diagnostic.get("figuresDetected")),
            "equationsDetected": _safe_int(diagnostic.get("equationsDetected")),
        },
        "requiredAuthority": _required_authority(affected),
        "expectedRepairedArtifactShape": _expected_repaired_artifact_shape(affected, components),
        "unsafeOrInsufficientProvenanceBlockers": blockers,
        "dryRunExecutorReadiness": readiness,
        "dryRunExecutorReadinessReasons": readiness_reasons,
        "reportOnly": True,
        "mutationCounters": _zero_counters(),
    }


def _top_level_authority_requirements() -> dict[str, Any]:
    return _required_authority(["table_numeric", "figure_caption", "equation_region"])


def _top_level_artifact_shape() -> dict[str, Any]:
    return _expected_repaired_artifact_shape(
        ["table_numeric", "figure_caption", "equation_region"],
        [
            "page_block_segmentation",
            "multi_column_reading_order_probe",
            "table_numeric_region_locator",
            "figure_caption_region_locator",
            "equation_region_locator",
        ],
    )


def build_pymupdf_quality_repair_design(*, degradation_audit: dict[str, Any]) -> dict[str, Any]:
    """Build a design-only report for a later PyMuPDF quality repair dry-run."""

    schema_violations = _input_schema_violations(degradation_audit)
    unsafe_flags = _unsafe_upstream_flags(degradation_audit)
    source_items = [dict(item) for item in list(degradation_audit.get("items") or []) if isinstance(item, dict)]
    degraded_items = [
        item
        for item in source_items
        if bool(item.get("parsedDegraded"))
        and _clean_text(item.get("recommendedRecoveryStrategy")) == "pymupdf_quality_repair_design"
    ]
    design_rows = [_design_row(index, item) for index, item in enumerate(degraded_items, start=1)]
    private_leak_rows = _private_path_leak_rows(design_rows)

    by_strategy: Counter[str] = Counter()
    by_evidence: Counter[str] = Counter()
    by_blocker: Counter[str] = Counter()
    by_readiness: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    for row in design_rows:
        by_strategy.update([row["primaryRepairStrategy"]])
        by_evidence.update(row["affectedEvidenceTypes"])
        by_blocker.update(row["unsafeOrInsufficientProvenanceBlockers"])
        by_readiness.update([row["dryRunExecutorReadiness"]])
        by_paper_id[row["paperId"]] = {
            "artifactId": row["artifactId"],
            "primaryRepairStrategy": row["primaryRepairStrategy"],
            "affectedEvidenceTypes": row["affectedEvidenceTypes"],
            "dryRunExecutorReadiness": row["dryRunExecutorReadiness"],
        }

    design_gap_rows = by_readiness.get("blocked_until_repair_authority_design_gap_resolved", 0)
    schema_violation_count = len(schema_violations)
    counts = {
        "inputCorpusRows": _safe_int(dict(degradation_audit.get("counts") or {}).get("inputCorpusRows"))
        or len(source_items),
        "coverageReadyRows": _safe_int(dict(degradation_audit.get("counts") or {}).get("coverageReadyRows")),
        "parserDegradedRows": len(degraded_items),
        "repairDesignRows": len(design_rows),
        "upstreamNoActionRows": max(0, len(source_items) - len(degraded_items)),
        "pageBlobSectionsOnlyRows": _safe_int(dict(degradation_audit.get("counts") or {}).get("pageBlobSectionsOnlyRows")),
        "tablesCaptionOnlyRows": _safe_int(dict(degradation_audit.get("counts") or {}).get("tablesCaptionOnlyRows")),
        "multiColumnProbeOnlyRows": _safe_int(dict(degradation_audit.get("counts") or {}).get("multiColumnProbeOnlyRows")),
        "tableNumericBlockedRows": by_evidence.get("table_numeric", 0),
        "figureCaptionBlockedRows": by_evidence.get("figure_caption", 0),
        "equationRegionBlockedRows": by_evidence.get("equation_region", 0),
        "dryRunReadyRows": by_readiness.get("ready_for_pymupdf_quality_repair_executor_dry_run", 0),
        "dryRunBlockedRows": design_gap_rows,
        **_zero_counters(
            private_path_leak_rows=private_leak_rows,
            schema_violation_count=schema_violation_count,
        ),
    }
    blocked = bool(schema_violations or unsafe_flags or private_leak_rows or design_gap_rows)
    gate_decision = (
        "blocked_until_repair_authority_design_gap_resolved"
        if blocked
        else "ready_for_pymupdf_quality_repair_executor_dry_run"
    )
    status = "blocked" if blocked else "design_ready"
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_design",
        "request": {
            "inputSchema": _clean_text(degradation_audit.get("schema")),
            "inputStatus": _clean_text(degradation_audit.get("status")),
            "requiredParentTranche": "parsed_artifact_parser_quality_degradation_audit",
            "reportOnly": True,
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
            "designGapRows": design_gap_rows,
            "nextTranche": "parsed_artifact_pymupdf_quality_repair_executor_dry_run"
            if gate_decision == "ready_for_pymupdf_quality_repair_executor_dry_run"
            else "resolve_pymupdf_quality_repair_authority_design_gap",
        },
        "requiredAuthority": _top_level_authority_requirements(),
        "expectedRepairedArtifactShape": _top_level_artifact_shape(),
        "byPaperId": by_paper_id,
        "byPrimaryRepairStrategy": dict(sorted(by_strategy.items())),
        "byAffectedEvidenceType": dict(sorted(by_evidence.items())),
        "byUnsafeOrInsufficientProvenanceBlocker": dict(sorted(by_blocker.items())),
        "byDryRunExecutorReadiness": dict(sorted(by_readiness.items())),
        "designRows": design_rows,
        "warnings": [],
    }


def build_blocked_pymupdf_quality_repair_design(*, reason: str) -> dict[str, Any]:
    return {
        "schema": PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_pymupdf_quality_repair_design",
        "request": {
            "inputSchema": "",
            "inputStatus": "",
            "requiredParentTranche": "parsed_artifact_parser_quality_degradation_audit",
            "reportOnly": True,
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
            "parserDegradedRows": 0,
            "repairDesignRows": 0,
            "upstreamNoActionRows": 0,
            "pageBlobSectionsOnlyRows": 0,
            "tablesCaptionOnlyRows": 0,
            "multiColumnProbeOnlyRows": 0,
            "tableNumericBlockedRows": 0,
            "figureCaptionBlockedRows": 0,
            "equationRegionBlockedRows": 0,
            "dryRunReadyRows": 0,
            "dryRunBlockedRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "gate": {
            "decision": "blocked_until_repair_authority_design_gap_resolved",
            "schemaViolations": [reason],
            "unsafeUpstreamFlags": [],
            "designGapRows": 0,
            "nextTranche": "resolve_pymupdf_quality_repair_authority_design_gap",
        },
        "requiredAuthority": _top_level_authority_requirements(),
        "expectedRepairedArtifactShape": _top_level_artifact_shape(),
        "byPaperId": {},
        "byPrimaryRepairStrategy": {},
        "byAffectedEvidenceType": {},
        "byUnsafeOrInsufficientProvenanceBlocker": {},
        "byDryRunExecutorReadiness": {},
        "designRows": [],
        "warnings": [reason],
    }


def render_pymupdf_quality_repair_design_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Parsed Artifact PyMuPDF Quality Repair Design",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate decision: `{gate.get('decision')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Input corpus rows: `{counts.get('inputCorpusRows', 0)}`",
        f"- Coverage-ready rows: `{counts.get('coverageReadyRows', 0)}`",
        f"- Parser-degraded rows: `{counts.get('parserDegradedRows', 0)}`",
        f"- Repair design rows: `{counts.get('repairDesignRows', 0)}`",
        f"- Table numeric blocked rows: `{counts.get('tableNumericBlockedRows', 0)}`",
        f"- Figure caption blocked rows: `{counts.get('figureCaptionBlockedRows', 0)}`",
        f"- Equation region blocked rows: `{counts.get('equationRegionBlockedRows', 0)}`",
        f"- Next tranche: `{gate.get('nextTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only design. No parser rerun, parsed artifact overwrite, parser routing change, DB/index mutation, reindex, reembed, vault scan, external download, SourceSpan, StrictEvidence, citation/runtime evidence, answer path invocation, or answerability promotion.",
        "",
        "## Primary Repair Strategies",
        "",
    ]
    for strategy, count in dict(report.get("byPrimaryRepairStrategy") or {}).items():
        lines.append(f"- `{strategy}`: `{count}`")
    if not report.get("byPrimaryRepairStrategy"):
        lines.append("- none")
    lines.extend(["", "## Affected Evidence Types", ""])
    for evidence_type, count in dict(report.get("byAffectedEvidenceType") or {}).items():
        lines.append(f"- `{evidence_type}`: `{count}`")
    lines.extend(["", "## Authority Requirements", ""])
    authority = dict(report.get("requiredAuthority") or {})
    for key, spec in authority.items():
        basis = dict(spec).get("basis")
        lines.append(f"- `{key}`: `{basis}`")
    lines.extend(["", "## Design Rows", ""])
    for row in list(report.get("designRows") or []):
        affected = ", ".join(row.get("affectedEvidenceTypes") or []) or "-"
        blockers = ", ".join(row.get("unsafeOrInsufficientProvenanceBlockers") or []) or "-"
        lines.append(
            f"- `{row.get('paperId')}` strategy=`{row.get('primaryRepairStrategy')}` "
            f"affected=`{affected}` readiness=`{row.get('dryRunExecutorReadiness')}` "
            f"blockers=`{blockers}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_pymupdf_quality_repair_design_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-pymupdf-quality-repair-design.json"
    markdown_path = root / "parsed-artifact-pymupdf-quality-repair-design.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_pymupdf_quality_repair_design_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID",
    "PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_pymupdf_quality_repair_design",
    "build_pymupdf_quality_repair_design",
    "render_pymupdf_quality_repair_design_markdown",
    "write_pymupdf_quality_repair_design_reports",
]
