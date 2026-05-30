"""Report-only parser-quality audit for coverage-ready parsed paper artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from knowledge_hub.application.corpus_artifacts import corpus_entry_ref, corpus_manifest_entries


PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-parser-quality-degradation-audit.v1"
)

ZERO_COUNTER_KEYS = (
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


def _manifest_refs(corpus_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for entry in corpus_manifest_entries(corpus_manifest):
        artifact_id = corpus_entry_ref(entry)
        if not artifact_id:
            continue
        refs[artifact_id.casefold()] = {
            "artifactId": artifact_id,
            "sourceIds": _as_list(entry.get("sourceIds")) or _as_list(entry.get("sourceId")),
            "corpusTier": _clean_text(entry.get("corpusTier")) or "local_corpus",
        }
    return refs


def _diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(item.get("parsedDiagnostic") or {})
    return dict(parsed.get("diagnostic") or {})


def _degradation_reasons(item: dict[str, Any]) -> list[str]:
    return [_clean_text(reason) for reason in list(_diagnostic(item).get("degradationReasons") or []) if _clean_text(reason)]


def _evidence_impacts(item: dict[str, Any]) -> list[str]:
    diagnostic = _diagnostic(item)
    reasons = set(_degradation_reasons(item))
    impacts: list[str] = []

    if {"tables_caption_only", "multi_column_probe_only"} & reasons:
        impacts.append("table_numeric_blocked")
    if "page_blob_sections_only" in reasons and _safe_int(diagnostic.get("figuresDetected")) > 0:
        impacts.append("figure_caption_blocked")
    if "page_blob_sections_only" in reasons and _safe_int(diagnostic.get("equationsDetected")) == 0:
        impacts.append("equation_region_blocked")
    if (
        _clean_text(item.get("coverageStatus")) == "ready"
        and _safe_int(diagnostic.get("pagesWithText")) > 0
        and bool(diagnostic.get("textLayerDetected"))
    ):
        impacts.append("section_span_usable")
    if not impacts:
        impacts.append("unknown")
    return impacts


def _primary_impact(impacts: list[str]) -> str:
    for candidate in (
        "table_numeric_blocked",
        "figure_caption_blocked",
        "equation_region_blocked",
        "section_span_usable",
        "unknown",
    ):
        if candidate in impacts:
            return candidate
    return "unknown"


def _recovery_strategy(item: dict[str, Any], impacts: list[str]) -> tuple[str, list[str]]:
    reasons = set(_degradation_reasons(item))
    if not bool(item.get("parsedDegraded")) and not reasons:
        return "no_action_coverage_ready", []

    secondary: list[str] = []
    if "page_blob_sections_only" in reasons:
        secondary.append("alternative_parser_probe_needed")
    if "table_numeric_blocked" in impacts:
        secondary.append("table_specific_extractor_needed")
    if "figure_caption_blocked" in impacts:
        secondary.append("figure_caption_location_identity_needed")
    return "pymupdf_quality_repair_design", secondary


def _quality_category(item: dict[str, Any], impacts: list[str]) -> str:
    if _clean_text(item.get("coverageStatus")) != "ready":
        return "coverage_not_ready_excluded_from_parser_quality_claim"
    if not bool(item.get("parsedDegraded")):
        return "coverage_ready_parser_quality_ok"
    if "table_numeric_blocked" in impacts and "figure_caption_blocked" in impacts:
        return "coverage_ready_multi_structure_quality_degraded"
    if "table_numeric_blocked" in impacts:
        return "coverage_ready_table_quality_degraded"
    if "figure_caption_blocked" in impacts:
        return "coverage_ready_figure_quality_degraded"
    if "equation_region_blocked" in impacts:
        return "coverage_ready_equation_quality_degraded"
    return "coverage_ready_parser_quality_degraded_unknown_impact"


def _zero_counters(private_path_leak_rows: int = 0, schema_violation_count: int = 0) -> dict[str, int]:
    counters = {key: 0 for key in ZERO_COUNTER_KEYS}
    counters["privatePathLeakRows"] = max(0, int(private_path_leak_rows))
    counters["schemaViolationCount"] = max(0, int(schema_violation_count))
    return counters


def _private_path_leak_rows(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if _PRIVATE_PATH_RE.search(json.dumps(item, ensure_ascii=False)))


def _next_tranche(counts: dict[str, int]) -> str:
    if counts.get("parserDegradedRows", 0) <= 0:
        return "none"
    return "parsed_artifact_pymupdf_quality_repair_design"


def build_parser_quality_degradation_audit(
    *,
    coverage_report: dict[str, Any],
    corpus_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Classify coverage-ready parser degradation without reparsing or promoting evidence."""

    manifest_refs = _manifest_refs(corpus_manifest)
    coverage_items = [dict(item) for item in list(coverage_report.get("items") or []) if isinstance(item, dict)]
    classified: list[dict[str, Any]] = []

    for item in coverage_items:
        artifact_id = _clean_text(item.get("artifactId"))
        manifest_ref = manifest_refs.get(artifact_id.casefold(), {})
        diagnostic = _diagnostic(item)
        reasons = _degradation_reasons(item)
        impacts = _evidence_impacts(item)
        primary_strategy, secondary_strategies = _recovery_strategy(item, impacts)
        row = {
            "artifactId": artifact_id,
            "sourceIds": _as_list(item.get("sourceIds")) or _as_list(manifest_ref.get("sourceIds")),
            "corpusTier": _clean_text(item.get("corpusTier")) or _clean_text(manifest_ref.get("corpusTier")),
            "paperId": _clean_text(item.get("paperId")),
            "paperTitle": _clean_text(item.get("paperTitle")),
            "coverageStatus": _clean_text(item.get("coverageStatus")),
            "parsedStatus": _clean_text(item.get("parsedStatus")),
            "parser": _clean_text(diagnostic.get("parser")) or "unknown",
            "parserQualityCategory": _quality_category(item, impacts),
            "parsedDegraded": bool(item.get("parsedDegraded")),
            "degradationReasons": reasons,
            "evidenceImpacts": impacts,
            "primaryEvidenceImpact": _primary_impact(impacts),
            "recommendedRecoveryStrategy": primary_strategy,
            "secondaryRecoveryStrategies": secondary_strategies,
            "diagnosticMetrics": {
                "pageCount": _safe_int(diagnostic.get("pageCount")),
                "pagesWithText": _safe_int(diagnostic.get("pagesWithText")),
                "textLayerDetected": bool(diagnostic.get("textLayerDetected")),
                "columnCountDetected": _safe_int(diagnostic.get("columnCountDetected")),
                "tablesDetected": _safe_int(diagnostic.get("tablesDetected")),
                "figuresDetected": _safe_int(diagnostic.get("figuresDetected")),
                "equationsDetected": _safe_int(diagnostic.get("equationsDetected")),
            },
            "reportOnly": True,
            "mutationCounters": _zero_counters(),
        }
        classified.append(row)

    by_reason: Counter[str] = Counter()
    by_impact: Counter[str] = Counter()
    by_primary_impact: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()
    by_paper_id: dict[str, dict[str, Any]] = {}
    private_leak_rows = _private_path_leak_rows(classified)

    for row in classified:
        by_reason.update(row["degradationReasons"])
        by_impact.update(row["evidenceImpacts"])
        by_primary_impact.update([row["primaryEvidenceImpact"]])
        by_strategy.update([row["recommendedRecoveryStrategy"]])
        by_paper_id[row["paperId"]] = {
            "artifactId": row["artifactId"],
            "coverageStatus": row["coverageStatus"],
            "parserQualityCategory": row["parserQualityCategory"],
            "degradationReasons": row["degradationReasons"],
            "evidenceImpacts": row["evidenceImpacts"],
            "recommendedRecoveryStrategy": row["recommendedRecoveryStrategy"],
        }

    coverage_ready_rows = sum(1 for row in classified if row["coverageStatus"] == "ready")
    parser_degraded_rows = sum(1 for row in classified if row["parsedDegraded"])
    counts = {
        "inputCorpusRows": len(classified),
        "coverageReadyRows": coverage_ready_rows,
        "parserDegradedRows": parser_degraded_rows,
        "pageBlobSectionsOnlyRows": by_reason.get("page_blob_sections_only", 0),
        "tablesCaptionOnlyRows": by_reason.get("tables_caption_only", 0),
        "multiColumnProbeOnlyRows": by_reason.get("multi_column_probe_only", 0),
        "classifiedRows": len(classified),
        **_zero_counters(private_path_leak_rows=private_leak_rows),
    }
    status = "degradation_audit_complete" if counts["classifiedRows"] == counts["inputCorpusRows"] else "blocked"
    if private_leak_rows:
        status = "blocked"
    return {
        "schema": PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID,
        "status": status,
        "generatedAt": _now_iso(),
        "scope": "eval_critical_coverage_ready_parser_quality",
        "request": {
            "coverageReportSchema": _clean_text(coverage_report.get("schema")),
            "coverageReportStatus": _clean_text(coverage_report.get("status")),
            "coverageReadyRequired": True,
            "reportOnly": True,
        },
        "safety": {
            "vaultScan": False,
            "externalDownload": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "canonicalParsedArtifactWrite": False,
            "parserRoutingChanged": False,
            "answerPathInvoked": False,
            "answerabilityPromoted": False,
            "runtimeEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "privatePathLeakAllowed": False,
        },
        "counts": counts,
        "byPaperId": by_paper_id,
        "byDegradationReason": dict(sorted(by_reason.items())),
        "byEvidenceImpact": dict(sorted(by_impact.items())),
        "byPrimaryEvidenceImpact": dict(sorted(by_primary_impact.items())),
        "byRecommendedRecoveryStrategy": dict(sorted(by_strategy.items())),
        "recommendedRecoveryStrategy": "pymupdf_quality_repair_design"
        if parser_degraded_rows
        else "no_action_coverage_ready",
        "nextRecommendedTranche": _next_tranche(counts),
        "coverageParserQualitySeparation": {
            "coverageReadyRows": coverage_ready_rows,
            "coverageProblemRows": len(classified) - coverage_ready_rows,
            "parserQualityProblemRows": parser_degraded_rows,
            "decision": "coverage_ready_parser_quality_remains_blocker"
            if parser_degraded_rows
            else "coverage_ready_parser_quality_ok",
        },
        "items": classified,
        "warnings": [],
    }


def build_blocked_parser_quality_degradation_audit(*, reason: str) -> dict[str, Any]:
    return {
        "schema": PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID,
        "status": "blocked",
        "generatedAt": _now_iso(),
        "scope": "eval_critical_coverage_ready_parser_quality",
        "request": {
            "coverageReportSchema": "",
            "coverageReportStatus": "",
            "coverageReadyRequired": True,
            "reportOnly": True,
        },
        "safety": {
            "vaultScan": False,
            "externalDownload": False,
            "dbMutation": False,
            "indexMutation": False,
            "reindex": False,
            "reembed": False,
            "canonicalParsedArtifactWrite": False,
            "parserRoutingChanged": False,
            "answerPathInvoked": False,
            "answerabilityPromoted": False,
            "runtimeEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "privatePathLeakAllowed": False,
        },
        "counts": {
            "inputCorpusRows": 0,
            "coverageReadyRows": 0,
            "parserDegradedRows": 0,
            "pageBlobSectionsOnlyRows": 0,
            "tablesCaptionOnlyRows": 0,
            "multiColumnProbeOnlyRows": 0,
            "classifiedRows": 0,
            **_zero_counters(schema_violation_count=1 if reason == "schema_validation_failed" else 0),
        },
        "byPaperId": {},
        "byDegradationReason": {},
        "byEvidenceImpact": {},
        "byPrimaryEvidenceImpact": {},
        "byRecommendedRecoveryStrategy": {},
        "recommendedRecoveryStrategy": "no_action_coverage_ready",
        "nextRecommendedTranche": "none",
        "coverageParserQualitySeparation": {
            "coverageReadyRows": 0,
            "coverageProblemRows": 0,
            "parserQualityProblemRows": 0,
            "decision": "audit_blocked",
        },
        "items": [],
        "warnings": [reason],
    }


def render_parser_quality_degradation_audit_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Parsed Artifact Parser Quality Degradation Audit",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Scope: `{report.get('scope')}`",
        f"- Input corpus rows: `{counts.get('inputCorpusRows', 0)}`",
        f"- Coverage-ready rows: `{counts.get('coverageReadyRows', 0)}`",
        f"- Parser-degraded rows: `{counts.get('parserDegradedRows', 0)}`",
        f"- Page-blob section rows: `{counts.get('pageBlobSectionsOnlyRows', 0)}`",
        f"- Tables-caption-only rows: `{counts.get('tablesCaptionOnlyRows', 0)}`",
        f"- Multi-column probe-only rows: `{counts.get('multiColumnProbeOnlyRows', 0)}`",
        f"- Recommended recovery strategy: `{report.get('recommendedRecoveryStrategy')}`",
        f"- Next recommended tranche: `{report.get('nextRecommendedTranche')}`",
        "",
        "## Safety",
        "",
        "Report-only audit. No parser rerun, parsed artifact overwrite, parser routing change, source recovery/download, DB/index mutation, reindex, reembed, vault scan, evidence promotion, answer path invocation, or answerability promotion.",
        "",
        "## Degradation Reasons",
        "",
    ]
    for reason, count in dict(report.get("byDegradationReason") or {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    if not report.get("byDegradationReason"):
        lines.append("- none")
    lines.extend(["", "## Evidence Impact", ""])
    for impact, count in dict(report.get("byEvidenceImpact") or {}).items():
        lines.append(f"- `{impact}`: `{count}`")
    lines.extend(["", "## Paper Classifications", ""])
    for item in list(report.get("items") or []):
        reasons = ", ".join(item.get("degradationReasons") or []) or "-"
        impacts = ", ".join(item.get("evidenceImpacts") or []) or "-"
        lines.append(
            f"- `{item.get('paperId')}` coverage=`{item.get('coverageStatus')}` "
            f"category=`{item.get('parserQualityCategory')}` reasons=`{reasons}` "
            f"impacts=`{impacts}` strategy=`{item.get('recommendedRecoveryStrategy')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_parser_quality_degradation_audit_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "parsed-artifact-parser-quality-degradation-audit.json"
    markdown_path = root / "parsed-artifact-parser-quality-degradation-audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_parser_quality_degradation_audit_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID",
    "ZERO_COUNTER_KEYS",
    "build_blocked_parser_quality_degradation_audit",
    "build_parser_quality_degradation_audit",
    "render_parser_quality_degradation_audit_markdown",
    "write_parser_quality_degradation_audit_reports",
]
