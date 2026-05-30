from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import (
    build_pymupdf_quality_repair_design,
)
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_executor_dry_run import (
    PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_pymupdf_quality_repair_executor_dry_run,
    render_pymupdf_quality_repair_executor_dry_run_markdown,
    write_pymupdf_quality_repair_executor_dry_run_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-executor-dry-run.v1.fixture.json"
)


def _audit_item(
    *,
    artifact_id: str,
    paper_id: str,
    reasons: list[str],
    impacts: list[str] | None = None,
    parsed_degraded: bool = True,
) -> dict:
    return {
        "artifactId": artifact_id,
        "sourceIds": [paper_id],
        "paperId": paper_id,
        "paperTitle": f"Paper {paper_id}",
        "coverageStatus": "ready",
        "parsedStatus": "degraded" if parsed_degraded else "ok",
        "parser": "pymupdf",
        "parserQualityCategory": "coverage_ready_multi_structure_quality_degraded"
        if parsed_degraded
        else "coverage_ready_parser_quality_ok",
        "parsedDegraded": parsed_degraded,
        "degradationReasons": reasons,
        "evidenceImpacts": impacts
        if impacts is not None
        else ["table_numeric_blocked", "figure_caption_blocked", "equation_region_blocked", "section_span_usable"],
        "primaryEvidenceImpact": "table_numeric_blocked" if parsed_degraded else "section_span_usable",
        "recommendedRecoveryStrategy": "pymupdf_quality_repair_design"
        if parsed_degraded
        else "no_action_coverage_ready",
        "secondaryRecoveryStrategies": [],
        "diagnosticMetrics": {
            "pageCount": 10,
            "pagesWithText": 10,
            "textLayerDetected": True,
            "columnCountDetected": 2 if "multi_column_probe_only" in reasons else 0,
            "tablesDetected": 0,
            "figuresDetected": 2,
            "equationsDetected": 0,
        },
        "reportOnly": True,
        "mutationCounters": {},
    }


def _degradation_audit() -> dict:
    items = [
        _audit_item(
            artifact_id="all_three",
            paper_id="2600.00001",
            reasons=["page_blob_sections_only", "multi_column_probe_only", "tables_caption_only"],
        ),
        _audit_item(
            artifact_id="table_only",
            paper_id="2600.00002",
            reasons=["page_blob_sections_only", "tables_caption_only"],
        ),
        _audit_item(
            artifact_id="multi_only",
            paper_id="2600.00003",
            reasons=["page_blob_sections_only", "multi_column_probe_only"],
        ),
        _audit_item(
            artifact_id="ok",
            paper_id="2600.00004",
            reasons=[],
            impacts=["section_span_usable"],
            parsed_degraded=False,
        ),
    ]
    return {
        "schema": "knowledge-hub.paper.parsed-artifact-parser-quality-degradation-audit.v1",
        "status": "degradation_audit_complete",
        "request": {"reportOnly": True},
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
            "inputCorpusRows": 4,
            "coverageReadyRows": 4,
            "parserDegradedRows": 3,
            "pageBlobSectionsOnlyRows": 3,
            "tablesCaptionOnlyRows": 2,
            "multiColumnProbeOnlyRows": 2,
            "canonicalParsedArtifactWriteRows": 0,
            "parserRoutingChangedRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "runtimeEvidenceCreatedRows": 0,
            "citationEvidenceCreatedRows": 0,
            "strictEvidenceCreatedRows": 0,
            "answerPathInvokedRows": 0,
            "answerabilityPromotionRows": 0,
            "schemaViolationCount": 0,
            "privatePathLeakRows": 0,
        },
        "nextRecommendedTranche": "parsed_artifact_pymupdf_quality_repair_design",
        "items": items,
    }


def _design_report() -> dict:
    return build_pymupdf_quality_repair_design(degradation_audit=_degradation_audit())


def test_pymupdf_quality_repair_executor_dry_run_plans_rows_without_apply() -> None:
    report = build_pymupdf_quality_repair_executor_dry_run(design_report=_design_report())

    assert report["schema"] == PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert report["status"] == "executor_dry_run_complete"
    assert report["gate"]["decision"] == "ready_for_pymupdf_quality_repair_sidecar_oracle_comparison"
    assert validate_payload(report, PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True).ok
    assert report["counts"]["repairDesignRows"] == 3
    assert report["counts"]["dryRunPlanRows"] == 3
    assert report["counts"]["dryRunReadyRows"] == 3
    assert report["counts"]["sidecarOracleRequiredBeforeApplyRows"] == 3
    assert report["counts"]["safeApplyCandidateRows"] == 0
    assert report["counts"]["tableNumericPlannedRows"] == 3
    assert report["counts"]["figureCaptionPlannedRows"] == 3
    assert report["counts"]["equationRegionPlannedRows"] == 3
    assert report["byDryRunDisposition"] == {"dry_run_ready": 3}
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0


def test_pymupdf_quality_repair_executor_dry_run_keeps_output_candidate_only() -> None:
    report = build_pymupdf_quality_repair_executor_dry_run(design_report=_design_report())
    row = report["executorPlanRows"][0]

    assert row["safeApplyCandidate"] is False
    assert row["sidecarOracleRequiredBeforeApply"] is True
    assert "sidecar_oracle_comparison_required_before_apply" in row["safeApplyBlockers"]
    assert {item["check"] for item in row["authorityCheckPlan"]} >= {
        "source_content_hash_authority",
        "page_locator_authority",
        "deterministic_identity_key",
        "table_cell_locator_authority",
        "figure_caption_identity_and_region_authority",
        "equation_region_authority",
    }
    for action in row["plannedRepairActions"]:
        assert action["dryRunOnly"] is True
        assert action["writesCanonicalParsedArtifact"] is False
        assert action["createsSourceSpan"] is False
        assert action["createsStrictEvidence"] is False
        assert action["createsCitationEvidence"] is False
        assert action["createsRuntimeEvidence"] is False
        assert action["invokesAnswerPath"] is False


def test_pymupdf_quality_repair_executor_dry_run_blocks_parent_gate_or_conflict() -> None:
    design = _design_report()
    design["gate"]["decision"] = "blocked_until_repair_authority_design_gap_resolved"
    blocked = build_pymupdf_quality_repair_executor_dry_run(design_report=design)

    assert blocked["status"] == "blocked"
    assert "pymupdf_quality_repair_design_gate_not_ready_for_executor" in blocked["gate"]["schemaViolations"]

    conflict_design = _design_report()
    conflict_design["designRows"][1] = copy.deepcopy(conflict_design["designRows"][0])
    conflict = build_pymupdf_quality_repair_executor_dry_run(design_report=conflict_design)

    assert conflict["status"] == "blocked"
    assert conflict["counts"]["blockedConflictRows"] == 2
    assert conflict["gate"]["duplicateArtifactIdRows"] == 1
    assert conflict["gate"]["duplicateIdempotencyKeyRows"] == 1


def test_pymupdf_quality_repair_executor_dry_run_schema_fixture_validates() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    result = validate_payload(fixture, PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True)

    assert result.ok, result.errors


def test_pymupdf_quality_repair_executor_dry_run_writer_outputs_path_safe_reports(tmp_path: Path) -> None:
    report = build_pymupdf_quality_repair_executor_dry_run(design_report=_design_report())
    paths = write_pymupdf_quality_repair_executor_dry_run_reports(report, tmp_path / "reports")
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    assert validate_payload(payload, PYMUPDF_QUALITY_REPAIR_EXECUTOR_DRY_RUN_SCHEMA_ID, strict=True).ok
    assert "ready_for_pymupdf_quality_repair_sidecar_oracle_comparison" in markdown
    assert "sidecarBeforeApply=`True`" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_pymupdf_quality_repair_executor_dry_run_markdown(report)
