from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import (
    build_pymupdf_quality_repair_design,
)
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_executor_dry_run import (
    build_pymupdf_quality_repair_executor_dry_run,
)
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_sidecar_oracle_comparison import (
    PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID,
    SIDECAR_ORACLE_PACK_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_pymupdf_quality_repair_sidecar_oracle_comparison,
    render_pymupdf_quality_repair_sidecar_oracle_comparison_markdown,
    write_pymupdf_quality_repair_sidecar_oracle_comparison_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "schemas"
    / "fixtures"
    / "paper-parsed-artifact-pymupdf-quality-repair-sidecar-oracle-comparison.v1.fixture.json"
)


def _audit_item(
    *,
    artifact_id: str,
    paper_id: str,
    reasons: list[str],
    impacts: list[str] | None = None,
) -> dict:
    return {
        "artifactId": artifact_id,
        "sourceIds": [paper_id],
        "paperId": paper_id,
        "paperTitle": f"Paper {paper_id}",
        "coverageStatus": "ready",
        "parsedStatus": "degraded",
        "parser": "pymupdf",
        "parserQualityCategory": "coverage_ready_multi_structure_quality_degraded",
        "parsedDegraded": True,
        "degradationReasons": reasons,
        "evidenceImpacts": impacts
        if impacts is not None
        else ["table_numeric_blocked", "figure_caption_blocked", "equation_region_blocked", "section_span_usable"],
        "primaryEvidenceImpact": "table_numeric_blocked",
        "recommendedRecoveryStrategy": "pymupdf_quality_repair_design",
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


def _degradation_audit(items: list[dict] | None = None) -> dict:
    if items is None:
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
            "inputCorpusRows": len(items),
            "coverageReadyRows": len(items),
            "parserDegradedRows": len(items),
            "pageBlobSectionsOnlyRows": len(items),
            "tablesCaptionOnlyRows": len(items),
            "multiColumnProbeOnlyRows": 1,
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


def _executor_report(items: list[dict] | None = None) -> dict:
    design = build_pymupdf_quality_repair_design(degradation_audit=_degradation_audit(items))
    return build_pymupdf_quality_repair_executor_dry_run(design_report=design)


def _agreeing_sidecar_pack(executor_report: dict) -> dict:
    oracle_rows: list[dict] = []
    for plan in executor_report["executorPlanRows"]:
        for requirement in plan["sidecarOracleRequirements"]:
            oracle_rows.append(
                {
                    "planId": plan["planId"],
                    "artifactId": plan["artifactId"],
                    "paperId": plan["paperId"],
                    "requirement": requirement["requirement"],
                    "component": requirement["component"],
                    "oracleStatus": "oracle_agrees",
                    "reasons": ["synthetic_local_oracle_fixture_agrees"],
                    "candidateParsers": requirement["candidateParsers"],
                    "sourceContentHashVerified": True,
                    "pageLocatorVerified": True,
                    "bboxOrLocatorVerified": True,
                    "deterministicIdentityVerified": True,
                    "reportOnly": True,
                    "mutationCounters": {key: 0 for key in ZERO_COUNTER_KEYS},
                }
            )
    return {
        "schema": SIDECAR_ORACLE_PACK_SCHEMA_ID,
        "status": "oracle_pack_ready",
        "request": {"reportOnly": True, "localOnly": True},
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
        "counts": {key: 0 for key in ZERO_COUNTER_KEYS} | {"oracleRows": len(oracle_rows)},
        "oracleRows": oracle_rows,
    }


def test_sidecar_oracle_comparison_blocks_when_pack_missing() -> None:
    executor_report = _executor_report()
    report = build_pymupdf_quality_repair_sidecar_oracle_comparison(executor_dry_run=executor_report)

    assert report["schema"] == PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID
    assert report["status"] == "comparison_complete"
    assert report["gate"]["decision"] == "blocked_until_sidecar_oracle_candidate_pack_available"
    assert validate_payload(report, PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID, strict=True).ok
    assert report["request"]["sidecarOraclePackState"] == "not_provided"
    assert report["counts"]["inputPlanRows"] == 2
    assert report["counts"]["oracleMissingPlanRows"] == 2
    assert report["counts"]["safeApplyCandidateRows"] == 0
    assert report["byOracleComparisonDisposition"] == {"oracle_missing": 2}
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0


def test_sidecar_oracle_comparison_can_mark_all_agreed_rows_safe_for_review() -> None:
    item = _audit_item(
        artifact_id="table_one",
        paper_id="2600.00001",
        reasons=["page_blob_sections_only", "tables_caption_only"],
        impacts=["table_numeric_blocked", "section_span_usable"],
    )
    executor_report = _executor_report([item])
    pack = _agreeing_sidecar_pack(executor_report)
    report = build_pymupdf_quality_repair_sidecar_oracle_comparison(
        executor_dry_run=executor_report,
        sidecar_oracle_pack=pack,
    )

    assert report["status"] == "comparison_complete"
    assert report["gate"]["decision"] == "ready_for_pymupdf_quality_repair_safe_apply_subset_review"
    assert report["counts"]["oracleAgreesPlanRows"] == 1
    assert report["counts"]["oracleAgreementSignalRows"] == report["counts"]["oracleRequirementRows"]
    assert report["counts"]["safeApplyCandidateRows"] == 1
    assert report["comparisonRows"][0]["safeApplyCandidate"] is True
    assert report["comparisonRows"][0]["safeApplyBlockers"] == []


def test_sidecar_oracle_comparison_classifies_conflict_and_unsafe_pack() -> None:
    item = _audit_item(
        artifact_id="table_one",
        paper_id="2600.00001",
        reasons=["page_blob_sections_only", "tables_caption_only"],
        impacts=["table_numeric_blocked", "section_span_usable"],
    )
    executor_report = _executor_report([item])
    pack = _agreeing_sidecar_pack(executor_report)
    pack["oracleRows"][0]["oracleStatus"] = "oracle_conflict"
    pack["oracleRows"][0]["conflictReasons"] = ["table_bbox_disagrees_with_caption_region"]
    conflict = build_pymupdf_quality_repair_sidecar_oracle_comparison(
        executor_dry_run=executor_report,
        sidecar_oracle_pack=pack,
    )

    assert conflict["gate"]["decision"] == "blocked_until_sidecar_oracle_conflicts_resolved"
    assert conflict["counts"]["oracleConflictPlanRows"] == 1

    unsafe_pack = copy.deepcopy(pack)
    unsafe_pack["counts"]["databaseMutationRows"] = 1
    unsafe = build_pymupdf_quality_repair_sidecar_oracle_comparison(
        executor_dry_run=executor_report,
        sidecar_oracle_pack=unsafe_pack,
    )

    assert unsafe["status"] == "blocked"
    assert "sidecar_pack_databaseMutationRows_nonzero" in unsafe["gate"]["unsafeUpstreamFlags"]


def test_sidecar_oracle_comparison_schema_fixture_validates() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    result = validate_payload(fixture, PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID, strict=True)

    assert result.ok, result.errors


def test_sidecar_oracle_comparison_writer_outputs_path_safe_reports(tmp_path: Path) -> None:
    report = build_pymupdf_quality_repair_sidecar_oracle_comparison(executor_dry_run=_executor_report())
    paths = write_pymupdf_quality_repair_sidecar_oracle_comparison_reports(report, tmp_path / "reports")
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    assert validate_payload(payload, PYMUPDF_QUALITY_REPAIR_SIDECAR_ORACLE_COMPARISON_SCHEMA_ID, strict=True).ok
    assert "blocked_until_sidecar_oracle_candidate_pack_available" in markdown
    assert "Sidecar oracle pack state: `not_provided`" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_pymupdf_quality_repair_sidecar_oracle_comparison_markdown(report)
