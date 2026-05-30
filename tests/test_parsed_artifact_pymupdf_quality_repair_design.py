from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_pymupdf_quality_repair_design import (
    PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_pymupdf_quality_repair_design,
    render_pymupdf_quality_repair_design_markdown,
    write_pymupdf_quality_repair_design_reports,
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


def _degradation_audit(*, status: str = "degradation_audit_complete", unsafe: bool = False) -> dict:
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
        "status": status,
        "request": {"reportOnly": True},
        "safety": {
            "vaultScan": False,
            "externalDownload": False,
            "dbMutation": unsafe,
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
            "databaseMutationRows": 1 if unsafe else 0,
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
        "byEvidenceImpact": {
            "table_numeric_blocked": 3,
            "figure_caption_blocked": 3,
            "equation_region_blocked": 3,
            "section_span_usable": 4,
        },
        "byRecommendedRecoveryStrategy": {
            "pymupdf_quality_repair_design": 3,
            "no_action_coverage_ready": 1,
        },
        "nextRecommendedTranche": "parsed_artifact_pymupdf_quality_repair_design",
        "items": items,
    }


def test_pymupdf_quality_repair_design_classifies_degraded_rows() -> None:
    report = build_pymupdf_quality_repair_design(degradation_audit=_degradation_audit())

    assert report["schema"] == PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID
    assert report["status"] == "design_ready"
    assert report["gate"]["decision"] == "ready_for_pymupdf_quality_repair_executor_dry_run"
    assert validate_payload(report, PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID, strict=True).ok
    assert report["counts"]["inputCorpusRows"] == 4
    assert report["counts"]["coverageReadyRows"] == 4
    assert report["counts"]["parserDegradedRows"] == 3
    assert report["counts"]["repairDesignRows"] == 3
    assert report["counts"]["upstreamNoActionRows"] == 1
    assert report["counts"]["tableNumericBlockedRows"] == 3
    assert report["counts"]["figureCaptionBlockedRows"] == 3
    assert report["counts"]["equationRegionBlockedRows"] == 3
    assert report["byPrimaryRepairStrategy"] == {
        "pymupdf_multi_column_figure_equation_locator_repair_design": 1,
        "pymupdf_multi_column_table_figure_equation_repair_design": 1,
        "pymupdf_table_figure_equation_locator_repair_design": 1,
    }
    assert report["byAffectedEvidenceType"] == {
        "equation_region": 3,
        "figure_caption": 3,
        "table_numeric": 3,
    }
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0


def test_pymupdf_quality_repair_design_declares_authority_and_non_evidence_shape() -> None:
    report = build_pymupdf_quality_repair_design(degradation_audit=_degradation_audit())
    row = report["designRows"][0]

    assert row["requiredAuthority"]["sourceContentHash"]["required"] is True
    assert row["requiredAuthority"]["sourceContentHash"]["materializedInThisDesign"] is False
    assert row["requiredAuthority"]["pageLocatorBasis"]["mustBeVerifiedByDryRun"] is True
    assert row["requiredAuthority"]["tableLocatorBasis"]["mustBeVerifiedByDryRun"] is True
    assert row["requiredAuthority"]["captionLocatorBasis"]["mustBeVerifiedByDryRun"] is True
    assert row["requiredAuthority"]["equationLocatorBasis"]["mustBeVerifiedByDryRun"] is True
    assert row["requiredAuthority"]["deterministicIdentityKeyBasis"]["components"] == [
        "artifactId",
        "sourceContentHash",
        "parser",
        "structureType",
        "page",
        "locator",
        "normalizedTextHash",
    ]
    assert row["expectedRepairedArtifactShape"]["canonicalParsedArtifactOverwrite"] is False
    assert row["expectedRepairedArtifactShape"]["sourceSpanCreated"] is False
    assert row["expectedRepairedArtifactShape"]["strictEvidenceCreated"] is False
    assert row["expectedRepairedArtifactShape"]["runtimeEvidenceCreated"] is False
    assert "SourceSpan" in row["expectedRepairedArtifactShape"]["mustNotContain"]
    assert "table_numeric_cells_need_page_bbox_row_column_or_cell_locator" in row[
        "unsafeOrInsufficientProvenanceBlockers"
    ]


def test_pymupdf_quality_repair_design_blocks_unsafe_or_not_ready_upstream() -> None:
    report = build_pymupdf_quality_repair_design(
        degradation_audit=_degradation_audit(status="blocked", unsafe=True)
    )

    assert report["status"] == "blocked"
    assert report["gate"]["decision"] == "blocked_until_repair_authority_design_gap_resolved"
    assert "parser_quality_degradation_audit_not_ready" in report["gate"]["schemaViolations"]
    assert "databaseMutationRows_nonzero" in report["gate"]["unsafeUpstreamFlags"]
    assert "dbMutation_true" in report["gate"]["unsafeUpstreamFlags"]


def test_pymupdf_quality_repair_design_writer_outputs_path_safe_reports(tmp_path: Path) -> None:
    report = build_pymupdf_quality_repair_design(degradation_audit=_degradation_audit())
    paths = write_pymupdf_quality_repair_design_reports(report, tmp_path / "reports")
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    assert validate_payload(payload, PYMUPDF_QUALITY_REPAIR_DESIGN_SCHEMA_ID, strict=True).ok
    assert "ready_for_pymupdf_quality_repair_executor_dry_run" in markdown
    assert "sourceContentHash" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_pymupdf_quality_repair_design_markdown(report)
