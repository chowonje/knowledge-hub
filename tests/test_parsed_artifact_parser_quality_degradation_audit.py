from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_parser_quality_degradation_audit import (
    PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    build_parser_quality_degradation_audit,
    render_parser_quality_degradation_audit_markdown,
    write_parser_quality_degradation_audit_reports,
)


def _coverage_item(
    *,
    artifact_id: str,
    paper_id: str,
    reasons: list[str],
    parsed_degraded: bool,
    tables: int = 0,
    figures: int = 2,
    equations: int = 0,
) -> dict:
    return {
        "artifactId": artifact_id,
        "sourceIds": [paper_id],
        "corpusTier": "local_corpus",
        "paperId": paper_id,
        "paperTitle": f"Paper {paper_id}",
        "coverageStatus": "ready",
        "sourceStatus": "ok",
        "parsedStatus": "degraded" if parsed_degraded else "ok",
        "parsedDegraded": parsed_degraded,
        "parsedDiagnostic": {
            "diagnostic": {
                "parser": "pymupdf",
                "pageCount": 10,
                "pagesWithText": 10,
                "textLayerDetected": True,
                "columnCountDetected": 2 if "multi_column_probe_only" in reasons else 1,
                "tablesDetected": tables,
                "figuresDetected": figures,
                "equationsDetected": equations,
                "extractionDegraded": parsed_degraded,
                "degradationReasons": reasons,
            }
        },
    }


def _coverage_report() -> dict:
    return {
        "schema": "knowledge-hub.paper.parsed-artifact-coverage-audit.v1",
        "status": "ready",
        "counts": {
            "totalCorpusArtifacts": 5,
            "ready": 5,
            "missingSource": 0,
            "missingParsed": 0,
            "hashMismatch": 0,
            "unknownStatus": 0,
            "parsedDegraded": 3,
        },
        "items": [
            _coverage_item(
                artifact_id="multi",
                paper_id="2600.00001",
                reasons=["page_blob_sections_only", "multi_column_probe_only", "tables_caption_only"],
                parsed_degraded=True,
            ),
            _coverage_item(
                artifact_id="table_only",
                paper_id="2600.00002",
                reasons=["page_blob_sections_only", "tables_caption_only"],
                parsed_degraded=True,
            ),
            _coverage_item(
                artifact_id="multi_only",
                paper_id="2600.00003",
                reasons=["page_blob_sections_only", "multi_column_probe_only"],
                parsed_degraded=True,
            ),
            _coverage_item(
                artifact_id="ok_a",
                paper_id="2600.00004",
                reasons=[],
                parsed_degraded=False,
                tables=4,
            ),
            _coverage_item(
                artifact_id="ok_b",
                paper_id="2600.00005",
                reasons=[],
                parsed_degraded=False,
                tables=2,
            ),
        ],
    }


def _manifest() -> dict:
    return {
        "schema": "knowledge-hub.corpus-manifest.v1",
        "artifacts": [
            {"artifactId": key, "sourceIds": [paper_id], "corpusTier": "local_corpus"}
            for key, paper_id in [
                ("multi", "2600.00001"),
                ("table_only", "2600.00002"),
                ("multi_only", "2600.00003"),
                ("ok_a", "2600.00004"),
                ("ok_b", "2600.00005"),
            ]
        ],
    }


def test_parser_quality_degradation_audit_classifies_ready_coverage_rows() -> None:
    report = build_parser_quality_degradation_audit(
        coverage_report=_coverage_report(),
        corpus_manifest=_manifest(),
    )

    assert report["status"] == "degradation_audit_complete"
    assert validate_payload(report, PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID, strict=True).ok
    assert report["counts"]["inputCorpusRows"] == 5
    assert report["counts"]["coverageReadyRows"] == 5
    assert report["counts"]["parserDegradedRows"] == 3
    assert report["counts"]["pageBlobSectionsOnlyRows"] == 3
    assert report["counts"]["tablesCaptionOnlyRows"] == 2
    assert report["counts"]["multiColumnProbeOnlyRows"] == 2
    assert report["byRecommendedRecoveryStrategy"] == {
        "no_action_coverage_ready": 2,
        "pymupdf_quality_repair_design": 3,
    }
    assert report["recommendedRecoveryStrategy"] == "pymupdf_quality_repair_design"
    assert report["nextRecommendedTranche"] == "parsed_artifact_pymupdf_quality_repair_design"
    assert report["coverageParserQualitySeparation"] == {
        "coverageReadyRows": 5,
        "coverageProblemRows": 0,
        "parserQualityProblemRows": 3,
        "decision": "coverage_ready_parser_quality_remains_blocker",
    }
    for key in ZERO_COUNTER_KEYS:
        assert report["counts"][key] == 0


def test_parser_quality_degradation_audit_keeps_path_safe_reports(tmp_path: Path) -> None:
    report = build_parser_quality_degradation_audit(
        coverage_report=_coverage_report(),
        corpus_manifest=_manifest(),
    )
    paths = write_parser_quality_degradation_audit_reports(report, tmp_path / "reports")
    written = Path(paths["json"]).read_text(encoding="utf-8")
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    payload = json.loads(written)

    assert validate_payload(payload, PARSER_QUALITY_DEGRADATION_AUDIT_SCHEMA_ID, strict=True).ok
    assert "pymupdf_quality_repair_design" in markdown
    assert "parsed_artifact_pymupdf_quality_repair_design" in markdown
    assert str(tmp_path) not in written
    assert str(tmp_path) not in markdown
    assert str(tmp_path) not in render_parser_quality_degradation_audit_markdown(report)
