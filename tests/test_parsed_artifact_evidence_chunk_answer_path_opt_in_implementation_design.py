from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design,
    write_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_opt_in_route_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
    READY_DECISION as ROUTE_REVIEW_READY_DECISION,
)


def _route_review_report(*, status: str = "ready") -> dict[str, object]:
    ready = status == "ready"
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_ROUTE_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": ROUTE_REVIEW_READY_DECISION if ready else "parsed_artifact_evidence_chunk_answer_path_opt_in_route_review_blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design",
        "inputs": {
            "answerQualitySmokeReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_real_answer_quality_smoke.v1.json",
            "answerQualitySmokeSchema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-real-answer-quality-smoke.v1",
            "answerQualitySmokeStatus": "ready",
        },
        "policy": {
            "reportOnly": True,
            "routeReviewOnly": True,
            "defaultAskPathChanged": False,
            "publicCliFlagAdded": False,
            "runtimeRouteWrite": False,
            "answerGenerationRun": False,
            "llmCalls": False,
            "judgeModelCalls": False,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
            "parserExecution": False,
            "databaseMutation": False,
            "indexMutation": False,
            "vaultScan": False,
            "externalDownload": False,
        },
        "counts": {
            "inputAnswerQualitySmokeRows": 1,
            "answerQualitySmokePassRows": 2,
            "routeReviewRows": 8,
            "routeReviewPassRows": 7,
            "routeReviewGapRows": 1,
            "routeReviewFailRows": 0,
            "requiredRouteRows": 7,
            "optionalGapRows": 1,
            "runtimeRouteWriteRows": 0,
            "publicCliFlagRows": 0,
            "defaultOnRows": 0,
            "answerVisibleDefaultRows": 0,
            "candidateStoreWriteRows": 0,
            "sourceSpanCreatedRows": 0,
            "strictEvidenceRows": 0,
            "parserExecutionRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
            "llmCallRows": 0,
            "judgeModelCallRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForOptInImplementationDesign": ready,
            "answerQualitySmokeReady": ready,
            "internalRuntimeQueryPlanIngressReady": ready,
            "publicSearcherIngressGap": True,
            "publicCliDefaultUnchanged": True,
            "semanticViolations": [],
        },
        "implementationNotes": [],
        "rows": [],
        "warnings": [],
    }


def _build(report: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design(
        route_review_report=report,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_implementation_design_ready_from_route_review_gap() -> None:
    report = _build(_route_review_report())

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["designRows"] == 7
    assert report["counts"]["designReadyRows"] == 7
    assert report["counts"]["plannedSearcherIngressRows"] == 2
    assert report["counts"]["plannedRuntimeForwardingRows"] == 2
    assert report["gate"]["readyForSearcherIngressImplementation"] is True
    assert report["gate"]["publicSearcherIngressGapConfirmed"] is True
    assert report["gate"]["plannedPublicCliChange"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_implementation_design_targets_generate_and_stream_without_cli_flag() -> None:
    report = _build(_route_review_report())
    rows = {row["rowId"]: row for row in report["rows"]}

    assert rows["rag_searcher_generate_answer_add_query_plan_param"]["fileRef"] == "knowledge_hub/ai/rag.py"
    assert rows["rag_searcher_generate_answer_forward_query_plan"]["status"] == "design_ready"
    assert rows["rag_searcher_stream_answer_add_query_plan_param"]["status"] == "design_ready"
    assert rows["rag_searcher_stream_answer_forward_query_plan"]["status"] == "design_ready"
    assert rows["khub_ask_no_public_flag"]["status"] == "design_ready"
    assert report["design"]["targetMethods"] == ["generate_answer", "stream_answer"]
    assert report["design"]["queryPlanDefault"] is None
    assert report["design"]["publicCliFlag"] is None


def test_implementation_design_blocks_when_route_review_not_ready() -> None:
    report = _build(_route_review_report(status="blocked"))

    assert report["status"] == "blocked"
    assert report["gate"]["routeReviewReady"] is False
    assert "route_review_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok


def test_implementation_design_blocks_when_public_searcher_gap_not_confirmed() -> None:
    source = copy.deepcopy(_route_review_report())
    source["gate"]["publicSearcherIngressGap"] = False
    report = _build(source)

    assert report["status"] == "blocked"
    assert "public_searcher_ingress_gap_not_confirmed" in report["gate"]["semanticViolations"]


def test_implementation_design_blocks_on_unsafe_route_review_counters() -> None:
    source = copy.deepcopy(_route_review_report())
    source["counts"]["publicCliFlagRows"] = 1
    report = _build(source)

    assert report["status"] == "blocked"
    assert "route_review_has_publicCliFlagRows" in report["gate"]["semanticViolations"]


def test_implementation_design_blocks_private_path_markers() -> None:
    source = copy.deepcopy(_route_review_report())
    source["rows"].append({"fileRef": "/Users/example/private"})
    report = _build(source)

    assert report["status"] == "blocked"
    assert "route_review_private_path_leak" in report["gate"]["semanticViolations"]


def test_implementation_design_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build(_route_review_report())
    paths = write_parsed_artifact_evidence_chunk_answer_path_opt_in_implementation_design(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Opt-in Implementation Design"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_OPT_IN_IMPLEMENTATION_DESIGN_SCHEMA_ID,
        strict=True,
    ).ok
