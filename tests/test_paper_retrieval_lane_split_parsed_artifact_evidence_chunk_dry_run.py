from __future__ import annotations

import copy
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_final_merge_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
    READY_DECISION as FINAL_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run import (
    PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run,
)


FINAL_REVIEW_REF = "eval/knowledgeos/reports/limited_visual_retrieval_hint_final_merge_review_005.v1.json"


def _final_review_report() -> dict[str, object]:
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-28T00:00:00Z",
        "decision": FINAL_REVIEW_READY_DECISION,
        "nextRecommendedTranche": "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run",
        "counts": {
            "reviewRows": 125,
            "plannedRouteBindingRows": 125,
            "candidateDiscoveryOnlyRows": 125,
            "qualityEvalProductionVectorHitAt5Rows": 234,
            "qualityEvalHybridHitAt5Rows": 241,
            "qualityEvalHybridHitAt5LiftRows": 108,
            "qualityEvalRankRegressedRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "branchDeletionRows": 0,
            "githubPrMutationRows": 0,
            "runtimeRouteWriteRows": 0,
            "runtimeConfigMutationRows": 0,
            "operationalSearchIndexQueryRows": 0,
            "runtimeVisibleRows": 0,
            "answerVisibleRows": 0,
            "answerGenerationRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "productionVectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "graphDbWriteRows": 0,
            "ontologyWriteRows": 0,
            "memoryCardWriteRows": 0,
            "clusterWriteRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
        },
        "gate": {"passed": True},
    }


def _build(source: dict[str, object] | None = None) -> dict[str, object]:
    return build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run(
        visual_retrieval_hint_final_merge_review=source or _final_review_report(),
        source_visual_retrieval_hint_final_merge_review_ref=FINAL_REVIEW_REF,
        generated_at="2026-05-28T00:00:00Z",
    )


def _lane(report: dict[str, object], lane_name: str) -> dict[str, object]:
    rows = list(report["retrievalLaneRows"])
    return next(row for row in rows if row["laneName"] == lane_name)


def test_lane_split_dry_run_plans_visual_hint_and_parsed_evidence_lanes() -> None:
    report = _build()

    assert report["schema"] == PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["plannedLaneSplitRows"] == 2
    assert report["counts"]["parsedArtifactEvidenceChunkLaneRows"] == 1
    assert report["counts"]["visualCandidateDiscoveryLaneRows"] == 1
    assert report["counts"]["answerEvidenceEligibleLaneRows"] == 1
    assert report["counts"]["answerabilityEligibleLaneRows"] == 1
    assert report["counts"]["visualHintRowsQuarantinedFromAnswerEvidence"] == 125
    assert report["counts"]["fallbackToVisualHintAsEvidenceAllowedRows"] == 0
    assert report["counts"]["runtimeRouteWriteRows"] == 0
    assert report["counts"]["operationalSearchIndexQueryRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["gate"]["passed"] is True

    parsed_lane = _lane(report, "parsed_artifact_evidence_chunk")
    visual_lane = _lane(report, "visual_retrieval_hint_candidate_discovery")
    assert parsed_lane["lanePolicy"]["maySupplyAnswerEvidence"] is True
    assert parsed_lane["lanePolicy"]["maySatisfyAnswerability"] is True
    assert parsed_lane["lanePolicy"]["requiresSourceContentHash"] is True
    assert parsed_lane["lanePolicy"]["requiresLocator"] is True
    assert visual_lane["lanePolicy"]["maySupplyAnswerEvidence"] is False
    assert visual_lane["lanePolicy"]["maySatisfyAnswerability"] is False
    assert visual_lane["lanePolicy"]["candidateDiscoveryOnly"] is True

    validation = validate_payload(
        report,
        PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_lane_split_blocks_if_final_review_is_not_ready() -> None:
    source = copy.deepcopy(_final_review_report())
    source["status"] = "blocked"
    source["decision"] = "blocked"
    source["gate"] = {"passed": False}

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["counts"]["plannedLaneSplitRows"] == 0
    assert "visual_retrieval_hint_final_merge_review_not_ready" in report["technicalBlockers"]
    assert "visual_retrieval_hint_final_merge_review_invalid_decision" in report["technicalBlockers"]
    assert "visual_retrieval_hint_final_merge_review_gate_not_passed" in report["technicalBlockers"]


def test_lane_split_blocks_if_visual_hint_rows_are_not_quarantined() -> None:
    source = copy.deepcopy(_final_review_report())
    source["counts"]["candidateDiscoveryOnlyRows"] = 124

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["counts"]["visualHintRowsQuarantinedFromAnswerEvidence"] == 0
    assert "visual_hint_candidate_discovery_only_rows_not_125" in report["technicalBlockers"]
    assert report["gate"]["checks"]["visualHintsQuarantinedFromAnswerEvidence"] is False


def test_lane_split_blocks_if_source_report_has_unsafe_counter() -> None:
    source = copy.deepcopy(_final_review_report())
    source["counts"]["answerVisibleRows"] = 1

    report = _build(source)

    assert report["status"] == "blocked"
    assert "visual_retrieval_hint_final_merge_review_has_answerVisibleRows" in report["technicalBlockers"]
    assert report["counts"]["answerVisibleRows"] == 0


def test_lane_split_blocks_private_path_source_without_leaking_it() -> None:
    source = copy.deepcopy(_final_review_report())
    source["debugPath"] = "Mobile Documents/private-paper.pdf"

    report = _build(source)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "visual_retrieval_hint_final_merge_review_has_private_path_leak" in report["technicalBlockers"]
    assert "Mobile Documents/private-paper.pdf" not in rendered
