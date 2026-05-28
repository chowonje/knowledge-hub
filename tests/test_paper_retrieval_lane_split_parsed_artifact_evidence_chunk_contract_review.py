from __future__ import annotations

import copy
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_final_merge_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
    READY_DECISION as FINAL_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review import (
    PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
    READY_DECISION,
    build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review,
)
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run import (
    build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run,
)


FINAL_REVIEW_REF = "eval/knowledgeos/reports/limited_visual_retrieval_hint_final_merge_review_005.v1.json"
LANE_SPLIT_REF = "eval/knowledgeos/reports/paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run.v1.json"


def _final_review_report() -> dict[str, object]:
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-29T00:00:00Z",
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


def _lane_split() -> dict[str, object]:
    return build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run(
        visual_retrieval_hint_final_merge_review=_final_review_report(),
        source_visual_retrieval_hint_final_merge_review_ref=FINAL_REVIEW_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def _build(source: dict[str, object] | None = None) -> dict[str, object]:
    return build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review(
        lane_split_dry_run=source or _lane_split(),
        source_lane_split_dry_run_ref=LANE_SPLIT_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def _contract(report: dict[str, object], artifact_type: str) -> dict[str, object]:
    rows = list(report["allowedArtifactContracts"])
    return next(row for row in rows if row["artifactType"] == artifact_type)


def _disallowed(report: dict[str, object], source_kind: str) -> dict[str, object]:
    rows = list(report["disallowedSourceContracts"])
    return next(row for row in rows if row["sourceKind"] == source_kind)


def test_contract_review_fixes_parsed_artifact_evidence_chunk_contracts() -> None:
    report = _build()

    assert report["schema"] == PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == "parsed_artifact_evidence_chunk_candidate_dry_run"
    assert report["counts"]["contractRows"] == 5
    assert report["counts"]["allowedArtifactTypeRows"] == 5
    assert report["counts"]["disallowedSourceRows"] == 5
    assert report["counts"]["textSpanContractRows"] == 2
    assert report["counts"]["structuredArtifactContractRows"] == 3
    assert report["counts"]["answerEvidenceEligibleContractRows"] == 5
    assert report["counts"]["answerabilityEligibleContractRows"] == 5
    assert report["counts"]["visualHintAnswerEvidenceAllowedRows"] == 0
    assert report["counts"]["fallbackChunkAnswerEvidenceAllowedRows"] == 0
    assert report["counts"]["locatorOnlyAnswerEvidenceAllowedRows"] == 0
    assert report["counts"]["runtimeRouteWriteRows"] == 0
    assert report["counts"]["operationalSearchIndexQueryRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["citationGradeRows"] == 0
    assert report["counts"]["databaseMutationRows"] == 0
    assert report["counts"]["indexMutationRows"] == 0
    assert report["gate"]["passed"] is True

    section = _contract(report, "section")
    paragraph = _contract(report, "paragraph")
    table = _contract(report, "table")
    assert section["locatorContract"] == "chars:start-end_required"
    assert paragraph["locatorContract"] == "chars:start-end_required"
    assert "sourceContentHash" in section["requiredFields"]
    assert "snippetHash" in section["requiredFields"]
    assert "sourceContentHash" in paragraph["requiredFields"]
    assert table["requiresStructuredArtifactReadback"] is True

    visual_hint = _disallowed(report, "visual_retrieval_hint_text")
    fallback = _disallowed(report, "fallback_chunk")
    locator_only = _disallowed(report, "locator_only_anchor")
    korean_summary = _disallowed(report, "korean_summary_or_paraphrase")
    assert visual_hint["answerEvidenceAllowed"] is False
    assert fallback["answerabilityAllowed"] is False
    assert locator_only["citationGradeAllowed"] is False
    assert korean_summary["strictEvidenceAllowed"] is False

    validation = validate_payload(
        report,
        PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_contract_review_blocks_if_lane_split_is_not_ready() -> None:
    source = copy.deepcopy(_lane_split())
    source["status"] = "blocked"
    source["decision"] = "blocked"
    source["gate"] = {"passed": False}

    report = _build(source)

    assert report["status"] == "blocked"
    assert report["counts"]["contractRows"] == 0
    assert "paper_retrieval_lane_split_dry_run_not_ready" in report["technicalBlockers"]
    assert "paper_retrieval_lane_split_dry_run_invalid_decision" in report["technicalBlockers"]
    assert "paper_retrieval_lane_split_dry_run_gate_not_passed" in report["technicalBlockers"]


def test_contract_review_blocks_if_lane_split_counts_drift() -> None:
    source = copy.deepcopy(_lane_split())
    source["counts"]["visualHintRowsQuarantinedFromAnswerEvidence"] = 124
    source["counts"]["plannedLaneSplitRows"] = 3

    report = _build(source)

    assert report["status"] == "blocked"
    assert "planned_lane_split_rows_not_2" in report["technicalBlockers"]
    assert "visual_hints_not_quarantined_from_answer_evidence" in report["technicalBlockers"]


def test_contract_review_blocks_if_source_has_unsafe_counter() -> None:
    source = copy.deepcopy(_lane_split())
    source["counts"]["answerVisibleRows"] = 1

    report = _build(source)

    assert report["status"] == "blocked"
    assert "paper_retrieval_lane_split_dry_run_has_answerVisibleRows" in report["technicalBlockers"]
    assert report["counts"]["answerVisibleRows"] == 0


def test_contract_review_blocks_if_visual_lane_can_supply_answer_evidence() -> None:
    source = copy.deepcopy(_lane_split())
    for row in source["retrievalLaneRows"]:
        if row["laneName"] == "visual_retrieval_hint_candidate_discovery":
            row["lanePolicy"]["maySupplyAnswerEvidence"] = True

    report = _build(source)

    assert report["status"] == "blocked"
    assert "visual_candidate_discovery_lane_can_supply_answer_evidence" in report["technicalBlockers"]


def test_contract_review_blocks_if_visual_hint_fallback_is_allowed() -> None:
    source = copy.deepcopy(_lane_split())
    source["counts"]["fallbackToVisualHintAsEvidenceAllowedRows"] = 1

    report = _build(source)

    assert report["status"] == "blocked"
    assert "fallback_to_visual_hint_as_evidence_allowed" in report["technicalBlockers"]


def test_contract_review_blocks_private_path_source_without_leaking_it() -> None:
    source = copy.deepcopy(_lane_split())
    source["debugPath"] = "/Users/won/private/paper.pdf"

    report = _build(source)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "paper_retrieval_lane_split_dry_run_has_private_path_leak" in report["technicalBlockers"]
    assert "/Users/won/private" not in rendered


def test_contract_review_write_and_evidence_counters_remain_zero() -> None:
    report = _build()
    for field in (
        "runtimeRouteWriteRows",
        "runtimeConfigMutationRows",
        "operationalSearchIndexQueryRows",
        "answerVisibleRows",
        "answerGenerationRows",
        "strictEvidenceRows",
        "citationGradeRows",
        "runtimeEvidenceRows",
        "sourceSpanCreatedRows",
        "sourceSpanCandidateCreatedRows",
        "parsedArtifactEvidenceChunkCreatedRows",
        "embeddingCallRows",
        "vectorIndexWriteRows",
        "productionVectorIndexWriteRows",
        "databaseMutationRows",
        "indexMutationRows",
        "reindexOrReembedRows",
        "parserExecutionRows",
        "canonicalParsedArtifactWriteRows",
        "vaultScanRows",
        "externalDownloadRows",
        "branchDeletionRows",
        "githubPrMutationRows",
    ):
        assert report["counts"][field] == 0
