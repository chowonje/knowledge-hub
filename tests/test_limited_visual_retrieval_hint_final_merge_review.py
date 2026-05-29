from __future__ import annotations

import copy
import hashlib
import json

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_final_merge_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
    NEXT_TRANCHE_READY,
    READY_DECISION,
    build_limited_visual_retrieval_hint_final_merge_review,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
    PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
    READY_DECISION as APPLY_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION as SEARCH_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_runtime_candidate_discovery_route_design import (
    build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design,
)


ROUTE_REF = "eval/knowledgeos/reports/route_design_fixture.v1.json"
SEARCH_REF = "eval/knowledgeos/reports/search_quality_fixture.v1.json"
APPLY_REF = "eval/knowledgeos/reports/apply_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int) -> dict[str, object]:
    text = f"Retrieval hint only: sample visual hint {index} about figure layout."
    return {
        "schema": PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "namespace": "production_visual_retrieval_hint_candidates_v1",
        "collectionName": "knowledge_hub_visual_retrieval_hints",
        "vectorDocumentId": f"visual-retrieval-hint-production-vector-doc:{index:04d}",
        "idempotencyKey": f"key-{index}",
        "hintCandidateId": f"visual-retrieval-hint:sample:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}",
        "sourceCandidateId": f"visual-layout:sample:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}",
        "sourceContentHash": "sha256:" + "1" * 64,
        "paperId": "sample",
        "paperRef": "papers_dir/sample.pdf",
        "page": index,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "candidateType": "figure_caption_region",
        "documentText": text,
        "documentTextHash": _hash_text(text),
        "embeddingText": f"allowed_use=retrieval_hint_only | keywords=figure layout | {text}",
        "embeddingTextHash": _hash_text(text + "|embedding"),
        "embeddingVectorSha256": _hash_text(text + "|vector"),
        "sourcePreviewRecordSha256": _hash_text(text + "|preview"),
        "metadata": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "productionIndexEligible": False,
            "candidateDiscoveryOnly": True,
        },
        "productionVectorRecordSha256": _hash_text(text + "|record"),
        "embeddingVectorPresent": True,
        "embeddingVectorLength": 256,
    }


def _records(count: int = 125) -> list[dict[str, object]]:
    return [_record(index) for index in range(1, count + 1)]


def _apply_report(
    records: list[dict[str, object]] | None = None,
    *,
    mutation_count: int = 0,
) -> dict[str, object]:
    records = _records() if records is None else records
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
        "status": "ready",
        "decision": APPLY_READY_DECISION,
        "counts": {
            "plannedProductionVectorRecordRows": len(records),
            "appliedProductionVectorRecordRows": 0,
            "readbackValidatedRows": 0,
            "candidateDiscoveryOnlyRows": len(records),
            "productionVectorIndexWriteRows": mutation_count,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "blockedRows": 0,
            "policyViolationRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "externalEmbeddingCallRows": 0,
            "embeddingCallRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
        },
        "productionVectorIndexRecordPreviews": records,
    }


def _search_quality_report(
    *,
    production_hit_rows: int = 234,
    hybrid_hit_rows: int = 241,
    hybrid_lift_rows: int = 108,
    rank_regressed_rows: int = 0,
    mutation_count: int = 0,
) -> dict[str, object]:
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        "status": "ready",
        "decision": SEARCH_READY_DECISION,
        "qualityGate": {
            "passed": True,
            "thresholds": {
                "minProductionVectorHitAt5Rows": 200,
                "minHybridHitAt5LiftRows": 25,
                "maxRankRegressedRows": 0,
            },
            "observed": {
                "productionVectorHitAt5Rows": production_hit_rows,
                "hybridHitAt5LiftRows": hybrid_lift_rows,
                "rankRegressedRows": rank_regressed_rows,
            },
        },
        "counts": {
            "layoutCandidateRows": 36664,
            "sourceProductionVectorRecordRows": 125,
            "matchedProductionVectorRecordRows": 125,
            "queryRows": 250,
            "textOnlyHitAt5Rows": 133,
            "productionVectorHitAt5Rows": production_hit_rows,
            "hybridHitAt5Rows": hybrid_hit_rows,
            "hybridHitAt5LiftRows": hybrid_lift_rows,
            "rankRegressedRows": rank_regressed_rows,
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "productionVectorIndexWriteRows": mutation_count,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "operationalSearchIndexQueryRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "blockedRows": 0,
            "policyViolationRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
    }


def _route_design_report(
    search_report: dict[str, object] | None = None,
    apply_report: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design(
        production_vector_search_quality_eval=search_report or _search_quality_report(),
        production_vector_apply_executor_report=apply_report or _apply_report(),
        source_production_vector_search_quality_eval_ref=SEARCH_REF,
        source_production_vector_apply_executor_report_ref=APPLY_REF,
        generated_at="2026-05-28T00:00:00Z",
    )


def _build(
    route_design: dict[str, object] | None = None,
    search_report: dict[str, object] | None = None,
    apply_report: dict[str, object] | None = None,
) -> dict[str, object]:
    search_report = search_report or _search_quality_report()
    apply_report = apply_report or _apply_report()
    route_design = route_design or _route_design_report(search_report=search_report, apply_report=apply_report)
    return build_limited_visual_retrieval_hint_final_merge_review(
        runtime_candidate_discovery_route_design=route_design,
        production_vector_search_quality_eval=search_report,
        production_vector_apply_executor_report=apply_report,
        source_runtime_candidate_discovery_route_design_ref=ROUTE_REF,
        source_production_vector_search_quality_eval_ref=SEARCH_REF,
        source_production_vector_apply_executor_report_ref=APPLY_REF,
        generated_at="2026-05-28T00:00:00Z",
    )


def test_final_merge_review_closes_candidate_discovery_tranche() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["nextRecommendedTranche"] == NEXT_TRANCHE_READY
    assert report["counts"]["plannedRouteBindingRows"] == 125
    assert report["counts"]["candidateDiscoveryOnlyRows"] == 125
    assert report["counts"]["qualityEvalProductionVectorHitAt5Rows"] == 234
    assert report["counts"]["qualityEvalHybridHitAt5Rows"] == 241
    assert report["counts"]["qualityEvalHybridHitAt5LiftRows"] == 108
    assert report["counts"]["qualityEvalRankRegressedRows"] == 0
    assert report["counts"]["blockedRows"] == 0
    assert report["counts"]["runtimeRouteWriteRows"] == 0
    assert report["counts"]["answerVisibleRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["counts"]["productionVectorIndexWriteRows"] == 0
    assert report["branchCleanupAudit"]["baseRef"] == "refs/remotes/origin/main"
    assert report["branchCleanupAudit"]["staleWorktreeReuseAllowed"] is False
    assert report["branchCleanupAudit"]["branchDeletionRows"] == 0
    assert report["gate"]["passed"] is True

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_final_merge_review_blocks_if_route_design_count_drifts() -> None:
    route_design = copy.deepcopy(_route_design_report())
    route_design["counts"]["plannedRouteBindingRows"] = 124

    report = _build(route_design=route_design)

    assert report["status"] == "blocked"
    assert "planned_route_binding_rows_not_125" in report["technicalBlockers"]
    assert report["gate"]["passed"] is False


def test_final_merge_review_blocks_if_search_quality_counts_drift() -> None:
    route_design = _route_design_report()
    search_report = _search_quality_report(production_hit_rows=233)

    report = _build(route_design=route_design, search_report=search_report)

    assert report["status"] == "blocked"
    assert "production_vector_hit_at_5_rows_not_234" in report["technicalBlockers"]
    assert "route_design_search_quality_production_hit_at_5_mismatch" in report["technicalBlockers"]


def test_final_merge_review_blocks_if_rank_regresses() -> None:
    search_report = _search_quality_report(rank_regressed_rows=1)
    route_design = _route_design_report(search_report=search_report)

    report = _build(route_design=route_design, search_report=search_report)

    assert report["status"] == "blocked"
    assert "quality_eval_rank_regressions_present" in report["technicalBlockers"]
    assert "search_quality_rank_regressions_present" in report["technicalBlockers"]


def test_final_merge_review_blocks_if_apply_report_would_mutate_index() -> None:
    apply_report = _apply_report(mutation_count=1)
    route_design = _route_design_report()

    report = _build(route_design=route_design, apply_report=apply_report)

    assert report["status"] == "blocked"
    assert "production_vector_apply_executor_has_productionVectorIndexWriteRows" in report["technicalBlockers"]
    assert report["counts"]["productionVectorIndexWriteRows"] == 0


def test_final_merge_review_blocks_private_path_source_without_leaking_it() -> None:
    route_design = copy.deepcopy(_route_design_report())
    route_design["debugPath"] = "Mobile Documents/private-paper.pdf"

    report = _build(route_design=route_design)
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "runtime_candidate_discovery_route_design_has_private_path_leak" in report["technicalBlockers"]
    assert "Mobile Documents/private-paper.pdf" not in rendered
