from __future__ import annotations

import hashlib

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
    PRODUCTION_VECTOR_INDEX_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION as SEARCH_READY_DECISION,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_runtime_candidate_discovery_route_design import (
    LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
    READY_DECISION,
    READY_ROW_STATUS,
    build_limited_visual_retrieval_hint_runtime_candidate_discovery_route_design,
)


SEARCH_REF = "eval/knowledgeos/reports/search_quality_fixture.v1.json"
APPLY_REF = "eval/knowledgeos/reports/apply_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    text = f"Retrieval hint only: sample visual hint {index} about CLIP prompt engineering."
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
        "embeddingText": f"allowed_use=retrieval_hint_only | keywords=CLIP, prompt engineering | {text}",
        "embeddingTextHash": _hash_text(text + "|embedding"),
        "embeddingVectorSha256": _hash_text(text + "|vector"),
        "sourcePreviewRecordSha256": _hash_text(text + "|preview"),
        "metadata": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
            "indexEligible": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
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


def _apply_report(records: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    records = _records() if records is None else records
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
        "status": status,
        "decision": (
            "ready_for_limited_visual_retrieval_hint_production_vector_db_integration_apply"
            if status == "ready"
            else "blocked"
        ),
        "counts": {
            "plannedProductionVectorRecordRows": len(records) if status == "ready" else 0,
            "appliedProductionVectorRecordRows": 0,
            "readbackValidatedRows": 0,
            "candidateDiscoveryOnlyRows": len(records) if status == "ready" else 0,
            "productionVectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
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


def _search_quality_report(*, status: str = "ready", mutation_count: int = 0) -> dict[str, object]:
    ready = status == "ready"
    return {
        "schema": LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        "status": status,
        "decision": SEARCH_READY_DECISION if ready else "blocked",
        "qualityGate": {
            "passed": ready,
            "thresholds": {
                "minProductionVectorHitAt5Rows": 200,
                "minHybridHitAt5LiftRows": 25,
                "maxRankRegressedRows": 0,
            },
            "observed": {
                "productionVectorHitAt5Rows": 234 if ready else 0,
                "hybridHitAt5LiftRows": 108 if ready else 0,
                "rankRegressedRows": 0,
            },
        },
        "counts": {
            "layoutCandidateRows": 36664,
            "sourceProductionVectorRecordRows": 125 if ready else 0,
            "matchedProductionVectorRecordRows": 125 if ready else 0,
            "queryRows": 250 if ready else 0,
            "textOnlyHitAt5Rows": 133 if ready else 0,
            "productionVectorHitAt5Rows": 234 if ready else 0,
            "hybridHitAt5Rows": 241 if ready else 0,
            "hybridHitAt5LiftRows": 108 if ready else 0,
            "rankRegressedRows": 0,
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


def _build(
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


def test_runtime_candidate_discovery_route_design_projects_125_ready_bindings() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["plannedRouteBindingRows"] == 125
    assert report["counts"]["candidateDiscoveryOnlyRows"] == 125
    assert report["counts"]["qualityEvalHybridHitAt5Rows"] == 241
    assert report["counts"]["qualityEvalRankRegressedRows"] == 0
    assert report["counts"]["runtimeRouteWriteRows"] == 0
    assert report["counts"]["operationalSearchIndexQueryRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["gate"]["passed"] is True
    assert len(report["plannedRuntimeRouteBindings"]) == 125
    assert all(row["status"] == READY_ROW_STATUS for row in report["routeBindingRowsDetail"])
    assert report["plannedRuntimeRouteBindings"][0]["candidateDiscoveryPolicy"]["candidateMaySupplyAnswerEvidence"] is False

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_RUNTIME_CANDIDATE_DISCOVERY_ROUTE_DESIGN_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_runtime_candidate_discovery_route_design_blocks_failed_quality_gate() -> None:
    report = _build(search_report=_search_quality_report(status="blocked"))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedRouteBindingRows"] == 0
    assert report["counts"]["blockedRows"] == 125
    assert "production_vector_search_quality_eval_not_ready" in report["technicalBlockers"]


def test_runtime_candidate_discovery_route_design_blocks_unsafe_policy_row() -> None:
    records = _records()
    records[0] = _record(1, runtime_visible=True)
    report = _build(apply_report=_apply_report(records))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedRouteBindingRows"] == 124
    assert report["counts"]["blockedRows"] == 1
    assert report["counts"]["policyViolationRows"] == 1
    assert "policy_not_retrieval_hint_only" in report["technicalBlockers"]


def test_runtime_candidate_discovery_route_design_blocks_source_mutation_counter() -> None:
    report = _build(search_report=_search_quality_report(mutation_count=1))

    assert report["status"] == "blocked"
    assert report["counts"]["plannedRouteBindingRows"] == 0
    assert report["counts"]["blockedRows"] == 125
    assert "production_vector_search_quality_eval_has_productionVectorIndexWriteRows" in report["technicalBlockers"]
