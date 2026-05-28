from __future__ import annotations

import hashlib

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_integration_apply_executor import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_INTEGRATION_APPLY_EXECUTOR_SCHEMA_ID,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_production_vector_db_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION,
    build_limited_visual_retrieval_hint_production_vector_db_search_quality_eval,
)


APPLY_REF = "eval/knowledgeos/reports/apply_fixture.v1.json"
LAYOUT_REF = "eval/knowledgeos/reports/layout_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    source_id = f"visual-layout:clip:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"
    vector_id = f"visual-retrieval-hint-production-vector-doc:{index:04d}"
    document = f"Retrieval hint only: unique_token_{index} CLIP zero shot chart."
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-production-vector-index-record.v1",
        "namespace": "production_visual_retrieval_hint_candidates_v1",
        "collectionName": "knowledge_hub_visual_retrieval_hints",
        "vectorDocumentId": vector_id,
        "idempotencyKey": f"key-{index}",
        "hintCandidateId": f"visual-retrieval-hint:clip:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}",
        "sourceCandidateId": source_id,
        "sourceContentHash": "sha256:" + "1" * 64,
        "paperId": "clip",
        "paperRef": "papers_dir/clip.pdf",
        "page": index,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "candidateType": "figure_caption_region",
        "documentText": document,
        "documentTextHash": _hash_text(document),
        "embeddingText": f"allowed_use=retrieval_hint_only | keywords=unique_token_{index}, CLIP | {document}",
        "embeddingTextHash": _hash_text(document + "embed"),
        "sourcePreviewRecordSha256": _hash_text(f"preview-{index}"),
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
        "productionVectorRecordSha256": _hash_text(f"record-{index}"),
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


def _layout_report(records: list[dict[str, object]]) -> dict[str, object]:
    rows = []
    for record in records:
        rows.append(
            {
                "schema": "knowledge-hub.paper.visual-layout-candidate-row.v1",
                "candidateId": record["sourceCandidateId"],
                "paperId": record["paperId"],
                "paperRef": record["paperRef"],
                "sourceContentHash": record["sourceContentHash"],
                "page": record["page"],
                "bbox": record["bbox"],
                "candidateType": record["candidateType"],
                "textContext": {
                    "nearbyText": record["documentText"],
                    "captionText": "",
                    "headingPath": [],
                },
                "visualContext": "",
                "retrievalHintPlan": {},
                "provenance": {},
                "blockerReason": "",
            }
        )
    return {
        "schema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
        "status": "ready",
        "counts": {"candidateRows": len(rows), "blockedRows": 0, "privatePathLeakRows": 0, "schemaViolationCount": 0},
        "candidateRowsDetail": rows,
    }


def _build(apply_report: dict[str, object] | None = None) -> dict[str, object]:
    records = _records()
    return build_limited_visual_retrieval_hint_production_vector_db_search_quality_eval(
        production_vector_apply_executor_report=apply_report or _apply_report(records),
        layout_candidate_report=_layout_report(records),
        source_production_vector_apply_executor_report_ref=APPLY_REF,
        source_layout_candidate_report_ref=LAYOUT_REF,
        min_production_hit_at5_rows=100,
        min_hybrid_hit_at5_lift_rows=0,
        generated_at="2026-05-28T00:00:00Z",
    )


def test_production_vector_search_quality_eval_ready() -> None:
    report = _build()

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["sourceProductionVectorRecordRows"] == 125
    assert report["counts"]["queryRows"] == 250
    assert report["counts"]["productionVectorHitAt5Rows"] >= 100
    assert report["counts"]["productionVectorIndexWriteRows"] == 0
    assert report["counts"]["runtimeVisibleRows"] == 0
    assert report["qualityGate"]["passed"] is True

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_PRODUCTION_VECTOR_DB_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_production_vector_search_quality_eval_blocks_bad_source() -> None:
    report = _build(_apply_report(status="blocked"))

    assert report["status"] == "blocked"
    assert "production_vector_apply_executor_not_ready_or_applied" in report["technicalBlockers"]


def test_production_vector_search_quality_eval_blocks_policy_violation() -> None:
    records = _records()
    records[0] = _record(1, runtime_visible=True)
    report = _build(_apply_report(records))

    assert report["status"] == "blocked"
    assert "policy_not_retrieval_hint_only" in report["technicalBlockers"]
