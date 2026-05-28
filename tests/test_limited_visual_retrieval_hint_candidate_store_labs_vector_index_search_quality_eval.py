from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_executor import (
    LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
    LOCAL_EMBEDDING_DIMENSIONS,
    LOCAL_EMBEDDING_MODEL_REF,
    LOCAL_EMBEDDING_PROVIDER_REF,
    _hashing_vector,
    _vector_hash,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run import (
    LABS_NAMESPACE,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
    READY_DECISION,
    build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval,
)


SOURCE_LAYOUT_REF = "eval/knowledgeos/reports/visual_layout_candidate_list_report_fixture.v1.json"
SOURCE_APPLY_REF = "eval/knowledgeos/reports/labs_vector_apply_fixture.v1.json"
SOURCE_DRY_RUN_REF = "eval/knowledgeos/reports/labs_vector_apply_dry_run_fixture.v1.json"


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash() -> str:
    return "sha256:" + "c" * 64


def _source_candidate_id(index: int) -> str:
    return f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"


def _hint_candidate_id(index: int) -> str:
    return f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:bbbbbbbbbbbb{index:04d}"


def _upsert_record(index: int = 1) -> dict[str, object]:
    document_text = "Retrieval hint only: ReLU CIFAR-10 training error plot with tanh comparison."
    embedding_text = (
        "allowed_use=retrieval_hint_only | paper=sample-paper | type=figure_caption_region | "
        f"page={index} | keywords=ReLU, CIFAR-10, training error, tanh | {document_text}"
    )
    bbox = [10.0, 20.0, 120.0, 180.0]
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-labs-vector-upsert-record.v1",
        "namespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": f"papers_dir/visual_retrieval_hints/labs_vector_index/{LABS_NAMESPACE}.v1.jsonl",
        "vectorDocumentId": f"visual-retrieval-hint-vector-doc:{index:04d}",
        "hintCandidateId": _hint_candidate_id(index),
        "sourceCandidateId": _source_candidate_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": bbox,
        "candidateType": "figure_caption_region",
        "sourceRecordSha256": _hash(),
        "documentText": document_text,
        "documentTextHash": _hash_text(document_text),
        "embeddingText": embedding_text,
        "embeddingTextHash": _hash_text(embedding_text),
        "metadata": {
            "retrieval_unit_schema": "visual_retrieval_hint_vector_document.v1",
            "namespace": LABS_NAMESPACE,
            "allowedUse": "retrieval_hint_only",
            "hintCandidateId": _hint_candidate_id(index),
            "sourceCandidateId": _source_candidate_id(index),
            "paperId": "sample-paper",
            "paperRef": "papers_dir/sample.pdf",
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": bbox,
            "candidateType": "figure_caption_region",
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
            "labsOnly": True,
        },
        "executionPlan": {},
    }


def _vector_record(upsert_record: dict[str, object], *, runtime_visible: bool = False) -> dict[str, object]:
    vector = _hashing_vector(str(upsert_record["embeddingText"]))
    policy = dict(upsert_record["policy"])
    policy["runtimeVisible"] = runtime_visible
    return {
        "schema": LABS_VECTOR_INDEX_RECORD_SCHEMA_ID,
        "namespace": LABS_NAMESPACE,
        "plannedVectorIndexRef": upsert_record["plannedVectorIndexRef"],
        "vectorDocumentId": upsert_record["vectorDocumentId"],
        "hintCandidateId": upsert_record["hintCandidateId"],
        "sourceCandidateId": upsert_record["sourceCandidateId"],
        "paperId": upsert_record["paperId"],
        "paperRef": upsert_record["paperRef"],
        "sourceContentHash": upsert_record["sourceContentHash"],
        "page": upsert_record["page"],
        "bbox": upsert_record["bbox"],
        "candidateType": upsert_record["candidateType"],
        "documentTextHash": upsert_record["documentTextHash"],
        "embeddingTextHash": upsert_record["embeddingTextHash"],
        "embeddingProviderRef": LOCAL_EMBEDDING_PROVIDER_REF,
        "embeddingModelRef": LOCAL_EMBEDDING_MODEL_REF,
        "embeddingDimensions": LOCAL_EMBEDDING_DIMENSIONS,
        "embeddingVector": vector,
        "embeddingVectorSha256": _vector_hash(vector),
        "sourceUpsertRecordSha256": _hash(),
        "sourceApplyReviewRowId": "apply-review:0001",
        "sourceExecutorDryRunRowId": "executor-dry-run:0001",
        "metadata": upsert_record["metadata"],
        "policy": policy,
        "execution": {
            "externalEmbeddingCall": False,
            "localEmbeddingComputed": True,
            "labsVectorIndexWrite": True,
            "productionVectorIndexWrite": False,
        },
    }


def _write_index(papers_dir: Path, rows: list[dict[str, object]]) -> None:
    path = papers_dir / "visual_retrieval_hints" / "labs_vector_index" / f"{LABS_NAMESPACE}.v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _layout_row(candidate_id: str, nearby_text: str, index: int = 1) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-layout-candidate-row.v1",
        "candidateId": candidate_id,
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "textContext": {
            "nearbyText": nearby_text,
            "captionText": "",
            "headingPath": ["Experiments"],
        },
        "visualContext": {},
        "retrievalHintPlan": {},
        "provenance": {},
        "blockerReason": "",
    }


def _layout_report() -> dict[str, object]:
    rows = [
        _layout_row(_source_candidate_id(1), "A generic caption area with implementation notes.", 1),
    ]
    for index in range(2, 10):
        rows.append(
            _layout_row(
                _source_candidate_id(index),
                "Distractor text repeats ReLU CIFAR-10 training error and tanh comparison.",
                index,
            )
        )
    return {
        "schema": "knowledge-hub.paper.visual-layout-candidate-list-report.v1",
        "status": "ready",
        "candidateRowsDetail": rows,
    }


def _apply_report(count: int = 1) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor.v1",
        "status": "applied",
        "decision": "applied_limited_visual_retrieval_hint_candidate_store_labs_vector_index",
        "counts": {
            "appliedLabsVectorRecordRows": count,
            "readbackValidatedRows": count,
            "vectorIndexWriteRows": count,
            "productionVectorIndexWriteRows": 0,
            "candidateStoreWriteRows": 0,
            "externalEmbeddingCallRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "indexEligibleRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
    }


def _dry_run_report(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-labs-vector-index-apply-executor-dry-run.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_apply_review",
        "counts": {
            "executorDryRunRows": len(records),
            "plannedVectorUpsertRows": len(records),
            "embeddingInputRows": len(records),
            "candidateStoreWriteRows": 0,
            "embeddingCallRows": 0,
            "embeddingVectorWriteRows": 0,
            "vectorIndexWriteRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "indexEligibleRows": 0,
            "runtimeVisibleRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "answerableWithoutTextEvidenceRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "plannedVectorUpsertRecords": records,
    }


def _build(papers_dir: Path, records: list[dict[str, object]]) -> dict[str, object]:
    return build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
        layout_candidate_report=_layout_report(),
        labs_vector_index_apply_report=_apply_report(len(records)),
        labs_vector_index_apply_executor_dry_run=_dry_run_report(records),
        source_layout_candidate_report_ref=SOURCE_LAYOUT_REF,
        source_labs_vector_index_apply_report_ref=SOURCE_APPLY_REF,
        source_labs_vector_index_apply_executor_dry_run_ref=SOURCE_DRY_RUN_REF,
        papers_dir=papers_dir,
        min_labs_hit_at5_rows=2,
        min_hybrid_hit_at5_lift_rows=2,
        generated_at="2026-05-28T00:00:00Z",
    )


def test_labs_vector_index_search_quality_eval_uses_actual_index_and_measures_lift(tmp_path: Path) -> None:
    upsert = _upsert_record(1)
    _write_index(tmp_path / "papers", [_vector_record(upsert)])

    report = _build(tmp_path / "papers", [upsert])

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["actualLabsVectorIndexRows"] == 1
    assert report["counts"]["matchedVectorRecordRows"] == 1
    assert report["counts"]["queryRows"] == 2
    assert report["counts"]["textOnlyHitAt5Rows"] == 0
    assert report["counts"]["labsVectorHitAt5Rows"] == 2
    assert report["counts"]["hybridHitAt5Rows"] == 2
    assert report["qualityGate"]["passed"] is True
    assert report["scope"]["productionVectorIndexWriteRows"] == 0
    assert report["scope"]["operationalSearchIndexQueryRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0

    validation = validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_LABS_VECTOR_INDEX_SEARCH_QUALITY_EVAL_SCHEMA_ID,
        strict=True,
    )
    assert validation.ok, validation.errors


def test_labs_vector_index_search_quality_eval_blocks_missing_index_row(tmp_path: Path) -> None:
    upsert = _upsert_record(1)
    _write_index(tmp_path / "papers", [])

    report = _build(tmp_path / "papers", [upsert])

    assert report["status"] == "blocked"
    assert report["counts"]["matchedVectorRecordRows"] == 0
    assert report["counts"]["vectorContractViolationRows"] == 1
    assert "actual_labs_vector_index_rows_missing" in report["technicalBlockers"]


def test_labs_vector_index_search_quality_eval_blocks_unsafe_policy_row(tmp_path: Path) -> None:
    upsert = _upsert_record(1)
    _write_index(tmp_path / "papers", [_vector_record(upsert, runtime_visible=True)])

    report = _build(tmp_path / "papers", [upsert])

    assert report["status"] == "blocked"
    assert report["counts"]["matchedVectorRecordRows"] == 0
    assert report["counts"]["vectorContractViolationRows"] == 1
    assert "rowPolicyRetrievalHintOnly" in report["vectorContractRows"][0]["contractBlockers"]


def test_labs_vector_index_search_quality_eval_blocks_non_applied_source_report(tmp_path: Path) -> None:
    upsert = _upsert_record(1)
    _write_index(tmp_path / "papers", [_vector_record(upsert)])
    apply_report = _apply_report(1)
    apply_report["status"] = "ready"

    report = build_limited_visual_retrieval_hint_candidate_store_labs_vector_index_search_quality_eval(
        layout_candidate_report=_layout_report(),
        labs_vector_index_apply_report=apply_report,
        labs_vector_index_apply_executor_dry_run=_dry_run_report([upsert]),
        source_layout_candidate_report_ref=SOURCE_LAYOUT_REF,
        source_labs_vector_index_apply_report_ref=SOURCE_APPLY_REF,
        source_labs_vector_index_apply_executor_dry_run_ref=SOURCE_DRY_RUN_REF,
        papers_dir=tmp_path / "papers",
        min_labs_hit_at5_rows=2,
        min_hybrid_hit_at5_lift_rows=2,
        generated_at="2026-05-28T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "labs_vector_index_apply_executor_not_applied" in report["technicalBlockers"]
