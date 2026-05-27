from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_readback_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_BLOCKED_DUPLICATE,
    READBACK_STATUS_BLOCKED_MISSING,
    READBACK_STATUS_BLOCKED_POLICY_VIOLATION,
    READBACK_STATUS_VALIDATED,
    review_limited_visual_retrieval_hint_candidate_store_apply_readback,
)


SOURCE_APPLY_REPORT_REF = "eval/knowledgeos/reports/apply_fixture.v1.json"


def _hash() -> str:
    return "sha256:" + "c" * 64


def _candidate_record(index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
        "hintCandidateId": f"visual-retrieval-hint:sample-paper:table_region:{index}:abcdefabcdef{index:04d}",
        "sourceCandidateId": f"visual-layout:sample-paper:table_region:{index}:aaaaaaaaaaaa{index:04d}",
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "table_region",
        "sourceAttachmentRef": "eval/knowledgeos/reports/pack/assets/01.png",
        "derivedTextForRetrieval": "Retrieval hint only: a sample table.",
        "visibleText": "Visible fragments include: sample table.",
        "retrievalKeywords": ["sample", "table"],
        "uncertainty": "Low.",
        "limitations": "Retrieval hint only, not evidence.",
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "targetDerivedTextField": "derivedTextForRetrieval",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": runtime_visible,
            "indexEligible": False,
            "requiresFutureIndexEligibilityGate": True,
            "answerabilityGateBypassAllowed": False,
        },
        "provenance": {
            "sourceCandidateId": f"visual-layout:sample-paper:table_region:{index}:aaaaaaaaaaaa{index:04d}",
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
        },
    }


def _apply_report(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-apply-executor.v1",
        "status": "applied",
        "decision": "applied_limited_visual_retrieval_hint_candidate_store_apply",
        "counts": {
            "plannedApplyRows": len(records),
            "appliedCandidateRecordRows": len(records),
            "candidateStoreWriteRows": len(records),
            "readbackValidatedRows": len(records),
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "candidateRecords": records,
    }


def _write_store(papers_dir: Path, records: list[dict[str, object]]) -> None:
    store = papers_dir / "visual_retrieval_hints" / "visual_retrieval_hint_candidates.v1.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_apply_readback_review_validates_store_records(tmp_path: Path) -> None:
    records = [_candidate_record(1), _candidate_record(2)]
    papers_dir = tmp_path / "papers"
    _write_store(papers_dir, records)

    report = review_limited_visual_retrieval_hint_candidate_store_apply_readback(
        apply_report=_apply_report(records),
        source_apply_report_ref=SOURCE_APPLY_REPORT_REF,
        papers_dir=papers_dir,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == "ready_for_limited_visual_retrieval_hint_candidate_store_labs_vector_index_dry_run"
    assert report["counts"]["expectedCandidateRows"] == 2
    assert report["counts"]["storeRows"] == 2
    assert report["counts"]["readbackValidatedRows"] == 2
    assert report["counts"]["blockedRows"] == 0
    assert {row["readbackStatus"] for row in report["rows"]} == {READBACK_STATUS_VALIDATED}
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_apply_readback_review_blocks_missing_store_record(tmp_path: Path) -> None:
    expected = [_candidate_record(1), _candidate_record(2)]
    _write_store(tmp_path / "papers", expected[:1])

    report = review_limited_visual_retrieval_hint_candidate_store_apply_readback(
        apply_report=_apply_report(expected),
        source_apply_report_ref=SOURCE_APPLY_REPORT_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["missingStoreRows"] == 1
    assert report["rows"][1]["readbackStatus"] == READBACK_STATUS_BLOCKED_MISSING


def test_apply_readback_review_blocks_duplicate_store_record(tmp_path: Path) -> None:
    expected = [_candidate_record(1)]
    _write_store(tmp_path / "papers", [expected[0], expected[0]])

    report = review_limited_visual_retrieval_hint_candidate_store_apply_readback(
        apply_report=_apply_report(expected),
        source_apply_report_ref=SOURCE_APPLY_REPORT_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["duplicateStoreRows"] == 1
    assert report["rows"][0]["readbackStatus"] == READBACK_STATUS_BLOCKED_DUPLICATE


def test_apply_readback_review_blocks_policy_violation(tmp_path: Path) -> None:
    expected = [_candidate_record(1)]
    _write_store(tmp_path / "papers", [_candidate_record(1, runtime_visible=True)])

    report = review_limited_visual_retrieval_hint_candidate_store_apply_readback(
        apply_report=_apply_report(expected),
        source_apply_report_ref=SOURCE_APPLY_REPORT_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["policyViolationRows"] == 1
    assert report["rows"][0]["readbackStatus"] == READBACK_STATUS_BLOCKED_POLICY_VIOLATION
