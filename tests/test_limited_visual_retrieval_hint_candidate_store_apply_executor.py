from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers import limited_visual_retrieval_hint_candidate_store_apply_executor as executor
from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT,
    EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH,
    EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION,
    EXECUTOR_STATUS_DRY_RUN_READY,
    LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
    execute_limited_visual_retrieval_hint_candidate_store_apply_executor,
    write_limited_visual_retrieval_hint_candidate_store_apply_executor,
)


PLANNED_STORE_REF = "papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl"
SOURCE_REPORT_REF = "eval/knowledgeos/reports/apply_executor_dry_run_fixture.v1.json"


def _hash() -> str:
    return "sha256:" + "b" * 64


def _hint_id(index: int) -> str:
    return f"visual-retrieval-hint:sample-paper:figure_caption_region:{index}:abcdefabcdef{index:04d}"


def _source_id(index: int) -> str:
    return f"visual-layout:sample-paper:figure_caption_region:{index}:aaaaaaaaaaaa{index:04d}"


def _canonical_hash(record: dict[str, object]) -> str:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


def _idempotency_key(record: dict[str, object]) -> str:
    basis = "|".join(
        [
            str(record["hintCandidateId"]),
            str(record["sourceCandidateId"]),
            str(record["sourceContentHash"]),
            str(record["page"]),
            json.dumps(record["bbox"], ensure_ascii=True, sort_keys=True),
        ]
    )
    return "visual-retrieval-hint-idempotency:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _candidate_record(index: int, *, unsafe_policy: bool = False) -> dict[str, object]:
    return {
        "schema": "knowledge-hub.paper.visual-retrieval-hint-candidate-row.v1",
        "hintCandidateId": _hint_id(index),
        "sourceCandidateId": _source_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "sourceAttachmentRef": "eval/knowledgeos/reports/pack/assets/01.png",
        "derivedTextForRetrieval": "Retrieval hint only: a ReLU tanh CIFAR-10 training error plot.",
        "visibleText": "Visible fragments include: ReLU, tanh, CIFAR-10, training error.",
        "retrievalKeywords": ["ReLU", "tanh", "CIFAR-10", "training error"],
        "uncertainty": "Low.",
        "limitations": "Retrieval hint only, not evidence.",
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "targetDerivedTextField": "derivedTextForRetrieval",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": unsafe_policy,
            "indexEligible": False,
            "requiresFutureIndexEligibilityGate": True,
            "answerabilityGateBypassAllowed": False,
        },
        "provenance": {
            "sourceCandidateId": _source_id(index),
            "sourceContentHash": _hash(),
            "page": index,
            "bbox": [10.0, 20.0, 120.0, 180.0],
        },
    }


def _source_row(index: int, *, unsafe_policy: bool = False, holdout: bool = False) -> dict[str, object]:
    record = _candidate_record(index, unsafe_policy=unsafe_policy)
    ready = not holdout
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-allowlist-apply-executor-dry-run-row.v1",
        "executorDryRunRowId": f"limited-visual-retrieval-hint-allowlist-apply-executor-dry-run:fixture{index}",
        "sourceReviewRowId": f"limited-visual-retrieval-hint-apply-allowlist-review:fixture{index}",
        "sourceApplyDesignRowId": f"limited-visual-retrieval-hint-apply-design:fixture{index}",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/dry_run_fixture.v1.json",
        "sourceDryRunRowId": f"visual-retrieval-hint-dry-run:fixture{index}",
        "hintCandidateId": _hint_id(index),
        "sourceCandidateId": _source_id(index),
        "paperId": "sample-paper",
        "paperRef": "papers_dir/sample.pdf",
        "sourceContentHash": _hash(),
        "page": index,
        "bbox": [10.0, 20.0, 120.0, 180.0],
        "candidateType": "figure_caption_region",
        "plannedStoreRef": PLANNED_STORE_REF,
        "idempotencyKey": _idempotency_key(record),
        "plannedJsonlRecordSha256": _canonical_hash(record),
        "recomputedPlannedJsonlRecordSha256": _canonical_hash(record),
        "plannedJsonlRecordPreview": record,
        "checks": {},
        "executorDryRunResult": {
            "wouldWriteOnSeparateExplicitApply": ready,
            "actualStoreWrite": False,
            "candidateStoreWrite": False,
            "jsonlSerializable": ready,
            "policyCompliant": ready,
            "applyExecutorPreviewOnly": True,
            "indexEligible": False,
            "runtimeVisible": False,
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
        },
        "policy": {
            "allowedUse": "retrieval_hint_only",
            "strictEvidence": False,
            "citationGrade": False,
            "answerableWithoutTextEvidence": False,
            "runtimeVisible": False,
            "indexEligible": False,
            "answerabilityGateBypassAllowed": False,
        },
        "provenance": {},
        "blockerReason": "holdout_pending_search_gate_review" if holdout else "",
    }


def _source_report(row_count: int = 1, *, unsafe_policy: bool = False, include_holdout: bool = False) -> dict[str, object]:
    rows = [_source_row(index + 1, unsafe_policy=unsafe_policy) for index in range(row_count)]
    excluded = []
    if include_holdout:
        rows.append(_source_row(row_count + 1, holdout=True))
        excluded.append(_hint_id(row_count + 1))
    return {
        "schema": "knowledge-hub.paper.limited-visual-retrieval-hint-candidate-store-allowlist-apply-executor-dry-run.v1",
        "status": "ready",
        "decision": "ready_for_limited_visual_retrieval_hint_candidate_store_apply_executor_review",
        "counts": {
            "executorDryRunRows": row_count,
            "plannedWriteRows": row_count,
            "candidateStoreWriteRows": 0,
            "candidateStoreApplyRows": 0,
            "blockedRows": 0,
            "excludedHoldoutRows": len(excluded),
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "executorDryRunRowsDetail": rows,
        "excludedHoldoutHintCandidateIds": excluded,
    }


def _read_store(papers_dir: Path) -> list[dict[str, object]]:
    path = papers_dir / "visual_retrieval_hints" / "visual_retrieval_hint_candidates.v1.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_apply_executor_dry_run_writes_nothing(tmp_path: Path) -> None:
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-dry",
        apply=False,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["readbackValidatedRows"] == 0
    assert report["rows"][0]["executionStatus"] == EXECUTOR_STATUS_DRY_RUN_READY
    assert not (tmp_path / "papers" / "visual_retrieval_hints").exists()
    assert validate_payload(
        report,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
        strict=True,
    ).ok


def test_apply_executor_apply_requires_papers_dir() -> None:
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        run_id="run-missing-dir",
        apply=True,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert "apply_requires_papers_dir" in report["warnings"]
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]


def test_apply_executor_writes_125_jsonl_records_idempotently(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    first = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(125),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
        generated_at="2026-05-27T00:00:00Z",
    )
    second = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(125),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
        generated_at="2026-05-27T00:00:00Z",
    )
    records = _read_store(papers_dir)
    manifest = json.loads((papers_dir / "visual_retrieval_hints" / "runs" / "run-apply.json").read_text())

    assert first["status"] == "applied"
    assert first["counts"]["appliedCandidateRecordRows"] == 125
    assert first["counts"]["candidateStoreWriteRows"] == 125
    assert first["counts"]["readbackValidatedRows"] == 125
    assert first["counts"]["blockedRows"] == 0
    assert {row["executionStatus"] for row in first["rows"]} == {EXECUTOR_STATUS_APPLIED}
    assert second["counts"]["readbackValidatedRows"] == 125
    assert len(records) == 125
    assert len({_idempotency_key(record) for record in records}) == 125
    assert manifest["schema"] == LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID
    assert validate_payload(
        first,
        LIMITED_VISUAL_RETRIEVAL_HINT_CANDIDATE_STORE_APPLY_EXECUTOR_SCHEMA_ID,
        strict=True,
    ).ok


def test_apply_executor_blocks_readback_mismatch(monkeypatch, tmp_path: Path) -> None:
    def _fake_apply_records(records: list[dict[str, object]], *, papers_dir: str | Path) -> tuple[int, int, list[str]]:
        return len(records), 0, [f"readback_mismatch:{records[0]['hintCandidateId']}"]

    monkeypatch.setattr(executor, "_apply_records", _fake_apply_records)
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-readback-mismatch",
        apply=True,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedReadbackMismatchRows"] == 1
    assert report["rows"][0]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_READBACK_MISMATCH


def test_apply_executor_blocks_unsafe_policy_row() -> None:
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1, unsafe_policy=True),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        run_id="run-unsafe",
        apply=False,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateInputRows"] == 0
    assert report["rows"][0]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_SCHEMA_VIOLATION
    assert "recordPolicyRetrievalHintOnly" in report["rows"][0]["executionBlockers"]


def test_apply_executor_ignores_holdout_row() -> None:
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1, include_holdout=True),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        run_id="run-holdout",
        apply=False,
        generated_at="2026-05-27T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["candidateInputRows"] == 1
    assert report["counts"]["heldInputRows"] == 1
    assert report["counts"]["plannedApplyRows"] == 1
    assert report["counts"]["blockedNonReadyInputRows"] == 1
    assert report["rows"][1]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_NON_READY_INPUT


def test_apply_executor_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = execute_limited_visual_retrieval_hint_candidate_store_apply_executor(
        apply_executor_dry_run_report=_source_report(1),
        source_apply_executor_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-writer",
        apply=False,
        generated_at="2026-05-27T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_limited_visual_retrieval_hint_candidate_store_apply_executor(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceApplyExecutorDryRunReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
