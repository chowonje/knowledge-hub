from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback import (
    APPLIED_DECISION as FULL_APPLY_APPLIED_DECISION,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readback_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READY_DECISION as FULL_APPLY_READBACK_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_adapter_design import (
    ADAPTER_STATUS_READY_CANDIDATE_ONLY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
    READY_DECISION as ADAPTER_DESIGN_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run import (
    DRY_RUN_STATUS_READY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run,
    write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run,
)


ADAPTER_DESIGN_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_adapter_design.v1.json"
)
FULL_APPLY_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback.v1.json"
)
READBACK_REVIEW_REF = (
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readback_review.v1.json"
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _design_row(index: int, *, paper_id: str, artifact_type: str = "paragraph") -> dict[str, object]:
    start = index * 100
    end = start + 80
    excerpt = f"Evidence chunk excerpt {index} for {paper_id}. " + ("supporting text " * 12)
    snippet_hash = _sha256_text(excerpt.strip())
    source_hash = "sha256:" + f"{index:064d}"[-64:]
    source_ref = f"papers_dir/parsed/{paper_id}/document.md"
    candidate_store_ref = f"papers_dir/structured_evidence_candidates/evidence_chunk/{paper_id}.jsonl"
    candidate_record_id = f"parsed-artifact-evidence-chunk-candidate:{paper_id}:{artifact_type}:{index}"
    return {
        "adapterDesignRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-adapter-design:{index:04d}",
        "sourceContractReviewRowId": f"parsed-artifact-evidence-chunk-candidate-runtime-contract-review:{index:04d}",
        "candidateRecordId": candidate_record_id,
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "paperId": paper_id,
        "artifactType": artifact_type,
        "sourceRef": source_ref,
        "sourceContentHash": source_hash,
        "spanLocator": f"chars:{start}-{end}",
        "charStart": start,
        "charEnd": end,
        "snippetHash": snippet_hash,
        "candidateStoreRef": candidate_store_ref,
        "adapterDesignStatus": ADAPTER_STATUS_READY_CANDIDATE_ONLY,
        "adapterDesignBlockers": [],
        "adapterDesignReady": True,
        "futureAdapterCandidate": True,
        "futureCandidateStoreReadRequired": True,
        "futureRuntimeEvidenceAllowed": False,
        "futureAnswerVisibleAllowed": False,
        "futureAnswerabilityAllowed": False,
        "candidateOnly": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "answerable": False,
        "plannedAdapterInputs": {},
        "plannedEvidenceItemShape": {},
        "plannedDiagnosticsShape": {},
        "checks": {},
        "_fixtureExcerpt": excerpt.strip(),
    }


def _record_from_row(row: dict[str, object]) -> dict[str, object]:
    start = int(row["charStart"])
    end = int(row["charEnd"])
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": row["candidateRecordId"],
        "runId": "test-run",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json",
        "sourceCandidateRowId": row["sourceCandidateRowId"],
        "paperId": row["paperId"],
        "sourceType": "paper",
        "artifactType": row["artifactType"],
        "sourceRef": row["sourceRef"],
        "sourceContentHash": row["sourceContentHash"],
        "locator": {
            "kind": "parsed_document_chars",
            "chars": {"start": start, "end": end, "basis": "parsed_document_text"},
            "page": 1,
        },
        "spanLocator": row["spanLocator"],
        "excerpt": row["_fixtureExcerpt"],
        "snippetHash": row["snippetHash"],
        "sectionTitle": "",
        "sectionPath": ["Page 1"],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "idempotencyKey": f"idempotency:{row['candidateRecordId']}",
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "evidenceTier": "parsed_artifact_evidence_chunk_candidate_only",
        "strictBlockers": ["candidate_store_record_not_strict_evidence"],
        "writePolicy": {
            "candidateStoreWrite": True,
            "sourceSpanCreated": False,
            "strictEvidenceCreated": False,
            "citationGradeEvidenceCreated": False,
            "runtimeEvidenceCreated": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "databaseMutation": False,
            "vaultScan": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
        "candidateRecordHash": "sha256:" + "a" * 64,
    }


def _rows() -> list[dict[str, object]]:
    return [
        _design_row(1, paper_id="paper-a"),
        _design_row(2, paper_id="paper-a"),
        _design_row(3, paper_id="paper-a"),
        _design_row(4, paper_id="paper-b", artifact_type="section"),
        _design_row(5, paper_id="paper-b"),
        _design_row(6, paper_id="paper-b"),
    ]


def _adapter_design_report(rows: list[dict[str, object]], *, status: str = "ready") -> dict[str, object]:
    ready = status == "ready"
    public_rows = [{k: v for k, v in row.items() if k != "_fixtureExcerpt"} for row in rows]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DESIGN_SCHEMA_ID,
        "status": status,
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": ADAPTER_DESIGN_READY_DECISION if ready else "blocked",
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run",
        "sourceRuntimeContractReview": {
            "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-contract-review.v1",
            "status": "ready",
            "decision": "parsed_artifact_evidence_chunk_candidate_runtime_contract_review_ready",
            "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_runtime_adapter_design",
            "reportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_runtime_contract_review.v1.json",
            "inputRows": len(rows),
            "contractReviewReadyRows": len(rows) if ready else 0,
            "runtimeAdapterDesignCandidateRows": len(rows) if ready else 0,
            "answerableRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "policy": {},
        "adapterDesign": {},
        "counts": {
            "inputRows": len(rows),
            "adapterDesignReadyInputRows": len(rows) if ready else 0,
            "contractReadyInputRows": len(rows) if ready else 0,
            "runtimeAdapterDesignReadyRows": len(rows) if ready else 0,
            "futureAdapterCandidateRows": len(rows) if ready else 0,
            "plannedCandidateStoreReadRows": len(rows) if ready else 0,
            "plannedEvidenceItemShapeRows": len(rows) if ready else 0,
            "plannedDiagnosticsRows": len(rows) if ready else 0,
            "answerableRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {"readyForRuntimeAdapterDryRun": ready},
        "rows": public_rows,
        "warnings": [],
    }


def _full_apply_report(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
        "status": "applied",
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": FULL_APPLY_APPLIED_DECISION,
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_full_apply_readback_review",
        "counts": {
            "plannedApplyRows": len(records),
            "appliedCandidateRecordRows": len(records),
            "alreadyCorrectRows": 0,
            "candidateStoreWriteRows": len(records),
            "readbackValidatedRows": len(records),
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "candidateRecords": records,
    }


def _readback_review(rows: int) -> dict[str, object]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
        "status": "ready",
        "generatedAt": "2026-05-29T00:00:00Z",
        "decision": FULL_APPLY_READBACK_REVIEW_READY_DECISION,
        "nextRecommendedTranche": "parsed_artifact_evidence_chunk_candidate_answerability_policy_gate",
        "counts": {
            "expectedCandidateRows": rows,
            "storeRows": rows,
            "matchingStoreRows": rows,
            "readbackValidatedRows": rows,
            "candidateStoreWriteRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "rows": [],
    }


def _build(
    *,
    rows: list[dict[str, object]] | None = None,
    records: list[dict[str, object]] | None = None,
    adapter_status: str = "ready",
) -> dict[str, object]:
    design_rows = rows or _rows()
    candidate_records = records if records is not None else [_record_from_row(row) for row in design_rows]
    return build_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
        runtime_adapter_design_report=_adapter_design_report(design_rows, status=adapter_status),
        full_apply_report=_full_apply_report(candidate_records),
        full_apply_readback_review_report=_readback_review(len(design_rows)),
        source_runtime_adapter_design_report_ref=ADAPTER_DESIGN_REF,
        source_full_apply_report_ref=FULL_APPLY_REF,
        source_full_apply_readback_review_report_ref=READBACK_REVIEW_REF,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_runtime_adapter_dry_run_selects_bounded_opt_in_rows() -> None:
    report = _build()

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["inputRows"] == 6
    assert report["counts"]["adapterDryRunReadyRows"] == 6
    assert report["counts"]["candidateStoreRecordMatchedRows"] == 6
    assert report["counts"]["excerptReadbackRows"] == 6
    assert report["counts"]["positiveScenarioSelectedRows"] == 4
    assert report["counts"]["answerableRows"] == 0
    assert report["counts"]["runtimeEvidenceRows"] == 0
    assert report["gate"]["readyForRuntimeAdapterImplementation"] is True
    selected = [row for row in report["rows"] if row["selectedForPositiveScenario"]]
    assert len(selected) == 4
    assert {row["adapterDryRunStatus"] for row in report["rows"]} == {DRY_RUN_STATUS_READY}
    assert max(
        sum(1 for row in selected if row["paperId"] == paper_id)
        for paper_id in {row["paperId"] for row in selected}
    ) == 2
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RUNTIME_ADAPTER_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_runtime_adapter_dry_run_skips_unsupported_scenarios() -> None:
    report = _build()

    scenarios = {item["scenarioId"]: item for item in report["scenarioResults"]}
    assert scenarios["opt_in_off"]["status"] == "skipped"
    assert scenarios["opt_in_off"]["rowsAdded"] == 0
    assert scenarios["opt_in_missing_resolved_paper_ids"]["skippedReason"] == "resolved_paper_ids_required"
    assert scenarios["source_type_not_paper"]["skippedReason"] == "source_type_not_paper"
    assert scenarios["opt_in_paper_resolved"]["status"] == "applied_dry_run"
    assert scenarios["opt_in_paper_resolved"]["rowsAdded"] == 4


def test_runtime_adapter_dry_run_blocks_non_ready_design_report() -> None:
    report = _build(adapter_status="blocked")

    assert report["status"] == "blocked"
    assert "runtime_adapter_design_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["adapterDryRunReadyRows"] == 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 6


def test_runtime_adapter_dry_run_blocks_missing_candidate_record() -> None:
    rows = _rows()
    records = [_record_from_row(row) for row in rows[:-1]]

    report = _build(rows=rows, records=records)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingCandidateRecordRows"] == 1
    assert "candidate_store_record_missing" in report["rows"][-1]["adapterDryRunBlockers"]


def test_runtime_adapter_dry_run_blocks_excerpt_hash_mismatch() -> None:
    rows = _rows()
    records = [_record_from_row(row) for row in rows]
    records[0]["excerpt"] = "changed excerpt"

    report = _build(rows=rows, records=records)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedRecordMismatchRows"] == 1
    assert "candidate_record_excerpt_hash_mismatch" in report["rows"][0]["adapterDryRunBlockers"]


def test_runtime_adapter_dry_run_blocks_locator_mismatch() -> None:
    rows = _rows()
    records = [_record_from_row(row) for row in rows]
    records[0]["locator"]["chars"]["start"] = 1

    report = _build(rows=rows, records=records)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedInvalidLocatorRows"] == 1
    assert "candidate_record_locator_mismatch" in report["rows"][0]["adapterDryRunBlockers"]


def test_runtime_adapter_dry_run_blocks_policy_violation() -> None:
    rows = _rows()
    records = [_record_from_row(row) for row in rows]
    records[0]["answerVisible"] = True

    report = _build(rows=rows, records=records)

    assert report["status"] == "blocked"
    assert report["counts"]["blockedPolicyQuarantineRows"] == 1
    assert "candidate_record_policy_quarantine_violation" in report["rows"][0]["adapterDryRunBlockers"]


def test_runtime_adapter_dry_run_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = _build()
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceRuntimeAdapterDesign"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
