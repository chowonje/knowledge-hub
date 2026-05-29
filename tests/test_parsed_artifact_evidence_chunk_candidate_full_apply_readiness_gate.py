from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    APPLIED_DECISION as CANARY_APPLIED_DECISION,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
    READINESS_STATUS_BLOCKED_MISSING_STORE,
    READINESS_STATUS_BLOCKED_POLICY_VIOLATION,
    READINESS_STATUS_VALIDATED,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
    write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
)


DRY_RUN_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
CANARY_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.json"


def _candidate_row(index: int, *, paper_id: str | None = None) -> dict[str, object]:
    pid = paper_id or f"paper-{index:02d}"
    return {
        "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-row.v1",
        "candidateRowId": f"parsed-artifact-evidence-chunk:{pid}:{index}",
        "paperId": pid,
        "artifactType": "paragraph",
        "sourceRef": f"papers_dir/parsed/{pid}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "spanLocator": f"chars:{index * 100}-{index * 100 + 120}",
        "excerpt": f"{index} Methods This is a long enough parsed artifact paragraph candidate for full apply readiness.",
        "snippetHash": "sha256:" + f"{index + 1:064d}"[-64:],
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
    }


def _candidate_record(row: dict[str, object], index: int, *, runtime_visible: bool = False) -> dict[str, object]:
    record = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:{row['paperId']}:paragraph:{index:04d}",
        "runId": "run-canary",
        "sourceDryRunReportRef": DRY_RUN_REF,
        "sourceCandidateRowId": row["candidateRowId"],
        "paperId": row["paperId"],
        "sourceType": "paper",
        "artifactType": row["artifactType"],
        "sourceRef": row["sourceRef"],
        "sourceContentHash": row["sourceContentHash"],
        "locator": {"kind": "parsed_document_chars", "chars": {"start": index * 100, "end": index * 100 + 120}},
        "spanLocator": row["spanLocator"],
        "excerpt": row["excerpt"],
        "snippetHash": row["snippetHash"],
        "sectionTitle": "",
        "sectionPath": ["Page 1"],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "idempotencyKey": f"parsed-artifact-evidence-chunk-candidate:key:{index:04d}",
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": runtime_visible,
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
    }
    return record


def _dry_run_report(*, status: str = "ready") -> dict[str, object]:
    rows = [_candidate_row(index) for index in range(1, 13)]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        "status": status,
        "decision": CANDIDATE_DRY_RUN_READY_DECISION if status == "ready" else "blocked",
        "gate": {"passed": status == "ready"},
        "counts": {
            "selectedCandidateRows": len(rows),
            "candidateRows": len(rows),
            "heldCandidateRows": 0,
            "paragraphCandidateRows": len(rows),
            "sectionCandidateRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            **{field: 0 for field in ZERO_COUNTER_FIELDS},
        },
        "candidateRows": rows,
    }


def _canary_report(dry_run: dict[str, object], *, status: str = "applied") -> dict[str, object]:
    rows = list(dry_run["candidateRows"])[:10]
    records = [_candidate_record(row, index) for index, row in enumerate(rows, start=1)]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
        "status": status,
        "decision": CANARY_APPLIED_DECISION if status == "applied" else "blocked",
        "counts": {
            "selectedCanaryRows": 10,
            "heldRows": 2,
            "candidateStoreWriteRows": 10,
            "alreadyCorrectRows": 0,
            "readbackValidatedRows": 10,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            "sourceSpanCreatedRows": 0,
            "strictEvidenceRows": 0,
            "citationGradeRows": 0,
            "runtimeEvidenceRows": 0,
            "answerVisibleRows": 0,
            "answerGenerationRows": 0,
            "databaseMutationRows": 0,
            "indexMutationRows": 0,
            "reindexOrReembedRows": 0,
            "parserExecutionRows": 0,
            "canonicalParsedArtifactWriteRows": 0,
            "vaultScanRows": 0,
            "externalDownloadRows": 0,
        },
        "candidateRecords": records,
    }


def _write_store(papers_dir: Path, records: list[dict[str, object]]) -> None:
    for record in records:
        path = (
            papers_dir
            / "structured_evidence_candidates"
            / "evidence_chunk"
            / f"{record['paperId']}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        path.write_text(
            "\n".join([*existing, json.dumps(record, ensure_ascii=False, sort_keys=True)]) + "\n",
            encoding="utf-8",
        )


def _build(tmp_path: Path, dry_run: dict[str, object], canary: dict[str, object]) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
        candidate_dry_run_report=dry_run,
        source_candidate_dry_run_report_ref=DRY_RUN_REF,
        canary_apply_readback_report=canary,
        source_canary_apply_readback_report_ref=CANARY_REF,
        papers_dir=tmp_path / "papers",
        generated_at="2026-05-29T00:00:00Z",
    )


def test_full_apply_readiness_gate_validates_canary_readback(tmp_path: Path) -> None:
    dry_run = _dry_run_report()
    canary = _canary_report(dry_run)
    _write_store(tmp_path / "papers", list(canary["candidateRecords"]))

    report = _build(tmp_path, dry_run, canary)

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["fullApplyCandidateRows"] == 12
    assert report["counts"]["canaryCandidateRows"] == 10
    assert report["counts"]["heldRows"] == 2
    assert report["counts"]["storeValidatedCanaryRows"] == 10
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["strictEvidenceRows"] == 0
    assert report["gate"]["readyForFullApplyExecutor"] is True
    assert report["gate"]["fullApplyPerformed"] is False
    assert {row["readinessStatus"] for row in report["rows"]} == {READINESS_STATUS_VALIDATED}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READINESS_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_full_apply_readiness_gate_blocks_missing_store_record(tmp_path: Path) -> None:
    dry_run = _dry_run_report()
    canary = _canary_report(dry_run)
    _write_store(tmp_path / "papers", list(canary["candidateRecords"])[:9])

    report = _build(tmp_path, dry_run, canary)

    assert report["status"] == "blocked"
    assert report["counts"]["missingStoreRows"] == 1
    assert report["rows"][-1]["readinessStatus"] == READINESS_STATUS_BLOCKED_MISSING_STORE


def test_full_apply_readiness_gate_blocks_policy_violation(tmp_path: Path) -> None:
    dry_run = _dry_run_report()
    canary = _canary_report(dry_run)
    records = copy.deepcopy(list(canary["candidateRecords"]))
    records[0]["answerVisible"] = True
    _write_store(tmp_path / "papers", records)

    report = _build(tmp_path, dry_run, canary)

    assert report["status"] == "blocked"
    assert report["counts"]["policyViolationRows"] == 1
    assert report["rows"][0]["readinessStatus"] == READINESS_STATUS_BLOCKED_POLICY_VIOLATION


def test_full_apply_readiness_gate_blocks_bad_source_reports(tmp_path: Path) -> None:
    dry_run = _dry_run_report(status="blocked")
    canary = _canary_report(_dry_run_report(), status="blocked")

    report = _build(tmp_path, dry_run, canary)

    assert report["status"] == "blocked"
    assert "candidate_dry_run_not_ready" in report["gate"]["schemaViolations"]
    assert "canary_apply_readback_not_applied" in report["gate"]["schemaViolations"]


def test_full_apply_readiness_gate_writer_outputs_sanitized_reports(tmp_path: Path) -> None:
    dry_run = _dry_run_report()
    canary = _canary_report(dry_run)
    _write_store(tmp_path / "papers", list(canary["candidateRecords"]))
    report = _build(tmp_path, dry_run, canary)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["input"]["sourceCandidateDryRunReportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
