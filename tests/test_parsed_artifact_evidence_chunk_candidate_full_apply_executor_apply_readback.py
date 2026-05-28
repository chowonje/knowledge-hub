from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback import (
    APPLIED_DECISION,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
    PENDING_APPLY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback,
    write_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate import (
    build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
)


DRY_RUN_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
CANARY_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.json"
READINESS_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate.v1.json"


def _candidate_row(index: int, *, paper_id: str | None = None, artifact_type: str = "paragraph") -> dict[str, object]:
    pid = paper_id or f"paper-{index:02d}"
    excerpt = (
        f"{index} Methods This parsed artifact evidence chunk candidate for {pid} "
        "contains enough text for full apply readback verification."
    )
    return {
        "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-row.v1",
        "candidateRowId": f"parsed-artifact-evidence-chunk:{pid}:{index}",
        "status": "candidate_ready",
        "paperId": pid,
        "sourceType": "paper",
        "artifactType": artifact_type,
        "sourceRef": f"papers_dir/parsed/{pid}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "locator": {
            "kind": "parsed_document_chars",
            "chars": {"start": index * 100, "end": index * 100 + len(excerpt), "basis": "parsed_document_text"},
            "page": index,
        },
        "spanLocator": f"chars:{index * 100}-{index * 100 + len(excerpt)}",
        "excerpt": excerpt,
        "snippetHash": "sha256:" + f"{index + 1:064d}"[-64:],
        "sectionTitle": "Methods" if artifact_type == "section" else "",
        "sectionPath": ["Page 1", "Methods"] if artifact_type == "section" else ["Page 1"],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "candidateOnly": True,
        "answerEvidenceCandidate": True,
        "answerabilityCandidate": True,
        "strictEvidence": False,
        "citationGrade": False,
        "runtimeEvidence": False,
        "answerVisible": False,
        "blockers": [],
    }


def _dry_run_report(rows: list[dict[str, object]] | None = None, *, status: str = "ready") -> dict[str, object]:
    candidate_rows = rows or [_candidate_row(index) for index in range(1, 13)]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        "status": status,
        "decision": CANDIDATE_DRY_RUN_READY_DECISION if status == "ready" else "blocked",
        "gate": {"passed": status == "ready"},
        "counts": {
            "selectedCandidateRows": len(candidate_rows),
            "candidateRows": len(candidate_rows),
            "heldCandidateRows": 0,
            "paragraphCandidateRows": len(candidate_rows),
            "sectionCandidateRows": 0,
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            **{field: 0 for field in ZERO_COUNTER_FIELDS},
        },
        "candidateRows": candidate_rows,
    }


def _reports(tmp_path: Path, *, rows: list[dict[str, object]] | None = None) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    papers_dir = tmp_path / "papers"
    dry_run = _dry_run_report(rows)
    canary = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=dry_run,
        source_candidate_dry_run_report_ref=DRY_RUN_REF,
        papers_dir=papers_dir,
        run_id="run-canary",
        apply=True,
        generated_at="2026-05-29T00:00:00Z",
    )
    readiness = build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate(
        candidate_dry_run_report=dry_run,
        source_candidate_dry_run_report_ref=DRY_RUN_REF,
        canary_apply_readback_report=canary,
        source_canary_apply_readback_report_ref=CANARY_REF,
        papers_dir=papers_dir,
        generated_at="2026-05-29T00:00:00Z",
    )
    assert canary["status"] == "applied"
    assert readiness["status"] == "ready"
    return dry_run, canary, readiness, papers_dir


def _build(
    *,
    dry_run: dict[str, object],
    canary: dict[str, object],
    readiness: dict[str, object],
    papers_dir: Path | None,
    apply: bool,
) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback(
        candidate_dry_run_report=dry_run,
        source_candidate_dry_run_report_ref=DRY_RUN_REF,
        full_apply_readiness_report=readiness,
        source_full_apply_readiness_report_ref=READINESS_REF,
        canary_apply_readback_report=canary,
        source_canary_apply_readback_report_ref=CANARY_REF,
        papers_dir=papers_dir,
        run_id="run-full",
        apply=apply,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_full_apply_executor_non_apply_plans_all_rows_and_writes_nothing(tmp_path: Path) -> None:
    dry_run, canary, readiness, papers_dir = _reports(tmp_path)

    report = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=False)

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == PENDING_APPLY_DECISION
    assert report["counts"]["inputRows"] == 12
    assert report["counts"]["plannedApplyRows"] == 12
    assert report["counts"]["reusedCanaryRecordRows"] == 10
    assert report["counts"]["dryRunCandidateRecordRows"] == 12
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["readbackValidatedRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["gate"]["readyForApply"] is True
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
        strict=True,
    ).ok


def test_full_apply_executor_apply_requires_papers_dir(tmp_path: Path) -> None:
    dry_run, canary, readiness, _papers_dir = _reports(tmp_path)

    report = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=None, apply=True)

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert "apply_requires_papers_dir" in report["warnings"]
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]


def test_full_apply_executor_apply_writes_remaining_records_and_is_idempotent(tmp_path: Path) -> None:
    dry_run, canary, readiness, papers_dir = _reports(tmp_path)

    first = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=True)
    second = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=True)

    assert first["status"] == "applied"
    assert first["decision"] == APPLIED_DECISION
    assert first["counts"]["candidateStoreWriteRows"] == 2
    assert first["counts"]["alreadyCorrectRows"] == 10
    assert first["counts"]["readbackValidatedRows"] == 12
    assert first["counts"]["runManifestWriteRows"] == 1
    assert first["counts"]["byExecutionStatus"] == {
        "already_correct_candidate_record": 10,
        "applied_candidate_record": 2,
    }
    assert second["status"] == "applied"
    assert second["counts"]["candidateStoreWriteRows"] == 0
    assert second["counts"]["alreadyCorrectRows"] == 12
    assert second["counts"]["readbackValidatedRows"] == 12
    assert second["counts"]["byExecutionStatus"] == {"already_correct_candidate_record": 12}
    manifest = json.loads(
        (papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "runs" / "run-full.json").read_text()
    )
    assert manifest["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID
    assert validate_payload(
        first,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_EXECUTOR_APPLY_READBACK_SCHEMA_ID,
        strict=True,
    ).ok


def test_full_apply_executor_blocks_bad_readiness_gate(tmp_path: Path) -> None:
    dry_run, canary, readiness, papers_dir = _reports(tmp_path)
    bad_readiness = copy.deepcopy(readiness)
    bad_readiness["status"] = "blocked"

    report = _build(dry_run=dry_run, canary=canary, readiness=bad_readiness, papers_dir=papers_dir, apply=False)

    assert report["status"] == "blocked"
    assert "full_apply_readiness_not_ready" in report["gate"]["schemaViolations"]
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_full_apply_executor_blocks_idempotency_conflict_without_overwrite(tmp_path: Path) -> None:
    dry_run, canary, readiness, papers_dir = _reports(tmp_path)
    first = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=True)
    record_path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "paper-11.jsonl"
    stored = json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])
    stored["excerpt"] = "conflicting stored payload"
    record_path.write_text(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    second = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=True)

    assert first["status"] == "applied"
    assert second["status"] == "blocked"
    assert second["counts"]["candidateStoreWriteRows"] == 0
    assert second["counts"]["blockedIdempotencyConflictRows"] == 1
    assert second["counts"]["readbackValidatedRows"] == 11
    assert json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])["excerpt"] == "conflicting stored payload"


def test_full_apply_executor_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    dry_run, canary, readiness, papers_dir = _reports(tmp_path)
    report = _build(dry_run=dry_run, canary=canary, readiness=readiness, papers_dir=papers_dir, apply=False)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback(
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
