from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    APPLIED_DECISION,
    EXECUTOR_STATUS_ALREADY_CORRECT,
    EXECUTOR_STATUS_APPLIED,
    EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT,
    EXECUTOR_STATUS_DRY_RUN_READY,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
    PENDING_APPLY_DECISION,
    build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback,
    write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_dry_run import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
    READY_DECISION as CANDIDATE_DRY_RUN_READY_DECISION,
    ZERO_COUNTER_FIELDS,
)


SOURCE_REPORT_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"


def _candidate_row(paper_id: str, index: int, *, artifact_type: str = "paragraph") -> dict[str, object]:
    excerpt = (
        f"{index} Methods This parsed artifact candidate for {paper_id} contains enough text "
        "to support a conservative evidence chunk canary readback test."
    )
    return {
        "schema": "knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-row.v1",
        "candidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "status": "candidate_ready",
        "paperId": paper_id,
        "sourceType": "paper",
        "artifactType": artifact_type,
        "sourceRef": f"papers_dir/parsed/{paper_id}/document.md",
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
    candidate_rows = rows or [_candidate_row("2005.11401", 1)]
    zero_counts = {field: 0 for field in ZERO_COUNTER_FIELDS}
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        "status": status,
        "decision": CANDIDATE_DRY_RUN_READY_DECISION if status == "ready" else "blocked",
        "gate": {"passed": status == "ready"},
        "counts": {
            "selectedCandidateRows": len(candidate_rows),
            "blockedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
            **zero_counts,
        },
        "candidateRows": candidate_rows,
    }


def _read_records(papers_dir: Path, paper_id: str) -> list[dict[str, object]]:
    path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / f"{paper_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_canary_apply_readback_non_apply_selects_10_and_writes_nothing(tmp_path: Path) -> None:
    preferred = [
        "2005.11401",
        "1706.03762",
        "1512.03385",
        "1810.04805",
        "2005.14165",
        "2010.11929",
        "1406.2661",
        "2404.16130",
        "2410.05779",
    ]
    rows = [_candidate_row(paper_id, index + 1) for index, paper_id in enumerate(preferred)]
    rows.extend([_candidate_row("1207.0580", 20), _candidate_row("1301.3781", 21)])

    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report(rows),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-dry",
        apply=False,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == PENDING_APPLY_DECISION
    assert report["counts"]["inputRows"] == 11
    assert report["counts"]["selectedCanaryRows"] == 10
    assert report["counts"]["heldRows"] == 1
    assert report["counts"]["preferredCanaryPaperRows"] == 9
    assert report["counts"]["topUpCanaryRows"] == 1
    assert report["counts"]["missingPreferredCanaryPaperRows"] == 1
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["readbackValidatedRows"] == 0
    assert {row["executionStatus"] for row in report["rows"]} == {EXECUTOR_STATUS_DRY_RUN_READY}
    assert not (tmp_path / "papers" / "structured_evidence_candidates").exists()
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
        strict=True,
    ).ok


def test_canary_apply_readback_apply_requires_papers_dir() -> None:
    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report([_candidate_row("2005.11401", 1)]),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        run_id="run-missing-dir",
        apply=True,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert "apply_requires_papers_dir" in report["warnings"]
    assert "apply_requires_papers_dir" in report["gate"]["schemaViolations"]


def test_canary_apply_readback_writes_jsonl_records_idempotently(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    rows = [_candidate_row("2005.11401", 1), _candidate_row("1706.03762", 2, artifact_type="section")]

    first = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report(rows),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
        canary_record_limit=2,
        generated_at="2026-05-29T00:00:00Z",
    )
    second = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report(rows),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-apply",
        apply=True,
        canary_record_limit=2,
        generated_at="2026-05-29T00:00:00Z",
    )

    records = _read_records(papers_dir, "2005.11401") + _read_records(papers_dir, "1706.03762")
    manifest = json.loads(
        (papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "runs" / "run-apply.json").read_text()
    )

    assert first["status"] == "applied"
    assert first["decision"] == APPLIED_DECISION
    assert first["counts"]["candidateStoreWriteRows"] == 2
    assert first["counts"]["readbackValidatedRows"] == 2
    assert first["counts"]["runManifestWriteRows"] == 1
    assert {row["executionStatus"] for row in first["rows"]} == {EXECUTOR_STATUS_APPLIED}
    assert second["status"] == "applied"
    assert second["counts"]["candidateStoreWriteRows"] == 0
    assert second["counts"]["alreadyCorrectRows"] == 2
    assert second["counts"]["readbackValidatedRows"] == 2
    assert {row["executionStatus"] for row in second["rows"]} == {EXECUTOR_STATUS_ALREADY_CORRECT}
    assert len(records) == 2
    assert len({record["idempotencyKey"] for record in records}) == 2
    assert manifest["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID
    assert validate_payload(
        first,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_CANARY_APPLY_READBACK_SCHEMA_ID,
        strict=True,
    ).ok


def test_canary_apply_readback_blocks_source_dry_run_not_ready(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report([_candidate_row("2005.11401", 1)], status="blocked"),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-blocked-source",
        apply=False,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["selectedCanaryRows"] == 0
    assert "candidate_dry_run_not_ready" in report["gate"]["schemaViolations"]


def test_canary_apply_readback_blocks_all_requested(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report([_candidate_row("2005.11401", 1)]),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-all",
        apply=False,
        all_requested=True,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "all_apply_not_allowed_in_canary_tranche" in report["warnings"]
    assert "all_apply_not_allowed_in_canary_tranche" in report["gate"]["schemaViolations"]


def test_canary_apply_readback_blocks_idempotency_conflict_without_overwrite(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    rows = [_candidate_row("2005.11401", 1)]
    first = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report(rows),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-conflict",
        apply=True,
        canary_record_limit=1,
        generated_at="2026-05-29T00:00:00Z",
    )
    record_path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "2005.11401.jsonl"
    stored = json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])
    stored["excerpt"] = "conflicting stored payload"
    record_path.write_text(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    second = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report(rows),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=papers_dir,
        run_id="run-conflict",
        apply=True,
        canary_record_limit=1,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert first["status"] == "applied"
    assert second["status"] == "blocked"
    assert second["counts"]["candidateStoreWriteRows"] == 0
    assert second["counts"]["blockedIdempotencyConflictRows"] == 1
    assert second["rows"][0]["executionStatus"] == EXECUTOR_STATUS_BLOCKED_IDEMPOTENCY_CONFLICT
    assert json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])["excerpt"] == "conflicting stored payload"


def test_canary_apply_readback_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report([_candidate_row("2005.11401", 1)]),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-writer",
        apply=False,
        generated_at="2026-05-29T00:00:00Z",
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
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


def test_canary_apply_readback_blocks_unsafe_candidate_row(tmp_path: Path) -> None:
    row = copy.deepcopy(_candidate_row("2005.11401", 1))
    row["strictEvidence"] = True

    report = build_parsed_artifact_evidence_chunk_candidate_canary_apply_readback(
        candidate_dry_run_report=_dry_run_report([row]),
        source_candidate_dry_run_report_ref=SOURCE_REPORT_REF,
        papers_dir=tmp_path / "papers",
        run_id="run-unsafe",
        apply=False,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedSchemaViolationRows"] == 1
    assert "strict_evidence_not_false" in report["rows"][0]["executionBlockers"]
