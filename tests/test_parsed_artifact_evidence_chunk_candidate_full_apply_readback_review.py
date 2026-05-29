from __future__ import annotations

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
    build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readback_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_BLOCKED_DUPLICATE,
    READBACK_STATUS_BLOCKED_MISSING,
    READBACK_STATUS_BLOCKED_POLICY_VIOLATION,
    READBACK_STATUS_VALIDATED,
    build_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review,
    write_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate import (
    build_parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate,
)


DRY_RUN_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json"
CANARY_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_canary_apply_readback.v1.json"
READINESS_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_readiness_gate.v1.json"
FULL_APPLY_REF = "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback.v1.json"


def _candidate_row(index: int, *, paper_id: str | None = None, artifact_type: str = "paragraph") -> dict[str, object]:
    pid = paper_id or f"paper-{index:02d}"
    excerpt = (
        f"{index} Methods This parsed artifact evidence chunk candidate for {pid} "
        "contains enough text for readback review verification."
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


def _dry_run_report() -> dict[str, object]:
    rows = [_candidate_row(index) for index in range(1, 13)]
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_DRY_RUN_SCHEMA_ID,
        "status": "ready",
        "decision": CANDIDATE_DRY_RUN_READY_DECISION,
        "gate": {"passed": True},
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


def _reports(tmp_path: Path, *, full_apply: bool = True) -> tuple[dict[str, object], Path]:
    papers_dir = tmp_path / "papers"
    dry_run = _dry_run_report()
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
    full_apply_report = build_parsed_artifact_evidence_chunk_candidate_full_apply_executor_apply_readback(
        candidate_dry_run_report=dry_run,
        source_candidate_dry_run_report_ref=DRY_RUN_REF,
        full_apply_readiness_report=readiness,
        source_full_apply_readiness_report_ref=READINESS_REF,
        canary_apply_readback_report=canary,
        source_canary_apply_readback_report_ref=CANARY_REF,
        papers_dir=papers_dir,
        run_id="run-full",
        apply=full_apply,
        generated_at="2026-05-29T00:00:00Z",
    )
    assert canary["status"] == "applied"
    assert readiness["status"] == "ready"
    return full_apply_report, papers_dir


def _review(full_apply_report: dict[str, object], papers_dir: Path) -> dict[str, object]:
    return build_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review(
        full_apply_report=full_apply_report,
        source_full_apply_report_ref=FULL_APPLY_REF,
        papers_dir=papers_dir,
        generated_at="2026-05-29T00:00:00Z",
    )


def test_full_apply_readback_review_validates_store_records(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path)

    report = _review(full_apply_report, papers_dir)

    assert report["schema"] == PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID
    assert report["status"] == "ready"
    assert report["decision"] == "parsed_artifact_evidence_chunk_candidate_full_apply_readback_review_ready"
    assert report["counts"]["expectedCandidateRows"] == 12
    assert report["counts"]["storeRows"] == 12
    assert report["counts"]["matchingStoreRows"] == 12
    assert report["counts"]["readbackValidatedRows"] == 12
    assert report["counts"]["blockedRows"] == 0
    assert report["gate"]["readyForAnswerabilityPolicyGate"] is True
    assert {row["readbackStatus"] for row in report["rows"]} == {READBACK_STATUS_VALIDATED}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_FULL_APPLY_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_full_apply_readback_review_blocks_missing_store_record(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path)
    path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "paper-12.jsonl"
    path.unlink()

    report = _review(full_apply_report, papers_dir)

    assert report["status"] == "blocked"
    assert report["counts"]["missingStoreRows"] == 1
    assert report["rows"][-1]["readbackStatus"] == READBACK_STATUS_BLOCKED_MISSING


def test_full_apply_readback_review_blocks_duplicate_store_record(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path)
    path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "paper-12.jsonl"
    line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")

    report = _review(full_apply_report, papers_dir)

    assert report["status"] == "blocked"
    assert report["counts"]["duplicateStoreRows"] == 1
    assert report["rows"][-1]["readbackStatus"] == READBACK_STATUS_BLOCKED_DUPLICATE


def test_full_apply_readback_review_blocks_policy_violation(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path)
    path = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / "paper-12.jsonl"
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    stored["answerVisible"] = True
    path.write_text(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    report = _review(full_apply_report, papers_dir)

    assert report["status"] == "blocked"
    assert report["counts"]["policyViolationRows"] == 1
    assert report["rows"][-1]["readbackStatus"] == READBACK_STATUS_BLOCKED_POLICY_VIOLATION


def test_full_apply_readback_review_blocks_non_applied_source_report(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path, full_apply=False)

    report = _review(full_apply_report, papers_dir)

    assert report["status"] == "blocked"
    assert "full_apply_executor_not_applied" in report["gate"]["schemaViolations"]
    assert report["counts"]["candidateStoreWriteRows"] == 0


def test_full_apply_readback_review_writer_uses_sanitized_refs(tmp_path: Path) -> None:
    full_apply_report, papers_dir = _reports(tmp_path)
    report = _review(full_apply_report, papers_dir)
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    write_parsed_artifact_evidence_chunk_candidate_full_apply_readback_review(
        report,
        report_json=json_path,
        report_md=md_path,
    )

    combined = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["sourceFullApplyExecutorReport"]["reportRef"].startswith("eval/knowledgeos/reports/")
    assert "/" + "Users" + "/" not in combined
    assert "/" + "Volumes" + "/" not in combined
    assert "Mobile " + "Documents" not in combined
    assert "i" + "Cloud" not in combined
