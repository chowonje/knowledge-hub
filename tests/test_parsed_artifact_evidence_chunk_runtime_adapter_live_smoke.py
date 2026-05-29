from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_runtime_adapter_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke,
    write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, paper_id: str, artifact_type: str = "paragraph") -> dict[str, Any]:
    excerpt = f"{paper_id} live smoke evidence chunk {index} with source hash and offset support."
    start = index * 100
    end = start + len(excerpt)
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "test-run",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json",
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "paperId": paper_id,
        "sourceType": "paper",
        "artifactType": artifact_type,
        "sourceRef": f"papers_dir/parsed/{paper_id}/document.md",
        "sourceContentHash": "sha256:" + f"{index:064d}"[-64:],
        "locator": {
            "kind": "parsed_document_chars",
            "chars": {"start": start, "end": end, "basis": "parsed_document_text"},
            "page": 1,
        },
        "spanLocator": f"chars:{start}-{end}",
        "excerpt": excerpt,
        "snippetHash": _sha256_text(excerpt),
        "sectionTitle": "Method",
        "sectionPath": ["Method"],
        "evidenceKind": "parsed_artifact_evidence_chunk_candidate",
        "idempotencyKey": f"idempotency:{paper_id}:{index}",
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


def _write_records(papers_dir: Path, paper_id: str, records: list[dict[str, Any]]) -> None:
    target = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / f"{paper_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_live_smoke_ready_builds_contract_grade_spans_without_excerpt_report_leak(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a"), _record(2, paper_id="paper-a"), _record(3, paper_id="paper-a")])
    _write_records(tmp_path, "paper-b", [_record(4, paper_id="paper-b", artifact_type="section"), _record(5, paper_id="paper-b")])

    report = build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a", "paper-b"],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["adapterRowsAdded"] == 4
    assert report["counts"]["evidencePacketContractSpanRows"] == 4
    assert report["counts"]["answerContractCitationRows"] == 4
    assert report["counts"]["answerableRows"] == 1
    assert report["counts"]["answerContractAbstainRows"] == 0
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["gate"]["readyForRealAnswerQualitySmoke"] is True
    assert all(row["excerptIncludedInReport"] is False for row in report["rows"])
    assert "live smoke evidence chunk" not in json.dumps(report, ensure_ascii=False)
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID, strict=True).ok


def test_live_smoke_blocks_when_candidate_store_missing(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a"],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["adapterDiagnostics"]["status"] == "skipped"
    assert report["adapterDiagnostics"]["skippedReason"] == "candidate_store_records_missing"
    assert "runtime_adapter_not_applied" in report["gate"]["schemaViolations"]
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID, strict=True).ok


def test_live_smoke_blocks_invalid_candidate_record(tmp_path: Path) -> None:
    record = _record(1, paper_id="paper-a")
    record["snippetHash"] = "sha256:" + "0" * 64
    _write_records(tmp_path, "paper-a", [record])

    report = build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a"],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["adapterDiagnostics"]["status"] == "blocked"
    assert report["adapterDiagnostics"]["blockedReasons"]["snippet_hash_mismatch"] == 1
    assert report["counts"]["adapterBlockedRows"] == 1
    assert report["rows"] == []
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID, strict=True).ok


def test_live_smoke_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a"), _record(2, paper_id="paper-a")])
    report = build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a"],
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Runtime Adapter Live Smoke"
    )
    assert validate_payload(parsed, PARSED_ARTIFACT_EVIDENCE_CHUNK_RUNTIME_ADAPTER_LIVE_SMOKE_SCHEMA_ID, strict=True).ok
