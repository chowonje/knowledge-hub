from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
    READY_DECISION as DESIGN_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, paper_id: str, artifact_type: str = "paragraph") -> dict[str, Any]:
    excerpt = f"{paper_id} labs preview evidence chunk {index} with source hash and offset support."
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


def _ready_design_report(**updates: Any) -> dict[str, Any]:
    report = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_DESIGN_SCHEMA_ID,
        "status": "ready",
        "decision": DESIGN_READY_DECISION,
        "counts": {
            "blockedRows": 0,
            "schemaViolationCount": 0,
            "privatePathLeakRows": 0,
        },
        "gate": {
            "readyForLabsOptInSurfaceImplementation": True,
            "publicKhubAskClosed": True,
            "defaultMcpAskClosed": True,
            "plannedLabsOnly": True,
        },
    }
    report.update(updates)
    return report


def _seed_candidate_store(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a"), _record(2, paper_id="paper-a")])
    _write_records(
        tmp_path,
        "paper-b",
        [_record(3, paper_id="paper-b", artifact_type="section"), _record(4, paper_id="paper-b")],
    )


def test_labs_opt_in_surface_live_smoke_ready_with_local_candidate_store(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a", "paper-b"],
        labs_surface_design_report=_ready_design_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["surfaceSmokePassRows"] == 1
    assert report["counts"]["surfacePayloadSchemaValidRows"] == 1
    assert report["counts"]["adapterAppliedRows"] == 1
    assert report["counts"]["adapterRowsAdded"] == 4
    assert report["counts"]["selectedEvidenceCount"] == 4
    assert report["counts"]["citationCount"] == 4
    assert report["counts"]["evidencePacketContractSpanRows"] == 4
    assert report["counts"]["localFakeLlmCallRows"] == 1
    assert report["counts"]["externalRequestRejectedRows"] == 1
    assert report["counts"]["externalRejectionLlmCallRows"] == 0
    assert report["counts"]["externalLlmCallRows"] == 0
    assert report["counts"]["modelApiCallRows"] == 0
    assert report["counts"]["labsMcpToolDefaultProfileRows"] == 0
    assert report["counts"]["defaultMcpAskAdapterArgRows"] == 0
    assert report["counts"]["defaultMcpAskQueryPlanArgRows"] == 0
    assert report["gate"]["readyForLabsOptInQualityEvalSeed"] is True
    assert report["gate"]["labsMcpToolHiddenFromDefault"] is True
    assert report["gate"]["defaultMcpAskClosed"] is True
    assert report["positiveSurfacePayload"]["answerIncludedInReport"] is False
    assert all(row["evidenceTextIncludedInReport"] is False for row in report["rows"])
    assert "labs preview evidence chunk" not in json.dumps(report, ensure_ascii=False)
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_live_smoke_blocks_when_design_report_not_ready(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    design_report = _ready_design_report(status="blocked")

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a", "paper-b"],
        labs_surface_design_report=design_report,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["gate"]["labsSurfaceDesignReady"] is False
    assert "labs_surface_design_not_ready" in report["gate"]["schemaViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_live_smoke_blocks_when_candidate_store_missing(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a"],
        labs_surface_design_report=_ready_design_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "positive_surface_smoke_failed" in report["gate"]["schemaViolations"]
    assert "adapter_not_applied" in report["rows"][0]["violations"]
    assert report["counts"]["candidateStoreWriteRows"] == 0
    assert report["counts"]["externalLlmCallRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_live_smoke_blocks_private_path_in_source_design(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    design_report = _ready_design_report(localOnlyDebugPath="/Users/example/private.pdf")

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a", "paper-b"],
        labs_surface_design_report=design_report,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "labs_surface_design_private_path_leak" in report["gate"]["schemaViolations"]
    assert report["counts"]["privatePathLeakRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_surface_live_smoke_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        papers_dir=tmp_path,
        resolved_paper_ids=["paper-a", "paper-b"],
        labs_surface_design_report=_ready_design_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Surface Live Smoke"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        strict=True,
    ).ok
