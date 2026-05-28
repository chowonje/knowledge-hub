from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION as LIVE_SMOKE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, paper_id: str, text: str, artifact_type: str = "paragraph") -> dict[str, Any]:
    start = index * 100
    end = start + len(text)
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
        "excerpt": text,
        "snippetHash": _sha256_text(text),
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


def _seed_candidate_store(tmp_path: Path) -> None:
    _write_records(
        tmp_path,
        "paper-a",
        [
            _record(1, paper_id="paper-a", text="paper-a alpha method evidence chunk source hash and offset support"),
            _record(2, paper_id="paper-a", text="paper-a alpha result evidence chunk source hash and offset support"),
        ],
    )
    _write_records(
        tmp_path,
        "paper-b",
        [
            _record(3, paper_id="paper-b", text="paper-b vector similarity evidence chunk source hash support"),
            _record(4, paper_id="paper-b", text="paper-b vector representation evidence chunk offset support"),
        ],
    )


def _ready_live_smoke_report() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        "status": "ready",
        "decision": LIVE_SMOKE_READY_DECISION,
        "counts": {
            "surfaceSmokePassRows": 1,
            "adapterRowsAdded": 4,
            "externalRequestRejectedRows": 1,
            "schemaViolationCount": 0,
            "privatePathLeakRows": 0,
        },
        "gate": {
            "readyForLabsOptInQualityEvalSeed": True,
        },
    }


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "caseId": "paper_a_positive",
            "evalFocus": "single_paper",
            "question": "What alpha method evidence exists?",
            "paperIds": ["paper-a"],
            "expectedStatus": "ok",
            "expectedAnswerable": True,
            "expectedSourceIds": ["paper-a"],
            "requiredEvidenceTerms": ["alpha", "method"],
            "minCitations": 2,
            "minSpanRows": 2,
        },
        {
            "caseId": "pair_positive",
            "evalFocus": "two_paper_compare_seed",
            "question": "Compare alpha and vector evidence.",
            "paperIds": ["paper-a", "paper-b"],
            "expectedStatus": "ok",
            "expectedAnswerable": True,
            "expectedSourceIds": ["paper-a", "paper-b"],
            "requiredEvidenceTerms": ["alpha", "vector", "similarity"],
            "minCitations": 4,
            "minSpanRows": 4,
        },
        {
            "caseId": "missing_no_evidence",
            "evalFocus": "expected_no_evidence_safety",
            "question": "What evidence exists for missing paper?",
            "paperIds": ["missing-paper"],
            "expectedStatus": "no_evidence",
            "expectedAnswerable": False,
            "expectedSourceIds": [],
            "requiredEvidenceTerms": [],
            "minCitations": 0,
            "minSpanRows": 0,
        },
    ]


def _seed_report(tmp_path: Path) -> dict[str, Any]:
    return build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed(
        papers_dir=tmp_path,
        live_smoke_report=_ready_live_smoke_report(),
        cases=_cases(),
        generated_at="2026-05-29T00:00:00Z",
    )


def test_labs_opt_in_quality_eval_runner_ready(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
        papers_dir=tmp_path,
        quality_eval_seed_report=_seed_report(tmp_path),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["inputCaseRows"] == 3
    assert report["counts"]["qualityPassRows"] == 3
    assert report["counts"]["qualityPartialRows"] == 0
    assert report["counts"]["qualityFailRows"] == 0
    assert report["counts"]["averageQualityScore"] == 1.0
    assert report["counts"]["minQualityScore"] == 1.0
    assert report["counts"]["noEvidenceSafetyPassRows"] == 1
    assert report["counts"]["noEvidenceLlmCallRows"] == 0
    assert report["gate"]["readyForLabsOptInUserTestPacket"] is True
    assert report["gate"]["qualityScoreThresholdPassed"] is True
    assert all(row["answerTextIncludedInReport"] is False for row in report["rows"])
    assert "source hash and offset support" not in json.dumps(report, ensure_ascii=False)
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_quality_eval_runner_blocks_when_seed_not_ready(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    seed_report = _seed_report(tmp_path)
    seed_report["status"] = "blocked"

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
        papers_dir=tmp_path,
        quality_eval_seed_report=seed_report,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["attemptedCaseRows"] == 0
    assert "quality_eval_seed_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_quality_eval_runner_blocks_when_rerun_quality_fails(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    seed_report = _seed_report(tmp_path)
    seed_report["rows"][0]["requiredEvidenceTerms"] = ["not-present-term"]

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
        papers_dir=tmp_path,
        quality_eval_seed_report=seed_report,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["qualityFailRows"] == 1
    assert report["rows"][0]["qualityGrade"] == "partial"
    assert "quality_dimensions_not_all_passed" in report["rows"][0]["failureReasons"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_quality_eval_runner_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    _seed_candidate_store(tmp_path)
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
        papers_dir=tmp_path,
        quality_eval_seed_report=_seed_report(tmp_path),
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in Quality Eval Runner"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        strict=True,
    ).ok
