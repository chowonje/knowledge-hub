from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_candidate_canary_apply_readback import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_CANDIDATE_RECORD_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_real_answer_quality_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_real_answer_quality_smoke,
    write_parsed_artifact_evidence_chunk_real_answer_quality_smoke,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_runtime_adapter_live_smoke import (
    build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke,
    write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(index: int, *, paper_id: str, excerpt: str, artifact_type: str = "paragraph") -> dict[str, Any]:
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


def _write_ready_live_smoke(tmp_path: Path, *, papers_dir: Path, paper_ids: list[str]) -> Path:
    live_report = build_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        papers_dir=papers_dir,
        resolved_paper_ids=paper_ids,
        generated_at="2026-05-29T00:00:00Z",
    )
    assert live_report["status"] == "ready"
    paths = write_parsed_artifact_evidence_chunk_runtime_adapter_live_smoke(
        live_report,
        report_json=tmp_path / "live.json",
        report_md=tmp_path / "live.md",
    )
    return Path(paths["json"])


def test_real_answer_quality_smoke_ready_scores_deterministic_answers_without_excerpt_leak(tmp_path: Path) -> None:
    alpha_excerpt = "AlphaFold CASP14 accuracy evidence states protein structure prediction and accuracy comparisons."
    vector_excerpt = "Continuous vector representations of words are evaluated through word similarity tasks."
    _write_records(
        tmp_path,
        "paper-a",
        [
            _record(1, paper_id="paper-a", excerpt=alpha_excerpt),
            _record(2, paper_id="paper-a", excerpt="AlphaFold CASP14 accuracy second supporting sentence."),
        ],
    )
    _write_records(
        tmp_path,
        "paper-b",
        [
            _record(3, paper_id="paper-b", excerpt=vector_excerpt),
            _record(4, paper_id="paper-b", excerpt="Word representations and similarity evidence second sentence."),
        ],
    )
    live_report = _write_ready_live_smoke(tmp_path, papers_dir=tmp_path, paper_ids=["paper-a", "paper-b"])
    cases = [
        {
            "caseId": "alpha",
            "query": "What does the selected evidence say about AlphaFold CASP14 accuracy?",
            "resolvedPaperIds": ["paper-a"],
            "answerClaim": "Selected evidence supports AlphaFold CASP14 accuracy comparisons.",
            "requiredEvidenceTerms": ["AlphaFold", "CASP14", "accuracy"],
            "requiredAnswerTerms": ["AlphaFold", "CASP14", "accuracy"],
            "forbiddenAnswerTerms": ["word similarity"],
            "minCitations": 2,
        },
        {
            "caseId": "vectors",
            "query": "What does selected evidence say about word vector representations?",
            "resolvedPaperIds": ["paper-b"],
            "answerClaim": "Selected evidence supports continuous vector representations of words and word similarity.",
            "requiredEvidenceTerms": ["continuous", "representations", "word", "similarity"],
            "requiredAnswerTerms": ["continuous", "representations", "word", "similarity"],
            "forbiddenAnswerTerms": ["AlphaFold"],
            "minCitations": 2,
        },
    ]

    report = build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        papers_dir=tmp_path,
        live_smoke_report=live_report,
        cases=cases,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["passRows"] == 2
    assert report["counts"]["deterministicAnswerGeneratedRows"] == 2
    assert report["counts"]["llmCallRows"] == 0
    assert report["counts"]["judgeModelCallRows"] == 0
    assert report["gate"]["readyForAnswerPathOptInRouteReview"] is True
    assert all(row["answerTextIncludedInReport"] is False for row in report["rows"])
    assert alpha_excerpt not in json.dumps(report, ensure_ascii=False)
    assert vector_excerpt not in json.dumps(report, ensure_ascii=False)
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID, strict=True).ok


def test_real_answer_quality_smoke_blocks_when_live_smoke_missing(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        papers_dir=tmp_path,
        live_smoke_report=tmp_path / "missing.json",
        cases=[
            {
                "caseId": "alpha",
                "query": "q",
                "resolvedPaperIds": ["paper-a"],
                "answerClaim": "claim",
                "requiredEvidenceTerms": ["AlphaFold"],
                "requiredAnswerTerms": ["AlphaFold"],
            }
        ],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["attemptedCaseRows"] == 0
    assert report["counts"]["blockedRows"] == 1
    assert "live_smoke_schema_mismatch" in report["gate"]["semanticViolations"]
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID, strict=True).ok


def test_real_answer_quality_smoke_fails_missing_evidence_term(tmp_path: Path) -> None:
    _write_records(
        tmp_path,
        "paper-a",
        [
            _record(1, paper_id="paper-a", excerpt="AlphaFold CASP14 accuracy evidence."),
            _record(2, paper_id="paper-a", excerpt="AlphaFold accuracy support."),
        ],
    )
    live_report = _write_ready_live_smoke(tmp_path, papers_dir=tmp_path, paper_ids=["paper-a"])

    report = build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        papers_dir=tmp_path,
        live_smoke_report=live_report,
        cases=[
            {
                "caseId": "missing-term",
                "query": "q",
                "resolvedPaperIds": ["paper-a"],
                "answerClaim": "Selected evidence supports AlphaFold CASP14 accuracy.",
                "requiredEvidenceTerms": ["nonexistent-term"],
                "requiredAnswerTerms": ["AlphaFold"],
                "minCitations": 2,
            }
        ],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["failRows"] == 1
    assert "missing_evidence_term:nonexistent-term" in report["rows"][0]["failureReasons"]
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID, strict=True).ok


def test_real_answer_quality_smoke_fails_forbidden_answer_term(tmp_path: Path) -> None:
    _write_records(
        tmp_path,
        "paper-a",
        [
            _record(1, paper_id="paper-a", excerpt="AlphaFold CASP14 accuracy evidence."),
            _record(2, paper_id="paper-a", excerpt="AlphaFold accuracy support."),
        ],
    )
    live_report = _write_ready_live_smoke(tmp_path, papers_dir=tmp_path, paper_ids=["paper-a"])

    report = build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        papers_dir=tmp_path,
        live_smoke_report=live_report,
        cases=[
            {
                "caseId": "forbidden",
                "query": "q",
                "resolvedPaperIds": ["paper-a"],
                "answerClaim": "Selected evidence supports AlphaFold CASP14 accuracy and hallucinated claim.",
                "requiredEvidenceTerms": ["AlphaFold"],
                "requiredAnswerTerms": ["AlphaFold"],
                "forbiddenAnswerTerms": ["hallucinated"],
                "minCitations": 2,
            }
        ],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "forbidden_answer_term:hallucinated" in report["rows"][0]["failureReasons"]
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID, strict=True).ok


def test_real_answer_quality_smoke_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    _write_records(
        tmp_path,
        "paper-a",
        [
            _record(1, paper_id="paper-a", excerpt="AlphaFold CASP14 accuracy evidence."),
            _record(2, paper_id="paper-a", excerpt="AlphaFold accuracy support."),
        ],
    )
    live_report = _write_ready_live_smoke(tmp_path, papers_dir=tmp_path, paper_ids=["paper-a"])
    report = build_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        papers_dir=tmp_path,
        live_smoke_report=live_report,
        cases=[
            {
                "caseId": "alpha",
                "query": "q",
                "resolvedPaperIds": ["paper-a"],
                "answerClaim": "Selected evidence supports AlphaFold CASP14 accuracy.",
                "requiredEvidenceTerms": ["AlphaFold", "accuracy"],
                "requiredAnswerTerms": ["AlphaFold", "accuracy"],
                "minCitations": 2,
            }
        ],
        generated_at="2026-05-29T00:00:00Z",
    )
    paths = write_parsed_artifact_evidence_chunk_real_answer_quality_smoke(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Real Answer Quality Smoke"
    )
    assert validate_payload(parsed, PARSED_ARTIFACT_EVIDENCE_CHUNK_REAL_ANSWER_QUALITY_SMOKE_SCHEMA_ID, strict=True).ok
