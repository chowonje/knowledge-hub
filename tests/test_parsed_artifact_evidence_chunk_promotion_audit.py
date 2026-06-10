from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_promotion_audit import (
    CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID,
    PROMOTION_BLOCKED,
    PROMOTION_CANDIDATE_NARROW_SCOPE,
    build_parsed_artifact_evidence_chunk_promotion_audit,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_promotion_report_io import (
    write_parsed_artifact_evidence_chunk_promotion_audit,
)


POSITIVE_COMPLETE_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.v1.json"
)


def _read_positive_report() -> dict[str, Any]:
    return json.loads(POSITIVE_COMPLETE_REPORT.read_text(encoding="utf-8"))


def _record(index: int, *, paper_id: str = "paper-a", source_hash: str | None = None) -> dict[str, Any]:
    excerpt = f"{paper_id} promotion audit evidence row {index} with section paragraph support."
    start = index * 100
    end = start + len(excerpt)
    return {
        "schema": CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": f"parsed-artifact-evidence-chunk-candidate:{paper_id}:paragraph:{index}",
        "runId": "test-run",
        "sourceDryRunReportRef": "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_candidate_dry_run.v1.json",
        "sourceCandidateRowId": f"parsed-artifact-evidence-chunk:{paper_id}:{index}",
        "paperId": paper_id,
        "sourceType": "paper",
        "artifactType": "paragraph",
        "sourceRef": f"papers_dir/parsed/{paper_id}/document.md",
        "sourceContentHash": source_hash if source_hash is not None else "sha256:" + f"{index:064d}"[-64:],
        "locator": {"kind": "parsed_document_chars", "chars": {"start": start, "end": end}},
        "spanLocator": f"chars:{start}-{end}",
        "excerpt": excerpt,
        "snippetHash": "sha256:" + hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        "sectionPath": ["Method"],
        "sectionTitle": "Method",
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
    }


def _write_records(papers_dir: Path, paper_id: str, rows: list[dict[str, Any]]) -> None:
    target = papers_dir / "structured_evidence_candidates" / "evidence_chunk" / f"{paper_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_promotion_audit_marks_valid_positive_slice_as_narrow_scope_candidate(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a")])
    _write_records(tmp_path, "paper-b", [_record(2, paper_id="paper-b", source_hash="")])

    report = build_parsed_artifact_evidence_chunk_promotion_audit(
        papers_dir=tmp_path,
        positive_complete_report=_read_positive_report(),
        generated_at="2026-06-11T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == PROMOTION_CANDIDATE_NARROW_SCOPE
    assert report["counts"]["papersEvaluated"] == 2
    assert report["counts"]["papersWithCandidateRows"] == 2
    assert report["counts"]["candidateRowsWithValidSourceContentHash"] == 1
    assert report["counts"]["candidateRowsWithValidCharLocators"] == 2
    assert report["counts"]["answerVisibleEvidenceRows"] == 20
    assert report["counts"]["candidateStoreAnswerVisibleRows"] == 0
    assert report["blockersByCategory"][0]["category"] == "source_content_hash_invalid"
    assert validate_payload(report, PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID, strict=True).ok


def test_promotion_audit_blocks_when_positive_quality_is_not_ready(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a")])
    positive = deepcopy(_read_positive_report())
    positive["status"] = "blocked"

    report = build_parsed_artifact_evidence_chunk_promotion_audit(
        papers_dir=tmp_path,
        positive_complete_report=positive,
        generated_at="2026-06-11T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["decision"] == PROMOTION_BLOCKED
    assert "positive_quality_not_ready" in report["gate"]["semanticViolations"]


def test_promotion_audit_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    _write_records(tmp_path, "paper-a", [_record(1, paper_id="paper-a")])
    report = build_parsed_artifact_evidence_chunk_promotion_audit(
        papers_dir=tmp_path,
        positive_complete_report=_read_positive_report(),
        generated_at="2026-06-11T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_promotion_audit(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )
    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert validate_payload(loaded, PARSED_ARTIFACT_EVIDENCE_CHUNK_PROMOTION_AUDIT_SCHEMA_ID, strict=True).ok
    assert "promotion_candidate_narrow_scope" in Path(paths["markdown"]).read_text(encoding="utf-8")
