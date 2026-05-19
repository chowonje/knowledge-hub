from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_PROMOTION_REVIEW_READY,
    build_parsed_artifact_source_span_candidate_readback_review,
    write_parsed_artifact_source_span_candidate_readback_review_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
)


def _candidate_record(
    *,
    paper_id: str = "paper-1",
    artifact_type: str = "section",
    index: int = 1,
    source_content_hash: str | None = "source-hash",
    strict_eligible: bool = False,
) -> dict:
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": "run-1",
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "sourceCandidateId": f"candidate-{index}",
        "sourceReadinessRowId": f"readiness-{index}",
        "sourceContentHash": source_content_hash or "",
        "sourceFile": f"{paper_id}.pdf",
        "locator": {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {"start": None, "end": None},
        },
        "idempotencyKey": f"idem-{index}",
        "evidenceTier": "source_span_candidate_only",
        "strictEligible": strict_eligible,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": ["candidate_store_record_not_strict_evidence"],
        "writePolicy": {
            "executorRequired": True,
            "databaseMutation": False,
            "parserRoutingChanged": False,
            "answerIntegrationChanged": False,
            "reindexOrReembed": False,
            "canonicalParsedArtifactsWritten": False,
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_source_span_candidate_readback_review_marks_valid_records_ready(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(index=1), _candidate_record(index=2)],
    )

    report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["candidateRecordRows"] == 2
    assert report["counts"]["promotionReviewReadyCandidateOnlyRows"] == 2
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["policy"]["reportOnly"] is True
    assert report["policy"]["candidateStoreWrite"] is False
    assert {row["readback_status"] for row in report["rows"]} == {
        READBACK_STATUS_PROMOTION_REVIEW_READY
    }
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID, strict=True).ok


def test_source_span_candidate_readback_review_blocks_invalid_records(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    missing_hash = _candidate_record(index=1, source_content_hash="")
    missing_locator = _candidate_record(index=2)
    missing_locator["locator"] = {
        "page": None,
        "bbox": [],
        "blockIndexes": [],
        "chars": {"start": None, "end": None},
    }
    runtime_flag = _candidate_record(index=3, strict_eligible=True)
    duplicate_a = _candidate_record(index=4)
    duplicate_b = _candidate_record(index=5)
    duplicate_b["idempotencyKey"] = duplicate_a["idempotencyKey"]
    duplicate_b["candidateRecordId"] = "source-span-candidate:paper-1:section:5"
    malformed = _candidate_record(index=6)
    malformed["candidateRecordId"] = ""

    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [missing_hash, missing_locator, runtime_flag, duplicate_a, duplicate_b, malformed],
    )

    report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedSchemaViolationRows"] == 1
    assert report["counts"]["blockedMissingSourceHashRows"] == 1
    assert report["counts"]["blockedMissingLocatorRows"] == 1
    assert report["counts"]["blockedRuntimeOrStrictFlagRows"] == 1
    assert report["counts"]["blockedDuplicateIdempotencyKeyRows"] == 2
    assert report["counts"]["promotionReviewReadyCandidateOnlyRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID, strict=True).ok


def test_source_span_candidate_readback_review_filters_by_paper_id(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(paper_id="paper-1", index=1)],
    )
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-2.jsonl",
        [_candidate_record(paper_id="paper-2", index=2)],
    )

    report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        paper_ids=["paper-2"],
    )

    assert report["counts"]["candidateRecordRows"] == 1
    assert report["counts"]["byPaperId"] == {"paper-2": 1}
    assert report["counts"]["promotionReviewReadyCandidateOnlyRows"] == 1
    assert validate_payload(report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID, strict=True).ok


def test_source_span_candidate_readback_review_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(index=1)],
    )
    report = build_parsed_artifact_source_span_candidate_readback_review(papers_dir=papers_dir)

    paths = write_parsed_artifact_source_span_candidate_readback_review_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    written_summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert written_report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID
    assert written_summary["counts"]["promotionReviewReadyCandidateOnlyRows"] == 1
    assert "Parsed Artifact SourceSpan Candidate Readback Review" in markdown
    assert "source spans created: 0" in markdown
    assert validate_payload(written_report, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID, strict=True).ok
    assert validate_payload(written_summary, PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_READBACK_REVIEW_SCHEMA_ID, strict=True).ok
