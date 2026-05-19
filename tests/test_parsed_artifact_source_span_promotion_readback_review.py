from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_promotion_readback_review import (
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
    READBACK_STATUS_VALIDATED,
    _resolve_record_run_ids_from_manifest,
    build_parsed_artifact_source_span_promotion_readback_review,
    write_parsed_artifact_source_span_promotion_readback_review_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_STORE,
)


def _source_span_record(
    *,
    paper_id: str = "paper-1",
    artifact_type: str = "section",
    index: int = 1,
    source_content_hash: str | None = "source-hash",
    strict_eligible: bool = False,
    run_id: str = "run-1",
) -> dict:
    return {
        "schema": PARSED_ARTIFACT_SOURCE_SPAN_RECORD_SCHEMA_ID,
        "sourceSpanId": f"source-span:{paper_id}:{artifact_type}:{index}",
        "candidateRecordId": f"source-span-candidate:{paper_id}:{artifact_type}:{index}",
        "runId": run_id,
        "plannedWriteTarget": PARSED_ARTIFACT_SOURCE_SPAN_STORE,
        "paperId": paper_id,
        "artifactType": artifact_type,
        "sourceCandidateId": f"candidate-{index}",
        "sourceContentHash": source_content_hash or "",
        "sourceFile": f"{paper_id}.pdf",
        "locator": {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {"start": None, "end": None},
        },
        "idempotencyKey": f"idem-{index}",
        "evidenceTier": "parsed_artifact_source_span",
        "strictEligible": strict_eligible,
        "citationGrade": False,
        "runtimeEvidence": False,
        "strictBlockers": ["source_span_store_record_not_strict_evidence"],
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


def test_source_span_promotion_readback_review_marks_valid_records_ready(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1), _source_span_record(index=2)],
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["sourceSpanRecordRows"] == 2
    assert report["counts"]["readbackValidatedRows"] == 2
    assert report["counts"]["schemaViolationCount"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["policy"]["reportOnly"] is True
    assert report["policy"]["sourceSpanStoreWrite"] is False
    assert {row["readback_status"] for row in report["rows"]} == {READBACK_STATUS_VALIDATED}
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_readback_review_blocks_invalid_records(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    missing_hash = _source_span_record(index=1, source_content_hash="")
    missing_locator = _source_span_record(index=2)
    missing_locator["locator"] = {
        "page": None,
        "bbox": [],
        "blockIndexes": [],
        "chars": {"start": None, "end": None},
    }
    runtime_flag = _source_span_record(index=3, strict_eligible=True)
    duplicate_a = _source_span_record(index=4)
    duplicate_b = _source_span_record(index=5)
    duplicate_b["idempotencyKey"] = duplicate_a["idempotencyKey"]
    duplicate_b["sourceSpanId"] = "source-span:paper-1:section:5"
    malformed = _source_span_record(index=6)
    malformed["sourceSpanId"] = ""

    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [missing_hash, missing_locator, runtime_flag, duplicate_a, duplicate_b, malformed],
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedSchemaViolationRows"] == 1
    assert report["counts"]["blockedMissingSourceHashRows"] == 1
    assert report["counts"]["blockedMissingLocatorRows"] == 1
    assert report["counts"]["blockedRuntimeOrStrictFlagRows"] == 1
    assert report["counts"]["blockedDuplicateIdempotencyKeyRows"] == 2
    assert report["counts"]["readbackValidatedRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_readback_review_filters_by_paper_id(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(paper_id="paper-1", index=1)],
    )
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-2.jsonl",
        [_source_span_record(paper_id="paper-2", index=2)],
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        paper_ids=["paper-2"],
    )

    assert report["counts"]["sourceSpanRecordRows"] == 1
    assert report["counts"]["byPaperId"] == {"paper-2": 1}
    assert report["counts"]["readbackValidatedRows"] == 1
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_readback_review_filters_by_record_run_id(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [
            _source_span_record(index=1, run_id="run-a"),
            _source_span_record(index=2, run_id="run-b"),
        ],
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        run_id="run-a",
    )

    assert report["status"] == "ok"
    assert report["counts"]["sourceSpanRecordRows"] == 1
    assert report["input"]["runIdentity"]["runIdFilterMode"] == "record_run_id_exact"
    assert report["input"]["runIdentity"]["resolvedRecordRunIds"] == ["run-a"]


def test_source_span_promotion_readback_review_resolves_manifest_alias_to_record_run_id(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    record = _source_span_record(index=1, run_id="actual-run-id")
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [record],
    )
    manifest_path = papers_dir / "structured_evidence" / "runs" / "alias-run.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input": {"runId": "alias-run"},
                "sourceSpanRecords": [
                    {
                        "sourceSpanId": record["sourceSpanId"],
                        "idempotencyKey": record["idempotencyKey"],
                        "runId": "alias-run",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        run_manifest=manifest_path,
    )

    assert report["status"] == "ok"
    assert report["counts"]["sourceSpanRecordRows"] == 1
    assert report["counts"]["readbackValidatedRows"] == 1
    assert report["input"]["runIdentity"]["resolution"] == "manifest_record_metadata_match"
    assert report["input"]["runIdentity"]["resolvedRecordRunIds"] == ["actual-run-id"]
    assert "alias-run" in report["input"]["runIdentity"]["manifestCandidateRunIds"]


def test_source_span_promotion_readback_review_blocks_manifest_run_id_mismatch(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1, run_id="actual-run-id")],
    )
    manifest_path = papers_dir / "structured_evidence" / "runs" / "missing-run.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"input": {"runId": "missing-run"}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = build_parsed_artifact_source_span_promotion_readback_review(
        papers_dir=papers_dir,
        run_manifest=manifest_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["sourceSpanRecordRows"] == 0
    assert "run_manifest_record_run_id_mismatch" in report["warnings"]
    assert report["input"]["runIdentity"]["resolution"] == "run_manifest_record_run_id_mismatch"
    assert report["input"]["runIdentity"]["observedRecordRunIds"] == ["actual-run-id"]
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0


def test_resolve_record_run_ids_from_manifest_prefers_direct_match() -> None:
    manifest = {"input": {"runId": "run-a"}, "output": {"runId": "run-b"}}
    raw_rows = [
        {"record": {"runId": "run-b", "sourceSpanId": "span-1", "idempotencyKey": "idem-1"}}
    ]
    resolved, resolution = _resolve_record_run_ids_from_manifest(
        manifest=manifest,
        manifest_path="/tmp/run-a.json",
        raw_rows=raw_rows,
    )
    assert resolved == ["run-b"]
    assert resolution["resolution"] == "manifest_run_id_direct_match"


def test_source_span_promotion_readback_review_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence" / "source_span" / "paper-1.jsonl",
        [_source_span_record(index=1)],
    )
    report = build_parsed_artifact_source_span_promotion_readback_review(papers_dir=papers_dir)

    paths = write_parsed_artifact_source_span_promotion_readback_review_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    written_summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert written_report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID
    assert written_summary["counts"]["readbackValidatedRows"] == 1
    assert "Parsed Artifact SourceSpan Promotion Readback Review" in markdown
    assert "source spans created: 0" in markdown
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        written_summary,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_READBACK_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
