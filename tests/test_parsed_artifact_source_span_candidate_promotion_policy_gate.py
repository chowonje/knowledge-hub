from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_promotion_policy_gate import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    build_parsed_artifact_source_span_candidate_promotion_policy_gate,
    write_parsed_artifact_source_span_candidate_promotion_policy_gate_reports,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_readback_review import (
    build_parsed_artifact_source_span_candidate_readback_review,
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
    source_content_hash: str = "source-hash",
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
        "sourceContentHash": source_content_hash,
        "sourceFile": f"{paper_id}.pdf",
        "locator": {
            "page": index,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "blockIndexes": [index],
            "chars": {"start": None, "end": None},
        },
        "idempotencyKey": f"idem-{index}",
        "evidenceTier": "source_span_candidate_only",
        "strictEligible": False,
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


def _write_readback_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ready_readback_report(tmp_path: Path) -> dict:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(index=1), _candidate_record(index=2)],
    )
    return build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )


def test_source_span_candidate_promotion_policy_gate_marks_readback_ready_rows(
    tmp_path: Path,
) -> None:
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, _ready_readback_report(tmp_path))

    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["candidateRecordRows"] == 2
    assert report["counts"]["promotionPolicyGateReadyRows"] == 2
    assert report["counts"]["sourceSpanPromotionExecutorDryRunReadyRows"] == 2
    assert report["counts"]["sourceSpanPromotionApprovedRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["strictEvidenceCreatedRows"] == 0
    assert report["gate"]["readyForSourceSpanPromotionExecutorDryRun"] is True
    assert report["gate"]["sourceSpanPromotionApproved"] is False
    assert {row["policy_gate_status"] for row in report["rows"]} == {
        POLICY_STATUS_READY_CANDIDATE_ONLY
    }
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_candidate_promotion_policy_gate_blocks_policy_violations(
    tmp_path: Path,
) -> None:
    readback_report = _ready_readback_report(tmp_path)
    readback_report["rows"][0]["sourceContentHash"] = ""
    readback_report["rows"][1]["artifact_type"] = "equation"
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["blockedMissingSourceHashRows"] == 1
    assert report["counts"]["blockedUnsupportedArtifactTypeRows"] == 1
    assert report["counts"]["promotionPolicyGateReadyRows"] == 0
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["gate"]["readyForSourceSpanPromotionExecutorDryRun"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_candidate_promotion_policy_gate_blocks_invalid_input_schema(
    tmp_path: Path,
) -> None:
    readback_report = _ready_readback_report(tmp_path)
    readback_report["schema"] = "wrong.schema"
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2
    assert report["counts"]["promotionPolicyGateReadyRows"] == 0
    assert report["gate"]["readyForSourceSpanPromotionExecutorDryRun"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_candidate_promotion_policy_gate_filters_by_paper_id(tmp_path: Path) -> None:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(paper_id="paper-1", index=1)],
    )
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-2.jsonl",
        [_candidate_record(paper_id="paper-2", index=2)],
    )
    readback_report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, readback_report)

    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
        paper_ids=["paper-2"],
    )

    assert report["counts"]["candidateRecordRows"] == 1
    assert report["counts"]["byPaperId"] == {"paper-2": 1}
    assert report["counts"]["promotionPolicyGateReadyRows"] == 1
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_candidate_promotion_policy_gate_writer_outputs_schema_valid_reports(
    tmp_path: Path,
) -> None:
    readback_report_path = tmp_path / "readback.json"
    _write_readback_report(readback_report_path, _ready_readback_report(tmp_path))
    report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )

    paths = write_parsed_artifact_source_span_candidate_promotion_policy_gate_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    written_summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert written_report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID
    assert written_summary["counts"]["promotionPolicyGateReadyRows"] == 2
    assert "Parsed Artifact SourceSpan Candidate Promotion Policy Gate" in markdown
    assert "source spans created: 0" in markdown
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        written_summary,
        PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
        strict=True,
    ).ok
