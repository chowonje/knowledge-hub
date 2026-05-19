from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_source_span_candidate_promotion_policy_gate import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_PROMOTION_POLICY_GATE_SCHEMA_ID,
    POLICY_STATUS_READY_CANDIDATE_ONLY,
    build_parsed_artifact_source_span_candidate_promotion_policy_gate,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_readback_review import (
    build_parsed_artifact_source_span_candidate_readback_review,
)
from knowledge_hub.papers.parsed_artifact_source_span_candidate_store_contract import (
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_RECORD_SCHEMA_ID,
    PARSED_ARTIFACT_SOURCE_SPAN_CANDIDATE_STORE,
)
from knowledge_hub.papers.parsed_artifact_source_span_promotion_executor_dry_run import (
    EXECUTOR_STATUS_PLANNED,
    PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
    build_parsed_artifact_source_span_promotion_executor_dry_run,
    write_parsed_artifact_source_span_promotion_executor_dry_run_reports,
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ready_policy_gate_report(tmp_path: Path) -> dict:
    papers_dir = tmp_path / "papers"
    _write_jsonl(
        papers_dir / "structured_evidence_candidates" / "source_span" / "paper-1.jsonl",
        [_candidate_record(index=1), _candidate_record(index=2, artifact_type="figure")],
    )
    readback_report = build_parsed_artifact_source_span_candidate_readback_review(
        papers_dir=papers_dir,
        run_id="run-1",
    )
    readback_report_path = tmp_path / "readback.json"
    _write_json(readback_report_path, readback_report)
    return build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )


def test_source_span_promotion_executor_dry_run_plans_policy_gate_ready_rows(tmp_path: Path) -> None:
    policy_gate_report_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_report_path, _ready_policy_gate_report(tmp_path))

    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=policy_gate_report_path,
    )

    assert report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert report["status"] == "ok"
    assert report["counts"]["inputRows"] == 2
    assert report["counts"]["dryRunPlannedSourceSpanRows"] == 2
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert report["counts"]["databaseMutationRows"] == 0
    assert report["gate"]["readyForSourceSpanPromotionApply"] is False
    assert {row["executor_dry_run_status"] for row in report["rows"]} == {EXECUTOR_STATUS_PLANNED}
    assert all(row["writeMatrix"]["writeEnabled"] is False for row in report["rows"])
    assert all(
        row["plannedSourceSpanKey"].startswith("source-span:")
        and row["plannedSourceSpanKey"] == row["plannedSourceSpanId"]
        for row in report["rows"]
    )
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_executor_dry_run_blocks_policy_gate_violations(tmp_path: Path) -> None:
    policy_gate_report = _ready_policy_gate_report(tmp_path)
    policy_gate_report["rows"][0]["policy_gate_status"] = "blocked_missing_source_hash"
    policy_gate_report["rows"][0]["sourceSpanPromotionExecutorDryRunReady"] = False
    policy_gate_report_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_report_path, policy_gate_report)

    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=policy_gate_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["dryRunPlannedSourceSpanRows"] == 1
    assert report["counts"]["blockedPolicyGateNotReadyRows"] == 1
    assert report["counts"]["sourceSpanCreatedRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_executor_dry_run_blocks_invalid_input_schema(tmp_path: Path) -> None:
    policy_gate_report = _ready_policy_gate_report(tmp_path)
    policy_gate_report["schema"] = "wrong.schema"
    policy_gate_report_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_report_path, policy_gate_report)

    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=policy_gate_report_path,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["schemaViolationCount"] > 0
    assert report["counts"]["blockedInputSchemaViolationRows"] == 2
    assert report["counts"]["dryRunPlannedSourceSpanRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_executor_dry_run_filters_by_paper_id(tmp_path: Path) -> None:
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
    _write_json(readback_report_path, readback_report)
    policy_gate_report = build_parsed_artifact_source_span_candidate_promotion_policy_gate(
        readback_report_path=readback_report_path,
    )
    policy_gate_report_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_report_path, policy_gate_report)

    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=policy_gate_report_path,
        paper_ids=["paper-2"],
    )

    assert report["counts"]["inputRows"] == 1
    assert report["counts"]["byPaperId"] == {"paper-2": 1}
    assert report["counts"]["dryRunPlannedSourceSpanRows"] == 1
    assert validate_payload(
        report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok


def test_source_span_promotion_executor_dry_run_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    policy_gate_report_path = tmp_path / "policy-gate.json"
    _write_json(policy_gate_report_path, _ready_policy_gate_report(tmp_path))
    report = build_parsed_artifact_source_span_promotion_executor_dry_run(
        policy_gate_report_path=policy_gate_report_path,
    )

    paths = write_parsed_artifact_source_span_promotion_executor_dry_run_reports(
        report,
        tmp_path / "reports",
    )

    written_report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    written_summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

    assert written_report["schema"] == PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID
    assert written_summary["counts"]["dryRunPlannedSourceSpanRows"] == 2
    assert "Parsed Artifact SourceSpan Promotion Executor Dry Run" in markdown
    assert "source spans created: 0" in markdown
    assert validate_payload(
        written_report,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
    assert validate_payload(
        written_summary,
        PARSED_ARTIFACT_SOURCE_SPAN_PROMOTION_EXECUTOR_DRY_RUN_SCHEMA_ID,
        strict=True,
    ).ok
