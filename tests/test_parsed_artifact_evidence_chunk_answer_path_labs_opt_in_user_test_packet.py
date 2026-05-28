from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_runner import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
    READY_DECISION as RUNNER_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet,
)


def _runner_row(
    case_id: str,
    *,
    paper_ids: list[str],
    status: str = "ok",
    answerable: bool = True,
    terms: list[str] | None = None,
    citations: int = 2,
    spans: int = 2,
    llm_calls: int = 1,
) -> dict[str, Any]:
    return {
        "caseId": case_id,
        "evalFocus": "single_paper" if len(paper_ids) == 1 else "two_paper_compare_seed",
        "expectedStatus": status,
        "observedStatus": status,
        "expectedAnswerable": answerable,
        "observedAnswerable": answerable,
        "paperIds": paper_ids,
        "observedSourceIds": paper_ids if answerable else [],
        "missingSourceIds": [],
        "requiredEvidenceTerms": terms or [],
        "observedEvidenceTerms": terms or [],
        "missingEvidenceTerms": [],
        "adapterStatus": "applied" if answerable else "skipped",
        "adapterRowsAdded": citations,
        "selectedEvidenceCount": citations,
        "citationCount": citations,
        "evidencePacketContractSpanRows": spans,
        "localFakeLlmCallRows": llm_calls,
        "dimensionStatuses": {
            "schemaValid": True,
            "answerabilityExpectation": True,
            "sourceCoverage": True,
            "evidenceTermSupport": True,
            "citationAndSpanPresence": True,
            "expectedNoEvidenceSafety": True,
        },
        "qualityScore": 1.0,
        "qualityGrade": "pass",
        "answerTextHash": "sha256:" + "a" * 64,
        "answerTextIncludedInReport": False,
        "citationPayloadIncludedInReport": False,
        "sourcePayloadIncludedInReport": False,
        "excerptIncludedInReport": False,
        "candidateRecordIds": ["candidate-a"] if answerable else [],
        "failureReasons": [],
        "pass": True,
        "question": f"What evidence exists for {case_id}?",
    }


def _runner_report(**updates: Any) -> dict[str, Any]:
    rows = [
        _runner_row("paper_a_positive", paper_ids=["paper-a"], terms=["alpha", "method"]),
        _runner_row("pair_positive", paper_ids=["paper-a", "paper-b"], terms=["alpha", "vector"], citations=4, spans=4),
        _runner_row(
            "missing_no_evidence",
            paper_ids=["missing-paper"],
            status="no_evidence",
            answerable=False,
            terms=[],
            citations=0,
            spans=0,
            llm_calls=0,
        ),
    ]
    report = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_QUALITY_EVAL_RUNNER_SCHEMA_ID,
        "status": "ready",
        "decision": RUNNER_READY_DECISION,
        "counts": {
            "inputCaseRows": 3,
            "qualityPassRows": 3,
            "qualityPartialRows": 0,
            "qualityFailRows": 0,
            "expectedNoEvidenceRows": 1,
            "noEvidenceLlmCallRows": 0,
            "minQualityScore": 1.0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForLabsOptInUserTestPacket": True,
        },
        "rows": rows,
    }
    report.update(updates)
    return report


def test_labs_opt_in_user_test_packet_ready() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
        quality_eval_runner_report=_runner_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["packetRows"] == 3
    assert report["counts"]["expectedAnswerableCommandRows"] == 2
    assert report["counts"]["expectedNoEvidenceCommandRows"] == 1
    assert report["counts"]["externalRejectionCommandRows"] == 1
    assert report["counts"]["answerTextIncludedRows"] == 0
    assert report["gate"]["readyForLabsOptInUserTestOutputCapture"] is True
    assert report["instructions"]["mcpProfileRequirement"] == "labs_or_all_only"
    assert report["rows"][0]["command"].startswith("khub labs paper evidence-chunk-ask")
    assert "--allow-external" not in report["rows"][0]["command"]
    assert "--allow-external" in report["externalRejection"]["command"]
    assert all(row["answerTextIncludedInPacket"] is False for row in report["rows"])
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_user_test_packet_blocks_when_runner_not_ready() -> None:
    runner = _runner_report(status="blocked")

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
        quality_eval_runner_report=runner,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["packetRows"] == 0
    assert "quality_eval_runner_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_user_test_packet_blocks_private_path_marker() -> None:
    runner = _runner_report()
    runner["rows"][0]["question"] = "/Users/example/private marker"

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
        quality_eval_runner_report=runner,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "private_path_leak" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        strict=True,
    ).ok


def test_labs_opt_in_user_test_packet_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
        quality_eval_runner_report=_runner_report(),
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Packet"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        strict=True,
    ).ok
