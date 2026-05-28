from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.evidence_chunk_answer_preview import PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_packet import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
    READY_DECISION as PACKET_READY_DECISION,
)


def _payload(*, status: str = "ok", answerable: bool = True, terms: list[str] | None = None) -> dict[str, Any]:
    text = " ".join(terms or ["alpha", "method"])
    citations = 2 if answerable else 0
    return {
        "schema": PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
        "status": status,
        "mode": "labs_opt_in_preview",
        "question": "Question?",
        "paperIds": ["paper-a"],
        "sourceType": "paper",
        "retrievalMode": "semantic",
        "alpha": 0.7,
        "topK": 8,
        "allowExternal": False,
        "queryPlan": {
            "family": "paper_lookup",
            "parsed_artifact_evidence_chunk_adapter": "runtime_v1",
            "parsedArtifactEvidenceChunkAdapter": "runtime_v1",
            "resolvedPaperIds": ["paper-a"],
        },
        "answer": "Synthetic answer",
        "answerable": answerable,
        "adapterDiagnostics": {"status": "applied" if answerable else "skipped", "rowsAdded": citations},
        "evidencePacketSummary": {
            "answerable": answerable,
            "selectedEvidenceCount": citations,
            "citationCount": citations,
            "adapterStatus": "applied" if answerable else "skipped",
            "adapterRowsAdded": citations,
            "adapterCandidateRowsConsidered": citations,
        },
        "evidencePacketContractSummary": {"answerable": answerable, "spanRows": citations},
        "citations": [{"source_id": "paper-a"}] if answerable else [],
        "sources": [{"source_type": "paper", "source_id": "paper-a", "text": text}] if answerable else [],
        "warnings": [],
        "safety": {
            "labsOnly": True,
            "publicKhubAskChanged": False,
            "defaultMcpAskChanged": False,
            "explicitPaperIdsRequired": True,
            "sourceTypeForced": "paper",
            "externalModelCallsAllowed": False,
        },
    }


def _packet_row(
    case_id: str,
    *,
    status: str = "ok",
    answerable: bool = True,
    terms: list[str] | None = None,
) -> dict[str, Any]:
    citations = 2 if answerable else 0
    return {
        "packetRowId": f"user-test:{case_id}",
        "caseId": case_id,
        "testType": "expected_answerable" if answerable else "expected_no_evidence",
        "surface": "khub_labs_cli",
        "command": f"khub labs paper evidence-chunk-ask 'Question {case_id}?' --paper-id paper-a --json",
        "expectedExitCode": 0,
        "paperIds": ["paper-a"],
        "expectedStatus": status,
        "expectedAnswerable": answerable,
        "expectedMinCitations": citations,
        "expectedMinSpanRows": citations,
        "expectedEvidenceTerms": terms or [],
        "expectedNoEvidence": not answerable,
        "expectedNoExternalCall": True,
        "expectedNoPrivatePathLeak": True,
        "jsonAssertions": [
            {"jsonPath": "$.schema", "operator": "equals", "expected": PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID},
            {"jsonPath": "$.status", "operator": "equals", "expected": status},
            {"jsonPath": "$.answerable", "operator": "equals", "expected": answerable},
            {"jsonPath": "$.sourceType", "operator": "equals", "expected": "paper"},
            {"jsonPath": "$.allowExternal", "operator": "equals", "expected": False},
            {
                "jsonPath": "$.queryPlan.parsed_artifact_evidence_chunk_adapter",
                "operator": "equals",
                "expected": "runtime_v1",
            },
            {"jsonPath": "$.evidencePacketSummary.citationCount", "operator": "atLeast", "expected": citations},
            {"jsonPath": "$.evidencePacketContractSummary.spanRows", "operator": "atLeast", "expected": citations},
            *[
                {"jsonPath": "$.sources[*].text", "operator": "containsCaseInsensitive", "expected": term}
                for term in (terms or [])
            ],
        ],
        "answerTextIncludedInPacket": False,
        "citationPayloadIncludedInPacket": False,
        "sourcePayloadIncludedInPacket": False,
        "excerptIncludedInPacket": False,
    }


def _packet_report(**updates: Any) -> dict[str, Any]:
    rows = [
        _packet_row("positive", terms=["alpha", "method"]),
        _packet_row("missing", status="no_evidence", answerable=False),
    ]
    report = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PACKET_SCHEMA_ID,
        "status": "ready",
        "decision": PACKET_READY_DECISION,
        "counts": {
            "packetRows": 2,
            "externalRejectionCommandRows": 1,
            "schemaViolationCount": 0,
            "privatePathLeakRows": 0,
        },
        "gate": {"readyForLabsOptInUserTestOutputCapture": True},
        "rows": rows,
        "externalRejection": {
            "packetRowId": "user-test:external-rejection",
            "caseId": "external_rejection",
            "testType": "external_rejection",
            "surface": "khub_labs_cli",
            "command": "khub labs paper evidence-chunk-ask 'Question?' --paper-id paper-a --allow-external --json",
            "expectedExitCode": 1,
            "paperIds": ["paper-a"],
            "expectedErrorContains": "--allow-external is not enabled",
            "expectedNoExternalCall": True,
            "expectedNoPrivatePathLeak": True,
        },
    }
    report.update(updates)
    return report


def _capture(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    _ = papers_dir
    answerable = bool(row.get("expectedAnswerable"))
    terms = list(row.get("expectedEvidenceTerms") or [])
    return {
        "exitCode": 0,
        "stdout": json.dumps(_payload(status=row["expectedStatus"], answerable=answerable, terms=terms)),
        "payload": _payload(status=row["expectedStatus"], answerable=answerable, terms=terms),
        "localFakeLlmCallRows": 1 if answerable else 0,
    }


def _external(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
    _ = row, papers_dir
    return {"exitCode": 1, "stdout": "Error: --allow-external is not enabled for evidence-chunk-ask.", "payload": {}}


def test_output_capture_ready_and_sanitized() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        user_test_packet_report=_packet_report(),
        generated_at="2026-05-29T00:00:00Z",
        invoke_row=_capture,
        invoke_external=_external,
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["counts"]["capturedCommandRows"] == 3
    assert report["counts"]["outputCapturePassRows"] == 3
    assert report["counts"]["jsonAssertionFailRows"] == 0
    assert report["counts"]["rawOutputPersistedRows"] == 0
    assert report["counts"]["answerTextIncludedRows"] == 0
    assert report["externalRejection"]["pass"] is True
    assert "answer" not in report["rows"][0]
    assert "sources" not in report["rows"][0]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        strict=True,
    ).ok


def test_output_capture_blocks_when_packet_not_ready() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        user_test_packet_report=_packet_report(status="blocked"),
        generated_at="2026-05-29T00:00:00Z",
        invoke_row=_capture,
        invoke_external=_external,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["capturedCommandRows"] == 0
    assert "user_test_packet_not_ready" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        strict=True,
    ).ok


def test_output_capture_blocks_failed_assertion() -> None:
    def bad_capture(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        captured = _capture(row, papers_dir)
        captured["payload"]["status"] = "wrong"
        return captured

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        user_test_packet_report=_packet_report(),
        generated_at="2026-05-29T00:00:00Z",
        invoke_row=bad_capture,
        invoke_external=_external,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["jsonAssertionFailRows"] > 0
    assert "user_test_output_capture_failures:2" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        strict=True,
    ).ok


def test_output_capture_blocks_private_path_payload() -> None:
    def private_capture(row: dict[str, Any], papers_dir: str | Path) -> dict[str, Any]:
        captured = _capture(row, papers_dir)
        captured["payload"]["warnings"] = ["/" + "Users/example/private"]
        return captured

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        user_test_packet_report=_packet_report(),
        generated_at="2026-05-29T00:00:00Z",
        invoke_row=private_capture,
        invoke_external=_external,
    )

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 2
    assert "private_path_leak" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        strict=True,
    ).ok


def test_output_capture_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        user_test_packet_report=_packet_report(),
        generated_at="2026-05-29T00:00:00Z",
        invoke_row=_capture,
        invoke_external=_external,
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Output Capture"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        strict=True,
    ).ok
