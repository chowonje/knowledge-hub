from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
    READY_DECISION as OUTPUT_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
    PUBLIC_DEFAULT_HOLD_REASON,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review,
    write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review,
)


PRODUCT_DEFINITION_TEXT = """
A local-first, evidence-first research knowledge runtime for auditable AI research workflows.
A section/paragraph evidence-first paper QA and compare runtime for a local AI-paper corpus.
Lack of evidence produces abstain/no-answer rather than unsupported synthesis.
Public CLI/MCP surfaces match the documented Research Preview promise.
"""


def _capture_report(**updates: Any) -> dict[str, Any]:
    report = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
        "status": "ready",
        "decision": OUTPUT_READY_DECISION,
        "counts": {
            "capturedCommandRows": 5,
            "outputCapturePassRows": 5,
            "outputCaptureFailRows": 0,
            "jsonAssertionRows": 47,
            "jsonAssertionPassRows": 47,
            "jsonAssertionFailRows": 0,
            "expectedAnswerableOutputRows": 3,
            "observedAnswerableOutputRows": 3,
            "expectedNoEvidenceOutputRows": 1,
            "observedNoEvidenceOutputRows": 1,
            "externalRejectionPassRows": 1,
            "rawOutputPersistedRows": 0,
            "answerTextIncludedRows": 0,
            "citationPayloadIncludedRows": 0,
            "sourcePayloadIncludedRows": 0,
            "excerptIncludedRows": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForLabsOptInUserTestPromotionReview": True,
            "allCapturedCommandsPassed": True,
            "allJsonAssertionsPassed": True,
            "expectedNoEvidenceCasesStayedNoEvidence": True,
            "externalRequestRejected": True,
            "noRawOutputPersisted": True,
        },
    }
    report.update(updates)
    return report


def test_promotion_review_marks_labs_limited_ready_and_public_default_held() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        output_capture_report=_capture_report(),
        product_definition_text=PRODUCT_DEFINITION_TEXT,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["releaseDecision"]["v01ScopeDecision"] == "labs_limited_rc_candidate_ready"
    assert report["releaseDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["releaseDecision"]["publicDefaultHoldReason"] == PUBLIC_DEFAULT_HOLD_REASON
    assert report["counts"]["labsLimitedPromotionReadyRows"] == 1
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["gate"]["readyForV01LabsLimitedReleaseGate"] is True
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_review_blocks_when_output_capture_not_ready() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        output_capture_report=_capture_report(status="blocked"),
        product_definition_text=PRODUCT_DEFINITION_TEXT,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "output_capture_not_ready" in report["gate"]["semanticViolations"]
    assert report["counts"]["labsLimitedPromotionReadyRows"] == 0
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_review_blocks_when_product_definition_missing_scope() -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        output_capture_report=_capture_report(),
        product_definition_text="local-first only",
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert any(item.startswith("product_definition_missing:") for item in report["gate"]["semanticViolations"])
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_review_blocks_private_path_marker() -> None:
    capture = _capture_report()
    capture["rows"] = [{"warning": "/Users/example/private"}]

    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        output_capture_report=capture,
        product_definition_text=PRODUCT_DEFINITION_TEXT,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert report["status"] == "blocked"
    assert "output_capture_private_path_marker" in report["gate"]["semanticViolations"]
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok


def test_promotion_review_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        output_capture_report=_capture_report(),
        product_definition_text=PRODUCT_DEFINITION_TEXT,
        generated_at="2026-05-29T00:00:00Z",
    )

    paths = write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Promotion Review"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        strict=True,
    ).ok
