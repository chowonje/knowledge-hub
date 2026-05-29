from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.public_release_hygiene import PUBLIC_RELEASE_HYGIENE_SCHEMA
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
    READY_DECISION as DEFAULT_OFF_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION as SURFACE_LIVE_SMOKE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
    READY_DECISION as PROMOTION_REVIEW_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate,
)


def _promotion_review() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        "status": "ready",
        "decision": PROMOTION_REVIEW_READY_DECISION,
        "counts": {
            "labsLimitedPromotionReadyRows": 1,
            "publicDefaultPromotionReadyRows": 0,
            "publicDefaultPromotionHeldRows": 1,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "readyForV01LabsLimitedReleaseGate": True,
            "publicDefaultPromotionAllowed": False,
            "publicDefaultPromotionHeld": True,
        },
    }


def _release_smoke() -> dict[str, Any]:
    return {
        "status": "ok",
        "checkedCount": 2,
        "passedCount": 2,
        "commands": [
            {
                "name": "top_help",
                "status": "ok",
                "summary": "top-level help surface is present",
                "durationSec": 0.1,
                "argv": ["/" + "Users/example/.pyenv/bin/python"],
            },
            {
                "name": "doctor",
                "status": "ok",
                "summary": "doctor returned accepted local status=needs_setup",
                "durationSec": 0.2,
                "details": {"repoRoot": "/" + "Users/example/repo"},
            },
        ],
    }


def _public_hygiene() -> dict[str, Any]:
    return {
        "schema": PUBLIC_RELEASE_HYGIENE_SCHEMA,
        "status": "ok",
        "trackedFileCount": 2490,
        "issueCount": 0,
        "issueCountsByKind": {},
        "issues": [],
    }


def _default_off_no_answer() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
        "status": "ready",
        "decision": DEFAULT_OFF_READY_DECISION,
        "counts": {
            "inputScenarioRows": 3,
            "passRows": 3,
            "noAnswerRows": 3,
            "answerableRows": 0,
            "adapterAppliedRows": 0,
            "adapterRowsAdded": 0,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "allScenariosNoAnswer": True,
            "answerabilityStayedFalse": True,
            "adapterNeverApplied": True,
            "noLlmCalls": True,
        },
    }


def _surface_live_smoke() -> dict[str, Any]:
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
        "status": "ready",
        "decision": SURFACE_LIVE_SMOKE_READY_DECISION,
        "counts": {
            "surfaceSmokePassRows": 1,
            "surfacePayloadSchemaValidRows": 1,
            "privatePathLeakRows": 0,
            "schemaViolationCount": 0,
        },
        "gate": {
            "surfacePayloadSchemaValid": True,
            "runtimeAdapterApplied": True,
            "answerabilityReachedLabsSurface": True,
            "externalRequestRejected": True,
            "publicDefaultUnchanged": True,
            "defaultMcpAskClosed": True,
            "labsMcpToolHiddenFromDefault": True,
        },
    }


def _build(**updates: Any) -> dict[str, Any]:
    inputs = {
        "promotion_review_report": _promotion_review(),
        "release_smoke_payload": _release_smoke(),
        "public_hygiene_payload": _public_hygiene(),
        "default_off_no_answer_report": _default_off_no_answer(),
        "surface_live_smoke_report": _surface_live_smoke(),
        "generated_at": "2026-05-29T00:00:00Z",
    }
    inputs.update(updates)
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(**inputs)


def test_v01_rc_labs_release_gate_ready() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["releaseDecision"]["v01LabsReleaseGateDecision"] == "ready"
    assert report["releaseDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["counts"]["releaseSmokePassedRows"] == 2
    assert report["counts"]["publicHygieneIssueRows"] == 0
    assert report["counts"]["noAnswerPassRows"] == 3
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["gate"]["readyForV01RcBranchPrReadinessReview"] is True
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_v01_rc_labs_release_gate_blocks_release_smoke_failure() -> None:
    smoke = _release_smoke()
    smoke["passedCount"] = 1
    smoke["commands"][1]["status"] = "failed"

    report = _build(release_smoke_payload=smoke)

    assert report["status"] == "blocked"
    assert "release_smoke_not_all_passed" in report["gate"]["semanticViolations"]
    assert "release_smoke_command_failed:doctor" in report["gate"]["semanticViolations"]


def test_v01_rc_labs_release_gate_blocks_public_hygiene_issues() -> None:
    hygiene = _public_hygiene()
    hygiene["status"] = "failed"
    hygiene["issueCount"] = 1
    hygiene["issueCountsByKind"] = {"absolute_user_path": 1}
    hygiene["issues"] = [{"kind": "absolute_user_path", "path": "tests/example.py", "detail": "local path"}]

    report = _build(public_hygiene_payload=hygiene)

    assert report["status"] == "blocked"
    assert "public_hygiene_not_ok" in report["gate"]["semanticViolations"]
    assert "public_hygiene_issue:absolute_user_path" in report["gate"]["semanticViolations"]
    assert report["publicHygieneIssueRows"][0]["path"] == "tests/example.py"


def test_v01_rc_labs_release_gate_blocks_no_answer_regression() -> None:
    no_answer = _default_off_no_answer()
    no_answer["counts"]["answerableRows"] = 1
    no_answer["gate"]["answerabilityStayedFalse"] = False

    report = _build(default_off_no_answer_report=no_answer)

    assert report["status"] == "blocked"
    assert "default_off_no_answer_answerability_changed" in report["gate"]["semanticViolations"]
    assert "default_off_no_answer_answerable_rows_present" in report["gate"]["semanticViolations"]


def test_v01_rc_labs_release_gate_does_not_persist_raw_local_paths() -> None:
    report = _build()
    encoded = json.dumps(report, ensure_ascii=False)

    assert "/" + "Users/example" not in encoded
    assert report["counts"]["privatePathLeakRows"] == 0
    assert report["releaseSmokeRows"][0]["checkId"] == "top_help"


def test_v01_rc_labs_release_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Labs Release Gate"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        strict=True,
    ).ok
