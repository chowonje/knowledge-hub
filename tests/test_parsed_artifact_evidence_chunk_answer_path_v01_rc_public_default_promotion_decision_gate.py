from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION as RELEASE_GATE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence import (
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
    READY_DECISION,
    build_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate,
    write_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate,
)


MERGE_SHA = "46db241275f85b5c7737ce7178a58ceae44aa0f8"


def _release_gate(**count_updates: Any) -> dict[str, Any]:
    counts = {
        "releaseGatePassRows": 1,
        "releaseSmokePassedRows": 10,
        "publicHygieneIssueRows": 0,
        "noAnswerPassRows": 3,
        "labsSurfaceSmokePassRows": 1,
        "labsLimitedPromotionReadyRows": 1,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1,
        "privatePathLeakRows": 0,
        "schemaViolationCount": 0,
    }
    counts.update(count_updates)
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        "status": "ready",
        "decision": RELEASE_GATE_READY_DECISION,
        "counts": counts,
    }


def _branch_report(**updates: Any) -> dict[str, Any]:
    payload = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_BRANCH_PR_READINESS_REVIEW_SCHEMA_ID,
        "status": "ready",
        "nextRecommendedTranche": "knowledge_hub_v01_rc_draft_pr_opening_or_handoff",
    }
    payload.update(updates)
    return payload


def _handoff_report(**updates: Any) -> dict[str, Any]:
    payload = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_OPENING_OR_HANDOFF_SCHEMA_ID,
        "status": "ready",
        "nextRecommendedTranche": "operator_push_and_open_draft_pr_or_request_codex_pr_creation",
    }
    payload.update(updates)
    return payload


def _git_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "branchName": "codex/v01-rc-post-merge-convergence-20260529",
        "headSha": MERGE_SHA,
        "headShortSha": "46db241",
        "originMainSha": MERGE_SHA,
        "originMainShortSha": "46db241",
        "remoteMainSha": MERGE_SHA,
        "statusRows": [],
    }
    payload.update(updates)
    return payload


def _pr_state(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "lookupStatus": "ok",
        "number": 169,
        "title": "KnowledgeOS v0.1 RC labs evidence chunk preview",
        "state": "MERGED",
        "isDraft": False,
        "mergedAt": "2026-05-29T02:03:02Z",
        "mergeCommit": {"oid": MERGE_SHA},
        "headRefName": "codex/next-implementation-20260528",
        "baseRefName": "main",
        "statusCheckRollup": [
            {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS", "name": "static-guards"},
            {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS", "name": "python-hermetic"},
        ],
    }
    payload.update(updates)
    return payload


def _post_merge_report(**updates: Any) -> dict[str, Any]:
    report = build_parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence(
        release_gate_report=_release_gate(),
        branch_pr_readiness_report=_branch_report(),
        draft_pr_handoff_report=_handoff_report(),
        git_state=_git_state(),
        github_pr_state=_pr_state(),
        hygiene_result={"status": "ok", "issueCount": 0},
        release_smoke_result={"status": "ok", "checkedCount": 10, "passedCount": 10},
        generated_at="2026-05-29T00:00:00Z",
    )
    for key, value in updates.items():
        report[key] = value
    return report


def _build(post_merge_report: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate(
        post_merge_convergence_report=post_merge_report or _post_merge_report(),
        generated_at="2026-05-29T00:00:00Z",
    )


def test_public_default_promotion_gate_holds_default_and_allows_release_notes_path() -> None:
    report = _build()

    assert report["status"] == "ready"
    assert report["decision"] == READY_DECISION
    assert report["promotionDecision"]["researchPreviewDecision"] == "release_notes_path_allowed"
    assert report["promotionDecision"]["publicDefaultDecision"] == "hold_public_default_promotion"
    assert report["promotionDecision"]["defaultSurfaceDecision"] == "do_not_enable_default_ask_or_default_mcp"
    assert report["counts"]["releaseNotesPathAllowedRows"] == 1
    assert report["counts"]["publicDefaultPromotionReadyRows"] == 0
    assert report["counts"]["publicDefaultPromotionHeldRows"] == 1
    assert report["counts"]["generalRcReadyRows"] == 0
    assert report["counts"]["corpusScaleClaimProvenRows"] == 0
    assert report["gate"]["publicDefaultPromotionAllowed"] is False
    assert validate_payload(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
        strict=True,
    ).ok


def test_public_default_promotion_gate_blocks_when_post_merge_report_is_not_ready() -> None:
    post_merge = _post_merge_report(status="blocked")
    report = _build(post_merge)

    assert report["status"] == "blocked"
    assert "post_merge_convergence_not_ready" in report["gate"]["semanticViolations"]


def test_public_default_promotion_gate_blocks_unexpected_public_default_ready_counter() -> None:
    post_merge = _post_merge_report()
    post_merge["counts"]["publicDefaultPromotionReadyRows"] = 1
    report = _build(post_merge)

    assert report["status"] == "blocked"
    assert "public_default_promotion_ready_unexpected" in report["gate"]["semanticViolations"]


def test_public_default_promotion_gate_blocks_unsafe_upstream_counters() -> None:
    post_merge = _post_merge_report()
    post_merge["counts"]["candidateStoreWriteRows"] = 1
    report = _build(post_merge)

    assert report["status"] == "blocked"
    assert "unsafe_counter_nonzero:candidateStoreWriteRows" in report["gate"]["semanticViolations"]


def test_public_default_promotion_gate_blocks_private_path_marker() -> None:
    post_merge = _post_merge_report()
    marker = "/" + "Users" + "/example/private"
    post_merge["warnings"] = [marker]
    report = _build(post_merge)

    assert report["status"] == "blocked"
    assert report["counts"]["privatePathLeakRows"] == 1
    assert "post_merge_convergence_private_path_marker" in report["gate"]["semanticViolations"]


def test_public_default_promotion_gate_writer_outputs_schema_valid_reports(tmp_path: Path) -> None:
    report = _build()

    paths = write_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate(
        report,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    parsed = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert parsed["status"] == "ready"
    assert Path(paths["markdown"]).read_text(encoding="utf-8").startswith(
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Public/Default Promotion Decision Gate"
    )
    assert validate_payload(
        parsed,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
        strict=True,
    ).ok
