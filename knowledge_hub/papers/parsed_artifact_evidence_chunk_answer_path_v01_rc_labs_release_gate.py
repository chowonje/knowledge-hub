"""v0.1 labs release gate for the parsed-artifact evidence chunk answer path."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.application.public_release_hygiene import PUBLIC_RELEASE_HYGIENE_SCHEMA
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID,
    READY_DECISION as DEFAULT_OFF_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID,
    READY_DECISION as SURFACE_LIVE_SMOKE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
    READY_DECISION as PROMOTION_REVIEW_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-labs-release-gate.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_blocked"
NEXT_TRANCHE_READY = "knowledge_hub_v01_rc_branch_pr_readiness_review"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_repair"
PUBLIC_DEFAULT_HOLD_REASON = "public_default_promotion_remains_held_after_labs_release_gate"

DEFAULT_PROMOTION_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review.v1.json"
)
DEFAULT_DEFAULT_OFF_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json"
)
DEFAULT_SURFACE_LIVE_SMOKE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke.v1.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _release_smoke_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("status") != "ok":
        blockers.append("release_smoke_not_ok")
    if _int(payload.get("checkedCount")) <= 0:
        blockers.append("release_smoke_no_checks")
    if _int(payload.get("checkedCount")) != _int(payload.get("passedCount")):
        blockers.append("release_smoke_not_all_passed")
    for item in list(payload.get("commands") or []):
        if dict(item).get("status") != "ok":
            blockers.append(f"release_smoke_command_failed:{_clean_text(dict(item).get('name'))}")
    return sorted(set(blockers))


def _public_hygiene_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("schema") != PUBLIC_RELEASE_HYGIENE_SCHEMA:
        blockers.append("public_hygiene_schema_mismatch")
    if payload.get("status") != "ok":
        blockers.append("public_hygiene_not_ok")
    if _int(payload.get("issueCount")) != 0:
        blockers.append("public_hygiene_issues_present")
    for kind, count in dict(payload.get("issueCountsByKind") or {}).items():
        if _int(count) > 0:
            blockers.append(f"public_hygiene_issue:{_clean_text(kind)}")
    return sorted(set(blockers))


def _promotion_review_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID:
        blockers.append("promotion_review_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("promotion_review_not_ready")
    if report.get("decision") != PROMOTION_REVIEW_READY_DECISION:
        blockers.append("promotion_review_decision_not_ready")
    if gate.get("readyForV01LabsLimitedReleaseGate") is not True:
        blockers.append("promotion_review_not_ready_for_v01_labs_gate")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("promotion_review_allowed_public_default")
    if gate.get("publicDefaultPromotionHeld") is not True:
        blockers.append("promotion_review_did_not_hold_public_default")
    if _int(counts.get("labsLimitedPromotionReadyRows")) < 1:
        blockers.append("promotion_review_labs_ready_missing")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("promotion_review_public_default_ready")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("promotion_review_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("promotion_review_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("promotion_review_schema_violations_present")
    return sorted(set(blockers))


def _default_off_no_answer_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_DEFAULT_OFF_NO_ANSWER_REGRESSION_SMOKE_SCHEMA_ID:
        blockers.append("default_off_no_answer_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("default_off_no_answer_not_ready")
    if report.get("decision") != DEFAULT_OFF_READY_DECISION:
        blockers.append("default_off_no_answer_decision_not_ready")
    if gate.get("allScenariosNoAnswer") is not True:
        blockers.append("default_off_no_answer_scenarios_not_all_no_answer")
    if gate.get("answerabilityStayedFalse") is not True:
        blockers.append("default_off_no_answer_answerability_changed")
    if gate.get("adapterNeverApplied") is not True:
        blockers.append("default_off_no_answer_adapter_applied")
    if gate.get("noLlmCalls") is not True:
        blockers.append("default_off_no_answer_llm_called")
    input_rows = _int(counts.get("inputScenarioRows"))
    if input_rows <= 0:
        blockers.append("default_off_no_answer_no_scenarios")
    if _int(counts.get("passRows")) != input_rows:
        blockers.append("default_off_no_answer_not_all_passed")
    if _int(counts.get("noAnswerRows")) != input_rows:
        blockers.append("default_off_no_answer_not_all_no_answer")
    if _int(counts.get("answerableRows")) != 0:
        blockers.append("default_off_no_answer_answerable_rows_present")
    if _int(counts.get("adapterAppliedRows")) != 0 or _int(counts.get("adapterRowsAdded")) != 0:
        blockers.append("default_off_no_answer_adapter_rows_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("default_off_no_answer_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("default_off_no_answer_schema_violations_present")
    return sorted(set(blockers))


def _surface_live_smoke_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_SURFACE_LIVE_SMOKE_SCHEMA_ID:
        blockers.append("surface_live_smoke_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("surface_live_smoke_not_ready")
    if report.get("decision") != SURFACE_LIVE_SMOKE_READY_DECISION:
        blockers.append("surface_live_smoke_decision_not_ready")
    required_true = {
        "surfacePayloadSchemaValid": "surface_live_smoke_payload_schema_invalid",
        "runtimeAdapterApplied": "surface_live_smoke_adapter_not_applied",
        "answerabilityReachedLabsSurface": "surface_live_smoke_answerability_not_reached",
        "externalRequestRejected": "surface_live_smoke_external_request_not_rejected",
        "publicDefaultUnchanged": "surface_live_smoke_public_default_changed",
        "defaultMcpAskClosed": "surface_live_smoke_default_mcp_open",
        "labsMcpToolHiddenFromDefault": "surface_live_smoke_labs_mcp_visible_in_default",
    }
    for key, blocker in required_true.items():
        if gate.get(key) is not True:
            blockers.append(blocker)
    if _int(counts.get("surfaceSmokePassRows")) < 1:
        blockers.append("surface_live_smoke_no_pass_rows")
    if _int(counts.get("surfacePayloadSchemaValidRows")) < 1:
        blockers.append("surface_live_smoke_no_schema_valid_rows")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("surface_live_smoke_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("surface_live_smoke_schema_violations_present")
    return sorted(set(blockers))


def _release_smoke_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(payload.get("commands") or []):
        command = dict(item)
        rows.append(
            {
                "checkId": _clean_text(command.get("name")),
                "status": "pass" if command.get("status") == "ok" else "fail",
                "summary": _clean_text(command.get("summary")),
                "durationSec": float(command.get("durationSec") or 0.0),
            }
        )
    return rows


def _hygiene_issue_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(payload.get("issues") or []):
        issue = dict(item)
        rows.append(
            {
                "kind": _clean_text(issue.get("kind")),
                "path": _clean_text(issue.get("path")),
                "detail": _clean_text(issue.get("detail")),
            }
        )
    return rows


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(
    *,
    promotion_review_report_path: str | Path = DEFAULT_PROMOTION_REVIEW_REPORT,
    default_off_no_answer_report_path: str | Path = DEFAULT_DEFAULT_OFF_REPORT,
    surface_live_smoke_report_path: str | Path = DEFAULT_SURFACE_LIVE_SMOKE_REPORT,
    promotion_review_report: dict[str, Any] | None = None,
    release_smoke_payload: dict[str, Any] | None = None,
    public_hygiene_payload: dict[str, Any] | None = None,
    default_off_no_answer_report: dict[str, Any] | None = None,
    surface_live_smoke_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    promotion_report = dict(promotion_review_report or _read_json(promotion_review_report_path))
    default_off_report = dict(default_off_no_answer_report or _read_json(default_off_no_answer_report_path))
    surface_report = dict(surface_live_smoke_report or _read_json(surface_live_smoke_report_path))
    release_payload = dict(release_smoke_payload or {})
    hygiene_payload = dict(public_hygiene_payload or {})

    promotion_blockers = _promotion_review_blockers(promotion_report)
    release_blockers = _release_smoke_blockers(release_payload)
    hygiene_blockers = _public_hygiene_blockers(hygiene_payload)
    no_answer_blockers = _default_off_no_answer_blockers(default_off_report)
    surface_blockers = _surface_live_smoke_blockers(surface_report)
    semantic_violations = sorted(
        set(promotion_blockers + release_blockers + hygiene_blockers + no_answer_blockers + surface_blockers)
    )
    status = "ready" if not semantic_violations else "blocked"

    promotion_counts = dict(promotion_report.get("counts") or {})
    default_off_counts = dict(default_off_report.get("counts") or {})
    surface_counts = dict(surface_report.get("counts") or {})
    release_rows = _release_smoke_rows(release_payload)
    hygiene_rows = _hygiene_issue_rows(hygiene_payload)
    counts = {
        "promotionReviewReadyRows": 1 if not promotion_blockers else 0,
        "releaseSmokeCheckedRows": _int(release_payload.get("checkedCount")),
        "releaseSmokePassedRows": _int(release_payload.get("passedCount")),
        "releaseSmokeFailedRows": max(
            0, _int(release_payload.get("checkedCount")) - _int(release_payload.get("passedCount"))
        ),
        "publicHygieneIssueRows": _int(hygiene_payload.get("issueCount")),
        "publicHygieneTrackedFileRows": _int(hygiene_payload.get("trackedFileCount")),
        "noAnswerScenarioRows": _int(default_off_counts.get("inputScenarioRows")),
        "noAnswerPassRows": _int(default_off_counts.get("passRows")),
        "noAnswerRows": _int(default_off_counts.get("noAnswerRows")),
        "answerableRows": _int(default_off_counts.get("answerableRows")),
        "adapterAppliedRows": _int(default_off_counts.get("adapterAppliedRows")),
        "adapterRowsAdded": _int(default_off_counts.get("adapterRowsAdded")),
        "labsSurfaceSmokePassRows": _int(surface_counts.get("surfaceSmokePassRows")),
        "labsSurfacePayloadSchemaValidRows": _int(surface_counts.get("surfacePayloadSchemaValidRows")),
        "labsLimitedPromotionReadyRows": _int(promotion_counts.get("labsLimitedPromotionReadyRows")),
        "publicDefaultPromotionReadyRows": _int(promotion_counts.get("publicDefaultPromotionReadyRows")),
        "publicDefaultPromotionHeldRows": _int(promotion_counts.get("publicDefaultPromotionHeldRows")),
        "releaseGatePassRows": 1 if status == "ready" else 0,
        "releaseGateBlockedRows": 0 if status == "ready" else 1,
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "rawOutputPersistedRows": 0,
        "answerTextIncludedRows": 0,
        "citationPayloadIncludedRows": 0,
        "sourcePayloadIncludedRows": 0,
        "excerptIncludedRows": 0,
        "privatePathLeakRows": 0,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "promotionReviewReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review.v1.json"
            ),
            "defaultOffNoAnswerReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke.v1.json"
            ),
            "surfaceLiveSmokeReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_live_smoke.v1.json"
            ),
            "releaseSmokeCommand": "python scripts/check_release_smoke.py --mode release --json",
            "publicHygieneCommand": "python scripts/check_public_release_hygiene.py --repo-root . --json",
        },
        "policy": {
            "reportOnly": True,
            "labsLimitedRcGate": True,
            "publicDefaultPromotionAllowed": False,
            "publicKhubAskChanged": False,
            "defaultMcpAskChanged": False,
            "runtimeDefaultChange": False,
            "rawReleaseSmokePayloadPersisted": False,
            "rawPublicHygienePayloadPersisted": False,
            "rawOutputPersisted": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
        },
        "releaseDecision": {
            "v01LabsReleaseGateDecision": "ready" if status == "ready" else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "publicDefaultHoldReason": PUBLIC_DEFAULT_HOLD_REASON,
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "readyForV01RcBranchPrReadinessReview": status == "ready",
            "promotionReviewReady": not promotion_blockers,
            "releaseSmokePassed": not release_blockers,
            "publicHygienePassed": not hygiene_blockers,
            "noAnswerRegressionPassed": not no_answer_blockers,
            "labsSurfaceSmokePassed": not surface_blockers,
            "publicDefaultPromotionAllowed": False,
            "publicDefaultPromotionHeld": status == "ready",
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {
                "checkId": "promotion_review",
                "status": "pass" if not promotion_blockers else "fail",
                "blockers": promotion_blockers,
            },
            {
                "checkId": "release_smoke",
                "status": "pass" if not release_blockers else "fail",
                "blockers": release_blockers,
            },
            {
                "checkId": "public_hygiene",
                "status": "pass" if not hygiene_blockers else "fail",
                "blockers": hygiene_blockers,
            },
            {
                "checkId": "no_answer_regression",
                "status": "pass" if not no_answer_blockers else "fail",
                "blockers": no_answer_blockers,
            },
            {
                "checkId": "labs_surface_smoke",
                "status": "pass" if not surface_blockers else "fail",
                "blockers": surface_blockers,
            },
        ],
        "releaseSmokeRows": release_rows,
        "publicHygieneIssueRows": hygiene_rows,
        "warnings": [
            "public_default_promotion_remains_held_even_when_labs_release_gate_is_ready",
            "release_smoke_and_public_hygiene_payloads_are_summarized_to_avoid_local_path_leaks",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("releaseDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Labs Release Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- v01LabsReleaseGateDecision: `{decision.get('v01LabsReleaseGateDecision')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- releaseSmokeCheckedRows: `{counts.get('releaseSmokeCheckedRows')}`",
        f"- releaseSmokePassedRows: `{counts.get('releaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- noAnswerScenarioRows: `{counts.get('noAnswerScenarioRows')}`",
        f"- noAnswerPassRows: `{counts.get('noAnswerPassRows')}`",
        f"- labsSurfaceSmokePassRows: `{counts.get('labsSurfaceSmokePassRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Release Smoke", ""])
    for row in list(report.get("releaseSmokeRows") or []):
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate",
]
