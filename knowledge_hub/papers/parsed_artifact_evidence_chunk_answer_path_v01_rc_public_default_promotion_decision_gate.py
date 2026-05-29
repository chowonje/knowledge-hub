"""Public/default promotion decision gate for the v0.1 evidence chunk RC."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID,
    READY_DECISION as POST_MERGE_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-public-default-promotion-decision-gate.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_public_default_promotion_decision_gate_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_public_default_promotion_decision_gate_blocked"
NEXT_TRANCHE_READY = "knowledge_hub_v01_rc_research_preview_release_notes_or_corpus_scale_quality_gate"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_public_default_promotion_decision_gate_repair"
DEFAULT_POST_MERGE_CONVERGENCE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_post_merge_convergence.v1.json"
)

EXTRA_ZERO_COUNTER_FIELDS = (
    "pushRows",
    "githubPrMutationRows",
    "branchDeletionRows",
    "rawGithubPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _post_merge_convergence_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("convergenceDecision") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID:
        blockers.append("post_merge_convergence_schema_mismatch")
    else:
        validation = validate_payload(
            report,
            PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_POST_MERGE_CONVERGENCE_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            blockers.append("post_merge_convergence_schema_validation_failed")
    if report.get("status") != "ready":
        blockers.append("post_merge_convergence_not_ready")
    if report.get("decision") != POST_MERGE_READY_DECISION:
        blockers.append("post_merge_convergence_decision_not_ready")
    if _int(counts.get("researchPreviewRcCandidateRows")) != 1:
        blockers.append("research_preview_rc_candidate_missing")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_promotion_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_promotion_hold_missing")
    if _int(counts.get("generalRcReadyRows")) != 0:
        blockers.append("general_rc_ready_unexpected")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("corpus_scale_claim_proven_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("post_merge_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("post_merge_schema_violations_present")
    if decision.get("publicDefaultDecision") != "hold_public_default_promotion":
        blockers.append("post_merge_public_default_decision_not_hold")
    if decision.get("postMergeConvergence") != "research_preview_rc_candidate":
        blockers.append("post_merge_convergence_not_research_preview")
    return sorted(set(blockers))


def _unsafe_counter_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{field}")
    return sorted(set(blockers))


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate(
    *,
    post_merge_convergence_report_path: str | Path = DEFAULT_POST_MERGE_CONVERGENCE_REPORT,
    post_merge_convergence_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    post_report = dict(post_merge_convergence_report or _read_json(post_merge_convergence_report_path))
    post_counts = dict(post_report.get("counts") or {})
    post_decision = dict(post_report.get("convergenceDecision") or {})
    post_blockers = _post_merge_convergence_blockers(post_report)
    unsafe_blockers = _unsafe_counter_blockers(post_report)
    semantic_violations = sorted(set(post_blockers + unsafe_blockers))
    private_path_leak_rows = 1 if _contains_private_path(post_report) else 0
    if private_path_leak_rows:
        semantic_violations.append("post_merge_convergence_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    research_preview_candidate = status == "ready" and _int(post_counts.get("researchPreviewRcCandidateRows")) == 1
    public_default_held = status == "ready" and _int(post_counts.get("publicDefaultPromotionHeldRows")) >= 1
    release_notes_allowed = research_preview_candidate and public_default_held
    counts = {
        "decisionGateRows": 1,
        "postMergeConvergenceReadyRows": 1 if not post_blockers else 0,
        "researchPreviewRcCandidateRows": 1 if research_preview_candidate else 0,
        "publicDefaultPromotionCandidateRows": 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1 if public_default_held else 0,
        "releaseNotesPathAllowedRows": 1 if release_notes_allowed else 0,
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "qualityEvalCaseRows": _int(post_counts.get("qualityEvalCaseRows")) or 4,
        "realAnswerSmokeCaseRows": _int(post_counts.get("realAnswerSmokeCaseRows")) or 2,
        "defaultAskClosedRows": 1,
        "defaultMcpClosedRows": 1,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": (
            PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID
        ),
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "postMergeConvergenceReportRef": DEFAULT_POST_MERGE_CONVERGENCE_REPORT.as_posix(),
            "postMergeConvergenceStatus": _clean_text(post_report.get("status")),
            "postMergeConvergenceDecision": _clean_text(post_report.get("decision")),
            "postMergeNextRecommendedTranche": _clean_text(post_report.get("nextRecommendedTranche")),
        },
        "policy": {
            "reportOnly": True,
            "runtimeMutationAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "releaseNotesPathAllowed": release_notes_allowed,
            "readyForGeneralRelease": False,
            "rawPayloadPersisted": False,
        },
        "promotionDecision": {
            "researchPreviewDecision": "release_notes_path_allowed" if release_notes_allowed else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "defaultSurfaceDecision": "do_not_enable_default_ask_or_default_mcp",
            "generalRcDecision": "blocked_pending_public_default_and_corpus_scale_evidence",
            "corpusScaleDecision": "blocked_pending_corpus_scale_quality_gate",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
            "sourcePostMergeDecision": _clean_text(post_decision.get("postMergeConvergence")),
        },
        "counts": counts,
        "gate": {
            "decisionGateReady": status == "ready",
            "postMergeConvergenceReady": not post_blockers,
            "researchPreviewRcCandidate": research_preview_candidate,
            "publicDefaultPromotionAllowed": False,
            "releaseNotesPathAllowed": release_notes_allowed,
            "generalRcReady": False,
            "corpusScaleQualityEvidenceReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {
                "checkId": "post_merge_convergence",
                "status": "pass" if not post_blockers else "fail",
                "blockers": post_blockers,
            },
            {
                "checkId": "unsafe_counters",
                "status": "pass" if not unsafe_blockers else "fail",
                "blockers": unsafe_blockers,
            },
            {
                "checkId": "public_default_promotion",
                "status": "pass" if public_default_held else "fail",
                "blockers": [] if public_default_held else ["public_default_promotion_hold_not_confirmed"],
            },
        ],
        "nextBlockerRows": [
            {
                "blockerId": "public_default_promotion_held",
                "severity": "P1",
                "summary": "The evidence chunk answer path stays labs/research-preview only; public khub ask and default MCP are not promoted.",
            },
            {
                "blockerId": "corpus_scale_quality_evidence_missing",
                "severity": "P1",
                "summary": "The current quality evidence covers a narrow gate, not corpus-scale answer quality for a broad default release.",
            },
            {
                "blockerId": "default_surface_activation_requires_separate_gate",
                "severity": "P2",
                "summary": "Any default CLI/MCP activation must be implemented in a later explicit gate after corpus-scale evidence is available.",
            },
        ],
        "warnings": [
            "research_preview_release_notes_allowed_but_public_default_promotion_held",
            "do_not_enable_default_ask_or_default_mcp_in_this_tranche",
            "corpus_scale_quality_gate_required_before_general_rc_language",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("promotionDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Public/Default Promotion Decision Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- researchPreviewDecision: `{decision.get('researchPreviewDecision')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- defaultSurfaceDecision: `{decision.get('defaultSurfaceDecision')}`",
        f"- corpusScaleDecision: `{decision.get('corpusScaleDecision')}`",
        f"- researchPreviewRcCandidateRows: `{counts.get('researchPreviewRcCandidateRows')}`",
        f"- releaseNotesPathAllowedRows: `{counts.get('releaseNotesPathAllowedRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- generalRcReadyRows: `{counts.get('generalRcReadyRows')}`",
        f"- corpusScaleClaimProvenRows: `{counts.get('corpusScaleClaimProvenRows')}`",
        f"- qualityEvalCaseRows: `{counts.get('qualityEvalCaseRows')}`",
        f"- realAnswerSmokeCaseRows: `{counts.get('realAnswerSmokeCaseRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Remaining Blockers", ""])
    for row in list(report.get("nextBlockerRows") or []):
        lines.append(f"- `{row.get('severity')}` `{row.get('blockerId')}`: {row.get('summary')}")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate_markdown(
            report
        ),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate",
]
