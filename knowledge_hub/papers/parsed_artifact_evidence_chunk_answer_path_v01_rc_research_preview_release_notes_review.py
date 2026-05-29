"""Release-note review for the v0.1 parsed-artifact evidence chunk Research Preview."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
    READY_DECISION as PROMOTION_GATE_READY_DECISION,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-research-preview-release-notes-review.v1"
)

READY_DECISION = "knowledge_hub_v01_rc_research_preview_release_notes_review_ready"
BLOCKED_DECISION = "knowledge_hub_v01_rc_research_preview_release_notes_review_blocked"
NEXT_TRANCHE_READY = "knowledge_hub_v01_rc_release_package_handoff_or_corpus_scale_quality_gate"
NEXT_TRANCHE_BLOCKED = "knowledge_hub_v01_rc_research_preview_release_notes_repair"
DEFAULT_PROMOTION_DECISION_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate.v1.json"
)
DEFAULT_RELEASE_NOTES_DOC = Path("docs/releases/knowledge-hub-v0.1-rc-research-preview.md")

REQUIRED_PHRASES = (
    "Knowledge Hub v0.1 RC Research Preview",
    "Research Preview",
    "discover -> index -> search/ask -> evidence review",
    "section/paragraph evidence",
    "public/default promotion remains held",
    "khub labs paper evidence-chunk-ask",
    "default MCP",
    "publicDefaultPromotionReadyRows=0",
    "publicDefaultPromotionHeldRows=1",
    "generalRcReadyRows=0",
    "corpusScaleClaimProvenRows=0",
)
FORBIDDEN_PROMOTION_PHRASES = (
    "production-ready",
    "enterprise-ready",
    "fully tested",
    "secure by default",
    "default-on evidence chunk enabled",
    "default-on parsed-artifact enabled",
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


def _promotion_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID:
        blockers.append("promotion_gate_schema_mismatch")
    else:
        validation = validate_payload(
            report,
            PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            blockers.append("promotion_gate_schema_validation_failed")
    if report.get("status") != "ready":
        blockers.append("promotion_gate_not_ready")
    if report.get("decision") != PROMOTION_GATE_READY_DECISION:
        blockers.append("promotion_gate_decision_not_ready")
    if _int(counts.get("releaseNotesPathAllowedRows")) != 1:
        blockers.append("release_notes_path_not_allowed")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_hold_missing")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_allowed_unexpected")
    if _int(counts.get("generalRcReadyRows")) != 0:
        blockers.append("general_rc_ready_unexpected")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("corpus_scale_claim_proven_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("promotion_gate_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("promotion_gate_schema_violations_present")
    return sorted(set(blockers))


def _unsafe_counter_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{field}")
    return sorted(set(blockers))


def _release_note_text(path: str | Path, provided_text: str | None) -> str:
    if provided_text is not None:
        return provided_text
    return Path(path).read_text(encoding="utf-8")


def build_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review(
    *,
    promotion_decision_gate_report_path: str | Path = DEFAULT_PROMOTION_DECISION_GATE_REPORT,
    release_notes_doc_path: str | Path = DEFAULT_RELEASE_NOTES_DOC,
    promotion_decision_gate_report: dict[str, Any] | None = None,
    release_notes_text: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    promotion_report = dict(promotion_decision_gate_report or _read_json(promotion_decision_gate_report_path))
    doc_text = _release_note_text(release_notes_doc_path, release_notes_text)
    promotion_blockers = _promotion_gate_blockers(promotion_report)
    unsafe_blockers = _unsafe_counter_blockers(promotion_report)
    missing_required = [phrase for phrase in REQUIRED_PHRASES if phrase not in doc_text]
    forbidden_hits = [phrase for phrase in FORBIDDEN_PROMOTION_PHRASES if phrase.lower() in doc_text.lower()]
    semantic_violations = sorted(
        set(
            promotion_blockers
            + unsafe_blockers
            + [f"missing_required_phrase:{phrase}" for phrase in missing_required]
            + [f"forbidden_phrase:{phrase}" for phrase in forbidden_hits]
        )
    )
    private_path_leak_rows = 1 if _contains_private_path(promotion_report) or _contains_private_path(doc_text) else 0
    if private_path_leak_rows:
        semantic_violations.append("release_notes_or_gate_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    promotion_counts = dict(promotion_report.get("counts") or {})
    counts = {
        "releaseNotesReviewRows": 1,
        "promotionGateReadyRows": 1 if not promotion_blockers else 0,
        "releaseNotesDocRows": 1,
        "releaseNotesReadyRows": 1 if status == "ready" else 0,
        "requiredPhraseRows": len(REQUIRED_PHRASES),
        "requiredPhrasePassRows": len(REQUIRED_PHRASES) - len(missing_required),
        "missingRequiredPhraseRows": len(missing_required),
        "forbiddenPhraseRows": len(forbidden_hits),
        "releaseNotesPathAllowedRows": _int(promotion_counts.get("releaseNotesPathAllowedRows")),
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": _int(promotion_counts.get("publicDefaultPromotionHeldRows")),
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "promotionDecisionGateReportRef": DEFAULT_PROMOTION_DECISION_GATE_REPORT.as_posix(),
            "releaseNotesDocRef": DEFAULT_RELEASE_NOTES_DOC.as_posix(),
        },
        "policy": {
            "reportOnly": True,
            "releaseNotesOnly": True,
            "runtimeMutationAllowed": False,
            "publicDefaultPromotionAllowed": False,
            "readyForGeneralRelease": False,
            "rawPayloadPersisted": False,
        },
        "releaseNotesDecision": {
            "researchPreviewReleaseNotesReady": status == "ready",
            "publicDefaultDecision": "hold_public_default_promotion",
            "defaultSurfaceDecision": "do_not_enable_default_ask_or_default_mcp",
            "generalRcDecision": "blocked_pending_corpus_scale_quality_gate",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "releaseNotesReviewReady": status == "ready",
            "promotionGateReady": not promotion_blockers,
            "requiredPhrasesPresent": not missing_required,
            "forbiddenPromotionPhrasesAbsent": not forbidden_hits,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {
                "checkId": "promotion_decision_gate",
                "status": "pass" if not promotion_blockers else "fail",
                "blockers": promotion_blockers,
            },
            {
                "checkId": "unsafe_counters",
                "status": "pass" if not unsafe_blockers else "fail",
                "blockers": unsafe_blockers,
            },
            {
                "checkId": "release_notes_required_phrases",
                "status": "pass" if not missing_required else "fail",
                "blockers": [f"missing_required_phrase:{phrase}" for phrase in missing_required],
            },
            {
                "checkId": "release_notes_forbidden_phrases",
                "status": "pass" if not forbidden_hits else "fail",
                "blockers": [f"forbidden_phrase:{phrase}" for phrase in forbidden_hits],
            },
        ],
        "requiredPhraseRows": [
            {"phrase": phrase, "present": phrase not in missing_required} for phrase in REQUIRED_PHRASES
        ],
        "warnings": [
            "release_notes_are_research_preview_only",
            "public_default_promotion_remains_held",
            "corpus_scale_quality_gate_required_before_general_rc_language",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("releaseNotesDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Research Preview Release Notes Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- researchPreviewReleaseNotesReady: `{decision.get('researchPreviewReleaseNotesReady')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- defaultSurfaceDecision: `{decision.get('defaultSurfaceDecision')}`",
        f"- releaseNotesPathAllowedRows: `{counts.get('releaseNotesPathAllowedRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- requiredPhrasePassRows: `{counts.get('requiredPhrasePassRows')}`",
        f"- forbiddenPhraseRows: `{counts.get('forbiddenPhraseRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Required Phrases", ""])
    for row in list(report.get("requiredPhraseRows") or []):
        lines.append(f"- `{row.get('phrase')}`: `{row.get('present')}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review_markdown(
            report
        ),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_RESEARCH_PREVIEW_RELEASE_NOTES_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review",
    "write_parsed_artifact_evidence_chunk_answer_path_v01_rc_research_preview_release_notes_review",
]
