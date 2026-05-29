"""Promotion review for the labs parsed-artifact evidence chunk answer path."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _clean_text,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID,
    READY_DECISION as OUTPUT_CAPTURE_READY_DECISION,
    _read_json,
)


PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-promotion-review.v1"
)

READY_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_ready"
BLOCKED_DECISION = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate"
NEXT_TRANCHE_BLOCKED = "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_repair"
DEFAULT_OUTPUT_CAPTURE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture.v1.json"
)
DEFAULT_PRODUCT_DEFINITION = Path("docs/knowledge_os_definition.md")
PUBLIC_DEFAULT_HOLD_REASON = "public_default_promotion_requires_release_smoke_hygiene_and_public_surface_design"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_text(path: str | Path) -> str:
    try:
        return Path(str(path)).read_text(encoding="utf-8")
    except Exception:
        return ""


def _output_capture_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_OUTPUT_CAPTURE_SCHEMA_ID:
        blockers.append("output_capture_schema_mismatch")
    if report.get("status") != "ready":
        blockers.append("output_capture_not_ready")
    if report.get("decision") != OUTPUT_CAPTURE_READY_DECISION:
        blockers.append("output_capture_decision_not_ready")
    if gate.get("readyForLabsOptInUserTestPromotionReview") is not True:
        blockers.append("output_capture_gate_not_ready_for_promotion_review")
    if gate.get("allCapturedCommandsPassed") is not True:
        blockers.append("output_capture_commands_not_all_passed")
    if gate.get("allJsonAssertionsPassed") is not True:
        blockers.append("output_capture_json_assertions_not_all_passed")
    if gate.get("expectedNoEvidenceCasesStayedNoEvidence") is not True:
        blockers.append("output_capture_no_evidence_safety_failed")
    if gate.get("externalRequestRejected") is not True:
        blockers.append("output_capture_external_rejection_failed")
    if gate.get("noRawOutputPersisted") is not True:
        blockers.append("output_capture_raw_output_persisted")
    if _int(counts.get("outputCaptureFailRows")) != 0:
        blockers.append("output_capture_failures_present")
    if _int(counts.get("jsonAssertionFailRows")) != 0:
        blockers.append("output_capture_json_assertion_failures_present")
    if _int(counts.get("rawOutputPersistedRows")) != 0:
        blockers.append("output_capture_raw_output_persisted_rows_present")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("output_capture_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("output_capture_schema_violations_present")
    if _contains_private_path(report):
        blockers.append("output_capture_private_path_marker")
    return sorted(set(blockers))


def _product_definition_blockers(text: str) -> list[str]:
    blockers: list[str] = []
    required_fragments = [
        "local-first, evidence-first research knowledge runtime",
        "section/paragraph evidence-first paper QA and compare runtime",
        "Lack of evidence produces abstain/no-answer",
        "Public CLI/MCP surfaces match the documented Research Preview promise",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            blockers.append(f"product_definition_missing:{fragment}")
    return blockers


def _promotion_review_rows() -> list[dict[str, Any]]:
    return [
        {
            "reviewRowId": "promotion-review:labs-limited-rc-candidate",
            "surface": "khub_labs_paper_evidence_chunk_ask",
            "decision": "ready_for_v01_labs_limited_release_gate",
            "allowedInThisTranche": True,
            "publicDefaultPromotionAllowed": False,
            "reason": "output_capture_green_and_product_scope_matches_section_paragraph_evidence_first_preview",
        },
        {
            "reviewRowId": "promotion-review:public-default-hold",
            "surface": "khub_ask_and_default_mcp",
            "decision": "hold_public_default_promotion",
            "allowedInThisTranche": False,
            "publicDefaultPromotionAllowed": False,
            "reason": PUBLIC_DEFAULT_HOLD_REASON,
        },
    ]


def _required_next_checks() -> list[dict[str, Any]]:
    return [
        {
            "checkId": "release_smoke",
            "status": "pending",
            "requiredBeforePublicDefaultPromotion": True,
            "description": "Run release smoke on the release-candidate branch.",
        },
        {
            "checkId": "public_hygiene",
            "status": "pending",
            "requiredBeforePublicDefaultPromotion": True,
            "description": "Run public hygiene and private-path leak checks.",
        },
        {
            "checkId": "public_surface_design",
            "status": "pending",
            "requiredBeforePublicDefaultPromotion": True,
            "description": "Design any public/default CLI or MCP exposure separately.",
        },
        {
            "checkId": "no_answer_regression",
            "status": "pending",
            "requiredBeforePublicDefaultPromotion": True,
            "description": "Confirm no-answer safety remains green after any exposure change.",
        },
    ]


def build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
    *,
    output_capture_report_path: str | Path = DEFAULT_OUTPUT_CAPTURE_REPORT,
    product_definition_path: str | Path = DEFAULT_PRODUCT_DEFINITION,
    output_capture_report: dict[str, Any] | None = None,
    product_definition_text: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    output_report = dict(output_capture_report or _read_json(output_capture_report_path))
    product_text = product_definition_text if product_definition_text is not None else _read_text(product_definition_path)
    output_blockers = _output_capture_blockers(output_report)
    product_blockers = _product_definition_blockers(product_text)
    semantic_violations = sorted(set(output_blockers + product_blockers))
    counts_in = dict(output_report.get("counts") or {})
    review_rows = [] if semantic_violations else _promotion_review_rows()
    status = "ready" if not semantic_violations else "blocked"
    counts = {
        "inputOutputCaptureRows": 1 if output_report else 0,
        "outputCaptureReadyInputRows": 1 if output_report and not output_blockers else 0,
        "capturedCommandRows": _int(counts_in.get("capturedCommandRows")),
        "outputCapturePassRows": _int(counts_in.get("outputCapturePassRows")),
        "outputCaptureFailRows": _int(counts_in.get("outputCaptureFailRows")),
        "jsonAssertionRows": _int(counts_in.get("jsonAssertionRows")),
        "jsonAssertionPassRows": _int(counts_in.get("jsonAssertionPassRows")),
        "jsonAssertionFailRows": _int(counts_in.get("jsonAssertionFailRows")),
        "expectedAnswerableOutputRows": _int(counts_in.get("expectedAnswerableOutputRows")),
        "observedAnswerableOutputRows": _int(counts_in.get("observedAnswerableOutputRows")),
        "expectedNoEvidenceOutputRows": _int(counts_in.get("expectedNoEvidenceOutputRows")),
        "observedNoEvidenceOutputRows": _int(counts_in.get("observedNoEvidenceOutputRows")),
        "externalRejectionPassRows": _int(counts_in.get("externalRejectionPassRows")),
        "labsLimitedPromotionReadyRows": sum(
            1 for row in review_rows if row.get("decision") == "ready_for_v01_labs_limited_release_gate"
        ),
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1 if status == "ready" else 0,
        "releaseGateRequiredRows": len(_required_next_checks()),
        "rawOutputPersistedRows": _int(counts_in.get("rawOutputPersistedRows")),
        "answerTextIncludedRows": _int(counts_in.get("answerTextIncludedRows")),
        "citationPayloadIncludedRows": _int(counts_in.get("citationPayloadIncludedRows")),
        "sourcePayloadIncludedRows": _int(counts_in.get("sourcePayloadIncludedRows")),
        "excerptIncludedRows": _int(counts_in.get("excerptIncludedRows")),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": _int(counts_in.get("privatePathLeakRows")),
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "outputCaptureReportRef": (
                "eval/knowledgeos/reports/"
                "parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture.v1.json"
            ),
            "outputCaptureSchema": _clean_text(output_report.get("schema")),
            "outputCaptureStatus": _clean_text(output_report.get("status")),
            "outputCaptureDecision": _clean_text(output_report.get("decision")),
            "productDefinitionRef": "docs/knowledge_os_definition.md",
        },
        "policy": {
            "reportOnly": True,
            "labsLimitedRcCandidateOnly": True,
            "publicDefaultPromotionAllowed": False,
            "publicKhubAskChanged": False,
            "defaultMcpAskChanged": False,
            "runtimeDefaultChange": False,
            "externalModelCallsAllowed": False,
            "rawOutputPersisted": False,
            "answerTextExcludedFromReport": True,
            "citationPayloadExcludedFromReport": True,
            "sourcePayloadExcludedFromReport": True,
            "excerptExcludedFromReport": True,
            "candidateStoreWrites": False,
            "sourceSpanCreation": False,
            "strictEvidenceCreation": False,
        },
        "releaseDecision": {
            "v01ScopeDecision": "labs_limited_rc_candidate_ready" if status == "ready" else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "publicDefaultHoldReason": PUBLIC_DEFAULT_HOLD_REASON,
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "readyForV01LabsLimitedReleaseGate": status == "ready",
            "outputCaptureReady": not output_blockers,
            "productDefinitionAligned": not product_blockers,
            "publicDefaultPromotionAllowed": False,
            "publicDefaultPromotionHeld": status == "ready",
            "releaseGateRequiredBeforePublicDefault": True,
            "semanticViolations": semantic_violations,
        },
        "promotionReviewRows": review_rows,
        "requiredNextChecks": _required_next_checks(),
        "warnings": [
            "public_default_promotion_is_not_enabled_by_this_review",
            "labs_limited_rc_candidate_still_requires_release_smoke_hygiene_before_release",
        ],
    }


def render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("releaseDecision") or {})
    lines = [
        "# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Promotion Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- v01ScopeDecision: `{decision.get('v01ScopeDecision')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- capturedCommandRows: `{counts.get('capturedCommandRows')}`",
        f"- outputCapturePassRows: `{counts.get('outputCapturePassRows')}`",
        f"- jsonAssertionPassRows: `{counts.get('jsonAssertionPassRows')}`",
        f"- labsLimitedPromotionReadyRows: `{counts.get('labsLimitedPromotionReadyRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Decisions",
        "",
    ]
    for row in list(report.get("promotionReviewRows") or []):
        lines.append(
            f"- `{row.get('surface')}`: `{row.get('decision')}`; "
            f"publicDefaultPromotionAllowed=`{row.get('publicDefaultPromotionAllowed')}`"
        )
    lines.extend(["", "## Required Next Checks", ""])
    for check in list(report.get("requiredNextChecks") or []):
        lines.append(f"- `{check.get('checkId')}`: `{check.get('status')}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in ZERO_COUNTER_FIELDS:
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_LABS_OPT_IN_USER_TEST_PROMOTION_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review",
    "write_parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review",
]
