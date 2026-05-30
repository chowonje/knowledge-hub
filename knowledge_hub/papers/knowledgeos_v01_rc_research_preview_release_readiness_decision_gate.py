"""Research Preview release-readiness decision gate for KnowledgeOS v0.1 RC."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review import (
    KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
    READY_DECISION as POSITIVE_COMPLETE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_quality_eval_seed import (
    ZERO_COUNTER_FIELDS,
    _contains_private_path,
    _int,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_output_capture import (
    _read_json,
)


KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-research-preview-release-readiness-decision-gate.v1"
)

READY_DECISION = "knowledgeos_v01_rc_research_preview_release_readiness_decision_gate_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_research_preview_release_readiness_decision_gate_blocked"
NEXT_TRANCHE_READY = "operator_release_package_handoff_or_branch_cleanup_decision"
NEXT_TRANCHE_BLOCKED = "knowledgeos_v01_rc_research_preview_release_readiness_repair"

DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/knowledgeos_v01_rc_positive_section_paragraph_quality_complete_review.v1.json"
)
DEFAULT_PRODUCT_DEFINITION_DOC = Path("docs/knowledge_os_definition.md")
DEFAULT_RELEASE_NOTES_DOC = Path("docs/releases/knowledge-hub-v0.1-rc-research-preview.md")

EXPECTED_INPUT_CASE_ROWS = 50
EXPECTED_POSITIVE_ROWS = 7
EXPECTED_HELD_EXPECTED_NO_ANSWER_ROWS = 17
EXPECTED_HELD_STRUCTURED_MODALITY_ROWS = 34
EXPECTED_STRICT_PROVENANCE_ROWS = 20
EXPECTED_RELEASE_SMOKE_ROWS = 10

REQUIRED_PRODUCT_DEFINITION_PHRASES = (
    "A local-first, evidence-first research knowledge runtime for auditable AI research workflows.",
    "A section/paragraph evidence-first paper QA and compare runtime for a local AI-paper corpus.",
    "Public CLI/MCP surfaces match the documented Research Preview promise.",
    "The release-candidate branch and PR state are clean enough to review, merge, or intentionally hold.",
)

REQUIRED_RELEASE_NOTES_PHRASES = (
    "Knowledge Hub v0.1 RC Research Preview",
    "Research Preview",
    "discover -> index -> search/ask -> evidence review",
    "section/paragraph evidence",
    "public/default promotion remains held",
    "No table, equation, or figure-caption default evidence promise",
    "No runtime promotion of visual retrieval hints as answer evidence",
)

FORBIDDEN_RELEASE_PHRASES = (
    "production-ready",
    "enterprise-ready",
    "fully tested",
    "default-on evidence chunk enabled",
    "default MCP activation is enabled",
    "public/default promotion is ready",
)

EXTRA_ZERO_COUNTER_FIELDS = (
    "pushRows",
    "githubPrMutationRows",
    "mergeRows",
    "branchDeletionRows",
    "releaseTagRows",
    "packagePublishRows",
    "rawGithubPayloadPersistedRows",
    "defaultMcpToolRows",
    "defaultKhubAskRouteRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _zero_counter_fields() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS)))


def _required_phrase_blockers(text: str, phrases: tuple[str, ...], prefix: str) -> list[str]:
    return [f"{prefix}_missing_required_phrase:{phrase}" for phrase in phrases if phrase not in text]


def _forbidden_phrase_blockers(text: str) -> list[str]:
    lower_text = text.lower()
    return [f"release_notes_forbidden_phrase:{phrase}" for phrase in FORBIDDEN_RELEASE_PHRASES if phrase.lower() in lower_text]


def _unsafe_counter_blockers(report: dict[str, Any], prefix: str) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers: list[str] = []
    for field in _zero_counter_fields():
        if _int(counts.get(field)) != 0:
            blockers.append(f"unsafe_counter_nonzero:{prefix}:{field}")
    return sorted(set(blockers))


def _positive_complete_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers: list[str] = []
    if report.get("schema") != KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID:
        blockers.append("positive_complete_schema_mismatch")
    else:
        validation = validate_payload(
            report,
            KNOWLEDGEOS_V01_RC_POSITIVE_SECTION_PARAGRAPH_QUALITY_COMPLETE_REVIEW_SCHEMA_ID,
            strict=True,
        )
        if not validation.ok:
            blockers.append("positive_complete_schema_validation_failed")
    if report.get("status") != "ready":
        blockers.append("positive_complete_not_ready")
    if report.get("decision") != POSITIVE_COMPLETE_READY_DECISION:
        blockers.append("positive_complete_decision_not_ready")
    if report.get("nextRecommendedTranche") != "knowledgeos_v01_rc_research_preview_release_readiness_decision_gate":
        blockers.append("positive_complete_next_tranche_not_release_readiness")
    if _int(counts.get("inputCaseRows")) != EXPECTED_INPUT_CASE_ROWS:
        blockers.append("positive_complete_input_case_rows_not_50")
    if _int(counts.get("positiveSeedRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_complete_seed_rows_not_7")
    if _int(counts.get("positiveAnswerPassRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_complete_answer_pass_rows_not_7")
    if _int(counts.get("positiveAnswerFailRows")) != 0:
        blockers.append("positive_complete_answer_fail_rows_present")
    if _int(counts.get("provenancePassRows")) != EXPECTED_POSITIVE_ROWS:
        blockers.append("positive_complete_provenance_pass_rows_not_7")
    if _int(counts.get("provenanceFailRows")) != 0:
        blockers.append("positive_complete_provenance_fail_rows_present")
    if _int(counts.get("heldExpectedNoAnswerRows")) != EXPECTED_HELD_EXPECTED_NO_ANSWER_ROWS:
        blockers.append("positive_complete_expected_no_answer_hold_mismatch")
    if _int(counts.get("heldStructuredModalityRows")) != EXPECTED_HELD_STRUCTURED_MODALITY_ROWS:
        blockers.append("positive_complete_structured_modality_hold_mismatch")
    if _int(counts.get("strictProvenanceSpanRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_complete_strict_provenance_span_rows_not_20")
    if _int(counts.get("sourceContentHashRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_complete_source_content_hash_rows_not_20")
    if _int(counts.get("charsLocatorRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_complete_chars_locator_rows_not_20")
    if _int(counts.get("answerContractCitationProvenanceRows")) != EXPECTED_STRICT_PROVENANCE_ROWS:
        blockers.append("positive_complete_answer_contract_provenance_rows_not_20")
    if _int(counts.get("prMergedRows")) != 3:
        blockers.append("positive_complete_quality_prs_not_all_merged")
    if _int(counts.get("ciCheckSuccessRows")) != 21:
        blockers.append("positive_complete_ci_not_green")
    if _int(counts.get("releaseSmokePassedRows")) < EXPECTED_RELEASE_SMOKE_ROWS:
        blockers.append("positive_complete_release_smoke_not_green")
    if _int(counts.get("publicHygieneIssueRows")) != 0:
        blockers.append("positive_complete_public_hygiene_issues_present")
    if _int(counts.get("positiveSectionParagraphQualityCompleteRows")) != 1:
        blockers.append("positive_complete_slice_not_marked_complete")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("positive_complete_public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("positive_complete_public_default_hold_missing")
    if _int(counts.get("generalRcReadyRows")) != 0:
        blockers.append("positive_complete_general_rc_ready_unexpected")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("positive_complete_corpus_scale_claim_proven_unexpected")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("positive_complete_public_default_allowed_unexpected")
    if gate.get("generalRcReady") is not False:
        blockers.append("positive_complete_general_rc_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("positive_complete_private_path_leak_reported")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("positive_complete_schema_violations_present")
    return sorted(set(blockers + _unsafe_counter_blockers(report, "positive_complete")))


def build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
    *,
    positive_complete_review_report_path: str | Path = DEFAULT_POSITIVE_COMPLETE_REVIEW_REPORT,
    product_definition_doc_path: str | Path = DEFAULT_PRODUCT_DEFINITION_DOC,
    release_notes_doc_path: str | Path = DEFAULT_RELEASE_NOTES_DOC,
    positive_complete_review_report: dict[str, Any] | None = None,
    product_definition_text: str | None = None,
    release_notes_text: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    positive_report = dict(positive_complete_review_report or _read_json(positive_complete_review_report_path))
    product_text = (
        product_definition_text
        if product_definition_text is not None
        else Path(product_definition_doc_path).read_text(encoding="utf-8")
    )
    release_text = (
        release_notes_text
        if release_notes_text is not None
        else Path(release_notes_doc_path).read_text(encoding="utf-8")
    )
    positive_blockers = _positive_complete_blockers(positive_report)
    product_blockers = _required_phrase_blockers(
        product_text,
        REQUIRED_PRODUCT_DEFINITION_PHRASES,
        "product_definition",
    )
    release_required_blockers = _required_phrase_blockers(
        release_text,
        REQUIRED_RELEASE_NOTES_PHRASES,
        "release_notes",
    )
    release_forbidden_blockers = _forbidden_phrase_blockers(release_text)
    private_path_leak_rows = (
        1
        if _contains_private_path(positive_report)
        or _contains_private_path(product_text)
        or _contains_private_path(release_text)
        else 0
    )
    privacy_blockers = (
        ["research_preview_release_readiness_private_path_marker"] if private_path_leak_rows else []
    )
    semantic_violations = sorted(
        set(
            positive_blockers
            + product_blockers
            + release_required_blockers
            + release_forbidden_blockers
            + privacy_blockers
        )
    )
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    positive_counts = dict(positive_report.get("counts") or {})
    research_preview_ready = status == "ready"
    counts = {
        "releaseReadinessDecisionRows": 1,
        "positiveCompleteReviewReadyRows": 1 if not positive_blockers else 0,
        "productDefinitionReadyRows": 1 if not product_blockers else 0,
        "releaseNotesReadyRows": 1 if not release_required_blockers and not release_forbidden_blockers else 0,
        "inputCaseRows": _int(positive_counts.get("inputCaseRows")),
        "positiveSeedRows": _int(positive_counts.get("positiveSeedRows")),
        "positiveAnswerPassRows": _int(positive_counts.get("positiveAnswerPassRows")),
        "positiveAnswerFailRows": _int(positive_counts.get("positiveAnswerFailRows")),
        "provenancePassRows": _int(positive_counts.get("provenancePassRows")),
        "provenanceFailRows": _int(positive_counts.get("provenanceFailRows")),
        "strictProvenanceSpanRows": _int(positive_counts.get("strictProvenanceSpanRows")),
        "sourceContentHashRows": _int(positive_counts.get("sourceContentHashRows")),
        "charsLocatorRows": _int(positive_counts.get("charsLocatorRows")),
        "answerContractCitationProvenanceRows": _int(positive_counts.get("answerContractCitationProvenanceRows")),
        "heldExpectedNoAnswerRows": _int(positive_counts.get("heldExpectedNoAnswerRows")),
        "heldStructuredModalityRows": _int(positive_counts.get("heldStructuredModalityRows")),
        "qualityPrMergedRows": _int(positive_counts.get("prMergedRows")),
        "ciCheckSuccessRows": _int(positive_counts.get("ciCheckSuccessRows")),
        "releaseSmokePassedRows": _int(positive_counts.get("releaseSmokePassedRows")),
        "publicHygieneIssueRows": _int(positive_counts.get("publicHygieneIssueRows")),
        "positiveSectionParagraphQualityCompleteRows": _int(
            positive_counts.get("positiveSectionParagraphQualityCompleteRows")
        ),
        "researchPreviewReleaseReadyRows": 1 if research_preview_ready else 0,
        "researchPreviewReleaseHeldRows": 0 if research_preview_ready else 1,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1,
        "generalRcReadyRows": 0,
        "corpusScalePositiveSliceProvenRows": _int(positive_counts.get("corpusScalePositiveSliceProvenRows")),
        "corpusScaleClaimProvenRows": 0,
        "tableEquationFigureDefaultEvidenceRows": 0,
        "visualHintAnswerEvidenceRows": 0,
        "branchCleanupRecommendedRows": _int(positive_counts.get("branchCleanupRecommendedRows")),
        "branchCleanupAppliedRows": 0,
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in _zero_counter_fields()},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "positiveCompleteReviewReportRef": Path(positive_complete_review_report_path).as_posix(),
            "productDefinitionDocRef": Path(product_definition_doc_path).as_posix(),
            "releaseNotesDocRef": Path(release_notes_doc_path).as_posix(),
        },
        "readinessDecision": {
            "researchPreviewDecision": "ready_for_controlled_research_preview_release"
            if research_preview_ready
            else "blocked",
            "publicDefaultDecision": "hold_public_default_promotion",
            "defaultSurfaceDecision": "do_not_enable_default_ask_or_default_mcp",
            "generalReleaseDecision": "not_ready_for_general_release",
            "strongestSupportedClaim": "section_paragraph_positive_slice_ready_with_conservative_no_answer_boundary",
            "nextGate": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        },
        "counts": counts,
        "gate": {
            "releaseReadinessDecisionReady": status == "ready",
            "positiveCompleteReviewReady": not positive_blockers,
            "productDefinitionReady": not product_blockers,
            "releaseNotesReady": not release_required_blockers and not release_forbidden_blockers,
            "researchPreviewReleaseAllowed": research_preview_ready,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "checkRows": [
            {
                "checkId": "positive_section_paragraph_quality_complete_review",
                "status": "pass" if not positive_blockers else "fail",
                "blockers": positive_blockers,
            },
            {
                "checkId": "product_definition",
                "status": "pass" if not product_blockers else "fail",
                "blockers": product_blockers,
            },
            {
                "checkId": "release_notes_required_phrases",
                "status": "pass" if not release_required_blockers else "fail",
                "blockers": release_required_blockers,
            },
            {
                "checkId": "release_notes_forbidden_phrases",
                "status": "pass" if not release_forbidden_blockers else "fail",
                "blockers": release_forbidden_blockers,
            },
            {
                "checkId": "private_path_hygiene",
                "status": "pass" if not privacy_blockers else "fail",
                "blockers": privacy_blockers,
            },
        ],
        "nextActionRows": [
            {
                "actionId": "operator_release_package_handoff",
                "status": "recommended_not_applied" if research_preview_ready else "blocked",
                "requiresExplicitApproval": True,
                "summary": "Prepare or review the Research Preview package handoff without enabling public/default surfaces.",
            },
            {
                "actionId": "branch_cleanup_decision",
                "status": "recommended_not_applied" if research_preview_ready else "blocked",
                "requiresExplicitApproval": True,
                "summary": "Decide whether to clean up merged quality branches; do not delete branches in this gate.",
            },
            {
                "actionId": "public_default_promotion",
                "status": "held",
                "requiresExplicitApproval": True,
                "summary": "Keep public/default `khub ask` and default MCP promotion held until corpus-scale quality evidence passes.",
            },
        ],
        "warnings": [
            "research_preview_release_ready_does_not_mean_general_release_ready",
            "public_default_promotion_remains_held",
            "table_equation_figure_evidence_remains_outside_v01_default_promise",
            "visual_retrieval_hints_remain_candidate_discovery_only",
            "branch_cleanup_not_applied_in_this_tranche",
        ],
    }


def render_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("readinessDecision") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Research Preview Release Readiness Decision Gate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- researchPreviewDecision: `{decision.get('researchPreviewDecision')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- defaultSurfaceDecision: `{decision.get('defaultSurfaceDecision')}`",
        f"- generalReleaseDecision: `{decision.get('generalReleaseDecision')}`",
        f"- strongestSupportedClaim: `{decision.get('strongestSupportedClaim')}`",
        f"- researchPreviewReleaseReadyRows: `{counts.get('researchPreviewReleaseReadyRows')}`",
        f"- positiveSectionParagraphQualityCompleteRows: `{counts.get('positiveSectionParagraphQualityCompleteRows')}`",
        f"- positiveAnswerPassRows: `{counts.get('positiveAnswerPassRows')}`",
        f"- provenancePassRows: `{counts.get('provenancePassRows')}`",
        f"- releaseSmokePassedRows: `{counts.get('releaseSmokePassedRows')}`",
        f"- publicHygieneIssueRows: `{counts.get('publicHygieneIssueRows')}`",
        f"- publicDefaultPromotionReadyRows: `{counts.get('publicDefaultPromotionReadyRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- generalRcReadyRows: `{counts.get('generalRcReadyRows')}`",
        f"- corpusScaleClaimProvenRows: `{counts.get('corpusScaleClaimProvenRows')}`",
        f"- tableEquationFigureDefaultEvidenceRows: `{counts.get('tableEquationFigureDefaultEvidenceRows')}`",
        f"- visualHintAnswerEvidenceRows: `{counts.get('visualHintAnswerEvidenceRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Checks",
        "",
    ]
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Next Actions", ""])
    for row in list(report.get("nextActionRows") or []):
        lines.append(
            f"- `{row.get('actionId')}`: `{row.get('status')}`; "
            f"requiresExplicitApproval=`{row.get('requiresExplicitApproval')}`; {row.get('summary')}"
        )
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in _zero_counter_fields():
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate_markdown(report),
        encoding="utf-8",
    )
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_RESEARCH_PREVIEW_RELEASE_READINESS_DECISION_GATE_SCHEMA_ID",
    "READY_DECISION",
    "BLOCKED_DECISION",
    "build_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate",
    "write_knowledgeos_v01_rc_research_preview_release_readiness_decision_gate",
]
