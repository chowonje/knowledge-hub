"""Vision bottleneck definition review for KnowledgeOS v0.1 RC."""

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
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
    READY_DECISION as DRAFT_PR_POST_OPEN_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
    READY_DECISION as LABS_RELEASE_GATE_READY_DECISION,
)
from knowledge_hub.papers.parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate import (
    PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
    READY_DECISION as PUBLIC_DEFAULT_PROMOTION_READY_DECISION,
)


KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID = (
    "knowledge-hub.product.knowledgeos-v01-rc-vision-bottleneck-definition-review.v1"
)

READY_DECISION = "knowledgeos_v01_rc_vision_bottleneck_definition_ready"
BLOCKED_DECISION = "knowledgeos_v01_rc_vision_bottleneck_definition_blocked"
NEXT_TRANCHE_READY = "operator_ready_or_merge_pr_171_or_corpus_scale_quality_gate"
NEXT_TRANCHE_BLOCKED = "knowledgeos_v01_rc_vision_bottleneck_definition_repair"

DEFAULT_PRODUCT_DEFINITION_DOC = Path("docs/knowledge_os_definition.md")
DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_post_open_review.v1.json"
)
DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_public_default_promotion_decision_gate.v1.json"
)
DEFAULT_LABS_RELEASE_GATE_REPORT = Path(
    "eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate.v1.json"
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

REQUIRED_PRODUCT_DEFINITION_PHRASES = (
    "A local-first, evidence-first research knowledge runtime for auditable AI research workflows.",
    "section/paragraph evidence-first paper QA and compare runtime",
    "Visual, table, equation, and figure-caption workflows are clearly marked as labs or limited support",
    "The product should prefer a conservative, inspectable no-answer",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _schema_blocker(report: dict[str, Any], schema_id: str, prefix: str) -> list[str]:
    if report.get("schema") != schema_id:
        return [f"{prefix}_schema_mismatch"]
    validation = validate_payload(report, schema_id, strict=True)
    if not validation.ok:
        return [f"{prefix}_schema_validation_failed"]
    return []


def _draft_pr_post_open_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_DRAFT_PR_POST_OPEN_REVIEW_SCHEMA_ID,
        "draft_pr_post_open_review",
    )
    if report.get("status") != "ready":
        blockers.append("draft_pr_post_open_review_not_ready")
    if report.get("decision") != DRAFT_PR_POST_OPEN_READY_DECISION:
        blockers.append("draft_pr_post_open_review_decision_not_ready")
    if _int(counts.get("readyForHumanReviewRows")) != 1:
        blockers.append("ready_for_human_review_missing")
    if _int(counts.get("readyForMergeRows")) != 0:
        blockers.append("ready_for_merge_unexpected")
    if _int(counts.get("ciCheckSuccessRows")) != 7:
        blockers.append("ci_checks_not_green")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("draft_pr_post_open_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("draft_pr_post_open_schema_violations")
    return sorted(set(blockers))


def _public_default_promotion_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_PUBLIC_DEFAULT_PROMOTION_DECISION_GATE_SCHEMA_ID,
        "public_default_promotion_gate",
    )
    if report.get("status") != "ready":
        blockers.append("public_default_promotion_gate_not_ready")
    if report.get("decision") != PUBLIC_DEFAULT_PROMOTION_READY_DECISION:
        blockers.append("public_default_promotion_gate_decision_not_ready")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("public_default_promotion_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("public_default_promotion_hold_missing")
    if _int(counts.get("corpusScaleClaimProvenRows")) != 0:
        blockers.append("corpus_scale_claim_proven_unexpected")
    if gate.get("publicDefaultPromotionAllowed") is not False:
        blockers.append("public_default_promotion_allowed_unexpected")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("public_default_promotion_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("public_default_promotion_schema_violations")
    return sorted(set(blockers))


def _labs_release_gate_blockers(report: dict[str, Any]) -> list[str]:
    counts = dict(report.get("counts") or {})
    blockers = _schema_blocker(
        report,
        PARSED_ARTIFACT_EVIDENCE_CHUNK_ANSWER_PATH_V01_RC_LABS_RELEASE_GATE_SCHEMA_ID,
        "labs_release_gate",
    )
    if report.get("status") != "ready":
        blockers.append("labs_release_gate_not_ready")
    if report.get("decision") != LABS_RELEASE_GATE_READY_DECISION:
        blockers.append("labs_release_gate_decision_not_ready")
    if _int(counts.get("releaseSmokePassedRows")) < 10:
        blockers.append("release_smoke_not_green")
    if _int(counts.get("publicHygieneIssueRows")) != 0:
        blockers.append("public_hygiene_issues_present")
    if _int(counts.get("noAnswerPassRows")) < 3:
        blockers.append("no_answer_smoke_not_green")
    if _int(counts.get("publicDefaultPromotionReadyRows")) != 0:
        blockers.append("labs_release_public_default_ready_unexpected")
    if _int(counts.get("publicDefaultPromotionHeldRows")) < 1:
        blockers.append("labs_release_public_default_hold_missing")
    if _int(counts.get("privatePathLeakRows")) != 0:
        blockers.append("labs_release_private_path_leak")
    if _int(counts.get("schemaViolationCount")) != 0:
        blockers.append("labs_release_schema_violations")
    return sorted(set(blockers))


def _product_definition_blockers(text: str) -> list[str]:
    blockers = [f"missing_product_definition_phrase:{phrase}" for phrase in REQUIRED_PRODUCT_DEFINITION_PHRASES if phrase not in text]
    if _contains_private_path(text):
        blockers.append("product_definition_private_path_marker")
    return sorted(set(blockers))


def _unsafe_counter_blockers(*reports: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for index, report in enumerate(reports, start=1):
        counts = dict(report.get("counts") or {})
        for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
            if _int(counts.get(field)) != 0:
                blockers.append(f"unsafe_counter_nonzero:report{index}:{field}")
    return sorted(set(blockers))


def _bottleneck_rows(*, status: str) -> list[dict[str, Any]]:
    return [
        {
            "bottleneckId": "pr_171_operator_ready_or_merge_decision",
            "layer": "v0.1_rc_release_flow",
            "severity": "P1",
            "status": "open" if status == "ready" else "blocked_by_review",
            "blocksV01Rc": True,
            "blocksFinalVision": False,
            "summary": "PR #171 is clean and CI-green but still draft; ready/merge/cleanup requires an explicit operator decision.",
            "nextTranche": "operator_mark_pr_ready_or_merge_decision_after_review",
        },
        {
            "bottleneckId": "corpus_scale_answer_quality_gate_missing",
            "layer": "v0.1_rc_quality",
            "severity": "P1",
            "status": "open",
            "blocksV01Rc": False,
            "blocksFinalVision": True,
            "summary": "The current evidence proves a narrow Research Preview path, not corpus-scale answer quality for public/default promotion.",
            "nextTranche": "corpus_scale_quality_gate",
        },
        {
            "bottleneckId": "public_default_surface_promotion_held",
            "layer": "public_default_surface",
            "severity": "P1",
            "status": "open",
            "blocksV01Rc": False,
            "blocksFinalVision": True,
            "summary": "Public/default khub ask and default MCP remain held; the evidence chunk route is still labs/Research Preview bounded.",
            "nextTranche": "public_default_promotion_gate_after_corpus_scale_quality",
        },
        {
            "bottleneckId": "table_equation_figure_structured_evidence_labs_only",
            "layer": "structured_evidence_modalities",
            "severity": "P2",
            "status": "open",
            "blocksV01Rc": False,
            "blocksFinalVision": True,
            "summary": "Table, equation, and figure-caption evidence remain labs/limited support and are outside the v0.1 default promise.",
            "nextTranche": "structured_evidence_modality_roadmap_after_v01_rc",
        },
        {
            "bottleneckId": "post_merge_convergence_and_release_cleanup_pending",
            "layer": "release_operations",
            "severity": "P2",
            "status": "open",
            "blocksV01Rc": True,
            "blocksFinalVision": False,
            "summary": "After PR #171 is merged, a post-merge convergence report, branch cleanup decision, and release posture check must close the candidate.",
            "nextTranche": "post_merge_convergence_after_pr_171",
        },
    ]


def build_knowledgeos_v01_rc_vision_bottleneck_definition_review(
    *,
    product_definition_doc_path: str | Path = DEFAULT_PRODUCT_DEFINITION_DOC,
    draft_pr_post_open_review_report_path: str | Path = DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT,
    public_default_promotion_gate_report_path: str | Path = DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT,
    labs_release_gate_report_path: str | Path = DEFAULT_LABS_RELEASE_GATE_REPORT,
    product_definition_text: str | None = None,
    draft_pr_post_open_review_report: dict[str, Any] | None = None,
    public_default_promotion_gate_report: dict[str, Any] | None = None,
    labs_release_gate_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    product_text = product_definition_text if product_definition_text is not None else Path(product_definition_doc_path).read_text(encoding="utf-8")
    pr_report = dict(draft_pr_post_open_review_report or _read_json(draft_pr_post_open_review_report_path))
    public_report = dict(public_default_promotion_gate_report or _read_json(public_default_promotion_gate_report_path))
    labs_report = dict(labs_release_gate_report or _read_json(labs_release_gate_report_path))
    product_blockers = _product_definition_blockers(product_text)
    pr_blockers = _draft_pr_post_open_blockers(pr_report)
    public_blockers = _public_default_promotion_blockers(public_report)
    labs_blockers = _labs_release_gate_blockers(labs_report)
    unsafe_blockers = _unsafe_counter_blockers(pr_report, public_report, labs_report)
    semantic_violations = sorted(set(product_blockers + pr_blockers + public_blockers + labs_blockers + unsafe_blockers))
    private_path_leak_rows = (
        1
        if _contains_private_path(product_text)
        or _contains_private_path(pr_report)
        or _contains_private_path(public_report)
        or _contains_private_path(labs_report)
        else 0
    )
    if private_path_leak_rows:
        semantic_violations.append("vision_bottleneck_review_private_path_marker")
    semantic_violations = sorted(set(semantic_violations))
    status = "ready" if not semantic_violations else "blocked"
    pr_counts = dict(pr_report.get("counts") or {})
    public_counts = dict(public_report.get("counts") or {})
    labs_counts = dict(labs_report.get("counts") or {})
    bottlenecks = _bottleneck_rows(status=status)
    v01_blockers = [row for row in bottlenecks if row["blocksV01Rc"]]
    final_blockers = [row for row in bottlenecks if row["blocksFinalVision"]]
    counts = {
        "visionBottleneckReviewRows": 1,
        "productDefinitionReadyRows": 1 if not product_blockers else 0,
        "draftPrPostOpenReadyRows": 1 if not pr_blockers else 0,
        "labsReleaseGateReadyRows": 1 if not labs_blockers else 0,
        "publicDefaultPromotionGateReadyRows": 1 if not public_blockers else 0,
        "releaseSmokePassedRows": _int(labs_counts.get("releaseSmokePassedRows")),
        "publicHygieneIssueRows": _int(labs_counts.get("publicHygieneIssueRows")),
        "noAnswerPassRows": _int(labs_counts.get("noAnswerPassRows")),
        "readyForHumanReviewRows": _int(pr_counts.get("readyForHumanReviewRows")),
        "readyForMergeRows": 0,
        "publicDefaultPromotionReadyRows": 0,
        "publicDefaultPromotionHeldRows": 1 if _int(public_counts.get("publicDefaultPromotionHeldRows")) >= 1 else 0,
        "generalRcReadyRows": 0,
        "corpusScaleClaimProvenRows": 0,
        "v01RcBottleneckRows": len(v01_blockers),
        "finalVisionBottleneckRows": len(final_blockers),
        "totalBottleneckRows": len(bottlenecks),
        "blockedRows": len(semantic_violations),
        **{field: 0 for field in ZERO_COUNTER_FIELDS},
        **{field: 0 for field in EXTRA_ZERO_COUNTER_FIELDS},
        "privatePathLeakRows": private_path_leak_rows,
        "schemaViolationCount": len(semantic_violations),
    }
    return {
        "schema": KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_BLOCKED,
        "inputs": {
            "productDefinitionDocRef": DEFAULT_PRODUCT_DEFINITION_DOC.as_posix(),
            "draftPrPostOpenReviewReportRef": DEFAULT_DRAFT_PR_POST_OPEN_REVIEW_REPORT.as_posix(),
            "publicDefaultPromotionGateReportRef": DEFAULT_PUBLIC_DEFAULT_PROMOTION_GATE_REPORT.as_posix(),
            "labsReleaseGateReportRef": DEFAULT_LABS_RELEASE_GATE_REPORT.as_posix(),
        },
        "productDecision": {
            "decisionLabel": "narrow_scope",
            "v01RcPromise": "Research Preview section/paragraph evidence-first paper QA and compare runtime",
            "finalVisionPromise": "Local-first evidence-first research knowledge runtime for auditable AI research workflows",
            "shipPosture": "pr_171_ready_for_human_review_not_merge",
            "publicDefaultDecision": "hold_public_default_promotion",
            "strongestNextAction": "operator_ready_or_merge_pr_171",
        },
        "counts": counts,
        "gate": {
            "visionBottleneckDefinitionReady": status == "ready",
            "productDefinitionReady": not product_blockers,
            "draftPrPostOpenReady": not pr_blockers,
            "labsReleaseGateReady": not labs_blockers,
            "publicDefaultPromotionGateReady": not public_blockers,
            "readyForHumanReview": _int(pr_counts.get("readyForHumanReviewRows")) == 1,
            "readyForMerge": False,
            "publicDefaultPromotionAllowed": False,
            "generalRcReady": False,
            "semanticViolations": semantic_violations,
        },
        "bottleneckRows": bottlenecks,
        "checkRows": [
            {"checkId": "product_definition", "status": "pass" if not product_blockers else "fail", "blockers": product_blockers},
            {"checkId": "draft_pr_post_open_review", "status": "pass" if not pr_blockers else "fail", "blockers": pr_blockers},
            {"checkId": "labs_release_gate", "status": "pass" if not labs_blockers else "fail", "blockers": labs_blockers},
            {"checkId": "public_default_promotion_gate", "status": "pass" if not public_blockers else "fail", "blockers": public_blockers},
            {"checkId": "unsafe_counters", "status": "pass" if not unsafe_blockers else "fail", "blockers": unsafe_blockers},
        ],
        "warnings": [
            "v01_rc_is_research_preview_not_full_final_vision",
            "ready_for_human_review_is_not_merge_approval",
            "public_default_promotion_remains_held",
            "table_equation_figure_default_support_remains_future_scope",
        ],
    }


def render_knowledgeos_v01_rc_vision_bottleneck_definition_review_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    decision = dict(report.get("productDecision") or {})
    lines = [
        "# KnowledgeOS v0.1 RC Vision Bottleneck Definition Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- productDecision: `{decision.get('decisionLabel')}`",
        f"- shipPosture: `{decision.get('shipPosture')}`",
        f"- publicDefaultDecision: `{decision.get('publicDefaultDecision')}`",
        f"- readyForHumanReviewRows: `{counts.get('readyForHumanReviewRows')}`",
        f"- readyForMergeRows: `{counts.get('readyForMergeRows')}`",
        f"- publicDefaultPromotionHeldRows: `{counts.get('publicDefaultPromotionHeldRows')}`",
        f"- corpusScaleClaimProvenRows: `{counts.get('corpusScaleClaimProvenRows')}`",
        f"- v01RcBottleneckRows: `{counts.get('v01RcBottleneckRows')}`",
        f"- finalVisionBottleneckRows: `{counts.get('finalVisionBottleneckRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        "",
        "## Bottlenecks",
        "",
    ]
    for row in list(report.get("bottleneckRows") or []):
        lines.append(
            f"- `{row.get('severity')}` `{row.get('bottleneckId')}` "
            f"layer=`{row.get('layer')}` v01=`{row.get('blocksV01Rc')}` "
            f"final=`{row.get('blocksFinalVision')}`: {row.get('summary')}"
        )
    lines.extend(["", "## Checks", ""])
    for row in list(report.get("checkRows") or []):
        blockers = ", ".join(list(row.get("blockers") or [])) or "none"
        lines.append(f"- `{row.get('checkId')}`: `{row.get('status')}`; blockers=`{blockers}`")
    lines.extend(["", "## Mutation Guarantees", ""])
    for field in (*ZERO_COUNTER_FIELDS, *EXTRA_ZERO_COUNTER_FIELDS):
        lines.append(f"- {field}: `{counts.get(field)}`")
    return "\n".join(lines).rstrip() + "\n"


def write_knowledgeos_v01_rc_vision_bottleneck_definition_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_knowledgeos_v01_rc_vision_bottleneck_definition_review_markdown(report), encoding="utf-8")
    return {"json": report_json.as_posix(), "markdown": report_md.as_posix()}


__all__ = [
    "KNOWLEDGEOS_V01_RC_VISION_BOTTLENECK_DEFINITION_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_knowledgeos_v01_rc_vision_bottleneck_definition_review",
    "write_knowledgeos_v01_rc_vision_bottleneck_definition_review",
]
