"""Dry-run paper retrieval lane split for parsed-artifact evidence chunks.

This report-only helper records the next safe boundary after the visual
retrieval-hint tranche: visual hints may participate only in candidate
discovery, while parsed-artifact evidence chunks remain the lane that can later
feed answerability once provenance checks pass. It does not change runtime
search, query indexes, or promote evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.papers.limited_visual_retrieval_hint_candidate_store_apply_executor import (
    _contains_private_path,
    normalize_text,
)
from knowledge_hub.papers.limited_visual_retrieval_hint_final_merge_review import (
    LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID,
    READY_DECISION as FINAL_MERGE_READY_DECISION,
    load_json,
    sanitized_report_ref,
)


PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID = (
    "knowledge-hub.paper.paper-retrieval-lane-split-parsed-artifact-evidence-chunk-dry-run.v1"
)
PAPER_RETRIEVAL_LANE_SPLIT_ROW_SCHEMA_ID = (
    "knowledge-hub.paper.paper-retrieval-lane-split-dry-run-row.v1"
)

READY_DECISION = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review"
NEXT_TRANCHE_HOLD = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_review"

EXPECTED_VISUAL_HINT_ROWS = 125
PARSED_ARTIFACT_EVIDENCE_LANE = "parsed_artifact_evidence_chunk"
VISUAL_HINT_CANDIDATE_DISCOVERY_LANE = "visual_retrieval_hint_candidate_discovery"
TARGET_RUNTIME_BOUNDARY = "knowledge_hub.ai.rag_search_runtime.RAGSearchRuntime.search_with_diagnostics"
TARGET_PIPELINE_BOUNDARY = "knowledge_hub.ai.retrieval_pipeline.RetrievalPipelineService.execute"
TARGET_EVIDENCE_BOUNDARY = "knowledge_hub.ai.evidence_assembly.EvidenceAssemblyService.assemble"

READY_ROW_STATUS = "paper_retrieval_lane_split_dry_run_ready"
BLOCKED_SOURCE_STATUS = "blocked_source_report_gate"

SOURCE_UNSAFE_COUNTER_FIELDS = (
    "branchDeletionRows",
    "githubPrMutationRows",
    "runtimeRouteWriteRows",
    "runtimeConfigMutationRows",
    "operationalSearchIndexQueryRows",
    "runtimeVisibleRows",
    "answerVisibleRows",
    "answerGenerationRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "answerableWithoutTextEvidenceRows",
    "candidateStoreWriteRows",
    "embeddingCallRows",
    "embeddingVectorWriteRows",
    "vectorIndexWriteRows",
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "graphDbWriteRows",
    "ontologyWriteRows",
    "memoryCardWriteRows",
    "clusterWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
)

DRY_RUN_ZERO_COUNTER_FIELDS = (
    "runtimeRouteWriteRows",
    "runtimeConfigMutationRows",
    "operationalSearchIndexQueryRows",
    "runtimeVisibleRows",
    "answerVisibleRows",
    "answerGenerationRows",
    "strictEvidenceRows",
    "citationGradeRows",
    "runtimeEvidenceRows",
    "answerableWithoutTextEvidenceRows",
    "candidateStoreWriteRows",
    "sourceSpanCreatedRows",
    "sourceSpanCandidateCreatedRows",
    "parsedArtifactEvidenceChunkCreatedRows",
    "embeddingCallRows",
    "embeddingVectorWriteRows",
    "vectorIndexWriteRows",
    "productionVectorIndexWriteRows",
    "databaseMutationRows",
    "indexMutationRows",
    "reindexOrReembedRows",
    "parserExecutionRows",
    "canonicalParsedArtifactWriteRows",
    "graphDbWriteRows",
    "ontologyWriteRows",
    "memoryCardWriteRows",
    "clusterWriteRows",
    "vaultScanRows",
    "externalDownloadRows",
    "branchDeletionRows",
    "githubPrMutationRows",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _short_hash(value: str, *, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _counts(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("counts") or {})


def _final_merge_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "counts": {
            "reviewRows": _int(counts.get("reviewRows")),
            "plannedRouteBindingRows": _int(counts.get("plannedRouteBindingRows")),
            "candidateDiscoveryOnlyRows": _int(counts.get("candidateDiscoveryOnlyRows")),
            "qualityEvalProductionVectorHitAt5Rows": _int(counts.get("qualityEvalProductionVectorHitAt5Rows")),
            "qualityEvalHybridHitAt5Rows": _int(counts.get("qualityEvalHybridHitAt5Rows")),
            "qualityEvalHybridHitAt5LiftRows": _int(counts.get("qualityEvalHybridHitAt5LiftRows")),
            "qualityEvalRankRegressedRows": _int(counts.get("qualityEvalRankRegressedRows")),
            "blockedRows": _int(counts.get("blockedRows")),
            "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
            "schemaViolationCount": _int(counts.get("schemaViolationCount")),
        },
    }


def _final_merge_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != LIMITED_VISUAL_RETRIEVAL_HINT_FINAL_MERGE_REVIEW_SCHEMA_ID:
        blockers.append("invalid_visual_retrieval_hint_final_merge_review_schema")
    if report.get("status") != "ready":
        blockers.append("visual_retrieval_hint_final_merge_review_not_ready")
    if report.get("decision") != FINAL_MERGE_READY_DECISION:
        blockers.append("visual_retrieval_hint_final_merge_review_invalid_decision")
    if dict(report.get("gate") or {}).get("passed") is not True:
        blockers.append("visual_retrieval_hint_final_merge_review_gate_not_passed")
    if _int(counts.get("plannedRouteBindingRows")) != EXPECTED_VISUAL_HINT_ROWS:
        blockers.append("visual_hint_planned_route_binding_rows_not_125")
    if _int(counts.get("candidateDiscoveryOnlyRows")) != EXPECTED_VISUAL_HINT_ROWS:
        blockers.append("visual_hint_candidate_discovery_only_rows_not_125")
    if _int(counts.get("qualityEvalRankRegressedRows")) != 0:
        blockers.append("visual_hint_quality_eval_rank_regressions_present")
    for field in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *SOURCE_UNSAFE_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"visual_retrieval_hint_final_merge_review_has_{field}")
    if _contains_private_path(report):
        blockers.append("visual_retrieval_hint_final_merge_review_has_private_path_leak")
    return blockers


def _lane_row(
    *,
    lane_name: str,
    lane_kind: str,
    source_blockers: list[str],
    may_expand_search_candidates: bool,
    may_supply_answer_evidence: bool,
    may_satisfy_answerability: bool,
    answer_visible_allowed_after_contract: bool,
    strict_text_evidence_required: bool,
    evidence_contract: str,
    retrieval_unit_kind: str,
    upstream_lane: str | None = None,
) -> dict[str, Any]:
    basis = "|".join([lane_name, lane_kind, retrieval_unit_kind, evidence_contract])
    row = {
        "schema": PAPER_RETRIEVAL_LANE_SPLIT_ROW_SCHEMA_ID,
        "rowId": "paper-retrieval-lane-split:" + _short_hash(basis),
        "status": READY_ROW_STATUS if not source_blockers else BLOCKED_SOURCE_STATUS,
        "laneName": lane_name,
        "laneKind": lane_kind,
        "sourceType": "paper",
        "retrievalUnitKind": retrieval_unit_kind,
        "targetRuntimeBoundary": TARGET_RUNTIME_BOUNDARY,
        "targetPipelineBoundary": TARGET_PIPELINE_BOUNDARY,
        "targetEvidenceBoundary": TARGET_EVIDENCE_BOUNDARY,
        "upstreamCandidateDiscoveryLane": upstream_lane or "",
        "lanePolicy": {
            "mayExpandSearchCandidates": may_expand_search_candidates,
            "maySupplyAnswerEvidence": may_supply_answer_evidence,
            "maySatisfyAnswerability": may_satisfy_answerability,
            "answerVisibleAllowedAfterContract": answer_visible_allowed_after_contract,
            "strictTextEvidenceRequired": strict_text_evidence_required,
            "visualHintTextMayBeQuotedAsEvidence": False,
            "candidateDiscoveryOnly": lane_name == VISUAL_HINT_CANDIDATE_DISCOVERY_LANE,
            "requiresSourceContentHash": lane_name == PARSED_ARTIFACT_EVIDENCE_LANE,
            "requiresLocator": lane_name == PARSED_ARTIFACT_EVIDENCE_LANE,
            "requiresEvidenceContract": lane_name == PARSED_ARTIFACT_EVIDENCE_LANE,
            "fallbackToVisualHintAsEvidenceAllowed": False,
        },
        "evidenceContract": {
            "contractName": evidence_contract,
            "requiredForAnswerability": may_satisfy_answerability,
            "requiredFields": (
                ["sourceContentHash", "source_type=paper", "chars:start-end_or_page_bbox", "excerpt_or_text_span"]
                if may_supply_answer_evidence
                else []
            ),
            "disallowedEvidenceSources": ["visual_retrieval_hint_text", "locator_only_anchor", "memory_unit_locator"],
        },
        "dryRunEffects": {
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "operationalSearchIndexQuery": False,
            "answerGeneration": False,
            "answerVisibleExposure": False,
            "strictEvidenceCreation": False,
            "citationGradePromotion": False,
            "sourceSpanCreation": False,
            "parsedArtifactEvidenceChunkCreation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "parserExecution": False,
        },
        "blockers": list(source_blockers),
    }
    row["laneSplitRowSha256"] = _sha256_json(row)
    return row


def _lane_rows(source_blockers: list[str]) -> list[dict[str, Any]]:
    return [
        _lane_row(
            lane_name=PARSED_ARTIFACT_EVIDENCE_LANE,
            lane_kind="answer_evidence",
            source_blockers=source_blockers,
            may_expand_search_candidates=True,
            may_supply_answer_evidence=True,
            may_satisfy_answerability=True,
            answer_visible_allowed_after_contract=True,
            strict_text_evidence_required=True,
            evidence_contract="parsed_artifact_source_hash_locator_text_span_contract",
            retrieval_unit_kind="parsed_artifact_evidence_chunk",
            upstream_lane=VISUAL_HINT_CANDIDATE_DISCOVERY_LANE,
        ),
        _lane_row(
            lane_name=VISUAL_HINT_CANDIDATE_DISCOVERY_LANE,
            lane_kind="candidate_discovery",
            source_blockers=source_blockers,
            may_expand_search_candidates=True,
            may_supply_answer_evidence=False,
            may_satisfy_answerability=False,
            answer_visible_allowed_after_contract=False,
            strict_text_evidence_required=False,
            evidence_contract="retrieval_hint_candidate_discovery_only_contract",
            retrieval_unit_kind="visual_retrieval_hint_candidate_discovery_signal",
        ),
    ]


def _review_counts(rows: list[dict[str, Any]], final_merge_review: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    final_counts = _counts(final_merge_review)
    ready_rows = [row for row in rows if row.get("status") == READY_ROW_STATUS]
    parsed_rows = [row for row in ready_rows if row.get("laneName") == PARSED_ARTIFACT_EVIDENCE_LANE]
    visual_rows = [row for row in ready_rows if row.get("laneName") == VISUAL_HINT_CANDIDATE_DISCOVERY_LANE]
    counts = {
        "sourceFinalMergeReviewRows": _int(final_counts.get("reviewRows")),
        "sourcePlannedRouteBindingRows": _int(final_counts.get("plannedRouteBindingRows")),
        "sourceCandidateDiscoveryOnlyRows": _int(final_counts.get("candidateDiscoveryOnlyRows")),
        "sourceQualityEvalHybridHitAt5Rows": _int(final_counts.get("qualityEvalHybridHitAt5Rows")),
        "sourceQualityEvalHybridHitAt5LiftRows": _int(final_counts.get("qualityEvalHybridHitAt5LiftRows")),
        "sourceQualityEvalRankRegressedRows": _int(final_counts.get("qualityEvalRankRegressedRows")),
        "retrievalLaneRows": len(rows),
        "plannedLaneSplitRows": len(ready_rows),
        "parsedArtifactEvidenceChunkLaneRows": len(parsed_rows),
        "visualCandidateDiscoveryLaneRows": len(visual_rows),
        "answerEvidenceEligibleLaneRows": sum(1 for row in ready_rows if dict(row.get("lanePolicy") or {}).get("maySupplyAnswerEvidence") is True),
        "answerEvidenceIneligibleLaneRows": sum(1 for row in ready_rows if dict(row.get("lanePolicy") or {}).get("maySupplyAnswerEvidence") is False),
        "answerabilityEligibleLaneRows": sum(1 for row in ready_rows if dict(row.get("lanePolicy") or {}).get("maySatisfyAnswerability") is True),
        "answerabilityIneligibleLaneRows": sum(1 for row in ready_rows if dict(row.get("lanePolicy") or {}).get("maySatisfyAnswerability") is False),
        "visualHintRowsQuarantinedFromAnswerEvidence": (
            _int(final_counts.get("candidateDiscoveryOnlyRows")) if visual_rows else 0
        ),
        "fallbackToVisualHintAsEvidenceAllowedRows": sum(
            1
            for row in rows
            if dict(row.get("lanePolicy") or {}).get("fallbackToVisualHintAsEvidenceAllowed") is True
        ),
        "blockedRows": len(blockers),
        "privatePathLeakRows": sum(1 for blocker in blockers if "private_path" in blocker),
        "schemaViolationCount": 0,
    }
    counts.update({field: 0 for field in DRY_RUN_ZERO_COUNTER_FIELDS})
    return counts


def _gate(counts: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    checks = {
        "sourceFinalMergeReviewReady": not blockers,
        "sourceVisualHintsExactly125": _int(counts.get("sourceCandidateDiscoveryOnlyRows")) == EXPECTED_VISUAL_HINT_ROWS,
        "twoRetrievalLanesPlanned": _int(counts.get("plannedLaneSplitRows")) == 2,
        "parsedArtifactEvidenceLanePresent": _int(counts.get("parsedArtifactEvidenceChunkLaneRows")) == 1,
        "visualCandidateDiscoveryLanePresent": _int(counts.get("visualCandidateDiscoveryLaneRows")) == 1,
        "onlyParsedLaneMaySupplyAnswerEvidence": _int(counts.get("answerEvidenceEligibleLaneRows")) == 1
        and _int(counts.get("answerEvidenceIneligibleLaneRows")) == 1,
        "onlyParsedLaneMaySatisfyAnswerability": _int(counts.get("answerabilityEligibleLaneRows")) == 1
        and _int(counts.get("answerabilityIneligibleLaneRows")) == 1,
        "visualHintsQuarantinedFromAnswerEvidence": _int(counts.get("visualHintRowsQuarantinedFromAnswerEvidence"))
        == EXPECTED_VISUAL_HINT_ROWS,
        "fallbackToVisualHintAsEvidenceDisallowed": _int(counts.get("fallbackToVisualHintAsEvidenceAllowedRows")) == 0,
        "noBlockedRows": _int(counts.get("blockedRows")) == 0,
        "noPrivatePathLeaks": _int(counts.get("privatePathLeakRows")) == 0,
        "noMutationOrRuntimeExposure": all(_int(counts.get(field)) == 0 for field in DRY_RUN_ZERO_COUNTER_FIELDS),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedVisualHintRows": EXPECTED_VISUAL_HINT_ROWS,
        "observed": {
            "sourceCandidateDiscoveryOnlyRows": _int(counts.get("sourceCandidateDiscoveryOnlyRows")),
            "plannedLaneSplitRows": _int(counts.get("plannedLaneSplitRows")),
            "parsedArtifactEvidenceChunkLaneRows": _int(counts.get("parsedArtifactEvidenceChunkLaneRows")),
            "visualCandidateDiscoveryLaneRows": _int(counts.get("visualCandidateDiscoveryLaneRows")),
            "blockedRows": _int(counts.get("blockedRows")),
        },
    }


def build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run(
    *,
    visual_retrieval_hint_final_merge_review: dict[str, Any],
    source_visual_retrieval_hint_final_merge_review_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    blockers = sorted(set(_final_merge_blockers(visual_retrieval_hint_final_merge_review)))
    rows = _lane_rows(blockers)
    counts = _review_counts(rows, visual_retrieval_hint_final_merge_review, blockers)
    gate = _gate(counts, blockers)
    status = "ready" if gate.get("passed") and not blockers else "blocked"
    return {
        "schema": PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD,
        "input": {
            "sourceVisualRetrievalHintFinalMergeReviewRef": normalize_text(
                source_visual_retrieval_hint_final_merge_review_ref
            ),
            "expectedVisualHintRows": EXPECTED_VISUAL_HINT_ROWS,
            "targetRuntimeBoundary": TARGET_RUNTIME_BOUNDARY,
            "targetPipelineBoundary": TARGET_PIPELINE_BOUNDARY,
            "targetEvidenceBoundary": TARGET_EVIDENCE_BOUNDARY,
        },
        "policy": {
            "dryRunOnly": True,
            "laneSplitDesignOnly": True,
            "candidateDiscoveryOnlyForVisualHints": True,
            "parsedArtifactEvidenceRequiredForAnswerability": True,
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "operationalSearchIndexQuery": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "strictEvidenceCreation": False,
            "citationGradePromotion": False,
            "sourceSpanCreation": False,
            "parsedArtifactEvidenceChunkCreation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "vaultScan": False,
            "externalDownload": False,
            "branchDeletion": False,
            "githubMutation": False,
        },
        "sourceVisualRetrievalHintFinalMergeReview": _final_merge_summary(
            visual_retrieval_hint_final_merge_review,
            report_ref=source_visual_retrieval_hint_final_merge_review_ref,
        ),
        "method": {
            "name": "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_v1",
            "description": (
                "Plans separate retrieval lanes so visual hints can only expand paper candidates "
                "and parsed-artifact evidence chunks remain the answerability lane."
            ),
            "completionBoundary": "dry_run_report_only",
            "nextEvidenceWork": NEXT_TRANCHE_READY,
        },
        "laneSemantics": {
            "candidateDiscoveryLane": VISUAL_HINT_CANDIDATE_DISCOVERY_LANE,
            "answerEvidenceLane": PARSED_ARTIFACT_EVIDENCE_LANE,
            "candidateDiscoveryMayPrecedeEvidenceResolution": True,
            "visualHintTextMaySatisfyAnswerability": False,
            "visualHintTextMayBeQuotedAsEvidence": False,
            "parsedArtifactEvidenceMustCarrySourceHashAndLocator": True,
        },
        "counts": counts,
        "gate": gate,
        "retrievalLaneRows": rows,
        "technicalBlockers": blockers,
        "warnings": [],
    }


def render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Paper Retrieval Lane Split Parsed Artifact Evidence Chunk Dry Run",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- sourceCandidateDiscoveryOnlyRows: `{counts.get('sourceCandidateDiscoveryOnlyRows')}`",
        f"- plannedLaneSplitRows: `{counts.get('plannedLaneSplitRows')}`",
        f"- parsedArtifactEvidenceChunkLaneRows: `{counts.get('parsedArtifactEvidenceChunkLaneRows')}`",
        f"- visualCandidateDiscoveryLaneRows: `{counts.get('visualCandidateDiscoveryLaneRows')}`",
        f"- answerEvidenceEligibleLaneRows: `{counts.get('answerEvidenceEligibleLaneRows')}`",
        f"- visualHintRowsQuarantinedFromAnswerEvidence: `{counts.get('visualHintRowsQuarantinedFromAnswerEvidence')}`",
        f"- fallbackToVisualHintAsEvidenceAllowedRows: `{counts.get('fallbackToVisualHintAsEvidenceAllowedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        f"- runtimeRouteWriteRows: `{counts.get('runtimeRouteWriteRows')}`",
        f"- operationalSearchIndexQueryRows: `{counts.get('operationalSearchIndexQueryRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- onlyParsedLaneMaySupplyAnswerEvidence: `{dict(gate.get('checks') or {}).get('onlyParsedLaneMaySupplyAnswerEvidence')}`",
        f"- visualHintsQuarantinedFromAnswerEvidence: `{dict(gate.get('checks') or {}).get('visualHintsQuarantinedFromAnswerEvidence')}`",
        f"- noMutationOrRuntimeExposure: `{dict(gate.get('checks') or {}).get('noMutationOrRuntimeExposure')}`",
        "",
        "## Lane Split",
        "",
        "- `visual_retrieval_hint_candidate_discovery`: can expand candidates only.",
        "- `parsed_artifact_evidence_chunk`: the only planned lane that may later satisfy answerability.",
        "",
        "## Non-Scope",
        "",
        "- No runtime route write.",
        "- No operational search index query.",
        "- No vector DB apply.",
        "- No answer-visible exposure.",
        "- No strict or citation-grade evidence creation.",
        "- No parser execution, reindex, DB/index mutation, vault scan, or external download.",
    ]
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID",
    "READY_DECISION",
    "build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run",
    "load_json",
    "render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run_markdown",
    "sanitized_report_ref",
    "write_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run",
]
