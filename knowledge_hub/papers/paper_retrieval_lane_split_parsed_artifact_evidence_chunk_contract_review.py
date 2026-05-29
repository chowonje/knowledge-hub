"""Contract review for parsed-artifact evidence chunks.

This report-only helper fixes the minimum contract for the
`parsed_artifact_evidence_chunk` lane introduced by the retrieval lane split.
It does not create chunks, mutate stores, change runtime routes, or promote
anything to strict/citation evidence.
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
from knowledge_hub.papers.paper_retrieval_lane_split_parsed_artifact_evidence_chunk_dry_run import (
    PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID,
    PARSED_ARTIFACT_EVIDENCE_LANE,
    READY_DECISION as LANE_SPLIT_READY_DECISION,
    VISUAL_HINT_CANDIDATE_DISCOVERY_LANE,
    load_json,
    sanitized_report_ref,
)


PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID = (
    "knowledge-hub.paper.paper-retrieval-lane-split-parsed-artifact-evidence-chunk-contract-review.v1"
)
PARSED_ARTIFACT_EVIDENCE_CHUNK_ARTIFACT_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-artifact-contract.v1"
)
PARSED_ARTIFACT_EVIDENCE_CHUNK_DISALLOWED_SOURCE_CONTRACT_SCHEMA_ID = (
    "knowledge-hub.paper.parsed-artifact-evidence-chunk-disallowed-source-contract.v1"
)

READY_DECISION = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_ready"
BLOCKED_DECISION = "blocked"
NEXT_TRANCHE_READY = "parsed_artifact_evidence_chunk_candidate_dry_run"
NEXT_TRANCHE_HOLD = "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_repair"

EXPECTED_VISUAL_HINT_ROWS = 125
EXPECTED_LANE_SPLIT_ROWS = 2
EXPECTED_ALLOWED_ARTIFACT_TYPES = ("section", "paragraph", "table", "equation", "figure_caption")
DISALLOWED_SOURCE_TYPES = (
    "visual_retrieval_hint_text",
    "fallback_chunk",
    "locator_only_anchor",
    "memory_unit_locator",
    "korean_summary_or_paraphrase",
)

ZERO_COUNTER_FIELDS = (
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


def _source_summary(report: dict[str, Any], *, report_ref: str) -> dict[str, Any]:
    counts = _counts(report)
    return {
        "schema": normalize_text(report.get("schema")),
        "status": normalize_text(report.get("status")),
        "decision": normalize_text(report.get("decision")),
        "nextRecommendedTranche": normalize_text(report.get("nextRecommendedTranche")),
        "reportRef": normalize_text(report_ref),
        "counts": {
            "sourceCandidateDiscoveryOnlyRows": _int(counts.get("sourceCandidateDiscoveryOnlyRows")),
            "plannedLaneSplitRows": _int(counts.get("plannedLaneSplitRows")),
            "parsedArtifactEvidenceChunkLaneRows": _int(counts.get("parsedArtifactEvidenceChunkLaneRows")),
            "visualCandidateDiscoveryLaneRows": _int(counts.get("visualCandidateDiscoveryLaneRows")),
            "answerEvidenceEligibleLaneRows": _int(counts.get("answerEvidenceEligibleLaneRows")),
            "answerabilityEligibleLaneRows": _int(counts.get("answerabilityEligibleLaneRows")),
            "visualHintRowsQuarantinedFromAnswerEvidence": _int(
                counts.get("visualHintRowsQuarantinedFromAnswerEvidence")
            ),
            "fallbackToVisualHintAsEvidenceAllowedRows": _int(
                counts.get("fallbackToVisualHintAsEvidenceAllowedRows")
            ),
            "blockedRows": _int(counts.get("blockedRows")),
            "privatePathLeakRows": _int(counts.get("privatePathLeakRows")),
            "schemaViolationCount": _int(counts.get("schemaViolationCount")),
        },
    }


def _lane_split_blockers(report: dict[str, Any]) -> list[str]:
    counts = _counts(report)
    blockers: list[str] = []
    if report.get("schema") != PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_DRY_RUN_SCHEMA_ID:
        blockers.append("invalid_paper_retrieval_lane_split_dry_run_schema")
    if report.get("status") != "ready":
        blockers.append("paper_retrieval_lane_split_dry_run_not_ready")
    if report.get("decision") != LANE_SPLIT_READY_DECISION:
        blockers.append("paper_retrieval_lane_split_dry_run_invalid_decision")
    if dict(report.get("gate") or {}).get("passed") is not True:
        blockers.append("paper_retrieval_lane_split_dry_run_gate_not_passed")
    if _int(counts.get("sourceCandidateDiscoveryOnlyRows")) != EXPECTED_VISUAL_HINT_ROWS:
        blockers.append("source_candidate_discovery_only_rows_not_125")
    if _int(counts.get("plannedLaneSplitRows")) != EXPECTED_LANE_SPLIT_ROWS:
        blockers.append("planned_lane_split_rows_not_2")
    if _int(counts.get("parsedArtifactEvidenceChunkLaneRows")) != 1:
        blockers.append("parsed_artifact_evidence_chunk_lane_missing")
    if _int(counts.get("visualCandidateDiscoveryLaneRows")) != 1:
        blockers.append("visual_candidate_discovery_lane_missing")
    if _int(counts.get("answerEvidenceEligibleLaneRows")) != 1:
        blockers.append("answer_evidence_eligible_lane_rows_not_1")
    if _int(counts.get("answerabilityEligibleLaneRows")) != 1:
        blockers.append("answerability_eligible_lane_rows_not_1")
    if _int(counts.get("visualHintRowsQuarantinedFromAnswerEvidence")) != EXPECTED_VISUAL_HINT_ROWS:
        blockers.append("visual_hints_not_quarantined_from_answer_evidence")
    if _int(counts.get("fallbackToVisualHintAsEvidenceAllowedRows")) != 0:
        blockers.append("fallback_to_visual_hint_as_evidence_allowed")
    for field in ("blockedRows", "privatePathLeakRows", "schemaViolationCount", *ZERO_COUNTER_FIELDS):
        if _int(counts.get(field)) != 0:
            blockers.append(f"paper_retrieval_lane_split_dry_run_has_{field}")
    if _contains_private_path(report):
        blockers.append("paper_retrieval_lane_split_dry_run_has_private_path_leak")
    blockers.extend(_lane_policy_blockers(report))
    return blockers


def _lane_policy_blockers(report: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    rows = list(report.get("retrievalLaneRows") or [])
    lanes = {normalize_text(row.get("laneName")): row for row in rows if isinstance(row, dict)}
    parsed_lane = lanes.get(PARSED_ARTIFACT_EVIDENCE_LANE)
    visual_lane = lanes.get(VISUAL_HINT_CANDIDATE_DISCOVERY_LANE)
    if parsed_lane is None:
        blockers.append("parsed_artifact_evidence_chunk_lane_row_missing")
    if visual_lane is None:
        blockers.append("visual_candidate_discovery_lane_row_missing")
    if parsed_lane:
        policy = dict(parsed_lane.get("lanePolicy") or {})
        if policy.get("maySupplyAnswerEvidence") is not True:
            blockers.append("parsed_artifact_evidence_chunk_lane_cannot_supply_answer_evidence")
        if policy.get("maySatisfyAnswerability") is not True:
            blockers.append("parsed_artifact_evidence_chunk_lane_cannot_satisfy_answerability")
        if policy.get("requiresSourceContentHash") is not True:
            blockers.append("parsed_artifact_evidence_chunk_lane_missing_source_hash_requirement")
        if policy.get("requiresLocator") is not True:
            blockers.append("parsed_artifact_evidence_chunk_lane_missing_locator_requirement")
        if policy.get("fallbackToVisualHintAsEvidenceAllowed") is True:
            blockers.append("parsed_artifact_evidence_chunk_lane_allows_visual_hint_fallback")
    if visual_lane:
        policy = dict(visual_lane.get("lanePolicy") or {})
        if policy.get("candidateDiscoveryOnly") is not True:
            blockers.append("visual_candidate_discovery_lane_not_candidate_only")
        if policy.get("maySupplyAnswerEvidence") is not False:
            blockers.append("visual_candidate_discovery_lane_can_supply_answer_evidence")
        if policy.get("maySatisfyAnswerability") is not False:
            blockers.append("visual_candidate_discovery_lane_can_satisfy_answerability")
        if policy.get("fallbackToVisualHintAsEvidenceAllowed") is True:
            blockers.append("visual_candidate_discovery_lane_allows_visual_hint_fallback")
    return blockers


def _artifact_contract(
    artifact_type: str,
    *,
    locator_contract: str,
    additional_required_fields: list[str],
    structured: bool,
) -> dict[str, Any]:
    required_fields = [
        "contractVersion",
        "paperId",
        "sourceType=paper",
        "artifactType",
        "sourceRef",
        "sourceContentHash",
        "locator",
        "excerpt",
        "snippetHash",
        "evidenceKind",
        "derivation.parentParsedArtifactRef",
        "answerEvidenceEligible",
        "answerabilityEligible",
    ]
    required_fields.extend(additional_required_fields)
    row = {
        "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_ARTIFACT_CONTRACT_SCHEMA_ID,
        "contractId": f"parsed-artifact-evidence-chunk-contract:{artifact_type}",
        "artifactType": artifact_type,
        "contractVersion": "parsed_artifact_evidence_chunk_contract_v1",
        "sourceType": "paper",
        "locatorContract": locator_contract,
        "requiredFields": required_fields,
        "requiredHashFields": ["sourceContentHash", "snippetHash"],
        "requiredSourceFields": ["paperId", "sourceRef", "sourceContentHash"],
        "answerEvidenceEligibleWhenContractSatisfied": True,
        "answerabilityEligibleWhenContractSatisfied": True,
        "requiresOriginalSourceText": artifact_type in {"section", "paragraph", "figure_caption"},
        "requiresStructuredArtifactReadback": structured,
        "strictEvidenceByDefault": False,
        "citationGradeByDefault": False,
        "runtimeVisibleByDefault": False,
        "fallbackAllowed": False,
        "visualHintTextAllowed": False,
        "locatorOnlyAllowed": False,
        "paraphraseAllowedAsOriginalEvidence": False,
        "notes": [
            "candidate_chunks_are_not_strict_evidence_until_promoted_by_a_separate_gate",
            "answerability_requires_contract_satisfied_plus_later_runtime_gate",
        ],
    }
    row["contractSha256"] = _sha256_json(row)
    return row


def _artifact_contracts() -> list[dict[str, Any]]:
    return [
        _artifact_contract(
            "section",
            locator_contract="chars:start-end_required",
            additional_required_fields=["sectionPath", "sectionTitle"],
            structured=False,
        ),
        _artifact_contract(
            "paragraph",
            locator_contract="chars:start-end_required",
            additional_required_fields=["sectionPath", "paragraphIndex"],
            structured=False,
        ),
        _artifact_contract(
            "table",
            locator_contract="page_bbox_plus_table_cell_or_caption_required",
            additional_required_fields=["tableId", "captionOrHeader", "cellOrRegionProvenance"],
            structured=True,
        ),
        _artifact_contract(
            "equation",
            locator_contract="chars_or_page_bbox_plus_equation_identity_required",
            additional_required_fields=["equationId", "equationTextOrLatexHash", "nearbyTextLocator"],
            structured=True,
        ),
        _artifact_contract(
            "figure_caption",
            locator_contract="chars_or_page_bbox_plus_caption_identity_required",
            additional_required_fields=["figureId", "captionTextHash"],
            structured=True,
        ),
    ]


def _disallowed_source_contracts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reasons = {
        "visual_retrieval_hint_text": "candidate_discovery_only_not_answer_evidence",
        "fallback_chunk": "fallback_chunk_missing_strict_provenance",
        "locator_only_anchor": "locator_without_verbatim_text_cannot_satisfy_answerability",
        "memory_unit_locator": "memory_unit_locator_is_not_original_paper_evidence",
        "korean_summary_or_paraphrase": "derived_language_or_paraphrase_is_not_original_evidence",
    }
    for source_type in DISALLOWED_SOURCE_TYPES:
        row = {
            "schema": PARSED_ARTIFACT_EVIDENCE_CHUNK_DISALLOWED_SOURCE_CONTRACT_SCHEMA_ID,
            "contractId": f"parsed-artifact-evidence-chunk-disallowed-source:{source_type}",
            "sourceKind": source_type,
            "reason": reasons[source_type],
            "answerEvidenceAllowed": False,
            "answerabilityAllowed": False,
            "strictEvidenceAllowed": False,
            "citationGradeAllowed": False,
            "fallbackToParsedArtifactEvidenceAllowed": False,
        }
        row["contractSha256"] = _sha256_json(row)
        rows.append(row)
    return rows


def _review_counts(
    *,
    lane_split_dry_run: dict[str, Any],
    artifact_contracts: list[dict[str, Any]],
    disallowed_source_contracts: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    source_counts = _counts(lane_split_dry_run)
    counts = {
        "sourceCandidateDiscoveryOnlyRows": _int(source_counts.get("sourceCandidateDiscoveryOnlyRows")),
        "sourcePlannedLaneSplitRows": _int(source_counts.get("plannedLaneSplitRows")),
        "sourceParsedArtifactEvidenceChunkLaneRows": _int(
            source_counts.get("parsedArtifactEvidenceChunkLaneRows")
        ),
        "sourceVisualCandidateDiscoveryLaneRows": _int(source_counts.get("visualCandidateDiscoveryLaneRows")),
        "sourceAnswerEvidenceEligibleLaneRows": _int(source_counts.get("answerEvidenceEligibleLaneRows")),
        "sourceAnswerabilityEligibleLaneRows": _int(source_counts.get("answerabilityEligibleLaneRows")),
        "sourceVisualHintRowsQuarantinedFromAnswerEvidence": _int(
            source_counts.get("visualHintRowsQuarantinedFromAnswerEvidence")
        ),
        "contractRows": len(artifact_contracts),
        "allowedArtifactTypeRows": len(artifact_contracts),
        "allowedArtifactContractRows": len(artifact_contracts),
        "disallowedSourceRows": len(disallowed_source_contracts),
        "disallowedSourceContractRows": len(disallowed_source_contracts),
        "textSpanContractRows": sum(
            1 for row in artifact_contracts if row.get("artifactType") in {"section", "paragraph"}
        ),
        "textSpanArtifactContractRows": sum(
            1 for row in artifact_contracts if row.get("artifactType") in {"section", "paragraph"}
        ),
        "structuredArtifactContractRows": sum(
            1 for row in artifact_contracts if bool(row.get("requiresStructuredArtifactReadback"))
        ),
        "answerEvidenceEligibleContractRows": sum(
            1 for row in artifact_contracts if row.get("answerEvidenceEligibleWhenContractSatisfied") is True
        ),
        "answerabilityEligibleContractRows": sum(
            1 for row in artifact_contracts if row.get("answerabilityEligibleWhenContractSatisfied") is True
        ),
        "visualHintAnswerEvidenceAllowedRows": sum(
            1
            for row in disallowed_source_contracts
            if row.get("sourceKind") == "visual_retrieval_hint_text" and row.get("answerEvidenceAllowed") is True
        ),
        "fallbackChunkAnswerEvidenceAllowedRows": sum(
            1
            for row in disallowed_source_contracts
            if row.get("sourceKind") == "fallback_chunk" and row.get("answerEvidenceAllowed") is True
        ),
        "locatorOnlyAnswerEvidenceAllowedRows": sum(
            1
            for row in disallowed_source_contracts
            if row.get("sourceKind") == "locator_only_anchor" and row.get("answerEvidenceAllowed") is True
        ),
        "memoryUnitAnswerEvidenceAllowedRows": sum(
            1
            for row in disallowed_source_contracts
            if row.get("sourceKind") == "memory_unit_locator" and row.get("answerEvidenceAllowed") is True
        ),
        "paraphraseAnswerEvidenceAllowedRows": sum(
            1
            for row in disallowed_source_contracts
            if row.get("sourceKind") == "korean_summary_or_paraphrase"
            and row.get("answerEvidenceAllowed") is True
        ),
        "blockedRows": len(blockers),
        "privatePathLeakRows": sum(1 for blocker in blockers if "private_path" in blocker),
        "schemaViolationCount": 0,
    }
    counts.update({field: 0 for field in ZERO_COUNTER_FIELDS})
    return counts


def _gate(counts: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    checks = {
        "sourceLaneSplitReady": not blockers,
        "sourceVisualHintsQuarantined": _int(counts.get("sourceVisualHintRowsQuarantinedFromAnswerEvidence"))
        == EXPECTED_VISUAL_HINT_ROWS,
        "sourceHasTwoLaneSplitRows": _int(counts.get("sourcePlannedLaneSplitRows")) == EXPECTED_LANE_SPLIT_ROWS,
        "allArtifactTypesCovered": _int(counts.get("allowedArtifactContractRows"))
        == len(EXPECTED_ALLOWED_ARTIFACT_TYPES),
        "allDisallowedSourcesCovered": _int(counts.get("disallowedSourceRows")) == len(DISALLOWED_SOURCE_TYPES),
        "allAllowedContractsCanBecomeAnswerEvidenceAfterSatisfaction": _int(
            counts.get("answerEvidenceEligibleContractRows")
        )
        == len(EXPECTED_ALLOWED_ARTIFACT_TYPES),
        "allAllowedContractsCanSatisfyAnswerabilityAfterSatisfaction": _int(
            counts.get("answerabilityEligibleContractRows")
        )
        == len(EXPECTED_ALLOWED_ARTIFACT_TYPES),
        "visualHintCannotBeAnswerEvidence": _int(counts.get("visualHintAnswerEvidenceAllowedRows")) == 0,
        "fallbackChunkCannotBeAnswerEvidence": _int(counts.get("fallbackChunkAnswerEvidenceAllowedRows")) == 0,
        "locatorOnlyCannotBeAnswerEvidence": _int(counts.get("locatorOnlyAnswerEvidenceAllowedRows")) == 0,
        "memoryUnitCannotBeAnswerEvidence": _int(counts.get("memoryUnitAnswerEvidenceAllowedRows")) == 0,
        "paraphraseCannotBeOriginalAnswerEvidence": _int(counts.get("paraphraseAnswerEvidenceAllowedRows")) == 0,
        "noBlockedRows": _int(counts.get("blockedRows")) == 0,
        "noPrivatePathLeaks": _int(counts.get("privatePathLeakRows")) == 0,
        "noMutationOrRuntimeExposure": all(_int(counts.get(field)) == 0 for field in ZERO_COUNTER_FIELDS),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expectedAllowedArtifactTypes": list(EXPECTED_ALLOWED_ARTIFACT_TYPES),
        "expectedDisallowedSourceTypes": list(DISALLOWED_SOURCE_TYPES),
        "observed": {
            "contractRows": _int(counts.get("contractRows")),
            "allowedArtifactTypeRows": _int(counts.get("allowedArtifactTypeRows")),
            "disallowedSourceRows": _int(counts.get("disallowedSourceRows")),
            "blockedRows": _int(counts.get("blockedRows")),
        },
    }


def build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review(
    *,
    lane_split_dry_run: dict[str, Any],
    source_lane_split_dry_run_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    blockers = sorted(set(_lane_split_blockers(lane_split_dry_run)))
    artifact_contracts = _artifact_contracts() if not blockers else []
    disallowed_source_contracts = _disallowed_source_contracts() if not blockers else []
    counts = _review_counts(
        lane_split_dry_run=lane_split_dry_run,
        artifact_contracts=artifact_contracts,
        disallowed_source_contracts=disallowed_source_contracts,
        blockers=blockers,
    )
    gate = _gate(counts, blockers)
    status = "ready" if gate.get("passed") and not blockers else "blocked"
    return {
        "schema": PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID,
        "status": status,
        "generatedAt": generated_at or utc_now_iso(),
        "decision": READY_DECISION if status == "ready" else BLOCKED_DECISION,
        "nextRecommendedTranche": NEXT_TRANCHE_READY if status == "ready" else NEXT_TRANCHE_HOLD,
        "input": {
            "sourceLaneSplitDryRunRef": normalize_text(source_lane_split_dry_run_ref),
            "expectedAllowedArtifactTypes": list(EXPECTED_ALLOWED_ARTIFACT_TYPES),
            "expectedDisallowedSourceTypes": list(DISALLOWED_SOURCE_TYPES),
        },
        "policy": {
            "reviewOnly": True,
            "contractOnly": True,
            "runtimeRouteWrite": False,
            "runtimeConfigMutation": False,
            "operationalSearchIndexQuery": False,
            "answerVisibleExposure": False,
            "answerGeneration": False,
            "strictEvidenceCreation": False,
            "citationGradePromotion": False,
            "runtimeEvidenceCreation": False,
            "sourceSpanCreation": False,
            "sourceSpanCandidateCreation": False,
            "parsedArtifactEvidenceChunkCreation": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "parserExecution": False,
            "canonicalParsedArtifactWrite": False,
            "vaultScan": False,
            "externalDownload": False,
            "branchDeletion": False,
            "githubMutation": False,
        },
        "sourceLaneSplitDryRun": _source_summary(lane_split_dry_run, report_ref=source_lane_split_dry_run_ref),
        "method": {
            "name": "paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_v1",
            "description": (
                "Fixes minimum answer-evidence and answerability contracts for parsed-artifact "
                "evidence chunks while keeping visual retrieval hints candidate-discovery-only."
            ),
            "completionBoundary": "contract_review_report_only",
            "nextEvidenceWork": NEXT_TRANCHE_READY,
        },
        "contract": {
            "contractName": "parsed_artifact_evidence_chunk_contract_v1",
            "contractVersion": "v1",
            "answerEvidenceLane": "parsed_artifact_evidence_chunk",
            "candidateDiscoveryLane": "visual_retrieval_hint_candidate_discovery",
            "minimumRequiredFields": [
                "paperId",
                "sourceType=paper",
                "artifactType",
                "sourceRef",
                "sourceContentHash",
                "locator",
                "excerpt",
                "snippetHash",
                "evidenceKind",
                "answerEvidenceEligible",
                "answerabilityEligible",
            ],
            "requiredCommonFields": [
                "paperId",
                "sourceType=paper",
                "artifactType",
                "sourceRef",
                "sourceContentHash",
                "locator",
                "excerpt",
                "snippetHash",
                "evidenceKind",
                "answerEvidenceEligible",
                "answerabilityEligible",
            ],
            "allowedArtifactTypes": list(EXPECTED_ALLOWED_ARTIFACT_TYPES),
            "disallowedEvidenceSources": list(DISALLOWED_SOURCE_TYPES),
            "promotionPrinciples": [
                "candidate_chunks_are_not_strict_evidence_by_default",
                "sourceContentHash_and_snippetHash_are_separate_fields",
                "visual_hints_can_expand_candidates_but_cannot_satisfy_answerability",
                "locator_only_or_fallback_sources_fail_closed",
                "answerability_requires_a_later_runtime_gate_after_candidate_generation",
            ],
            "promotionRules": [
                "only_parsed_artifact_evidence_chunk_lane_may_satisfy_answerability",
                "sourceContentHash_and_snippetHash_required_before_answerability",
                "chars_or_page_bbox_locator_required_before_answerability",
                "verbatim_excerpt_required_before_answerability",
                "runtime_visibility_requires_later_separate_gate",
            ],
            "failureClassifications": [
                "missing_source_content_hash",
                "missing_locator",
                "missing_verbatim_excerpt",
                "fallback_or_locator_only_source",
                "candidate_discovery_only_visual_hint",
            ],
        },
        "allowedArtifactContracts": artifact_contracts,
        "disallowedSourceContracts": disallowed_source_contracts,
        "counts": counts,
        "gate": gate,
        "technicalBlockers": blockers,
        "warnings": [],
    }


def render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_markdown(
    report: dict[str, Any],
) -> str:
    counts = dict(report.get("counts") or {})
    gate = dict(report.get("gate") or {})
    lines = [
        "# Paper Retrieval Lane Split Parsed Artifact Evidence Chunk Contract Review",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- status: `{report.get('status')}`",
        f"- decision: `{report.get('decision')}`",
        f"- nextRecommendedTranche: `{report.get('nextRecommendedTranche')}`",
        f"- contractRows: `{counts.get('contractRows')}`",
        f"- allowedArtifactTypeRows: `{counts.get('allowedArtifactTypeRows')}`",
        f"- disallowedSourceRows: `{counts.get('disallowedSourceRows')}`",
        f"- answerEvidenceEligibleContractRows: `{counts.get('answerEvidenceEligibleContractRows')}`",
        f"- answerabilityEligibleContractRows: `{counts.get('answerabilityEligibleContractRows')}`",
        f"- visualHintAnswerEvidenceAllowedRows: `{counts.get('visualHintAnswerEvidenceAllowedRows')}`",
        f"- fallbackChunkAnswerEvidenceAllowedRows: `{counts.get('fallbackChunkAnswerEvidenceAllowedRows')}`",
        f"- locatorOnlyAnswerEvidenceAllowedRows: `{counts.get('locatorOnlyAnswerEvidenceAllowedRows')}`",
        f"- blockedRows: `{counts.get('blockedRows')}`",
        f"- privatePathLeakRows: `{counts.get('privatePathLeakRows')}`",
        f"- schemaViolationCount: `{counts.get('schemaViolationCount')}`",
        f"- runtimeRouteWriteRows: `{counts.get('runtimeRouteWriteRows')}`",
        f"- answerVisibleRows: `{counts.get('answerVisibleRows')}`",
        f"- strictEvidenceRows: `{counts.get('strictEvidenceRows')}`",
        f"- citationGradeRows: `{counts.get('citationGradeRows')}`",
        f"- parsedArtifactEvidenceChunkCreatedRows: `{counts.get('parsedArtifactEvidenceChunkCreatedRows')}`",
        f"- databaseMutationRows: `{counts.get('databaseMutationRows')}`",
        f"- indexMutationRows: `{counts.get('indexMutationRows')}`",
        "",
        "## Gate",
        "",
        f"- passed: `{gate.get('passed')}`",
        f"- allArtifactTypesCovered: `{dict(gate.get('checks') or {}).get('allArtifactTypesCovered')}`",
        f"- visualHintCannotBeAnswerEvidence: `{dict(gate.get('checks') or {}).get('visualHintCannotBeAnswerEvidence')}`",
        f"- noMutationOrRuntimeExposure: `{dict(gate.get('checks') or {}).get('noMutationOrRuntimeExposure')}`",
        "",
        "## Allowed Artifact Contracts",
        "",
    ]
    for row in list(report.get("allowedArtifactContracts") or []):
        lines.append(
            f"- `{row.get('artifactType')}`: `{row.get('locatorContract')}`, "
            f"structuredReadback=`{row.get('requiresStructuredArtifactReadback')}`"
        )
    lines.extend(["", "## Disallowed Sources", ""])
    for row in list(report.get("disallowedSourceContracts") or []):
        lines.append(f"- `{row.get('sourceKind')}`: `{row.get('reason')}`")
    lines.extend(
        [
            "",
            "## Non-Scope",
            "",
            "- No parsed-artifact evidence chunk creation.",
            "- No SourceSpan, StrictEvidence, citation-grade, or runtime evidence creation.",
            "- No answer-visible exposure or answer generation.",
            "- No runtime route write, operational search query, DB/index mutation, parser execution, reindex, vault scan, or external download.",
        ]
    )
    blockers = list(report.get("technicalBlockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def write_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review(
    report: dict[str, Any],
    *,
    report_json: Path,
    report_md: Path,
) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(report_json), "md": str(report_md)}


__all__ = [
    "PAPER_RETRIEVAL_LANE_SPLIT_PARSED_ARTIFACT_EVIDENCE_CHUNK_CONTRACT_REVIEW_SCHEMA_ID",
    "READY_DECISION",
    "build_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review",
    "load_json",
    "render_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review_markdown",
    "sanitized_report_ref",
    "write_paper_retrieval_lane_split_parsed_artifact_evidence_chunk_contract_review",
]
