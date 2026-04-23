from __future__ import annotations

import re
from typing import Any

from knowledge_hub.ai.rag_support import normalize_source_type


def normalize_enrichment(context_expansion: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(context_expansion or {})
    return {
        "eligible": bool(payload.get("eligible")),
        "used": bool(payload.get("used")),
        "mode": str(payload.get("mode") or "none"),
        "reason": str(payload.get("reason") or ""),
        "queryIntent": str(payload.get("queryIntent") or ""),
        "enrichmentRoute": str(payload.get("enrichmentRoute") or ""),
        "ontologyEligible": bool(payload.get("ontologyEligible")),
        "clusterEligible": bool(payload.get("clusterEligible")),
        "ontologyUsed": bool(payload.get("ontologyUsed")),
        "clusterUsed": bool(payload.get("clusterUsed")),
    }


def retrieval_objects_available(
    *,
    source_type: str | None,
    query_frame: dict[str, Any],
    v2_diagnostics: dict[str, Any],
) -> list[str]:
    normalized_source = str(normalize_source_type(source_type or query_frame.get("source_type")) or "").strip().lower()
    original_source = str(query_frame.get("source_type") or source_type or "").strip().lower()
    available: list[str] = ["RawEvidenceUnit"]
    if normalized_source == "paper":
        available.append("DocSummary")
        available.append("SectionCard")
        if str(query_frame.get("family") or "").strip().lower() == "paper_compare":
            available.append("ClaimCard")
    elif normalized_source == "web":
        available.extend(["DocSummary", "SectionCard"])
        if original_source != "youtube" and str(query_frame.get("family") or "").strip().lower() == "relation_explainer":
            available.append("ClaimCard")
    elif normalized_source == "vault":
        available.append("DocSummary")
    if (
        normalized_source not in {"vault"}
        and original_source != "youtube"
        and list(v2_diagnostics.get("claimCards") or [])
        and "ClaimCard" not in available
    ):
        available.append("ClaimCard")
    if list(v2_diagnostics.get("sectionCards") or []) and "SectionCard" not in available:
        available.append("SectionCard")
    return available


def retrieval_objects_used(
    *,
    source_type: str | None,
    query_frame: dict[str, Any],
    evidence: list[dict[str, Any]],
    v2_diagnostics: dict[str, Any],
) -> list[str]:
    family = str(query_frame.get("family") or "").strip().lower()
    normalized_source = str(normalize_source_type(source_type or query_frame.get("source_type")) or "").strip().lower()
    original_source = str(query_frame.get("source_type") or source_type or "").strip().lower()
    used: list[str] = ["RawEvidenceUnit"]
    if family == "paper_compare" and list(v2_diagnostics.get("claimCards") or []):
        used.insert(0, "ClaimCard")
    if family in {"paper_discover", "concept_explainer"} and any(
        str(item.get("unit_type") or "").strip().lower() == "document_summary"
        for item in evidence
    ):
        used.insert(0, "DocSummary")
    if family == "concept_explainer" and list(v2_diagnostics.get("sectionCards") or []):
        used.append("SectionCard")
    if normalized_source == "vault":
        used.insert(0, "DocSummary")
    if normalized_source == "web":
        if original_source == "youtube":
            if family in {"video_lookup", "video_explainer"}:
                used.insert(0, "DocSummary")
            if family in {"section_lookup", "timestamp_lookup"}:
                used.insert(0, "DocSummary")
                if list(v2_diagnostics.get("sectionCards") or []):
                    used.append("SectionCard")
        elif family == "relation_explainer" and list(v2_diagnostics.get("claimCards") or []):
            used.insert(0, "ClaimCard")
        elif family == "source_disambiguation":
            used.insert(0, "DocSummary")
        elif family == "temporal_update":
            used.insert(0, "DocSummary")
            if list(v2_diagnostics.get("sectionCards") or []):
                used.append("SectionCard")
        elif family == "reference_explainer" and list(v2_diagnostics.get("sectionCards") or []):
            used.append("SectionCard")
    return list(dict.fromkeys(used))


def representative_role(*, paper_family: str, representative_paper: dict[str, Any]) -> str:
    normalized_family = str(paper_family or "").strip().lower()
    title = str(representative_paper.get("title") or "").strip().lower()
    if not title:
        return ""
    if re.search(r"\b(survey|overview)\b|서베이|개관|정리", title):
        return "survey"
    if re.search(r"\bbenchmark\b|벤치마크", title):
        return "benchmark"
    if normalized_family != "concept_explainer" and re.search(
        r"\b(using|application|applications)\b|적용",
        title,
    ):
        return "application"
    return "anchor"


def representative_lookup_fallback(
    *,
    query_frame: dict[str, Any],
    query_plan: dict[str, Any],
    paper_answer_scope: dict[str, Any],
    evidence: list[dict[str, Any]],
    source_count: int,
) -> dict[str, Any]:
    family = str(query_frame.get("family") or query_plan.get("family") or "").strip().lower()
    if family != "paper_lookup":
        return {}
    resolved_ids = [
        str(item or "").strip()
        for item in [
            *list(query_frame.get("resolved_source_ids") or []),
            *list(query_plan.get("resolved_paper_ids") or []),
            *list(query_plan.get("resolvedPaperIds") or []),
            *list(paper_answer_scope.get("matchedPaperIds") or []),
        ]
        if str(item or "").strip()
    ]
    if not resolved_ids:
        return {}
    first_id = resolved_ids[0]
    matched_source = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("arxiv_id") or item.get("paper_id") or "").strip()
        if source_id and source_id == first_id:
            matched_source = item
            break
    if not matched_source and evidence:
        matched_source = dict(evidence[0] or {})
    fallback_title = ""
    for term in list(query_frame.get("expanded_terms") or []):
        token = str(term or "").strip()
        if token and (len(token.split()) >= 4 or re.search(r"\d", token)):
            fallback_title = token
            break
    return {
        "paperId": first_id,
        "title": fallback_title or str(matched_source.get("title") or "").strip(),
        "citationLabel": str(matched_source.get("citation_label") or "").strip(),
        "sourceCount": int(source_count or 0),
    }


def planner_fallback_payload(query_plan: dict[str, Any]) -> dict[str, Any]:
    planner_status = str(query_plan.get("plannerStatus") or query_plan.get("planner_status") or "not_attempted")
    return {
        "attempted": planner_status in {"attempted", "used"},
        "used": bool(query_plan.get("plannerUsed") or query_plan.get("planner_used")),
        "status": planner_status,
        "reason": str(query_plan.get("plannerReason") or query_plan.get("planner_reason") or ""),
        "warnings": list(query_plan.get("plannerWarnings") or query_plan.get("planner_warnings") or []),
        "route": dict(query_plan.get("plannerRoute") or query_plan.get("planner_route") or {}),
    }


def family_route_diagnostics(
    *,
    public_paper_family: str,
    query_frame: dict[str, Any],
    query_plan: dict[str, Any],
    retrieval_plan_payload: dict[str, Any],
    runtime_execution: dict[str, Any],
    v2_diagnostics: dict[str, Any],
    normalized_source: str,
) -> dict[str, Any]:
    planner_status = str(query_plan.get("plannerStatus") or query_plan.get("planner_status") or "not_attempted")
    metadata_filter_applied = dict(retrieval_plan_payload.get("metadataFilterApplied") or {})
    original_source = str(query_frame.get("source_type") or "").strip().lower()
    return {
        "paperFamily": public_paper_family,
        "queryIntent": str(query_frame.get("query_intent") or retrieval_plan_payload.get("queryIntent") or ""),
        "answerMode": str(query_plan.get("answerMode") or query_plan.get("answer_mode") or ""),
        "confidence": float(query_plan.get("confidence") or 0.0),
        "frameProvenance": str(query_frame.get("frame_provenance") or ""),
        "frameTrusted": bool(query_frame.get("trusted")),
        "lockMask": list(query_frame.get("lock_mask") or []),
        "familySource": str(query_frame.get("family_source") or ""),
        "overridesApplied": list(query_frame.get("overrides_applied") or []),
        "routeSource": "frame" if str(query_frame.get("family_source") or "") == "from_frame" else "pack",
        "plannerStatus": planner_status,
        "runtimeUsed": str(runtime_execution.get("used") or ""),
        "resolvedSourceScopeApplied": bool(retrieval_plan_payload.get("resolvedSourceScopeApplied")),
        "canonicalEntitiesApplied": list(retrieval_plan_payload.get("canonicalEntitiesApplied") or []),
        "metadataFilterApplied": metadata_filter_applied,
        "prefilterReason": str(retrieval_plan_payload.get("prefilterReason") or ""),
        "temporalSignalsApplied": bool(
            retrieval_plan_payload.get("temporalRouteApplied")
            or dict(retrieval_plan_payload.get("temporalSignals") or {}).get("enabled")
            or metadata_filter_applied.get("temporal_required")
            or metadata_filter_applied.get("timeline_required")
        ),
        "referenceSourceApplied": bool(retrieval_plan_payload.get("referenceSourceApplied")),
        "watchlistScopeApplied": bool(retrieval_plan_payload.get("watchlistScopeApplied")),
        "internalReferenceApplied": bool(dict(v2_diagnostics.get("routing") or {}).get("internalReferenceApplied")),
        "videoScopeApplied": bool(
            normalized_source == "web"
            and original_source == "youtube"
            and (
                retrieval_plan_payload.get("resolvedSourceScopeApplied")
                or metadata_filter_applied.get("canonical_url")
                or metadata_filter_applied.get("video_id")
                or metadata_filter_applied.get("document_id")
            )
        ),
        "chapterScopeApplied": bool(
            original_source == "youtube" and metadata_filter_applied.get("section_preferred")
        ),
        "vaultScopeApplied": bool(
            original_source == "vault"
            and (
                retrieval_plan_payload.get("resolvedSourceScopeApplied")
                or metadata_filter_applied.get("note_id")
                or metadata_filter_applied.get("file_path")
            )
        ),
    }


__all__ = [
    "family_route_diagnostics",
    "normalize_enrichment",
    "planner_fallback_payload",
    "representative_lookup_fallback",
    "representative_role",
    "retrieval_objects_available",
    "retrieval_objects_used",
]
