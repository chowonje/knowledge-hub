from __future__ import annotations

from typing import Any

from knowledge_hub.ai.answer_contracts import build_evidence_packet_contract
from knowledge_hub.ai.answer_policy_support import (
    apply_claim_consensus_to_verification as _apply_claim_consensus_to_verification,
    evaluate_policy as _evaluate_policy,
    policy_payload as _policy_payload,
    with_claim_context as _with_claim_context,
)
from knowledge_hub.ai.answer_payload_support import (
    family_route_diagnostics as _family_route_diagnostics,
    normalize_enrichment as _normalize_enrichment,
    planner_fallback_payload as _planner_fallback_payload,
    representative_lookup_fallback as _representative_lookup_fallback,
    representative_role as _representative_role,
    retrieval_objects_available as _retrieval_objects_available,
    retrieval_objects_used as _retrieval_objects_used,
)
from knowledge_hub.ai.rag_support import normalize_source_type


class AnswerPayloadBuilder:
    def __init__(self, searcher: Any):
        self.searcher = searcher

    @staticmethod
    def normalize_enrichment(context_expansion: dict[str, Any] | None) -> dict[str, Any]:
        return _normalize_enrichment(context_expansion)

    def base_payload(
        self,
        *,
        query: str,
        retrieval_mode: str,
        pipeline_result: Any,
        evidence_packet: Any,
        answer: str,
        status: str = "ok",
    ) -> dict[str, Any]:
        context_expansion = dict(pipeline_result.context_expansion or {})
        retrieval_plan_payload = pipeline_result.plan.to_dict()
        query_plan = dict(retrieval_plan_payload.get("queryPlan") or {})
        query_frame = dict(retrieval_plan_payload.get("queryFrame") or {})
        source_type = str(
            query_frame.get("source_type")
            or retrieval_plan_payload.get("sourceType")
            or retrieval_plan_payload.get("source_type")
            or ""
        ).strip().lower()
        normalized_source = str(normalize_source_type(source_type) or source_type or "").strip().lower()
        canonical_family = str(query_frame.get("family") or retrieval_plan_payload.get("paperFamily") or "general")
        public_paper_family = canonical_family if normalized_source == "paper" else "general"
        representative_paper = dict((evidence_packet.answer_signals or {}).get("representative_paper") or {})
        evidence_policy = dict(getattr(evidence_packet, "evidence_policy", {}) or {})
        if not representative_paper and bool((evidence_packet.paper_answer_scope or {}).get("applied")):
            scope = dict(evidence_packet.paper_answer_scope or {})
            first_source = dict((evidence_packet.evidence or [{}])[0] or {}) if evidence_packet.evidence else {}
            representative_paper = {
                "paperId": str(scope.get("selectedPaperId") or "").strip(),
                "title": str(first_source.get("title") or "").strip(),
                "citationLabel": str(first_source.get("citation_label") or "").strip(),
                "sourceCount": int((evidence_packet.evidence_packet or {}).get("uniquePaperCount") or 0),
            }
        if not representative_paper:
            representative_paper = self._representative_lookup_fallback(
                query_frame=query_frame,
                query_plan=query_plan,
                paper_answer_scope=dict(evidence_packet.paper_answer_scope or {}),
                evidence=list(evidence_packet.evidence or []),
                source_count=int((evidence_packet.evidence_packet or {}).get("uniquePaperCount") or 0),
            )
        runtime_execution = dict((getattr(pipeline_result, "v2_diagnostics", {}) or {}).get("runtimeExecution") or {})
        retrieval_objects_available = self._retrieval_objects_available(
            source_type=source_type,
            query_frame=query_frame,
            v2_diagnostics=dict(getattr(pipeline_result, "v2_diagnostics", {}) or {}),
        )
        retrieval_objects_used = self._retrieval_objects_used(
            source_type=source_type,
            query_frame=query_frame,
            evidence=list(evidence_packet.evidence or []),
            v2_diagnostics=dict(getattr(pipeline_result, "v2_diagnostics", {}) or {}),
        )
        v2 = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        representative_role = self._representative_role(
            paper_family=canonical_family,
            representative_paper=representative_paper,
        )
        planner_status = str(query_plan.get("plannerStatus") or query_plan.get("planner_status") or "not_attempted")
        planner_fallback = _planner_fallback_payload(query_plan)
        payload = {
            "status": status,
            "answer": answer,
            "query": query,
            "retrievalMode": retrieval_mode,
            "paperFamily": public_paper_family,
            "queryPlan": query_plan,
            "queryFrame": query_frame,
            "representativePaper": representative_paper,
            "representativeRole": representative_role,
            "evidencePolicy": evidence_policy,
            "retrievalObjectsAvailable": retrieval_objects_available,
            "retrievalObjectsUsed": retrieval_objects_used,
            "plannerFallback": planner_fallback,
            "familyRouteDiagnostics": _family_route_diagnostics(
                public_paper_family=public_paper_family,
                query_frame=query_frame,
                query_plan=query_plan,
                retrieval_plan_payload=retrieval_plan_payload,
                runtime_execution=runtime_execution,
                v2_diagnostics=v2,
                normalized_source=normalized_source,
            ),
            "sources": list(evidence_packet.evidence),
            "evidence": list(evidence_packet.evidence),
            "citations": list(evidence_packet.citations),
            "answerSignals": dict(evidence_packet.answer_signals),
            "related_clusters": list(pipeline_result.related_clusters),
            "active_profile": pipeline_result.active_profile,
            "paperMemoryPrefilter": dict(pipeline_result.paper_memory_prefilter),
            "memoryRoute": dict(pipeline_result.memory_route),
            "memoryPrefilter": dict(pipeline_result.memory_prefilter),
            "paperAnswerScope": dict(evidence_packet.paper_answer_scope),
            "evidenceBudget": dict(evidence_packet.evidence_budget),
            "memoryRelationsUsed": list(pipeline_result.memory_prefilter.get("memoryRelationsUsed") or []),
            "temporalSignals": dict(pipeline_result.memory_prefilter.get("temporalSignals") or {}),
            "retrievalPlan": retrieval_plan_payload,
            "candidateSources": list(pipeline_result.candidate_sources),
            "rerankSignals": dict(pipeline_result.rerank_signals),
            "contextExpansion": context_expansion,
            "enrichment": self.normalize_enrichment(context_expansion),
            "evidencePacket": dict(evidence_packet.evidence_packet),
            "evidencePacketContract": build_evidence_packet_contract(
                query=query,
                retrieval_mode=retrieval_mode,
                pipeline_result=pipeline_result,
                evidence_packet=evidence_packet,
            ),
            "contextBudget": dict((evidence_packet.evidence_packet or {}).get("contextBudget") or {}),
            "collapsedRelatedEvidence": list((evidence_packet.evidence_packet or {}).get("collapsedRelatedEvidence") or []),
            "semanticFamilyCount": int((evidence_packet.evidence_packet or {}).get("semanticFamilyCount") or 0),
            "sourceScopeEnforced": bool(pipeline_result.source_scope_enforced),
            "mixedFallbackUsed": bool(pipeline_result.mixed_fallback_used),
            "v2": v2,
            "ontology_entities": self.searcher._resolve_query_entities(query),
            "supporting_beliefs": list(evidence_packet.supporting_beliefs),
            "contradicting_beliefs": list(evidence_packet.contradicting_beliefs),
            "belief_updates_suggested": list(evidence_packet.belief_updates_suggested),
            "claim_count": len(evidence_packet.claims),
        }
        v2 = payload["v2"]
        if v2.get("sectionCards"):
            payload["sectionCards"] = list(v2.get("sectionCards") or [])
        if v2.get("claimCards"):
            payload["claimCards"] = list(v2.get("claimCards") or [])
        if v2.get("claimAlignment"):
            payload["claimAlignment"] = dict(v2.get("claimAlignment") or {})
        if v2.get("answerProvenance"):
            payload["answerProvenance"] = dict(v2.get("answerProvenance") or {})
        if v2.get("scopeWarnings"):
            payload.setdefault("v2", {})["scopeWarnings"] = list(v2.get("scopeWarnings") or [])
        return payload

    @staticmethod
    def _retrieval_objects_available(
        *,
        source_type: str | None,
        query_frame: dict[str, Any],
        v2_diagnostics: dict[str, Any],
    ) -> list[str]:
        return _retrieval_objects_available(
            source_type=source_type,
            query_frame=query_frame,
            v2_diagnostics=v2_diagnostics,
        )

    @staticmethod
    def _retrieval_objects_used(
        *,
        source_type: str | None,
        query_frame: dict[str, Any],
        evidence: list[dict[str, Any]],
        v2_diagnostics: dict[str, Any],
    ) -> list[str]:
        return _retrieval_objects_used(
            source_type=source_type,
            query_frame=query_frame,
            evidence=evidence,
            v2_diagnostics=v2_diagnostics,
        )

    @staticmethod
    def _representative_role(*, paper_family: str, representative_paper: dict[str, Any]) -> str:
        return _representative_role(
            paper_family=paper_family,
            representative_paper=representative_paper,
        )

    @staticmethod
    def _representative_lookup_fallback(
        *,
        query_frame: dict[str, Any],
        query_plan: dict[str, Any],
        paper_answer_scope: dict[str, Any],
        evidence: list[dict[str, Any]],
        source_count: int,
    ) -> dict[str, Any]:
        return _representative_lookup_fallback(
            query_frame=query_frame,
            query_plan=query_plan,
            paper_answer_scope=paper_answer_scope,
            evidence=evidence,
            source_count=source_count,
        )

    @staticmethod
    def with_claim_context(context: str, claim_context: str) -> str:
        return _with_claim_context(context, claim_context)

    @staticmethod
    def apply_claim_consensus_to_verification(
        verification: dict[str, Any],
        claim_consensus: dict[str, Any],
        *,
        force_weak_caution: bool = True,
        merge_mode: str = "strict",
    ) -> dict[str, Any]:
        return _apply_claim_consensus_to_verification(
            verification,
            claim_consensus,
            force_weak_caution=force_weak_caution,
            merge_mode=merge_mode,
        )

    @staticmethod
    def evaluate_policy(
        *,
        context: str,
        allow_external: bool,
        route_mode: str,
    ) -> tuple[str, Any, str]:
        return _evaluate_policy(
            context=context,
            allow_external=allow_external,
            route_mode=route_mode,
        )

    @staticmethod
    def policy_payload(
        *,
        original_classification: str,
        effective_policy: Any,
        safe_context: str,
        original_context: str,
        allow_external: bool,
    ) -> dict[str, Any]:
        return _policy_payload(
            original_classification=original_classification,
            effective_policy=effective_policy,
            safe_context=safe_context,
            original_context=original_context,
            allow_external=allow_external,
        )


__all__ = ["AnswerPayloadBuilder"]
