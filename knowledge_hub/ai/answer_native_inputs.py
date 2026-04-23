from __future__ import annotations

import json
from typing import Any

from knowledge_hub.ai.claim_adjudication import adjudicate_claims, render_claim_adjudication_context
from knowledge_hub.ai.rag_support import extract_json_payload


class AnswerNativeInputBuilder:
    def __init__(self, searcher: Any):
        self.searcher = searcher

    def adjudicate_claims(self, *, evidence_packet: Any) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        claim_verification, claim_consensus = adjudicate_claims(
            getattr(self.searcher, "sqlite_db", None),
            claims=list(getattr(evidence_packet, "claims", []) or []),
            evidence=list(getattr(evidence_packet, "evidence", []) or []),
        )
        claim_context = render_claim_adjudication_context(claim_verification, claim_consensus)
        return claim_verification, claim_consensus, claim_context

    @staticmethod
    def claim_consensus_from_verification(claim_verification: list[dict[str, Any]]) -> dict[str, Any]:
        support_count = 0
        weak_count = 0
        unsupported_count = 0
        conflict_count = 0
        for item in claim_verification:
            status = str(item.get("status") or "").strip().lower()
            if status == "supported":
                support_count += 1
            elif status in {"weakly_supported", "context_ambiguous", "version_mismatch", "metric_mismatch"}:
                weak_count += 1
            elif status in {"contradicted", "direction_conflict"}:
                conflict_count += 1
            else:
                unsupported_count += 1
        return {
            "supportCount": support_count,
            "conflictCount": conflict_count,
            "weakClaimCount": weak_count,
            "unsupportedClaimCount": unsupported_count,
            "claimVerificationSummary": "conflicted" if conflict_count else ("weak" if weak_count or unsupported_count else "supported"),
            "conflicts": [],
        }

    @staticmethod
    def v2_claim_bundle(pipeline_result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        claim_cards = list(diagnostics.get("claimCards") or [])
        claim_alignment = list((diagnostics.get("claimAlignment") or {}).get("groups") or [])
        claim_verification = list(diagnostics.get("claimVerification") or [])
        answer_provenance = dict(diagnostics.get("answerProvenance") or {})
        return claim_cards, claim_alignment, claim_verification, answer_provenance

    @staticmethod
    def v2_section_bundle(pipeline_result: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        section_cards = list(diagnostics.get("sectionCards") or [])
        section_selection = dict(diagnostics.get("sectionSelection") or {})
        section_coverage = dict(diagnostics.get("sectionCoverage") or {})
        routing = dict(diagnostics.get("routing") or {})
        return section_cards, section_selection, section_coverage, routing

    def structured_verify_selected_claims(
        self,
        *,
        llm: Any,
        claim_cards: list[dict[str, Any]],
        claim_verification: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        by_id = {
            str(item.get("claimCardId") or "").strip(): dict(item)
            for item in claim_verification
            if str(item.get("claimCardId") or "").strip()
        }
        extracted = [
            item for item in claim_cards
            if str(item.get("origin") or "").strip() == "extracted"
        ][:6]
        if not extracted or llm is None:
            return claim_verification, []
        claims_payload = []
        for item in extracted:
            claims_payload.append(
                {
                    "claimCardId": item.get("claimCardId"),
                    "claimText": item.get("summaryText") or item.get("claimText"),
                    "task": item.get("taskCanonical") or item.get("task"),
                    "dataset": item.get("datasetCanonical") or item.get("dataset"),
                    "metric": item.get("metricCanonical") or item.get("metric"),
                    "comparator": item.get("comparatorCanonical") or item.get("comparator"),
                    "resultValueText": item.get("resultValueText") or "",
                    "conditionText": item.get("conditionText") or "",
                    "scopeText": item.get("scopeText") or "",
                    "anchors": list(item.get("anchorExcerpts") or [])[:3],
                }
            )
        prompt = (
            "Verify the selected claims against their anchor excerpts.\n"
            "Return only valid JSON in the form "
            "{\"claims\": [{\"claimCardId\": str, \"claim_supported\": bool, \"evidence_covers_claim\": \"full|partial|none\", "
            "\"scope_match\": \"true|false|unknown\", \"numeric_match\": \"true|false|not_applicable\", "
            "\"condition_match\": \"true|false|unknown\", \"issues\": [str], \"confidence\": float}]}\n"
        )
        context = json.dumps({"claims": claims_payload}, ensure_ascii=False, indent=2)
        try:
            raw = llm.generate(prompt, context)
            parsed = extract_json_payload(raw)
        except Exception:
            parsed = {}
        scope_warnings: list[str] = []
        for result in list(parsed.get("claims") or []):
            claim_card_id = str(result.get("claimCardId") or "").strip()
            existing = by_id.get(claim_card_id)
            if not existing:
                continue
            issues = [str(item).strip() for item in list(result.get("issues") or []) if str(item).strip()]
            supported = bool(result.get("claim_supported"))
            coverage = str(result.get("evidence_covers_claim") or "").strip().lower()
            scope_match = str(result.get("scope_match") or "").strip().lower()
            condition_match = str(result.get("condition_match") or "").strip().lower()
            if not supported or coverage == "none":
                existing["status"] = "unsupported"
            elif scope_match == "false":
                existing["status"] = "weakly_supported"
                scope_warnings.append(f"{claim_card_id}: scope mismatch")
            elif condition_match == "false":
                existing["status"] = "weakly_supported"
                scope_warnings.append(f"{claim_card_id}: condition mismatch")
            existing["stage2"] = {
                "claimSupported": supported,
                "evidenceCoversClaim": coverage or "none",
                "scopeMatch": scope_match or "unknown",
                "numericMatch": str(result.get("numeric_match") or "").strip().lower() or "not_applicable",
                "conditionMatch": condition_match or "unknown",
                "issues": issues,
                "confidence": result.get("confidence"),
            }
            if issues:
                existing["reasons"] = list(dict.fromkeys([*(existing.get("reasons") or []), *issues]))
        ordered = []
        for item in claim_verification:
            key = str(item.get("claimCardId") or "").strip()
            ordered.append(by_id.get(key, item))
        return ordered, list(dict.fromkeys(scope_warnings))

    @staticmethod
    def comparison_verification(*, claim_alignment: list[dict[str, Any]]) -> dict[str, Any]:
        disagreements = []
        scope_differences = []
        temporal_ordering = []
        for group in claim_alignment:
            if int(group.get("conflictingClaimCount") or 0) > 0:
                disagreements.append(
                    {
                        "groupKey": str(group.get("groupKey") or ""),
                        "claimCardIds": list(group.get("claimCardIds") or []),
                    }
                )
            if str(group.get("conditionText") or "").strip():
                scope_differences.append(
                    {
                        "groupKey": str(group.get("groupKey") or ""),
                        "conditionText": str(group.get("conditionText") or ""),
                    }
                )
            frame = dict(group.get("canonicalFrame") or {})
            if str(frame.get("datasetVersion") or "").strip():
                temporal_ordering.append(
                    {
                        "groupKey": str(group.get("groupKey") or ""),
                        "datasetVersion": str(frame.get("datasetVersion") or ""),
                    }
                )
        return {
            "agreements": max(0, len(claim_alignment) - len(disagreements)),
            "disagreements": disagreements,
            "scopeDifferences": scope_differences,
            "temporalOrdering": temporal_ordering,
        }

    def claim_native_inputs(
        self,
        *,
        query: str,
        pipeline_result: Any,
        evidence_packet: Any,
        llm: Any,
    ) -> tuple[str, str, list[dict[str, Any]], dict[str, Any], list[str]] | None:
        claim_cards, claim_alignment, claim_verification, answer_provenance = self.v2_claim_bundle(pipeline_result)
        if not claim_cards:
            return None
        enriched_verification, scope_warnings = self.structured_verify_selected_claims(
            llm=llm,
            claim_cards=claim_cards,
            claim_verification=claim_verification,
        )
        comparison_verification = self.comparison_verification(claim_alignment=claim_alignment)
        consensus = self.claim_consensus_from_verification(enriched_verification)
        alignment_conflicts = [
            {
                "groupKey": str(group.get("groupKey") or ""),
                "claimCardIds": list(group.get("claimCardIds") or []),
                "reason": "aligned_claim_conflict",
            }
            for group in claim_alignment
            if int(group.get("conflictingClaimCount") or 0) > 0
        ]
        if alignment_conflicts:
            consensus["conflictCount"] = int(consensus.get("conflictCount") or 0) + len(alignment_conflicts)
            consensus["claimVerificationSummary"] = "conflicted"
            merged_conflicts: list[dict[str, Any]] = []
            seen_conflicts: set[tuple[str, tuple[str, ...], str]] = set()
            for item in [*(consensus.get("conflicts") or []), *alignment_conflicts]:
                payload = dict(item or {})
                token = (
                    str(payload.get("groupKey") or payload.get("reason") or ""),
                    tuple(str(value) for value in list(payload.get("claimCardIds") or payload.get("claimIds") or [])),
                    str(payload.get("reason") or ""),
                )
                if token in seen_conflicts:
                    continue
                seen_conflicts.add(token)
                merged_conflicts.append(payload)
            consensus["conflicts"] = merged_conflicts
        weak_only = all(str(item.get("trustLevel") or "").strip().lower() == "low" for item in claim_cards)
        usable_claims = [
            item
            for item in claim_cards
            if str(item.get("trustLevel") or "").strip().lower() != "low"
            and str(item.get("status") or "").strip().lower() not in {"unsupported", "direction_conflict", "numeric_mismatch"}
        ]
        supplemental_context = evidence_packet.context if len(usable_claims) < 3 or weak_only else ""
        abstention_conditions = list((evidence_packet.evidence_packet or {}).get("insufficientEvidenceReasons") or [])
        prompt = self.searcher._build_claim_native_prompt(
            query=query,
            answer_provenance=str(answer_provenance.get("mode") or "claim_cards_verified"),
        )
        context = self.searcher._build_claim_native_context(
            claim_cards=claim_cards,
            claim_alignment=claim_alignment,
            claim_verification=enriched_verification,
            comparison_verification=comparison_verification,
            scope_warnings=scope_warnings,
            abstention_conditions=abstention_conditions,
            supplemental_context=supplemental_context,
        )
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        diagnostics["claimVerification"] = enriched_verification
        diagnostics["consensus"] = consensus
        diagnostics["comparisonVerification"] = comparison_verification
        diagnostics["scopeWarnings"] = scope_warnings
        diagnostics["answerProvenance"] = {
            "mode": (
                "claim_cards_conflicted"
                if consensus.get("conflictCount")
                else "weak_claim_fallback"
                if weak_only or consensus.get("unsupportedClaimCount") or consensus.get("weakClaimCount")
                else "claim_cards_verified"
            )
        }
        pipeline_result.v2_diagnostics = diagnostics
        return prompt, context, enriched_verification, consensus, scope_warnings

    def section_native_inputs(
        self,
        *,
        query: str,
        pipeline_result: Any,
        evidence_packet: Any,
    ) -> tuple[str, str, dict[str, Any]] | None:
        section_cards, _section_selection, section_coverage, routing = self.v2_section_bundle(pipeline_result)
        if not section_cards:
            return None
        if str(routing.get("intent") or "").strip() in {"comparison", "evaluation"}:
            return None
        if str(routing.get("memoryForm") or "").strip() != "section_cards":
            return None
        supplemental_context = evidence_packet.context if str(section_coverage.get("status") or "").strip() != "strong" else ""
        prompt = self.searcher._build_section_native_prompt(
            query=query,
            answer_provenance="section_cards_verified",
        )
        context = self.searcher._build_section_native_context(
            section_cards=section_cards,
            section_coverage=section_coverage,
            supplemental_context=supplemental_context,
        )
        diagnostics = dict(getattr(pipeline_result, "v2_diagnostics", {}) or {})
        diagnostics["answerProvenance"] = {
            "mode": "section_cards_verified" if str(section_coverage.get("status") or "").strip() == "strong" else "section_cards_weak_fallback"
        }
        diagnostics["scopeWarnings"] = [f"missing_section:{item}" for item in list(section_coverage.get("missingRoles") or []) if str(item).strip()]
        pipeline_result.v2_diagnostics = diagnostics
        return prompt, context, section_coverage


__all__ = ["AnswerNativeInputBuilder"]
