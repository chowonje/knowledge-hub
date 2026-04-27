from __future__ import annotations

import re
from typing import Any

from knowledge_hub.ai.ask_v2_support import classify_project_query_profile, clean_text, slot_coverage


def _anchor_has_document_temporal_marker(anchor: dict[str, Any]) -> bool:
    if clean_text(anchor.get("document_date") or anchor.get("event_date") or anchor.get("published_at") or anchor.get("evidence_window")):
        return True
    identity_text = " ".join(
        clean_text(anchor.get(name))
        for name in ("section_path", "source_ref", "source_url", "canonical_url", "citation_target", "title")
    ).casefold()
    excerpt_text = clean_text(anchor.get("excerpt")).casefold()
    text = f"{identity_text} {excerpt_text}"
    return bool(
        re.search(r"\b(version\s*\d+|v\d+|updated?|latest|newest|release)\b", identity_text, re.IGNORECASE)
        or re.search(r"\b(version\s*\d+(?:\.\d+)?|v\d+|20\d{2})\b", excerpt_text, re.IGNORECASE)
        or re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)
        or re.search(r"버전|업데이트|최신|최근|개정", text)
    )


class AskV2Verifier:
    def __init__(self, sqlite_db: Any):
        self.sqlite_db = sqlite_db

    def claim_normalizations(self, selected_claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        claim_ids = [clean_text(item.get("claim_id")) for item in selected_claims if clean_text(item.get("claim_id"))]
        if not claim_ids:
            return {}
        rows = self.sqlite_db.list_claim_normalizations(
            claim_ids=claim_ids,
            status="normalized",
            limit=max(10, len(claim_ids) * 2),
        )
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            claim_id = clean_text(row.get("claim_id"))
            if claim_id and claim_id not in result:
                result[claim_id] = dict(row)
        return result

    def claim_verification(
        self,
        *,
        selected_claims: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        normalizations = self.claim_normalizations(selected_claims)
        anchors_by_claim: dict[str, list[dict[str, Any]]] = {}
        for anchor in anchors:
            claim_id = clean_text(anchor.get("claim_id"))
            if claim_id:
                anchors_by_claim.setdefault(claim_id, []).append(dict(anchor))
        verification_items: list[dict[str, Any]] = []
        support_count = 0
        conflict_count = 0
        weak_count = 0
        unsupported_count = 0
        grouped_claims: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}

        for claim in selected_claims:
            claim_id = clean_text(claim.get("claim_id"))
            role = clean_text(claim.get("role") or claim.get("claim_type"))
            normalization = normalizations.get(claim_id, {})
            evidence_items = anchors_by_claim.get(claim_id) or []
            evidence_text = " ".join(clean_text(item.get("excerpt")) for item in evidence_items).casefold()
            numeric_value = normalization.get("result_value_numeric")
            numeric_token = ""
            if numeric_value is not None:
                try:
                    numeric_float = float(numeric_value)
                    numeric_token = str(int(numeric_float)) if numeric_float.is_integer() else str(numeric_float)
                except Exception:
                    numeric_token = clean_text(numeric_value)
            expected_terms = [
                clean_text(normalization.get("task")),
                clean_text(normalization.get("dataset")),
                clean_text(normalization.get("metric")),
                clean_text(normalization.get("comparator")),
                clean_text(normalization.get("result_value_text")),
                clean_text(normalization.get("condition_text")),
                clean_text(normalization.get("scope_text")),
                clean_text(normalization.get("limitation_text")),
            ]
            if numeric_token:
                expected_terms.append(numeric_token)
            expected_terms = [term for term in expected_terms if term]
            matched_terms = [term for term in expected_terms if term.casefold() in evidence_text]
            evidence_strength = clean_text(normalization.get("evidence_strength") or "weak")
            result_direction = clean_text(normalization.get("result_direction") or "unknown")
            verdict = "supported"
            reasons: list[str] = []
            evidence_numbers = {token.rstrip("%") for token in re.findall(r"\d+(?:\.\d+)?%?", evidence_text)}
            numeric_conflict = bool(numeric_token and evidence_numbers and numeric_token.rstrip("%") not in evidence_numbers)
            dataset_token = clean_text(normalization.get("dataset"))
            dataset_version = clean_text(claim.get("dataset_version"))
            metric_token = clean_text(normalization.get("metric"))
            version_mismatch = bool(dataset_version and dataset_token and dataset_token.casefold() in evidence_text and dataset_version.casefold() not in evidence_text)
            metric_mismatch = bool(metric_token and metric_token.casefold() not in evidence_text and any(token in evidence_text for token in ("accuracy", "f1", "bleu", "exact match", "precision", "recall")))
            if not evidence_items:
                verdict = "unsupported"
                reasons.append("no_anchor_backed_evidence")
            elif version_mismatch:
                verdict = "version_mismatch"
                reasons.append("dataset_version_mismatch")
            elif metric_mismatch:
                verdict = "metric_mismatch"
                reasons.append("metric_name_missing")
            elif numeric_conflict:
                verdict = "numeric_mismatch"
                reasons.append("numeric_value_conflict")
            elif evidence_strength == "weak":
                verdict = "context_ambiguous"
                reasons.append("weak_evidence_strength")
            elif expected_terms and len(matched_terms) < (
                1 if (numeric_token and numeric_token in matched_terms) or role in {"result", "metric"} else max(1, min(2, len(expected_terms)))
            ):
                verdict = "context_ambiguous"
                reasons.append("expected_terms_under_grounded")
            if result_direction == "better" and any(token in evidence_text for token in ("worse", "lower", "decrease", "감소", "악화")):
                verdict = "direction_conflict"
                reasons.append("direction_conflict")
            if result_direction == "worse" and any(token in evidence_text for token in ("improves", "better", "higher", "향상", "개선")):
                verdict = "direction_conflict"
                reasons.append("direction_conflict")
            final_status = "supported"
            if verdict in {"direction_conflict", "numeric_mismatch"}:
                final_status = "contradicted"
            elif verdict in {"context_ambiguous", "version_mismatch", "metric_mismatch"}:
                final_status = "weakly_supported"
            elif verdict == "unsupported":
                final_status = "unsupported"
            if verdict == "supported":
                support_count += 1
            elif verdict in {"context_ambiguous", "version_mismatch", "metric_mismatch"}:
                weak_count += 1
            elif verdict in {"direction_conflict"}:
                conflict_count += 1
            else:
                unsupported_count += 1
            key = (
                clean_text(normalization.get("task")),
                clean_text(normalization.get("dataset")),
                clean_text(normalization.get("metric")),
                clean_text(normalization.get("comparator")),
            )
            if any(key):
                grouped_claims.setdefault(key, []).append({"claim_id": claim_id, "normalization": normalization})
            verification_items.append(
                {
                    "claimCardId": clean_text(claim.get("claim_card_id")),
                    "claimId": claim_id,
                    "role": role,
                    "claimType": clean_text(claim.get("claim_type")),
                    "sourceKind": clean_text(claim.get("source_kind")),
                    "status": final_status,
                    "verdict": verdict,
                    "stage1Status": verdict,
                    "matchedAnchorCount": len(evidence_items),
                    "matchedTerms": matched_terms,
                    "reasons": reasons,
                    "normalized": {
                        "task": clean_text(normalization.get("task")),
                        "dataset": clean_text(normalization.get("dataset")),
                        "datasetVersion": dataset_version,
                        "metric": clean_text(normalization.get("metric")),
                        "comparator": clean_text(normalization.get("comparator")),
                        "resultDirection": clean_text(normalization.get("result_direction")),
                        "resultValueText": clean_text(normalization.get("result_value_text")),
                        "resultValueNumeric": normalization.get("result_value_numeric"),
                        "conditionText": clean_text(normalization.get("condition_text")),
                        "scopeText": clean_text(normalization.get("scope_text")),
                        "limitationText": clean_text(normalization.get("limitation_text")),
                        "evidenceStrength": evidence_strength,
                    },
                }
            )

        consensus_conflicts: list[dict[str, Any]] = []
        for key, items in grouped_claims.items():
            directions = {
                clean_text(item["normalization"].get("result_direction"))
                for item in items
                if clean_text(item["normalization"].get("result_direction")) not in {"", "unknown"}
            }
            numeric_values = {
                item["normalization"].get("result_value_numeric")
                for item in items
                if item["normalization"].get("result_value_numeric") is not None
            }
            if len(directions) > 1 or len(numeric_values) > 1:
                conflict_count += 1
                consensus_conflicts.append(
                    {
                        "task": key[0],
                        "dataset": key[1],
                        "metric": key[2],
                        "comparator": key[3],
                        "claimIds": [clean_text(item["claim_id"]) for item in items],
                        "reason": "normalized_claim_conflict",
                    }
                )

        summary = {
            "supportCount": support_count,
            "conflictCount": conflict_count,
            "weakClaimCount": weak_count,
            "unsupportedClaimCount": unsupported_count,
            "claimVerificationSummary": "conflicted" if conflict_count else ("weak" if weak_count or unsupported_count else "supported"),
            "conflicts": consensus_conflicts,
        }
        return verification_items, summary

    def comparison_verification(
        self,
        *,
        selected_claims: list[dict[str, Any]],
        alignment_groups: list[dict[str, Any]],
    ) -> dict[str, Any]:
        disagreements: list[dict[str, Any]] = []
        scope_differences: list[dict[str, Any]] = []
        temporal_ordering: list[dict[str, Any]] = []
        cards_by_id = {
            clean_text(item.get("claim_card_id")): dict(item)
            for item in selected_claims
            if clean_text(item.get("claim_card_id"))
        }
        for group in alignment_groups:
            claim_card_ids = [clean_text(item) for item in list(group.get("claimCardIds") or []) if clean_text(item)]
            cards = [cards_by_id[item] for item in claim_card_ids if item in cards_by_id]
            if len(cards) < 2:
                continue
            directions = {
                clean_text(item.get("result_direction")).casefold()
                for item in cards
                if clean_text(item.get("result_direction")) not in {"", "unknown"}
            }
            if len(directions) > 1 or int(group.get("conflictingClaimCount") or 0) > 0:
                disagreements.append(
                    {
                        "groupKey": clean_text(group.get("groupKey")),
                        "claimCardIds": claim_card_ids,
                        "reason": "aligned_direction_or_value_conflict",
                    }
                )
            conditions = [clean_text(item.get("condition_text")) for item in cards if clean_text(item.get("condition_text"))]
            if len(conditions) >= 2 and any(left != right for left in conditions for right in conditions):
                scope_differences.append(
                    {
                        "groupKey": clean_text(group.get("groupKey")),
                        "claimCardIds": claim_card_ids,
                        "conditions": conditions,
                    }
                )
            timestamps = sorted(
                {
                    clean_text(item.get("updated_at") or item.get("source_updated_at_snapshot"))
                    for item in cards
                    if clean_text(item.get("updated_at") or item.get("source_updated_at_snapshot"))
                }
            )
            if len(timestamps) >= 2:
                temporal_ordering.append(
                    {
                        "groupKey": clean_text(group.get("groupKey")),
                        "oldest": timestamps[0],
                        "newest": timestamps[-1],
                    }
                )
        return {
            "agreements": max(0, len(alignment_groups) - len(disagreements)),
            "disagreements": disagreements,
            "scopeDifferences": scope_differences,
            "temporalOrdering": temporal_ordering,
        }

    def verification_summary(
        self,
        *,
        query: str,
        route: Any,
        cards: list[dict[str, Any]],
        anchors: list[dict[str, Any]],
        evidence_packet: Any,
        claim_consensus: dict[str, Any],
    ) -> dict[str, Any]:
        _ = query
        unsupported_fields: list[str] = []
        evidence_text = " ".join(clean_text(anchor.get("excerpt")) for anchor in anchors).casefold()
        weak_slots: list[str] = []
        if route.intent == "temporal":
            has_document_temporal_marker = any(_anchor_has_document_temporal_marker(anchor) for anchor in anchors)
            if route.source_kind == "web":
                observed_only = bool(anchors) and all(
                    clean_text(anchor.get("observed_at")) and not _anchor_has_document_temporal_marker(anchor)
                    for anchor in anchors
                )
                if not has_document_temporal_marker or observed_only:
                    unsupported_fields.append("temporal_version_grounding")
            elif not has_document_temporal_marker and not any(token in evidence_text for token in ("202", "updated", "latest", "recent", "version", "최신", "업데이트")):
                unsupported_fields.append("temporal")
        if route.source_kind == "project":
            weak_slots = sorted(
                {
                    slot
                    for card in cards
                    for slot, status in slot_coverage(card).items()
                    if clean_text(status).casefold() != "complete"
                }
            )
            profile = classify_project_query_profile(query)
            selected_roles = {
                clean_text((card.get("diagnostics") or {}).get("fileRole") or card.get("file_role_core")).casefold()
                for card in cards
                if clean_text((card.get("diagnostics") or {}).get("fileRole") or card.get("file_role_core"))
            }
            if profile.get("architecture") and selected_roles and selected_roles <= {"docs", "test"}:
                unsupported_fields.append("project_structure")
            if weak_slots:
                unsupported_fields.append("weak_project_slots")
        if route.intent == "evaluation" and not any(token in evidence_text for token in ("accuracy", "f1", "auc", "benchmark", "%", "결과", "성능")):
            unsupported_fields.append("metrics")
        flagged_cards = sorted(
            {
                clean_text(item.get("card_id"))
                for item in cards
                if str(item.get("quality_flag") or "").strip().lower() not in {"", "ok"}
            }
        )
        if evidence_packet.filtered_results and not unsupported_fields and not flagged_cards and not claim_consensus.get("conflictCount") and not claim_consensus.get("unsupportedClaimCount"):
            status = "strong"
        elif evidence_packet.filtered_results:
            status = "weak"
        else:
            status = "missing"
        return {
            "anchorIdsUsed": [clean_text(item.get("anchor_id")) for item in anchors if clean_text(item.get("anchor_id"))],
            "unitsChecked": sorted(
                {
                    clean_text(item.get("unit_id") or item.get("document_id") or item.get("note_id"))
                    for item in anchors
                    if clean_text(item.get("unit_id") or item.get("document_id") or item.get("note_id"))
                }
            ),
            "verificationStatus": status,
            "unsupportedFields": sorted(set(unsupported_fields)),
            "weakSlots": weak_slots,
            "anchorRolesUsed": sorted(
                {
                    clean_text(item.get("evidence_role"))
                    for item in anchors
                    if clean_text(item.get("evidence_role"))
                }
            ),
            "flaggedCardIds": flagged_cards,
            "weakClaimCount": int(claim_consensus.get("weakClaimCount") or 0),
            "sourceKind": route.source_kind,
        }


__all__ = ["AskV2Verifier"]
