from __future__ import annotations

from typing import Any

from knowledge_hub.application.query_frame import normalize_query_frame_dict
from knowledge_hub.domain.web_knowledge.families import (
    WEB_FAMILY_REFERENCE_EXPLAINER,
    WEB_FAMILY_RELATION_EXPLAINER,
    WEB_FAMILY_SOURCE_DISAMBIGUATION,
    WEB_FAMILY_TEMPORAL_UPDATE,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


_FAMILY_POLICIES: dict[str, dict[str, Any]] = {
    WEB_FAMILY_REFERENCE_EXPLAINER: {
        "policyKey": "web_reference_explainer_policy",
        "family": WEB_FAMILY_REFERENCE_EXPLAINER,
        "selectionMode": "reference_first_raw_verify",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": False,
        "preferReferenceSource": True,
        "preferTemporalGrounding": False,
    },
    WEB_FAMILY_TEMPORAL_UPDATE: {
        "policyKey": "web_temporal_update_policy",
        "family": WEB_FAMILY_TEMPORAL_UPDATE,
        "selectionMode": "temporal_grounded_guarded",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": False,
        "preferReferenceSource": False,
        "preferTemporalGrounding": True,
    },
    WEB_FAMILY_RELATION_EXPLAINER: {
        "policyKey": "web_relation_explainer_policy",
        "family": WEB_FAMILY_RELATION_EXPLAINER,
        "selectionMode": "claim_relation_supported",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": False,
        "preferReferenceSource": True,
        "preferTemporalGrounding": False,
    },
    WEB_FAMILY_SOURCE_DISAMBIGUATION: {
        "policyKey": "web_source_disambiguation_policy",
        "family": WEB_FAMILY_SOURCE_DISAMBIGUATION,
        "selectionMode": "source_class_compare_guarded",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": True,
        "preferReferenceSource": False,
        "preferTemporalGrounding": True,
    },
}


def policy_key_for_family(family: str) -> str:
    normalized = _clean_text(family).lower()
    return str(dict(_FAMILY_POLICIES.get(normalized) or {}).get("policyKey") or "")


def policy_for_family(family: str) -> dict[str, Any]:
    normalized = _clean_text(family).lower()
    payload = dict(_FAMILY_POLICIES.get(normalized) or {})
    if payload:
        return payload
    return {
        "policyKey": "",
        "family": normalized,
        "selectionMode": "default",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": False,
    }


def select_evidence_policy(frame: Any) -> dict[str, Any]:
    normalized = normalize_query_frame_dict(frame)
    family = _clean_text(normalized.get("family")).lower()
    selected = policy_for_family(family)
    frame_key = _clean_text(normalized.get("evidence_policy_key"))
    if frame_key and frame_key == str(selected.get("policyKey") or ""):
        return selected
    if frame_key and not str(selected.get("policyKey") or ""):
        selected["policyKey"] = frame_key
    return selected


__all__ = [
    "policy_for_family",
    "policy_key_for_family",
    "select_evidence_policy",
]
