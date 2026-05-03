from __future__ import annotations

from typing import Any

from knowledge_hub.application.query_frame import normalize_query_frame_dict
from knowledge_hub.domain.ai_papers.families import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


_FAMILY_POLICIES: dict[str, dict[str, Any]] = {
    PAPER_FAMILY_CONCEPT_EXPLAINER: {
        "policyKey": "concept_explainer_policy",
        "family": PAPER_FAMILY_CONCEPT_EXPLAINER,
        "selectionMode": "representative_high_trust",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": False,
        "preferRepresentative": True,
        "preferHighTrust": True,
        "preferHighAuthority": True,
        "allowWeakSingleMatchAnchorOnly": True,
    },
    PAPER_FAMILY_LOOKUP: {
        "policyKey": "paper_lookup_policy",
        "family": PAPER_FAMILY_LOOKUP,
        "selectionMode": "exact_lookup_single_scope",
        "requiresMultipleSources": False,
        "singleScopeRequired": True,
        "shortlistOnly": False,
        "preferRepresentative": False,
        "preferHighTrust": True,
        "preferHighAuthority": True,
        "allowWeakSingleMatchAnchorOnly": False,
    },
    PAPER_FAMILY_COMPARE: {
        "policyKey": "paper_compare_policy",
        "family": PAPER_FAMILY_COMPARE,
        "selectionMode": "claim_aligned_multi_source",
        "requiresMultipleSources": True,
        "singleScopeRequired": False,
        "shortlistOnly": False,
        "preferRepresentative": False,
        "preferHighTrust": True,
        "preferHighAuthority": True,
        "allowWeakSingleMatchAnchorOnly": False,
    },
    PAPER_FAMILY_DISCOVER: {
        "policyKey": "paper_discover_policy",
        "family": PAPER_FAMILY_DISCOVER,
        "selectionMode": "shortlist_summary",
        "requiresMultipleSources": False,
        "singleScopeRequired": False,
        "shortlistOnly": True,
        "preferRepresentative": False,
        "preferHighTrust": True,
        "preferHighAuthority": True,
        "allowWeakSingleMatchAnchorOnly": False,
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


def normalize_evidence_policy(value: Any, *, family: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return policy_for_family(family)
    normalized_family = _clean_text(value.get("family") or family).lower()
    payload = policy_for_family(normalized_family)
    payload.update(
        {
            "policyKey": _clean_text(value.get("policyKey") or value.get("policy_key") or payload.get("policyKey")),
            "family": normalized_family or str(payload.get("family") or ""),
            "selectionMode": _clean_text(value.get("selectionMode") or value.get("selection_mode") or payload.get("selectionMode")),
            "requiresMultipleSources": bool(value.get("requiresMultipleSources", payload.get("requiresMultipleSources"))),
            "singleScopeRequired": bool(value.get("singleScopeRequired", payload.get("singleScopeRequired"))),
            "shortlistOnly": bool(value.get("shortlistOnly", payload.get("shortlistOnly"))),
        }
    )
    return payload


__all__ = [
    "normalize_evidence_policy",
    "policy_for_family",
    "policy_key_for_family",
    "select_evidence_policy",
]
