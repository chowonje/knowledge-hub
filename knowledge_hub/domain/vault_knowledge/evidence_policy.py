from __future__ import annotations

from typing import Any

from knowledge_hub.application.query_frame import NormalizedQueryFrame, normalize_query_frame_dict
from knowledge_hub.domain.vault_knowledge.families import (
    VAULT_FAMILY_NOTE_LOOKUP,
    VAULT_FAMILY_VAULT_COMPARE,
    VAULT_FAMILY_VAULT_EXPLAINER,
    VAULT_FAMILY_VAULT_TIMELINE,
)


_POLICIES: dict[str, dict[str, Any]] = {
    VAULT_FAMILY_NOTE_LOOKUP: {
        "policyKey": "vault_note_lookup_policy",
        "family": VAULT_FAMILY_NOTE_LOOKUP,
        "selectionMode": "note_scoped",
        "singleScopeRequired": True,
        "timelineRequired": False,
    },
    VAULT_FAMILY_VAULT_EXPLAINER: {
        "policyKey": "vault_explainer_policy",
        "family": VAULT_FAMILY_VAULT_EXPLAINER,
        "selectionMode": "note_summary_first",
        "singleScopeRequired": False,
        "timelineRequired": False,
    },
    VAULT_FAMILY_VAULT_COMPARE: {
        "policyKey": "vault_compare_policy",
        "family": VAULT_FAMILY_VAULT_COMPARE,
        "selectionMode": "compare_summary",
        "singleScopeRequired": False,
        "timelineRequired": False,
    },
    VAULT_FAMILY_VAULT_TIMELINE: {
        "policyKey": "vault_timeline_policy",
        "family": VAULT_FAMILY_VAULT_TIMELINE,
        "selectionMode": "timeline_grounded",
        "singleScopeRequired": False,
        "timelineRequired": True,
    },
}


def policy_key_for_family(family: str) -> str:
    return str(_POLICIES.get(str(family or "").strip().lower(), {}).get("policyKey") or "vault_explainer_policy")


def policy_for_family(family: str) -> dict[str, Any]:
    key = str(family or "").strip().lower()
    return dict(_POLICIES.get(key) or _POLICIES[VAULT_FAMILY_VAULT_EXPLAINER])


def select_evidence_policy(frame: NormalizedQueryFrame | dict[str, Any]) -> dict[str, Any]:
    if isinstance(frame, NormalizedQueryFrame):
        family = str(frame.family or "").strip().lower()
    else:
        payload = normalize_query_frame_dict(frame)
        family = str(payload.get("family") or "").strip().lower()
    return policy_for_family(family)


__all__ = [
    "policy_for_family",
    "policy_key_for_family",
    "select_evidence_policy",
]
