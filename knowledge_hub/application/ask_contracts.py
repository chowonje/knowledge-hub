"""Shared public ask payload contract helpers."""

from __future__ import annotations

from typing import Any

from knowledge_hub.ai.memory_prefilter import (
    memory_route_payload,
    normalize_memory_route_mode_details,
)
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.papers.prefilter import normalize_paper_memory_mode_details


def external_policy_contract(
    *,
    surface: str,
    allow_external: bool,
    requested: bool | None,
    decision_source: str,
) -> dict[str, Any]:
    policy_mode = "external-allowed" if allow_external else "local-only"
    return {
        "contractRole": "answer_generation_external_policy",
        "surface": str(surface or "").strip() or "unknown",
        "scope": "answer_generation",
        "allowExternal": bool(allow_external),
        "allowExternalRequested": None if requested is None else bool(requested),
        "decisionSource": str(decision_source or "").strip() or "unknown",
        "policyMode": policy_mode,
        "mode": policy_mode,
    }


def _default_memory_prefilter_payload(
    *,
    requested_mode: str,
    source_type: str | None,
    paper_memory_mode: Any = None,
) -> dict[str, Any]:
    requested, effective, alias_applied = normalize_memory_route_mode_details(
        requested_mode,
        paper_memory_mode=paper_memory_mode,
    )
    return {
        "contractRole": "retrieval_memory_prefilter",
        "requestedMode": requested,
        "effectiveMode": effective,
        "modeAliasApplied": alias_applied,
        "aliasDeprecated": bool(alias_applied and requested == "prefilter"),
        "sourceType": str(normalize_source_type(source_type) or "all"),
        "applied": False,
        "fallbackUsed": False,
        "matchedMemoryIds": [],
        "matchedDocumentIds": [],
        "formsTried": [],
        "reason": "not_reported",
        "memoryInfluenceApplied": False,
        "verificationCouplingApplied": False,
        "fallbackReason": "",
    }


def _default_paper_memory_prefilter_payload(
    *,
    paper_memory_mode: Any,
    source_type: str | None,
) -> dict[str, Any]:
    requested, effective, alias_applied = normalize_paper_memory_mode_details(paper_memory_mode)
    normalized_source = str(normalize_source_type(source_type) or "").strip()
    enabled = effective in {"compat", "on"}
    if not enabled:
        reason = "disabled"
    elif normalized_source == "paper":
        reason = "not_reported"
    else:
        reason = "source_not_paper"
    return {
        "contractRole": "paper_source_memory_prefilter",
        "requestedMode": requested,
        "effectiveMode": effective,
        "modeAliasApplied": alias_applied,
        "aliasDeprecated": bool(alias_applied and requested == "prefilter"),
        "sourceType": normalized_source or "all",
        "applied": False,
        "fallbackUsed": False,
        "matchedPaperIds": [],
        "matchedMemoryIds": [],
        "reason": reason,
        "memoryInfluenceApplied": False,
        "verificationCouplingApplied": False,
        "fallbackReason": "",
    }


def _merge_contract_defaults(defaults: dict[str, Any], current: Any) -> dict[str, Any]:
    payload = dict(defaults)
    if isinstance(current, dict):
        payload.update(current)
    if payload.get("modeAliasApplied") and payload.get("requestedMode") == "prefilter":
        payload["aliasDeprecated"] = True
    payload.setdefault("contractRole", defaults.get("contractRole", ""))
    return payload


def ensure_ask_contract_payload(
    payload: dict[str, Any] | None,
    *,
    source_type: str | None,
    memory_route_mode: str,
    paper_memory_mode: Any,
    external_policy: dict[str, Any],
) -> dict[str, Any]:
    data = dict(payload or {})

    route_defaults = memory_route_payload(
        requested_mode=memory_route_mode,
        source_type=source_type,
        paper_memory_mode=paper_memory_mode,
    )
    route_defaults.update(
        {
            "applied": False,
            "matchedForms": [],
            "fallbackReason": "",
        }
    )
    data["memoryRoute"] = _merge_contract_defaults(route_defaults, data.get("memoryRoute"))
    if not data["memoryRoute"].get("matchedForms"):
        data["memoryRoute"]["matchedForms"] = list(data["memoryRoute"].get("formsTried") or [])

    data["memoryPrefilter"] = _merge_contract_defaults(
        _default_memory_prefilter_payload(
            requested_mode=memory_route_mode,
            source_type=source_type,
            paper_memory_mode=paper_memory_mode,
        ),
        data.get("memoryPrefilter"),
    )
    data["paperMemoryPrefilter"] = _merge_contract_defaults(
        _default_paper_memory_prefilter_payload(
            paper_memory_mode=paper_memory_mode,
            source_type=source_type,
        ),
        data.get("paperMemoryPrefilter"),
    )
    data["allowExternal"] = bool(external_policy.get("allowExternal"))
    data["externalPolicy"] = dict(external_policy)
    return data


__all__ = [
    "ensure_ask_contract_payload",
    "external_policy_contract",
]
