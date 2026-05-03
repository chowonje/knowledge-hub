from __future__ import annotations

from typing import Any

from knowledge_hub.application.query_frame import NormalizedQueryFrame, normalize_query_frame_dict

from knowledge_hub.domain.youtube_knowledge.families import (
    YOUTUBE_FAMILY_SECTION_LOOKUP,
    YOUTUBE_FAMILY_TIMESTAMP_LOOKUP,
    YOUTUBE_FAMILY_VIDEO_EXPLAINER,
    YOUTUBE_FAMILY_VIDEO_LOOKUP,
)


_POLICIES: dict[str, dict[str, Any]] = {
    YOUTUBE_FAMILY_VIDEO_LOOKUP: {
        "policyKey": "youtube_video_lookup_policy",
        "family": YOUTUBE_FAMILY_VIDEO_LOOKUP,
        "selectionMode": "single_video",
        "singleScopeRequired": True,
        "timelineRequired": False,
        "sectionPreferred": False,
    },
    YOUTUBE_FAMILY_VIDEO_EXPLAINER: {
        "policyKey": "youtube_video_explainer_policy",
        "family": YOUTUBE_FAMILY_VIDEO_EXPLAINER,
        "selectionMode": "video_summary",
        "singleScopeRequired": False,
        "timelineRequired": False,
        "sectionPreferred": True,
    },
    YOUTUBE_FAMILY_SECTION_LOOKUP: {
        "policyKey": "youtube_section_lookup_policy",
        "family": YOUTUBE_FAMILY_SECTION_LOOKUP,
        "selectionMode": "section_first",
        "singleScopeRequired": True,
        "timelineRequired": False,
        "sectionPreferred": True,
    },
    YOUTUBE_FAMILY_TIMESTAMP_LOOKUP: {
        "policyKey": "youtube_timestamp_lookup_policy",
        "family": YOUTUBE_FAMILY_TIMESTAMP_LOOKUP,
        "selectionMode": "timeline_grounded",
        "singleScopeRequired": True,
        "timelineRequired": True,
        "sectionPreferred": True,
    },
}


def policy_key_for_family(family: str) -> str:
    return str(_POLICIES.get(str(family or "").strip().lower(), {}).get("policyKey") or "youtube_video_explainer_policy")


def policy_for_family(family: str) -> dict[str, Any]:
    key = str(family or "").strip().lower()
    return dict(_POLICIES.get(key) or _POLICIES[YOUTUBE_FAMILY_VIDEO_EXPLAINER])


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
