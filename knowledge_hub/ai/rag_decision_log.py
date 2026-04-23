from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _as_frame_dict(query_frame: Any) -> dict[str, Any]:
    if query_frame is None:
        return {}
    if isinstance(query_frame, dict):
        return dict(query_frame)
    to_dict = getattr(query_frame, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def emit_rag_decision_log(
    *,
    stage: str,
    query: str,
    source_type: str | None = None,
    query_plan: dict[str, Any] | None = None,
    query_frame: Any = None,
    memory_route: dict[str, Any] | None = None,
    memory_prefilter: dict[str, Any] | None = None,
    runtime_execution: dict[str, Any] | None = None,
) -> None:
    frame = _as_frame_dict(query_frame)
    plan = dict(query_plan or {})
    route = dict(memory_route or {})
    prefilter = dict(memory_prefilter or {})
    runtime = dict(runtime_execution or {})
    event = {
        "stage": str(stage or "").strip(),
        "query": str(query or ""),
        "sourceType": str(source_type or "").strip(),
        "frameFamily": frame.get("family") or plan.get("family"),
        "frameQueryIntent": frame.get("query_intent") or frame.get("queryIntent") or plan.get("queryIntent"),
        "classifiedIntent": frame.get("classified_intent") or frame.get("classifiedIntent") or plan.get("classifiedIntent"),
        "resolvedIntent": frame.get("resolved_intent") or frame.get("resolvedIntent") or plan.get("resolvedIntent"),
        "sourceKind": frame.get("source_kind") or frame.get("sourceKind") or source_type,
        "frameProvenance": frame.get("frame_provenance") or frame.get("frameProvenance"),
        "frameLockedFields": list(frame.get("frame_locked_fields") or frame.get("frameLockedFields") or []),
        "overrideReason": frame.get("override_reason") or frame.get("overrideReason") or plan.get("overrideReason"),
        "overrideApplied": bool(frame.get("override_applied") or frame.get("overrideApplied") or plan.get("overrideApplied")),
        "requestedMemoryMode": route.get("requestedMode") or prefilter.get("requestedMode"),
        "effectiveMemoryMode": route.get("effectiveMode") or prefilter.get("effectiveMode"),
        "memoryPrefilterApplied": bool(prefilter.get("applied") or route.get("memoryPrefilterApplied")),
        "compatFallbackUsed": bool(prefilter.get("compatFallbackUsed") or route.get("compatFallbackUsed")),
        "memoryInfluenceApplied": bool(route.get("memoryInfluenceApplied") or prefilter.get("memoryInfluenceApplied")),
        "runtimeUsed": runtime.get("used"),
        "runtimeFallbackReason": runtime.get("fallbackReason"),
    }
    logger.debug("rag_decision %s", json.dumps(event, ensure_ascii=False, sort_keys=True))
