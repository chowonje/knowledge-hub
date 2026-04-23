"""Conservative paper-memory prefilter helpers."""

from __future__ import annotations

from typing import Any

from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever

PAPER_MEMORY_MODE_OFF = "off"
PAPER_MEMORY_MODE_COMPAT = "compat"
PAPER_MEMORY_MODE_ON = "on"
PAPER_MEMORY_MODE_PREFILTER = "prefilter"
PAPER_MEMORY_MODE_ALIASES = {
    PAPER_MEMORY_MODE_PREFILTER: PAPER_MEMORY_MODE_COMPAT,
}
PAPER_MEMORY_MODES = {
    PAPER_MEMORY_MODE_OFF,
    PAPER_MEMORY_MODE_COMPAT,
    PAPER_MEMORY_MODE_ON,
}


def normalize_paper_memory_mode_details(value: Any) -> tuple[str, str, bool]:
    requested = str(value or PAPER_MEMORY_MODE_OFF).strip().lower() or PAPER_MEMORY_MODE_OFF
    if requested in PAPER_MEMORY_MODES:
        return requested, requested, False
    if requested in PAPER_MEMORY_MODE_ALIASES:
        return requested, PAPER_MEMORY_MODE_ALIASES[requested], True
    return requested, PAPER_MEMORY_MODE_OFF, False


def normalize_paper_memory_mode(value: Any) -> str:
    _, effective, _ = normalize_paper_memory_mode_details(value)
    return effective


def is_weak_paper_memory_card(item: dict[str, Any] | None) -> bool:
    if not item:
        return True
    fields = [
        str(item.get("paperCore") or "").strip(),
        str(item.get("problemContext") or "").strip(),
        str(item.get("methodCore") or "").strip(),
        str(item.get("evidenceCore") or "").strip(),
        str(item.get("limitations") or "").strip(),
    ]
    nonempty = sum(1 for value in fields if value)
    return nonempty < 2 and not list(item.get("claimRefs") or []) and not list(item.get("conceptLinks") or [])


def resolve_paper_memory_prefilter(
    sqlite_db: Any,
    *,
    query: str,
    source_type: str | None,
    requested_mode: str,
    limit: int = 3,
) -> dict[str, Any]:
    requested_token, mode, mode_alias_applied = normalize_paper_memory_mode_details(requested_mode)
    diagnostics = {
        "requestedMode": requested_token,
        "effectiveMode": mode,
        "modeAliasApplied": mode_alias_applied,
        "applied": False,
        "fallbackUsed": False,
        "matchedPaperIds": [],
        "matchedMemoryIds": [],
        "memoryRelationsUsed": [],
        "temporalSignals": {},
        "temporalRouteApplied": False,
        "updatesPreferred": False,
        "fallbackCandidates": [],
        "reason": "",
        "memoryInfluenceApplied": False,
        "verificationCouplingApplied": False,
    }
    if mode == PAPER_MEMORY_MODE_OFF:
        diagnostics["reason"] = "disabled"
        return diagnostics
    if str(source_type or "").strip().lower() != "paper":
        diagnostics["reason"] = "source_not_paper"
        return diagnostics
    if sqlite_db is None:
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "sqlite_unavailable"
        return diagnostics

    try:
        items = PaperMemoryRetriever(sqlite_db).search(query, limit=max(1, int(limit)), include_refs=False)
    except Exception:
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "prefilter_search_failed"
        return diagnostics

    if not items:
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "no_memory_hits"
        return diagnostics

    usable: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("qualityFlag") or "unscored").strip().lower() == "reject":
            continue
        if is_weak_paper_memory_card(item):
            continue
        usable.append(item)

    if not usable:
        diagnostics["fallbackUsed"] = True
        diagnostics["reason"] = "hits_not_usable"
        return diagnostics

    primary: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    memory_ids = {str(item.get("memoryId") or "").strip() for item in usable if str(item.get("memoryId") or "").strip()}
    for item in usable:
        memory_id = str(item.get("memoryId") or "").strip()
        if not memory_id:
            continue
        outgoing = sqlite_db.list_memory_relations(src_form="paper_memory", src_id=memory_id, relation_type="updates", limit=5)
        newer_ids = {str(row.get("dst_id") or "").strip() for row in outgoing if str(row.get("dst_id") or "").strip()}
        if newer_ids & memory_ids:
            fallback_candidates.append(item)
            continue
        primary.append(item)
    if not primary:
        primary = usable[:1]
        fallback_candidates = usable[1:]

    diagnostics["applied"] = True
    diagnostics["memoryInfluenceApplied"] = True
    diagnostics["reason"] = "matched_cards"
    diagnostics["matchedPaperIds"] = [
        str(item.get("paperId") or "").strip()
        for item in primary
        if str(item.get("paperId") or "").strip()
    ]
    diagnostics["matchedMemoryIds"] = [
        str(item.get("memoryId") or "").strip()
        for item in primary
        if str(item.get("memoryId") or "").strip()
    ]
    diagnostics["memoryRelationsUsed"] = [
        str(relation_id).strip()
        for item in primary
        for relation_id in list((item.get("retrievalSignals") or {}).get("memoryRelationsUsed") or [])
        if str(relation_id).strip()
    ]
    temporal_signals = next(
        (
            dict((item.get("retrievalSignals") or {}).get("temporalSignals") or {})
            for item in primary
            if isinstance((item.get("retrievalSignals") or {}).get("temporalSignals"), dict)
        ),
        {},
    )
    diagnostics["temporalSignals"] = temporal_signals
    diagnostics["temporalRouteApplied"] = bool(temporal_signals.get("enabled"))
    diagnostics["updatesPreferred"] = any(
        bool((item.get("retrievalSignals") or {}).get("updatesPreferred"))
        for item in primary
    )
    diagnostics["fallbackCandidates"] = [
        {
            "paperId": str(item.get("paperId") or "").strip(),
            "memoryId": str(item.get("memoryId") or "").strip(),
        }
        for item in fallback_candidates
        if str(item.get("paperId") or "").strip()
    ]
    return diagnostics
