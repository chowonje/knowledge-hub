"""Reusable helpers for bridging filed Dinger results into OS inbox items."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

SOURCE_REF_PRIMARY_KEYS = ("paperId", "url", "noteId", "stableScopeId", "documentScopeId")
DINGER_RESULT_SCHEMAS = {
    "knowledge-hub.dinger.capture.result.v1",
    "knowledge-hub.dinger.file.result.v1",
}


class DingerOsBridgeError(ValueError):
    """Raised when a Dinger payload cannot be bridged into OS."""


class FoundryProjectRunner(Protocol):
    def __call__(
        self,
        command_args: Sequence[str],
        timeout_sec: int = 120,
    ) -> tuple[dict[str, Any] | None, str | None]: ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_ref_primary(ref: Mapping[str, Any]) -> tuple[str, str]:
    source_type = str(ref.get("sourceType") or "").strip()
    primary_value = ""
    for key in SOURCE_REF_PRIMARY_KEYS:
        candidate = str(ref.get(key) or "").strip()
        if candidate:
            primary_value = candidate
            break
    return source_type, primary_value


def dedupe_source_refs(refs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for ref in refs:
        marker = source_ref_primary(ref)
        if not marker[0] or not marker[1]:
            skipped += 1
            continue
        if marker in seen:
            skipped += 1
            continue
        seen.add(marker)
        deduped.append({key: value for key, value in dict(ref).items() if value not in (None, "")})
    return deduped, skipped


def relative_note_id(payload: Mapping[str, Any]) -> str:
    relative_path = str(payload.get("relativePath") or "").strip().replace("\\", "/")
    if relative_path:
        return relative_path
    file_path = str(payload.get("filePath") or "").strip()
    vault_path = str(payload.get("vaultPath") or "").strip()
    if not file_path or not vault_path:
        return ""
    try:
        return str(Path(file_path).resolve().relative_to(Path(vault_path).resolve()).as_posix())
    except Exception:
        return ""


def build_dinger_bridge_payload(dinger_payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_id = str(dinger_payload.get("schema") or "").strip()
    if schema_id not in DINGER_RESULT_SCHEMAS:
        raise DingerOsBridgeError("dinger payload must be a capture/file result")
    if str(dinger_payload.get("status") or "").strip().lower() not in {"ok", "filed", "linked_to_os"}:
        raise DingerOsBridgeError("dinger payload must be successful before OS bridge")

    title = str(dinger_payload.get("title") or "").strip()
    resolved_note_id = relative_note_id(dinger_payload)
    if not resolved_note_id:
        raise DingerOsBridgeError("dinger payload must include a relativePath or a vault-relative filePath")
    note_ref = {
        "sourceType": "vault",
        "noteId": resolved_note_id,
        "title": title or Path(resolved_note_id).stem,
    }
    bridge_refs = list(dinger_payload.get("sourceRefs") or [])
    bridge_refs.append(note_ref)
    capture_url = str(dinger_payload.get("captureUrl") or dinger_payload.get("sourceUrl") or "").strip()
    if capture_url:
        bridge_refs.append(
            {
                "sourceType": "web",
                "url": capture_url,
                "title": title or capture_url,
            }
        )
    source_refs, duplicate_count = dedupe_source_refs(
        [dict(item) for item in bridge_refs if isinstance(item, Mapping)]
    )
    summary = (
        f"Dinger capture: {title or capture_url or resolved_note_id or 'captured item'}"
        if schema_id == "knowledge-hub.dinger.capture.result.v1"
        else f"Dinger file: {title or resolved_note_id or 'filed item'}"
    )
    trace = {
        "bridge": "dinger",
        "schema": schema_id,
        "title": title,
        "slug": str(dinger_payload.get("slug") or "").strip(),
        "kind": str(dinger_payload.get("kind") or ("capture" if capture_url else "file")).strip(),
        "relativePath": resolved_note_id,
        "filePath": str(dinger_payload.get("filePath") or "").strip(),
        "captureUrl": capture_url,
        "createdAt": str(dinger_payload.get("createdAt") or "").strip(),
    }
    capture_id = str(dinger_payload.get("captureId") or "").strip()
    packet_path = str(dinger_payload.get("packetPath") or dinger_payload.get("queuePath") or "").strip()
    if capture_id:
        trace["captureId"] = capture_id
    if packet_path:
        trace["packetPath"] = packet_path
    if resolved_note_id:
        trace["filingOutputPointer"] = resolved_note_id
    return {
        "summary": summary,
        "kind": "dinger_capture" if schema_id == "knowledge-hub.dinger.capture.result.v1" else "dinger_file",
        "sourceRefs": source_refs,
        "noteSourceRef": note_ref,
        "trace": trace,
        "duplicateSourceRefsSkipped": duplicate_count,
    }


def attach_capture_trace(
    payload: dict[str, Any],
    *,
    trace: Mapping[str, Any],
    duplicate_count: int,
    link_action: str | None = None,
) -> dict[str, Any]:
    item = payload.get("item")
    if isinstance(item, dict) and trace:
        item["trace"] = dict(trace)
    inbox_items = payload.get("inbox")
    if isinstance(inbox_items, list) and trace:
        item_id = str((item or {}).get("id") or "").strip() if isinstance(item, dict) else ""
        for entry in inbox_items:
            if isinstance(entry, dict) and (not item_id or str(entry.get("id") or "").strip() == item_id):
                entry["trace"] = dict(trace)
    if trace:
        payload["captureTrace"] = dict(trace)
    payload["duplicateSourceRefsSkipped"] = max(0, int(duplicate_count))
    if str(link_action or "").strip():
        payload["linkAction"] = str(link_action).strip()
    return payload


def is_capture_linked_to_os(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip().lower()
    link_action = str(payload.get("linkAction") or "").strip()
    trace = payload.get("captureTrace")
    item = payload.get("item")
    return (
        status == "ok"
        and link_action in {"created", "reused_existing"}
        and isinstance(trace, Mapping)
        and str(trace.get("bridge") or "").strip() == "dinger"
        and str(trace.get("relativePath") or "").strip() != ""
        and isinstance(item, Mapping)
        and str(item.get("id") or "").strip() != ""
    )


def build_capture_os_link_record(
    *,
    capture_payload: Mapping[str, Any] | None,
    dinger_payload: Mapping[str, Any],
    os_payload: Mapping[str, Any],
    project_id: str | None,
    slug: str | None,
) -> dict[str, Any]:
    bridge = build_dinger_bridge_payload(dinger_payload)
    item = dict(os_payload.get("item") or {}) if isinstance(os_payload.get("item"), Mapping) else {}
    trace = dict(os_payload.get("captureTrace") or {}) if isinstance(os_payload.get("captureTrace"), Mapping) else {}
    if not trace:
        trace = dict(bridge.get("trace") or {})
    capture_mapping = dict(capture_payload or {})
    capture_id = str(
        capture_mapping.get("captureId")
        or dinger_payload.get("captureId")
        or trace.get("captureId")
        or ""
    ).strip()
    packet_path = str(
        capture_mapping.get("packetPath")
        or capture_mapping.get("queuePath")
        or dinger_payload.get("packetPath")
        or dinger_payload.get("queuePath")
        or trace.get("packetPath")
        or ""
    ).strip()
    source_refs = [
        dict(ref)
        for ref in list(item.get("sourceRefs") or bridge.get("sourceRefs") or [])
        if isinstance(ref, Mapping)
    ]
    replay = dict(os_payload.get("replay") or {}) if isinstance(os_payload.get("replay"), Mapping) else {}
    dedupe_key = dict(os_payload.get("dedupeKey") or {}) if isinstance(os_payload.get("dedupeKey"), Mapping) else {}
    reason = dict(os_payload.get("reason") or {}) if isinstance(os_payload.get("reason"), Mapping) else {}
    if not reason and dedupe_key and replay:
        open_matches = [item] if str(replay.get("action") or "").strip() == "reused_existing_open_item" else []
        replayed_items: list[dict[str, Any]] = []
        if not open_matches:
            matched_states = [
                str(state).strip()
                for state in list(replay.get("matchedStates") or [])
                if str(state).strip()
            ]
            default_state = matched_states[0] if len(matched_states) == 1 else ""
            replayed_items = [
                {
                    "id": str(item_id or "").strip(),
                    "state": default_state,
                }
                for item_id in list(replay.get("matchedItemIds") or [])
                if str(item_id or "").strip()
            ]
        reason = build_candidate_reason(
            dedupe_key=dedupe_key,
            replay_action=str(replay.get("action") or "").strip(),
            open_matches=open_matches,
            replayed_items=replayed_items,
            trace=trace,
        )
    explanation = str(os_payload.get("explanation") or "").strip()
    if not explanation and reason:
        explanation = build_candidate_explanation(reason)
    return {
        "captureId": capture_id,
        "packetPath": packet_path,
        "sourceSchema": str(dinger_payload.get("schema") or "").strip(),
        "relativePath": str(trace.get("relativePath") or bridge.get("trace", {}).get("relativePath") or "").strip(),
        "filingOutputPointer": str(
            trace.get("filingOutputPointer")
            or bridge.get("trace", {}).get("filingOutputPointer")
            or ""
        ).strip(),
        "filePath": str(trace.get("filePath") or dinger_payload.get("filePath") or "").strip(),
        "captureUrl": str(trace.get("captureUrl") or bridge.get("trace", {}).get("captureUrl") or "").strip(),
        "title": str(trace.get("title") or bridge.get("trace", {}).get("title") or "").strip(),
        "summary": str(item.get("summary") or bridge.get("summary") or "").strip(),
        "kind": str(item.get("kind") or bridge.get("kind") or "").strip(),
        "itemId": str(item.get("id") or "").strip(),
        "projectId": str(item.get("projectId") or project_id or "").strip(),
        "projectSlug": str(slug or "").strip(),
        "linkAction": str(os_payload.get("linkAction") or "").strip(),
        "linkedAt": str(os_payload.get("createdAt") or item.get("updatedAt") or item.get("createdAt") or "").strip(),
        "duplicateSourceRefsSkipped": max(0, int(os_payload.get("duplicateSourceRefsSkipped") or 0)),
        "sourceRefs": source_refs,
        "captureTrace": trace,
        "dedupeKey": dedupe_key,
        "replay": replay,
        "reason": reason,
        "explanation": explanation,
    }


def _is_dinger_note_ref(ref: Mapping[str, Any]) -> bool:
    note_id = str(ref.get("noteId") or "").strip().replace("\\", "/")
    return note_id.startswith("KnowledgeOS/Dinger/")


def build_dinger_evidence_candidate_reason(
    *,
    inbox_id: str,
    summary: str,
    source_refs: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    dedupe_key: Mapping[str, Any],
    open_matches: Sequence[Mapping[str, Any]],
    replayed_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deduped_refs, _skipped = dedupe_source_refs(list(source_refs or []))
    supporting_refs = [dict(ref) for ref in deduped_refs if not _is_dinger_note_ref(ref)]
    relative_path = str(trace.get("relativePath") or "").strip()
    source_types = sorted(
        {
            str(ref.get("sourceType") or "").strip()
            for ref in deduped_refs
            if str(ref.get("sourceType") or "").strip()
        }
    )
    supporting_count = len(supporting_refs)
    supporting_label = f"{supporting_count} supporting source ref"
    if supporting_count != 1:
        supporting_label += "s"
    replay = build_replay_metadata(
        dedupe_key=dedupe_key,
        reused_existing=bool(open_matches),
        replayed_items=replayed_items,
    )
    bridge_reason = build_candidate_reason(
        dedupe_key=dedupe_key,
        replay_action=str(replay.get("action") or "").strip(),
        open_matches=open_matches,
        replayed_items=replayed_items,
        trace=trace,
    )
    matched_open = list(bridge_reason.get("matchedOpenItems") or [])
    matched_resolved = list(bridge_reason.get("matchedResolvedItems") or [])
    matched_other = list(bridge_reason.get("matchedOtherItems") or [])
    open_summary = _format_match_list(matched_open)
    resolved_summary = _format_match_list(matched_resolved)
    other_summary = _format_match_list(matched_other)
    reason_summary = (
        f"Open inbox item {inbox_id} is linked to filed Dinger page {relative_path} "
        f"and keeps {supporting_label} attached for later manual promotion. "
        f"rerun reuse={open_summary}; replay resolved={resolved_summary}; replay other={other_summary}."
    )
    return {
        "kind": "dinger_linked_inbox_candidate",
        "summary": reason_summary,
        "inboxState": "open",
        "bridge": "dinger",
        "inboxSummary": str(summary or "").strip(),
        "relativePath": relative_path,
        "supportingSourceRefCount": supporting_count,
        "sourceTypes": source_types,
        "upstreamSourceRefs": supporting_refs,
        "replayAction": str(replay.get("action") or "").strip(),
        "replayPolicy": dict(replay.get("policy") or {}),
        "matchedOpenItems": matched_open,
        "matchedResolvedItems": matched_resolved,
        "matchedOtherItems": matched_other,
        "dedupeKeySummary": dict(bridge_reason.get("dedupeKeySummary") or {}),
        "bridgeTraceSummary": dict(bridge_reason.get("bridgeTraceSummary") or {}),
    }


def build_dinger_evidence_candidates(
    project_evidence: Mapping[str, Any],
    *,
    inbox_items: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for link in list(project_evidence.get("inboxLinkedRefs") or []):
        if not isinstance(link, Mapping):
            continue
        inbox_id = str(link.get("inboxId") or "").strip()
        summary = str(link.get("summary") or "").strip()
        source_ref = dict(link.get("sourceRef") or {}) if isinstance(link.get("sourceRef"), Mapping) else {}
        if not inbox_id or not source_ref:
            continue
        bucket = grouped.setdefault(
            inbox_id,
            {
                "inboxId": inbox_id,
                "summary": summary,
                "sourceRefs": [],
                "trace": {},
            },
        )
        bucket["sourceRefs"].append(source_ref)
        if _is_dinger_note_ref(source_ref):
            note_id = str(source_ref.get("noteId") or "").strip().replace("\\", "/")
            bucket["trace"] = {
                "bridge": "dinger",
                "relativePath": note_id,
                "title": str(source_ref.get("title") or Path(note_id).stem).strip(),
            }

    candidates: list[dict[str, Any]] = []
    normalized_inbox_items = [dict(item) for item in list(inbox_items or []) if isinstance(item, Mapping)]
    for bucket in grouped.values():
        if not bucket.get("trace"):
            continue
        deduped_refs, _skipped = dedupe_source_refs(list(bucket.get("sourceRefs") or []))
        trace = dict(bucket["trace"])
        current_item = next(
            (item for item in normalized_inbox_items if str(item.get("id") or "").strip() == str(bucket["inboxId"]).strip()),
            {},
        )
        current_kind = str(current_item.get("kind") or "dinger_file").strip() or "dinger_file"
        dedupe_key = build_bridge_dedupe_key(kind=current_kind, source_refs=deduped_refs)
        matched_open: list[dict[str, Any]] = []
        replayed_items: list[dict[str, Any]] = []
        for item in normalized_inbox_items:
            if _item_dedupe_key(item) != dedupe_key:
                continue
            if str(item.get("state") or "").strip() == "open":
                matched_open.append(dict(item))
            else:
                replayed_items.append(dict(item))
        if not matched_open:
            fallback_item = {
                "id": str(bucket["inboxId"]),
                "state": "open",
                "summary": str(bucket.get("summary") or ""),
                "kind": current_kind,
                "sourceRefs": deduped_refs,
            }
            matched_open = [fallback_item]
        reason = build_dinger_evidence_candidate_reason(
            inbox_id=str(bucket["inboxId"]),
            summary=str(bucket.get("summary") or ""),
            source_refs=deduped_refs,
            trace=trace,
            dedupe_key=dedupe_key,
            open_matches=matched_open,
            replayed_items=replayed_items,
        )
        candidates.append(
            {
                "inboxId": bucket["inboxId"],
                "summary": bucket["summary"],
                "sourceRefs": deduped_refs,
                "trace": trace,
                "reason": reason,
                "explanation": str(reason.get("summary") or "").strip(),
            }
        )
    candidates.sort(key=lambda item: (str(item.get("summary") or ""), str(item.get("inboxId") or "")))
    return candidates


def _run_foundry_os(
    runner: FoundryProjectRunner,
    args: list[str],
    *,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    payload, error = runner(args, timeout_sec=timeout_sec)
    if payload is None:
        raise DingerOsBridgeError(error or "foundry project bridge failed")
    return payload


def _normalized_ref_markers(refs: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    markers: set[tuple[str, str]] = set()
    for ref in refs:
        marker = source_ref_primary(ref)
        if marker[0] and marker[1]:
            markers.add(marker)
    return markers


def _sorted_ref_markers(refs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    return sorted(_normalized_ref_markers(refs), key=lambda item: (item[0], item[1]))


def _marker_label(marker: Mapping[str, Any]) -> str:
    source_type = str(marker.get("sourceType") or "").strip()
    primary_value = str(marker.get("primary") or "").strip()
    if not source_type or not primary_value:
        return ""
    return f"{source_type}:{primary_value}"


def build_bridge_dedupe_key(
    *,
    kind: str,
    source_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_kind = str(kind or "").strip()
    markers = _sorted_ref_markers(source_refs)
    dinger_note_markers = [
        marker
        for marker in markers
        if marker[0] == "vault" and marker[1].startswith("KnowledgeOS/Dinger/")
    ]
    key_markers = dinger_note_markers or markers
    strategy = "vault_note" if dinger_note_markers else "source_refs"
    marker_payload = [
        {
            "sourceType": source_type,
            "primary": primary_value,
        }
        for source_type, primary_value in key_markers
    ]
    fingerprint = "|".join([normalized_kind, strategy, *(f"{source_type}:{primary_value}" for source_type, primary_value in key_markers)])
    return {
        "kind": normalized_kind,
        "strategy": strategy,
        "markers": marker_payload,
        "fingerprint": fingerprint,
    }


def _item_dedupe_key(item: Mapping[str, Any]) -> dict[str, Any]:
    return build_bridge_dedupe_key(
        kind=str(item.get("kind") or "").strip(),
        source_refs=list(item.get("sourceRefs") or []),
    )


def find_matching_inbox_item(
    inbox_items: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    source_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    desired_key = build_bridge_dedupe_key(kind=kind, source_refs=source_refs)
    if not desired_key["markers"]:
        return None
    for item in inbox_items:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("state") or "").strip() not in {"", "open"}:
            continue
        if _item_dedupe_key(item) != desired_key:
            continue
        return dict(item)
    return None


def find_replayed_inbox_items(
    inbox_items: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    source_refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    desired_key = build_bridge_dedupe_key(kind=kind, source_refs=source_refs)
    if not desired_key["markers"]:
        return []
    matches: list[dict[str, Any]] = []
    for item in inbox_items:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("state") or "").strip() in {"", "open"}:
            continue
        if _item_dedupe_key(item) != desired_key:
            continue
        matches.append(dict(item))
    return matches


def build_replay_metadata(
    *,
    dedupe_key: Mapping[str, Any],
    reused_existing: bool,
    replayed_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matched_states = sorted(
        {
            str(item.get("state") or "").strip()
            for item in replayed_items
            if str(item.get("state") or "").strip()
        }
    )
    matched_item_ids = [
        str(item.get("id") or "").strip()
        for item in replayed_items
        if str(item.get("id") or "").strip()
    ]
    if reused_existing:
        action = "reused_existing_open_item"
    elif replayed_items:
        action = "created_new_after_resolved_match"
    else:
        action = "created_new_without_prior_match"
    return {
        "policy": {
            "open": "reuse",
            "resolved": "create_new",
            "triaged": "create_new",
        },
        "dedupeKey": dict(dedupe_key),
        "action": action,
        "matchedItemIds": matched_item_ids,
        "matchedStates": matched_states,
    }


def _dedupe_key_summary(dedupe_key: Mapping[str, Any]) -> dict[str, Any]:
    markers = [
        {
            "sourceType": str(item.get("sourceType") or "").strip(),
            "primary": str(item.get("primary") or "").strip(),
        }
        for item in list(dedupe_key.get("markers") or [])
        if isinstance(item, Mapping)
        and str(item.get("sourceType") or "").strip()
        and str(item.get("primary") or "").strip()
    ]
    return {
        "kind": str(dedupe_key.get("kind") or "").strip(),
        "strategy": str(dedupe_key.get("strategy") or "").strip(),
        "markerCount": len(markers),
        "markers": markers,
        "fingerprint": str(dedupe_key.get("fingerprint") or "").strip(),
    }


def _matched_item_summary(item: Mapping[str, Any]) -> dict[str, str] | None:
    item_id = str(item.get("id") or "").strip()
    state = str(item.get("state") or "").strip()
    if not item_id or not state:
        return None
    payload = {"id": item_id, "state": state}
    summary = str(item.get("summary") or "").strip()
    if summary:
        payload["summary"] = summary
    return payload


def _matched_item_groups(
    *,
    open_matches: Sequence[Mapping[str, Any]],
    replayed_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    matched_open: list[dict[str, str]] = []
    matched_resolved: list[dict[str, str]] = []
    matched_other: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*list(open_matches), *list(replayed_items)]:
        payload = _matched_item_summary(item)
        if payload is None:
            continue
        marker = (payload["id"], payload["state"])
        if marker in seen:
            continue
        seen.add(marker)
        if payload["state"] == "open":
            matched_open.append(payload)
        elif payload["state"] == "resolved":
            matched_resolved.append(payload)
        else:
            matched_other.append(payload)
    return matched_open, matched_resolved, matched_other


def _bridge_trace_summary(trace: Mapping[str, Any]) -> dict[str, str]:
    summary = {
        "bridge": str(trace.get("bridge") or "").strip() or "dinger",
        "sourceSchema": str(trace.get("schema") or "").strip(),
        "kind": str(trace.get("kind") or "").strip(),
        "relativePath": str(trace.get("relativePath") or "").strip(),
    }
    for key in ("title", "captureUrl", "captureId"):
        value = str(trace.get(key) or "").strip()
        if value:
            summary[key] = value
    return summary


def build_candidate_reason(
    *,
    dedupe_key: Mapping[str, Any],
    replay_action: str,
    open_matches: Sequence[Mapping[str, Any]],
    replayed_items: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    matched_open, matched_resolved, matched_other = _matched_item_groups(
        open_matches=open_matches,
        replayed_items=replayed_items,
    )
    return {
        "dedupeKeySummary": _dedupe_key_summary(dedupe_key),
        "replayAction": str(replay_action or "").strip(),
        "matchedOpenItems": matched_open,
        "matchedResolvedItems": matched_resolved,
        "matchedOtherItems": matched_other,
        "bridgeTraceSummary": _bridge_trace_summary(trace),
    }


def _format_match_list(items: Sequence[Mapping[str, Any]]) -> str:
    labels: list[str] = []
    for item in items:
        item_id = str(item.get("id") or "").strip()
        state = str(item.get("state") or "").strip()
        if not item_id:
            continue
        labels.append(f"{item_id}({state})" if state else item_id)
    return ", ".join(labels) or "none"


def build_candidate_explanation(reason: Mapping[str, Any]) -> str:
    dedupe_key = dict(reason.get("dedupeKeySummary") or {}) if isinstance(reason.get("dedupeKeySummary"), Mapping) else {}
    bridge_trace = dict(reason.get("bridgeTraceSummary") or {}) if isinstance(reason.get("bridgeTraceSummary"), Mapping) else {}
    replay_action = str(reason.get("replayAction") or "").strip()
    markers = [
        str(item.get("primary") or "").strip()
        for item in list(dedupe_key.get("markers") or [])
        if isinstance(item, Mapping) and str(item.get("primary") or "").strip()
    ]
    marker_summary = ", ".join(markers) or str(dedupe_key.get("fingerprint") or "").strip() or "no dedupe marker"
    dedupe_summary = (
        f"{str(dedupe_key.get('kind') or '').strip()}/{str(dedupe_key.get('strategy') or '').strip()} [{marker_summary}]"
    ).strip()
    open_matches = _format_match_list(list(reason.get("matchedOpenItems") or []))
    resolved_matches = _format_match_list(list(reason.get("matchedResolvedItems") or []))
    other_matches = _format_match_list(list(reason.get("matchedOtherItems") or []))
    trace_summary = str(bridge_trace.get("relativePath") or bridge_trace.get("captureUrl") or bridge_trace.get("sourceSchema") or "").strip()
    source_schema = str(bridge_trace.get("sourceSchema") or "").strip() or "unknown_schema"

    if replay_action == "reused_existing_open_item":
        return (
            f"Reused existing open evidence candidate {open_matches} because dedupe key {dedupe_summary} "
            f"matched an open inbox item. replay.action={replay_action}; resolved matches={resolved_matches}; "
            f"other matches={other_matches}; bridge trace={source_schema} -> {trace_summary or 'unknown_target'}."
        )
    if replay_action == "created_new_after_resolved_match":
        return (
            f"Created a new evidence candidate because dedupe key {dedupe_summary} matched only resolved or non-open "
            f"items. replay.action={replay_action}; open matches={open_matches}; resolved matches={resolved_matches}; "
            f"other matches={other_matches}; bridge trace={source_schema} -> {trace_summary or 'unknown_target'}."
        )
    return (
        f"Created a new evidence candidate because dedupe key {dedupe_summary} had no prior open or resolved match. "
        f"replay.action={replay_action or 'created_new_without_prior_match'}; open matches={open_matches}; "
        f"resolved matches={resolved_matches}; other matches={other_matches}; "
        f"bridge trace={source_schema} -> {trace_summary or 'unknown_target'}."
    )


def bridge_dinger_result_to_os_capture(
    *,
    dinger_payload: Mapping[str, Any],
    project_id: str | None,
    slug: str | None,
    summary: str | None,
    kind: str | None,
    severity: str,
    extra_source_refs: Sequence[Mapping[str, Any]],
    ops_alerts_json: str,
    runner: FoundryProjectRunner,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    if not str(project_id or slug or "").strip():
        raise DingerOsBridgeError("project_id or slug is required")
    dinger_bridge = build_dinger_bridge_payload(dinger_payload)
    merged_source_refs, duplicate_count = dedupe_source_refs(
        list(extra_source_refs) + list(dinger_bridge.get("sourceRefs") or [])
    )
    duplicate_count += int(dinger_bridge.get("duplicateSourceRefsSkipped") or 0)
    resolved_summary = str(summary or dinger_bridge.get("summary") or "").strip()
    if not resolved_summary:
        raise DingerOsBridgeError("summary is required")
    resolved_kind = str(dinger_bridge.get("kind") or kind or "captured").strip() or "captured"
    dedupe_key = build_bridge_dedupe_key(kind=resolved_kind, source_refs=merged_source_refs)
    selector_args = [
        *(["--project-id", str(project_id)] if project_id else []),
        *(["--slug", str(slug)] if slug else []),
    ]
    list_payload = _run_foundry_os(
        runner,
        [
            "inbox",
            "list",
            "--state",
            "open",
            *selector_args,
            "--ops-alerts-json",
            ops_alerts_json,
        ],
        timeout_sec=timeout_sec,
    )
    inbox_items = [dict(item) for item in list(list_payload.get("items") or []) if isinstance(item, Mapping)]
    existing_item = find_matching_inbox_item(
        inbox_items,
        kind=resolved_kind,
        source_refs=merged_source_refs,
    )
    if existing_item is not None:
        reused_payload = {
            "schema": "knowledge-hub.os.capture.result.v1",
            "status": "ok",
            "item": existing_item,
            "inbox": inbox_items,
            "createdAt": str(list_payload.get("createdAt") or existing_item.get("updatedAt") or _now_iso()),
        }
        replay = build_replay_metadata(
            dedupe_key=dedupe_key,
            reused_existing=True,
            replayed_items=[],
        )
        reason = build_candidate_reason(
            dedupe_key=dedupe_key,
            replay_action=str(replay.get("action") or "").strip(),
            open_matches=[existing_item],
            replayed_items=[],
            trace=dict(dinger_bridge.get("trace") or {}),
        )
        result = attach_capture_trace(
            reused_payload,
            trace=dict(dinger_bridge.get("trace") or {}),
            duplicate_count=duplicate_count,
            link_action="reused_existing",
        )
        result["replay"] = replay
        result["dedupeKey"] = dict(dedupe_key)
        result["reason"] = reason
        result["explanation"] = build_candidate_explanation(reason)
        return result

    replay_items: list[dict[str, Any]] = []
    if dedupe_key["markers"]:
        replay_payload = _run_foundry_os(
            runner,
            [
                "inbox",
                "list",
                "--state",
                "all",
                *selector_args,
                "--ops-alerts-json",
                ops_alerts_json,
            ],
            timeout_sec=timeout_sec,
        )
        replay_candidates = [
            dict(item)
            for item in list(replay_payload.get("items") or [])
            if isinstance(item, Mapping)
        ]
        replay_items = find_replayed_inbox_items(
            replay_candidates,
            kind=resolved_kind,
            source_refs=merged_source_refs,
        )

    ref_args: list[str] = []
    for ref in merged_source_refs:
        ref_args.extend(["--source-ref", json.dumps(ref, ensure_ascii=False)])
    created_payload = _run_foundry_os(
        runner,
        [
            "capture",
            *selector_args,
            "--summary",
            resolved_summary,
            "--kind",
            resolved_kind,
            "--severity",
            str(severity),
            *ref_args,
            "--ops-alerts-json",
            ops_alerts_json,
        ],
        timeout_sec=timeout_sec,
    )
    result = attach_capture_trace(
        created_payload,
        trace=dict(dinger_bridge.get("trace") or {}),
        duplicate_count=duplicate_count,
        link_action="created",
    )
    replay = build_replay_metadata(
        dedupe_key=dedupe_key,
        reused_existing=False,
        replayed_items=replay_items,
    )
    reason = build_candidate_reason(
        dedupe_key=dedupe_key,
        replay_action=str(replay.get("action") or "").strip(),
        open_matches=[],
        replayed_items=replay_items,
        trace=dict(dinger_bridge.get("trace") or {}),
    )
    result["replay"] = replay
    result["dedupeKey"] = dict(dedupe_key)
    result["reason"] = reason
    result["explanation"] = build_candidate_explanation(reason)
    return result
