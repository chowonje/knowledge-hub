"""Project OS CLI backed by personal-foundry state."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from typing import Any

import click
from rich.console import Console

from knowledge_hub.application.agent.foundry_bridge import run_foundry_project_cli
from knowledge_hub.application.dinger_os_bridge import (
    DingerOsBridgeError,
    build_dinger_evidence_candidates,
    bridge_dinger_result_to_os_capture,
)
from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.learning.obsidian_writeback import (
    _upsert_marked_section,
    resolve_vault_write_adapter,
)

console = Console()


def _validate_cli_payload(config, payload: dict[str, Any], schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


def _collect_source_ref_jsons(
    *,
    source_ref_jsons: tuple[str, ...],
    paper_ids: tuple[str, ...],
    urls: tuple[str, ...],
    note_ids: tuple[str, ...],
    stable_scope_ids: tuple[str, ...],
    document_scope_ids: tuple[str, ...],
) -> list[str]:
    refs: list[dict[str, Any]] = []
    for raw in source_ref_jsons:
        text = str(raw or "").strip()
        if not text:
            continue
        refs.append(json.loads(text))
    for value in paper_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "paper", "paperId": text})
    for value in urls:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "web", "url": text})
    for value in note_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "vault", "noteId": text})
    for value in stable_scope_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "scope", "stableScopeId": text})
    for value in document_scope_ids:
        text = str(value or "").strip()
        if text:
            refs.append({"sourceType": "document", "documentScopeId": text})
    return [json.dumps(item, ensure_ascii=False) for item in refs]


def _source_ref_from_ops_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    target_kind = str(item.get("target_kind") or item.get("targetKind") or "").strip().lower()
    target_key = str(item.get("target_key") or item.get("targetKey") or "").strip()
    if target_kind == "paper" and target_key.startswith("paper:"):
        return [{"sourceType": "paper", "paperId": target_key.split(":", 1)[1]}]
    if target_kind == "note" and target_key.startswith("note:"):
        return [{"sourceType": "vault", "noteId": target_key.split(":", 1)[1]}]
    return []


def _ops_alerts_json(khub, *, limit: int = 100) -> str:
    sqlite_db = khub.sqlite_db()
    items = sqlite_db.list_ops_actions(status="pending", scope=None, limit=max(1, int(limit)))
    alerts: list[dict[str, Any]] = []
    for item in items:
        alert_payloads = list(item.get("alert_json") or [])
        severity = "medium"
        if alert_payloads:
            severities = [str(alert.get("severity") or "").strip().lower() for alert in alert_payloads]
            if "critical" in severities:
                severity = "critical"
            elif "high" in severities:
                severity = "high"
            elif "medium" in severities:
                severity = "medium"
            elif "low" in severities:
                severity = "low"
        alerts.append(
            {
                "alertId": str(item.get("action_id") or ""),
                "kind": str(item.get("action_type") or "ops_alert"),
                "severity": severity,
                "summary": str(item.get("summary") or ""),
                "sourceRefs": _source_ref_from_ops_item(item),
            }
        )
    return json.dumps(alerts, ensure_ascii=False)


def _run_foundry_os(args: list[str], *, timeout_sec: int = 120) -> dict[str, Any]:
    payload, error = run_foundry_project_cli(args, timeout_sec=timeout_sec)
    if payload is None:
        raise click.ClickException(error or "foundry project bridge failed")
    return payload


def _project_evidence_payload(*, khub, project_id: str | None, slug: str | None) -> dict[str, Any]:
    selector_args = [
        *(["--project-id", str(project_id)] if project_id else []),
        *(["--slug", str(slug)] if slug else []),
    ]
    payload = _run_foundry_os(
        [
            "project",
            "evidence",
            *selector_args,
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    project_evidence = payload.get("projectEvidence")
    if isinstance(project_evidence, dict):
        inbox_items: list[dict[str, Any]] = []
        if list(project_evidence.get("inboxLinkedRefs") or []):
            inbox_payload = _run_foundry_os(
                [
                    "inbox",
                    "list",
                    "--state",
                    "all",
                    *selector_args,
                    "--ops-alerts-json",
                    _ops_alerts_json(khub),
                ]
            )
            inbox_items = [dict(item) for item in list(inbox_payload.get("items") or []) if isinstance(item, dict)]
        project_evidence["evidenceCandidates"] = build_dinger_evidence_candidates(
            project_evidence,
            inbox_items=inbox_items,
        )
        payload["exploration"] = _build_project_evidence_exploration(
            candidates=list(project_evidence.get("evidenceCandidates") or []),
            inbox_items=inbox_items,
        )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.project.evidence.result.v1")
    return payload


def _current_utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _current_utc_datetime().isoformat()


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _seconds_since(value: Any, *, now: datetime | None = None) -> float | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    current = now or _current_utc_datetime()
    return max(0.0, (current - parsed).total_seconds())


def _relative_time_label(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    if seconds < 604800:
        return f"{int(seconds // 86400)}d"
    return f"{int(seconds // 604800)}w"


def _aging_hint(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 21600:
        return "fresh"
    if seconds < 172800:
        return "active"
    if seconds < 604800:
        return "cooling"
    return "stale"


def _path_tail(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def _primary_source_ref_label(refs: list[dict[str, Any]]) -> str:
    ranked_refs = sorted(
        [ref for ref in refs if isinstance(ref, dict)],
        key=lambda ref: (
            0
            if str(ref.get("sourceType") or "").strip() in {"paper", "web"}
            else 1 if str(ref.get("sourceType") or "").strip() == "vault" else 2,
            _source_ref_label(ref),
        ),
    )
    for ref in ranked_refs:
        if isinstance(ref, dict):
            label = _source_ref_label(ref)
            if label:
                return label
    return "none"


def _candidate_priority_hint(
    *,
    matched_open_count: int,
    matched_resolved_count: int,
    supporting_ref_count: int,
    source_type_count: int,
    updated_seconds: float | None,
) -> tuple[str, int, list[str]]:
    reasons: list[str] = []
    score = 0
    aging_hint = _aging_hint(updated_seconds)
    if matched_resolved_count > 0:
        score += 4
        reasons.append(f"resolved-match:{matched_resolved_count}")
    if aging_hint == "stale":
        score += 3
        reasons.append("stale-touch")
    elif aging_hint == "cooling":
        score += 2
        reasons.append("cooling-touch")
    elif aging_hint == "fresh":
        score += 1
        reasons.append("fresh-touch")
    if matched_open_count > 0:
        score += 2
        reasons.append(f"open-match:{matched_open_count}")
    if supporting_ref_count > 1:
        score += 1
        reasons.append(f"support:{supporting_ref_count}")
    if source_type_count > 1:
        score += 1
        reasons.append(f"source-types:{source_type_count}")
    if matched_resolved_count > 0:
        return "replayed", score, reasons
    if matched_open_count > 0 and aging_hint in {"cooling", "stale"}:
        return "stale-open", score, reasons
    if matched_open_count > 0:
        return "linked-open", score, reasons
    if source_type_count > 1 or supporting_ref_count > 1:
        return "multi-signal", score, reasons
    if aging_hint == "fresh":
        return "fresh", score, reasons
    return "queued", score, reasons


def _attention_level(
    *,
    priority_hint: str,
    attention_score: int,
    matched_open_count: int,
    matched_resolved_count: int,
) -> str:
    if matched_resolved_count > 0 or priority_hint in {"replayed", "stale-open"} or attention_score >= 6:
        return "hot"
    if matched_open_count > 0 or attention_score >= 4:
        return "warm"
    if attention_score >= 2:
        return "watch"
    return "fresh"


def _attention_scan_label(level: str) -> str:
    return {
        "hot": "now",
        "warm": "soon",
        "watch": "track",
        "fresh": "new",
    }.get(str(level or "").strip(), "new")


def _attention_basis(
    *,
    matched_open_count: int,
    matched_resolved_count: int,
    supporting_ref_count: int,
    source_type_count: int,
    updated_label: str,
) -> str:
    parts: list[str] = []
    if matched_resolved_count > 0:
        parts.append(f"resolved-history={matched_resolved_count}")
    if matched_open_count > 0:
        parts.append(f"open-links={matched_open_count}")
    if supporting_ref_count > 1:
        parts.append(f"supporting-refs={supporting_ref_count}")
    if source_type_count > 1:
        parts.append(f"source-types={source_type_count}")
    if updated_label and updated_label != "-":
        parts.append(f"updated={updated_label}")
    return ", ".join(parts) or "single-signal"


def _source_types_from_refs(refs: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(ref.get("sourceType") or "").strip()
            for ref in refs
            if isinstance(ref, dict) and str(ref.get("sourceType") or "").strip()
        }
    )


def _is_dinger_backed_inbox_item(item: dict[str, Any]) -> bool:
    trace = dict(item.get("trace") or {})
    if str(trace.get("bridge") or "").strip() == "dinger" and str(trace.get("relativePath") or "").strip():
        return True
    for ref in list(item.get("sourceRefs") or []):
        if not isinstance(ref, dict):
            continue
        note_id = str(ref.get("noteId") or "").strip().replace("\\", "/")
        if str(ref.get("sourceType") or "").strip() == "vault" and note_id.startswith("KnowledgeOS/Dinger/"):
            return True
    return False


def _candidate_activity_card(candidate: dict[str, Any], inbox_item: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    reason = dict(candidate.get("reason") or {})
    source_refs = [dict(ref) for ref in list(candidate.get("sourceRefs") or []) if isinstance(ref, dict)]
    upstream_refs = [dict(ref) for ref in list(reason.get("upstreamSourceRefs") or []) if isinstance(ref, dict)]
    created_at = str(inbox_item.get("createdAt") or "").strip()
    updated_at = str(inbox_item.get("updatedAt") or "").strip() or created_at
    matched_resolved = [
        dict(item) for item in list(reason.get("matchedResolvedItems") or []) if isinstance(item, dict)
    ]
    matched_open = [
        dict(item) for item in list(reason.get("matchedOpenItems") or []) if isinstance(item, dict)
    ]
    created_seconds = _seconds_since(created_at, now=now)
    updated_seconds = _seconds_since(updated_at, now=now)
    source_types = list(reason.get("sourceTypes") or _source_types_from_refs(source_refs))
    supporting_ref_count = int(reason.get("supportingSourceRefCount") or 0)
    priority_hint, attention_score, attention_reasons = _candidate_priority_hint(
        matched_open_count=len(matched_open),
        matched_resolved_count=len(matched_resolved),
        supporting_ref_count=supporting_ref_count,
        source_type_count=len(source_types),
        updated_seconds=updated_seconds,
    )
    page = str(((candidate.get("trace") or {}).get("relativePath") or "")).strip()
    inspect_token = str(candidate.get("inboxId") or "").strip() or page
    primary_source_ref = _primary_source_ref_label(upstream_refs or source_refs)
    attention_level = _attention_level(
        priority_hint=priority_hint,
        attention_score=attention_score,
        matched_open_count=len(matched_open),
        matched_resolved_count=len(matched_resolved),
    )
    return {
        "candidate": candidate,
        "inboxId": str(candidate.get("inboxId") or "").strip(),
        "summary": str(candidate.get("summary") or "").strip(),
        "page": page,
        "pageShort": _truncate_text(_path_tail(page) or page, 36),
        "createdAt": created_at,
        "updatedAt": updated_at,
        "ageLabel": _relative_time_label(created_seconds),
        "updatedLabel": _relative_time_label(updated_seconds),
        "ageHint": _aging_hint(created_seconds),
        "updatedHint": _aging_hint(updated_seconds),
        "replayAction": str(reason.get("replayAction") or "").strip() or "unknown",
        "inboxState": str(reason.get("inboxState") or inbox_item.get("state") or "open").strip() or "open",
        "sourceTypes": source_types,
        "supportingSourceRefCount": supporting_ref_count,
        "matchedOpenCount": len(matched_open),
        "matchedResolvedCount": len(matched_resolved),
        "recentResolvedMatch": matched_resolved[0] if matched_resolved else None,
        "priorityHint": priority_hint,
        "attentionScore": attention_score,
        "attentionLevel": attention_level,
        "scanLabel": _attention_scan_label(attention_level),
        "attentionReasons": attention_reasons,
        "attentionBasis": _attention_basis(
            matched_open_count=len(matched_open),
            matched_resolved_count=len(matched_resolved),
            supporting_ref_count=supporting_ref_count,
            source_type_count=len(source_types),
            updated_label=_relative_time_label(updated_seconds),
        ),
        "inspectToken": inspect_token,
        "primarySourceRef": primary_source_ref,
        "sourceRefPreview": _summarize_source_refs(upstream_refs or source_refs, limit=2),
    }


def _reviewed_activity_card(item: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    source_refs = [dict(ref) for ref in list(item.get("sourceRefs") or []) if isinstance(ref, dict)]
    trace = dict(item.get("trace") or {})
    created_at = str(item.get("createdAt") or "").strip()
    updated_at = str(item.get("updatedAt") or "").strip() or created_at
    created_seconds = _seconds_since(created_at, now=now)
    updated_seconds = _seconds_since(updated_at, now=now)
    page = str(trace.get("relativePath") or "").strip()
    return {
        "inboxId": str(item.get("id") or "").strip(),
        "summary": str(item.get("summary") or "").strip(),
        "state": str(item.get("state") or "").strip() or "resolved",
        "severity": str(item.get("severity") or "").strip(),
        "sourceRefs": source_refs,
        "sourceTypes": _source_types_from_refs(source_refs),
        "trace": trace,
        "page": page,
        "pageShort": _truncate_text(_path_tail(page) or page, 36),
        "createdAt": created_at,
        "updatedAt": updated_at,
        "ageLabel": _relative_time_label(created_seconds),
        "updatedLabel": _relative_time_label(updated_seconds),
        "ageHint": _aging_hint(created_seconds),
        "updatedHint": _aging_hint(updated_seconds),
        "primarySourceRef": _primary_source_ref_label(source_refs),
        "sourceRefPreview": _summarize_source_refs(source_refs, limit=2),
        "decisionSummary": "resolved_only_decision_not_persisted",
    }


def _candidate_related_key(card: dict[str, Any]) -> tuple[str, str]:
    return (
        str(card.get("page") or "").strip(),
        str(card.get("primarySourceRef") or "").strip(),
    )


def _related_open_candidate_ids(
    *,
    card: dict[str, Any],
    candidate_cards: list[dict[str, Any]],
) -> list[str]:
    page, primary_source_ref = _candidate_related_key(card)
    related_ids: list[str] = []
    for candidate_card in candidate_cards:
        candidate_page, candidate_source = _candidate_related_key(candidate_card)
        if not page and not primary_source_ref:
            continue
        if (page and candidate_page == page) or (
            primary_source_ref and candidate_source == primary_source_ref
        ):
            inbox_id = str(candidate_card.get("inboxId") or "").strip()
            if inbox_id and inbox_id not in related_ids:
                related_ids.append(inbox_id)
    return related_ids


def _reviewed_revisit_level(*, related_open_count: int, updated_hint: str) -> str:
    if related_open_count > 0:
        return "watch"
    if updated_hint in {"fresh", "active"}:
        return "recent"
    return "quiet"


def _revisit_scan_label(level: str) -> str:
    return {
        "watch": "follow",
        "recent": "recent",
        "quiet": "quiet",
    }.get(str(level or "").strip(), "quiet")


def _build_project_evidence_exploration(
    *,
    candidates: list[dict[str, Any]],
    inbox_items: list[dict[str, Any]],
) -> dict[str, Any]:
    now = _current_utc_datetime()
    inbox_by_id = {
        str(item.get("id") or "").strip(): dict(item)
        for item in inbox_items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    candidate_cards = [
        _candidate_activity_card(candidate, inbox_by_id.get(str(candidate.get("inboxId") or "").strip(), {}), now=now)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    candidate_cards.sort(
        key=lambda card: (
            str(card.get("updatedAt") or ""),
            str(card.get("createdAt") or ""),
            str(card.get("inboxId") or ""),
        ),
        reverse=True,
    )
    recent_reviewed_items = [
        _reviewed_activity_card(dict(item), now=now)
        for item in inbox_items
        if isinstance(item, dict)
        and str(item.get("state") or "").strip() not in {"", "open"}
        and _is_dinger_backed_inbox_item(dict(item))
    ]
    for card in recent_reviewed_items:
        related_open_ids = _related_open_candidate_ids(card=card, candidate_cards=candidate_cards)
        card["relatedOpenCandidateIds"] = related_open_ids
        card["relatedOpenCandidateCount"] = len(related_open_ids)
        card["revisitLevel"] = _reviewed_revisit_level(
            related_open_count=len(related_open_ids),
            updated_hint=str(card.get("updatedHint") or "").strip(),
        )
        card["scanLabel"] = _revisit_scan_label(str(card.get("revisitLevel") or "").strip())
        card["revisitBasis"] = (
            f"related-open={len(related_open_ids)} updated={card.get('updatedLabel') or '-'}"
        )
    recent_reviewed_items.sort(
        key=lambda card: (
            str(card.get("updatedAt") or ""),
            str(card.get("createdAt") or ""),
            str(card.get("inboxId") or ""),
        ),
        reverse=True,
    )
    replay_counts = Counter(str(card.get("replayAction") or "unknown") for card in candidate_cards)
    source_counts = Counter(
        source_type
        for card in candidate_cards
        for source_type in list(card.get("sourceTypes") or [])
    )
    attention_level_counts = Counter(str(card.get("attentionLevel") or "fresh") for card in candidate_cards)
    revisit_level_counts = Counter(str(card.get("revisitLevel") or "quiet") for card in recent_reviewed_items)
    return {
        "candidateCards": candidate_cards,
        "recentReviewedItems": recent_reviewed_items,
        "summary": {
            "openCandidateCount": len(candidate_cards),
            "recentReviewedCount": len(recent_reviewed_items),
            "inboxStateCounts": {
                "open": len(candidate_cards),
                "resolved": len(recent_reviewed_items),
            },
            "replayActionCounts": dict(sorted(replay_counts.items())),
            "sourceTypeCounts": dict(sorted(source_counts.items())),
            "attentionLevelCounts": dict(sorted(attention_level_counts.items())),
            "revisitLevelCounts": dict(sorted(revisit_level_counts.items())),
            "reviewDecisionPersistence": "resolved inbox items preserve state only; approve vs dismiss is not stored after review",
        },
    }


def _review_action_metadata(action: str) -> dict[str, str]:
    metadata = {
        "approve": {
            "semanticMeaning": "reviewed_for_manual_promotion",
            "operatorMeaning": "Resolve this inbox item as reviewed while keeping later task or decision promotion explicit.",
            "followUpExpectation": "No task or decision is created here; use explicit promotion later if this evidence should drive work.",
            "resultingInboxState": "resolved",
        },
        "dismiss": {
            "semanticMeaning": "closed_as_not_pursued",
            "operatorMeaning": "Resolve this inbox item as intentionally not pursued, without implying approval or promotion.",
            "followUpExpectation": "No promotion follows from this closure; any later follow-up should arrive through a new or replayed inbox item.",
            "resultingInboxState": "resolved",
        },
        "explain": {
            "semanticMeaning": "read_only_review_guidance",
            "operatorMeaning": "Inspect the candidate and compare approve versus dismiss semantics without mutating OS state.",
            "followUpExpectation": "Choose approve or dismiss explicitly later; the inbox item remains open until then.",
            "resultingInboxState": "open",
        },
    }
    return dict(metadata[action])


def _shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part).strip())


def _project_selector_args(*, project: dict[str, Any]) -> list[str]:
    slug = str(project.get("slug") or "").strip()
    if slug:
        return ["--slug", slug]
    project_id = str(project.get("id") or "").strip()
    if project_id:
        return ["--project-id", project_id]
    return []


def _source_ref_label(ref: dict[str, Any]) -> str:
    source_type = str(ref.get("sourceType") or "").strip()
    primary = (
        str(ref.get("paperId") or "").strip()
        or str(ref.get("url") or "").strip()
        or str(ref.get("noteId") or "").strip()
        or str(ref.get("stableScopeId") or "").strip()
        or str(ref.get("documentScopeId") or "").strip()
    )
    title = str(ref.get("title") or "").strip()
    base = f"{source_type}:{primary}" if source_type and primary else primary or source_type or "unknown"
    if title and title != primary:
        return _truncate_text(f"{base} ({title})", 88)
    return _truncate_text(base, 88)


def _format_source_refs(refs: list[dict[str, Any]], *, limit: int = 3) -> str:
    labels = [_source_ref_label(ref) for ref in refs if isinstance(ref, dict) and _source_ref_label(ref)]
    if not labels:
        return "none"
    if len(labels) <= limit:
        return ", ".join(labels)
    return f"{', '.join(labels[:limit])}, +{len(labels) - limit} more"


def _format_reason_matches(reason: dict[str, Any], key: str, *, limit: int = 2) -> str:
    labels: list[str] = []
    for item in list(reason.get(key) or [])[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        state = str(item.get("state") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not item_id:
            continue
        label = item_id
        if state:
            label += f"({state})"
        if summary:
            label += f": {summary}"
        labels.append(label)
    total = len(list(reason.get(key) or []))
    if not labels:
        return "none"
    if total > limit:
        labels.append(f"+{total - limit} more")
    return "; ".join(labels)


def _counter_summary(counter_map: dict[str, int], *, limit: int = 4) -> str:
    if not counter_map:
        return "none"
    items = sorted(counter_map.items(), key=lambda item: (-int(item[1]), str(item[0])))
    rendered = [f"{key}:{value}" for key, value in items[:limit] if str(key).strip()]
    remaining = max(0, len(items) - len(rendered))
    if remaining:
        rendered.append(f"+{remaining} more")
    return ", ".join(rendered) or "none"


def _filter_candidate_cards(
    cards: list[dict[str, Any]],
    *,
    replay_actions: tuple[str, ...],
    source_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    replay_filter = {str(value).strip() for value in replay_actions if str(value).strip()}
    source_filter = {str(value).strip() for value in source_types if str(value).strip()}
    filtered: list[dict[str, Any]] = []
    for card in cards:
        if replay_filter and str(card.get("replayAction") or "").strip() not in replay_filter:
            continue
        card_source_types = {str(value).strip() for value in list(card.get("sourceTypes") or []) if str(value).strip()}
        if source_filter and not (card_source_types & source_filter):
            continue
        filtered.append(card)
    return filtered


def _filter_reviewed_cards(
    cards: list[dict[str, Any]],
    *,
    source_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    source_filter = {str(value).strip() for value in source_types if str(value).strip()}
    if not source_filter:
        return list(cards)
    filtered: list[dict[str, Any]] = []
    for card in cards:
        card_source_types = {str(value).strip() for value in list(card.get("sourceTypes") or []) if str(value).strip()}
        if card_source_types & source_filter:
            filtered.append(card)
    return filtered


def _sort_candidate_cards(cards: list[dict[str, Any]], *, sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "attention":
        return sorted(
            cards,
            key=lambda card: (
                int(card.get("attentionScore") or 0),
                int(card.get("matchedResolvedCount") or 0),
                int(card.get("matchedOpenCount") or 0),
                int(card.get("supportingSourceRefCount") or 0),
                str(card.get("updatedAt") or ""),
                str(card.get("summary") or ""),
            ),
            reverse=True,
        )
    if sort_key == "oldest":
        return sorted(
            cards,
            key=lambda card: (
                str(card.get("createdAt") or ""),
                str(card.get("updatedAt") or ""),
                str(card.get("summary") or ""),
            ),
        )
    if sort_key == "replay-action":
        return sorted(
            cards,
            key=lambda card: (
                str(card.get("replayAction") or ""),
                str(card.get("updatedAt") or ""),
                str(card.get("summary") or ""),
            ),
            reverse=True,
        )
    if sort_key == "supporting-refs":
        return sorted(
            cards,
            key=lambda card: (
                int(card.get("supportingSourceRefCount") or 0),
                str(card.get("updatedAt") or ""),
                str(card.get("summary") or ""),
            ),
            reverse=True,
        )
    if sort_key == "summary":
        return sorted(
            cards,
            key=lambda card: (
                str(card.get("summary") or ""),
                str(card.get("inboxId") or ""),
            ),
        )
    return sorted(
        cards,
        key=lambda card: (
            str(card.get("updatedAt") or ""),
            str(card.get("createdAt") or ""),
            str(card.get("summary") or ""),
        ),
        reverse=True,
    )


def _build_project_evidence_view(
    *,
    payload: dict[str, Any],
    replay_actions: tuple[str, ...],
    inbox_state: str,
    source_types: tuple[str, ...],
    sort_key: str,
    view_mode: str,
    limit: int,
    recent_reviewed_limit: int,
) -> dict[str, Any]:
    project = dict(payload.get("project") or {})
    exploration = dict(payload.get("exploration") or {})
    candidate_cards = [
        dict(card) for card in list(exploration.get("candidateCards") or []) if isinstance(card, dict)
    ]
    recent_reviewed_items = [
        dict(card) for card in list(exploration.get("recentReviewedItems") or []) if isinstance(card, dict)
    ]
    selector_args = _project_selector_args(project=project)
    filtered_candidates = _sort_candidate_cards(
        _filter_candidate_cards(candidate_cards, replay_actions=replay_actions, source_types=source_types),
        sort_key=sort_key,
    )
    filtered_reviewed = _filter_reviewed_cards(recent_reviewed_items, source_types=source_types)
    visible_candidates = filtered_candidates[: max(1, int(limit))]
    visible_reviewed = filtered_reviewed[: max(0, int(recent_reviewed_limit))]
    for card in visible_candidates:
        inspect_token = str(card.get("inspectToken") or "").strip()
        if inspect_token:
            card["drillDown"] = {
                "candidateId": inspect_token,
                "selectorArg": _shell_command(["--candidate-id", inspect_token]),
                "page": str(card.get("page") or "").strip(),
                "primarySourceRef": str(card.get("primarySourceRef") or "").strip(),
                "show": _shell_command(
                    ["khub", "os", "evidence", "show", *selector_args, "--candidate-id", inspect_token]
                ),
                "explain": _shell_command(
                    ["khub", "os", "evidence", "review", *selector_args, "--candidate-id", inspect_token, "--action", "explain"]
                ),
            }
    for card in visible_reviewed:
        card["drillDown"] = {
            "inboxId": str(card.get("inboxId") or "").strip(),
            "selectorArg": _shell_command(["--candidate-id", str(card.get("inboxId") or "").strip()]),
            "page": str(card.get("page") or "").strip(),
            "primarySourceRef": str(card.get("primarySourceRef") or "").strip(),
            "relatedOpenPreview": ", ".join(list(card.get("relatedOpenCandidateIds") or [])[:2]) or "none",
            "projectEvidence": _shell_command(
                [
                    "khub",
                    "os",
                    "project",
                    "evidence",
                    *selector_args,
                    "--inbox-state",
                    "all",
                ]
            ),
        }
    visible_sections = {
        "openCandidates": inbox_state in {"open", "all"},
        "recentReviewed": inbox_state in {"resolved", "all"},
    }
    return {
        **exploration,
        "filtersApplied": {
            "replayAction": list(replay_actions),
            "inboxState": inbox_state,
            "sourceType": list(source_types),
            "sort": sort_key,
            "view": view_mode,
            "limit": max(1, int(limit)),
            "recentReviewedLimit": max(0, int(recent_reviewed_limit)),
        },
        "visibleSections": visible_sections,
        "visibleCandidates": visible_candidates if visible_sections["openCandidates"] else [],
        "visibleRecentReviewedItems": visible_reviewed if visible_sections["recentReviewed"] else [],
        "viewSummary": {
            "visibleCandidateCount": len(visible_candidates if visible_sections["openCandidates"] else []),
            "visibleRecentReviewedCount": len(visible_reviewed if visible_sections["recentReviewed"] else []),
            "replayActions": _counter_summary(
                Counter(str(card.get("replayAction") or "unknown") for card in filtered_candidates)
            ),
            "sourceTypes": _counter_summary(
                Counter(
                    source_type
                    for card in filtered_candidates
                    for source_type in list(card.get("sourceTypes") or [])
                )
            ),
            "priorityHints": _counter_summary(
                Counter(str(card.get("priorityHint") or "queued") for card in filtered_candidates)
            ),
            "attentionLevels": _counter_summary(
                Counter(str(card.get("attentionLevel") or "fresh") for card in filtered_candidates)
            ),
            "agingHints": _counter_summary(
                Counter(str(card.get("updatedHint") or "unknown") for card in filtered_candidates)
            ),
            "revisitLevels": _counter_summary(
                Counter(str(card.get("revisitLevel") or "quiet") for card in filtered_reviewed)
            ),
        },
    }


def _render_project_evidence_filters(view: dict[str, Any]) -> str:
    filters = dict(view.get("filtersApplied") or {})
    replay_value = ",".join(list(filters.get("replayAction") or [])) or "all"
    source_value = ",".join(list(filters.get("sourceType") or [])) or "all"
    return (
        f"filters: inbox-state={filters.get('inboxState') or 'open'} "
        f"replay-action={replay_value} "
        f"source-type={source_value} "
        f"sort={filters.get('sort') or 'recency'} "
        f"view={filters.get('view') or 'compact'} "
        f"limit={filters.get('limit')} "
        f"recent-reviewed-limit={filters.get('recentReviewedLimit')}"
    )


def _render_project_evidence_summary(view: dict[str, Any]) -> str:
    summary = dict(view.get("summary") or {})
    view_summary = dict(view.get("viewSummary") or {})
    return (
        f"summary: visible-open={view_summary.get('visibleCandidateCount') or 0} "
        f"visible-recent-reviewed={view_summary.get('visibleRecentReviewedCount') or 0} "
        f"all-open={summary.get('openCandidateCount') or 0} "
        f"all-reviewed={summary.get('recentReviewedCount') or 0} "
        f"replay-actions={view_summary.get('replayActions') or 'none'} "
        f"source-types={view_summary.get('sourceTypes') or 'none'} "
        f"priority-hints={view_summary.get('priorityHints') or 'none'} "
        f"attention={view_summary.get('attentionLevels') or 'none'} "
        f"aging={view_summary.get('agingHints') or 'none'} "
        f"revisit={view_summary.get('revisitLevels') or 'none'}"
    )


def _build_evidence_guidance(project: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    selector_args = _project_selector_args(project=project)
    candidate_token = str(candidate.get("inboxId") or "").strip() or str(((candidate.get("trace") or {}).get("relativePath") or "")).strip()
    explain_command = _shell_command(["khub", "os", "evidence", "review", *selector_args, "--candidate-id", candidate_token, "--action", "explain"])
    approve_command = _shell_command(["khub", "os", "evidence", "review", *selector_args, "--candidate-id", candidate_token, "--action", "approve"])
    dismiss_command = _shell_command(["khub", "os", "evidence", "review", *selector_args, "--candidate-id", candidate_token, "--action", "dismiss"])
    show_command = _shell_command(["khub", "os", "evidence", "show", *selector_args, "--candidate-id", candidate_token])
    reason = dict(candidate.get("reason") or {})
    replay_action = str(reason.get("replayAction") or "").strip() or "unknown"
    why_reused = (
        f"replayAction={replay_action}; matched open={_format_reason_matches(reason, 'matchedOpenItems', limit=2)}"
    )
    why_replayed = (
        f"matched resolved={_format_reason_matches(reason, 'matchedResolvedItems', limit=2)}; "
        f"matched other={_format_reason_matches(reason, 'matchedOtherItems', limit=2)}"
    )
    review_next_action = (
        f"Open details with {show_command}, then inspect {explain_command}. "
        f"If the evidence is still actionable use {approve_command}; otherwise use {dismiss_command}."
    )
    return {
        "whyReused": why_reused,
        "whyReplayed": why_replayed,
        "reviewNextAction": review_next_action,
        "commands": {
            "show": show_command,
            "explain": explain_command,
            "approve": approve_command,
            "dismiss": dismiss_command,
        },
    }


def _build_os_evidence_show_payload(*, project: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "knowledge-hub.os.evidence.show.result.v1",
        "status": "ok",
        "project": {
            "id": str(project.get("id") or "").strip(),
            "slug": str(project.get("slug") or "").strip(),
            "title": str(project.get("title") or "").strip(),
        },
        "candidate": candidate,
        "guidance": _build_evidence_guidance(project, candidate),
        "createdAt": _now_iso(),
    }


def _print_os_evidence_detail(
    *,
    label: str,
    project: dict[str, Any],
    candidate: dict[str, Any],
    guidance: dict[str, Any],
    review: dict[str, Any] | None = None,
    candidate_card: dict[str, Any] | None = None,
) -> None:
    reason = dict(candidate.get("reason") or {})
    trace = dict(candidate.get("trace") or {})
    console.print(
        f"[bold]{label}[/bold] project={project.get('slug') or project.get('id')} "
        f"inbox={candidate.get('inboxId')} state={reason.get('inboxState') or '-'}"
    )
    console.print(f"summary: {candidate.get('summary') or '-'}")
    console.print(f"dinger page: {trace.get('relativePath') or reason.get('relativePath') or '-'}")
    if candidate_card:
        console.print(
            f"activity: age={candidate_card.get('ageLabel') or '-'} "
            f"updated={candidate_card.get('updatedLabel') or '-'} "
            f"replay={candidate_card.get('replayAction') or 'unknown'} "
            f"source-types={','.join(list(candidate_card.get('sourceTypes') or [])) or 'none'}"
        )
    console.print(f"source refs: {_format_source_refs(list(candidate.get('sourceRefs') or []), limit=4)}")
    console.print(f"why reused: {guidance.get('whyReused') or '-'}")
    console.print(f"why replayed: {guidance.get('whyReplayed') or '-'}")
    console.print(f"matched open: {_format_reason_matches(reason, 'matchedOpenItems', limit=3)}")
    console.print(f"matched resolved: {_format_reason_matches(reason, 'matchedResolvedItems', limit=3)}")
    console.print(f"matched other: {_format_reason_matches(reason, 'matchedOtherItems', limit=3)}")
    console.print(f"review next: {guidance.get('reviewNextAction') or '-'}")
    commands = dict(guidance.get("commands") or {})
    console.print(f"show command: {commands.get('show') or '-'}")
    console.print(f"explain command: {commands.get('explain') or '-'}")
    console.print(f"approve command: {commands.get('approve') or '-'}")
    console.print(f"dismiss command: {commands.get('dismiss') or '-'}")
    if review:
        console.print(
            f"state change: {review.get('beforeState') or '-'} -> {review.get('afterState') or '-'} "
            f"(changed={review.get('stateChanged')})"
        )


def _find_evidence_candidate(project_payload: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    token = str(candidate_id or "").strip()
    if not token:
        return None
    project_evidence = project_payload.get("projectEvidence") or {}
    for item in list(project_evidence.get("evidenceCandidates") or []):
        if not isinstance(item, dict):
            continue
        inbox_id = str(item.get("inboxId") or "").strip()
        trace = item.get("trace") or {}
        relative_path = str(trace.get("relativePath") or "").strip()
        if token in {inbox_id, relative_path}:
            return dict(item)
    return None


def _find_candidate_card(project_payload: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    token = str(candidate_id or "").strip()
    if not token:
        return None
    exploration = dict(project_payload.get("exploration") or {})
    for card in list(exploration.get("candidateCards") or []):
        if not isinstance(card, dict):
            continue
        inbox_id = str(card.get("inboxId") or "").strip()
        page = str(card.get("page") or "").strip()
        if token in {inbox_id, page}:
            return dict(card)
    return None


def _build_os_evidence_review_payload(
    *,
    action: str,
    project: dict[str, Any],
    candidate: dict[str, Any],
    item: dict[str, Any],
    command: str,
    state_changed: bool,
    before_state: str,
    after_state: str,
) -> dict[str, Any]:
    decision = {
        "approve": "approved",
        "dismiss": "dismissed",
        "explain": "explained",
    }[action]
    receipt = {
        "chosenAction": action,
        **_review_action_metadata(action),
    }
    return {
        "schema": "knowledge-hub.os.evidence.review.result.v1",
        "status": "ok",
        "action": action,
        "project": {
            "id": str(project.get("id") or "").strip(),
            "slug": str(project.get("slug") or "").strip(),
            "title": str(project.get("title") or "").strip(),
        },
        "candidate": candidate,
        "item": item,
        "review": {
            "decision": decision,
            "command": command,
            "stateChanged": bool(state_changed),
            "beforeState": str(before_state or "").strip(),
            "afterState": str(after_state or "").strip(),
            "receipt": receipt,
            "preservesFlow": {
                "projectEvidence": "derived_read_model",
                "stateChange": "existing_inbox_state_only",
                "promotion": "manual_only",
            },
            "traceability": {
                "inboxId": str(candidate.get("inboxId") or "").strip(),
                "relativePath": str(((candidate.get("trace") or {}).get("relativePath") or "")).strip(),
                "bridge": str(((candidate.get("trace") or {}).get("bridge") or "")).strip(),
            },
        },
        "createdAt": _now_iso(),
    }


def _project_scope_args(project: dict[str, Any]) -> str:
    slug = str(project.get("slug") or "").strip()
    if slug:
        return f"--slug {slug}"
    project_id = str(project.get("id") or "").strip()
    if project_id:
        return f"--project-id {project_id}"
    return ""


def _truncate_text(text: str, limit: int = 96) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)].rstrip()}..."


def _summarize_source_refs(refs: list[dict[str, Any]], *, limit: int = 2) -> str:
    labels = [_source_ref_label(ref) for ref in refs[:limit] if isinstance(ref, dict)]
    remaining = max(0, len(refs) - len(labels))
    if remaining:
        labels.append(f"+{remaining} more")
    return ", ".join(label for label in labels if label) or "none"


def _matched_item_label(item: dict[str, Any]) -> str:
    item_id = str(item.get("id") or "").strip()
    state = str(item.get("state") or "").strip()
    summary = _truncate_text(str(item.get("summary") or "").strip(), 48)
    label = f"{item_id}({state})" if state else item_id
    if summary:
        label = f"{label}:{summary}"
    return label


def _summarize_matched_items(items: list[dict[str, Any]], *, limit: int = 2) -> str:
    labels = [_matched_item_label(item) for item in items[:limit] if isinstance(item, dict)]
    remaining = max(0, len(items) - len(labels))
    if remaining:
        labels.append(f"+{remaining} more")
    return ", ".join(label for label in labels if label) or "none"


def _candidate_compact_cue(card: dict[str, Any]) -> str:
    parts: list[str] = []
    matched_open = int(card.get("matchedOpenCount") or 0)
    matched_resolved = int(card.get("matchedResolvedCount") or 0)
    if matched_open > 0:
        parts.append(f"open+{matched_open}")
    if matched_resolved > 0:
        parts.append(f"resolved+{matched_resolved}")
    source_types = ",".join(list(card.get("sourceTypes") or []))
    if source_types:
        parts.append(f"types={source_types}")
    supporting_ref_count = int(card.get("supportingSourceRefCount") or 0)
    if supporting_ref_count > 1:
        parts.append(f"refs={supporting_ref_count}")
    parts.append(f"upd={card.get('updatedLabel') or '-'}")
    return " ".join(parts) or "upd=-"


def _related_open_preview(card: dict[str, Any], *, limit: int = 2) -> str:
    related_ids = [str(value).strip() for value in list(card.get("relatedOpenCandidateIds") or []) if str(value).strip()]
    if not related_ids:
        return "none"
    visible = related_ids[:limit]
    remaining = max(0, len(related_ids) - len(visible))
    preview = ",".join(visible)
    if remaining:
        preview = f"{preview},+{remaining}"
    return preview


def _render_project_evidence_candidate_compact(card: dict[str, Any]) -> str:
    candidate = dict(card.get("candidate") or {})
    drill_down = dict(card.get("drillDown") or {})
    return (
        f"- {str(card.get('inboxId') or '').strip()} "
        f"{card.get('scanLabel') or _attention_scan_label(str(card.get('attentionLevel') or 'fresh'))} "
        f"{_truncate_text(str(candidate.get('summary') or '').strip(), 52)}\n"
        f"  cue={_candidate_compact_cue(card)} age={card.get('ageLabel') or '-'} "
        f"replay={card.get('replayAction') or 'unknown'}\n"
        f"  token={drill_down.get('selectorArg') or '--candidate-id -'} "
        f"source={card.get('primarySourceRef') or 'none'} "
        f"page={card.get('pageShort') or 'unknown'}"
    )


def _render_project_evidence_candidate_readable(card: dict[str, Any]) -> str:
    candidate = dict(card.get("candidate") or {})
    reason = dict(candidate.get("reason") or {})
    drill_down = dict(card.get("drillDown") or {})
    upstream_refs = [dict(ref) for ref in list(reason.get("upstreamSourceRefs") or []) if isinstance(ref, dict)]
    open_matches = [dict(item) for item in list(reason.get("matchedOpenItems") or []) if isinstance(item, dict)]
    resolved_matches = [dict(item) for item in list(reason.get("matchedResolvedItems") or []) if isinstance(item, dict)]
    return (
        f"- {str(candidate.get('inboxId') or '').strip()} "
        f"{card.get('scanLabel') or _attention_scan_label(str(card.get('attentionLevel') or 'fresh'))} "
        f"{_truncate_text(str(candidate.get('summary') or '').strip(), 60)}\n"
        f"  activity: state={card.get('inboxState') or 'open'} age={card.get('ageLabel') or '-'} "
        f"updated={card.get('updatedLabel') or '-'} replay={card.get('replayAction') or 'unknown'} "
        f"attention={card.get('attentionScore') or 0}\n"
        f"  basis: {card.get('attentionBasis') or 'single-signal'}\n"
        f"  reasons: {', '.join(list(card.get('attentionReasons') or [])) or 'none'}\n"
        f"  matches: open={_summarize_matched_items(open_matches, limit=2)} "
        f"resolved={_summarize_matched_items(resolved_matches, limit=2)}\n"
        f"  drill-down: token={drill_down.get('selectorArg') or '--candidate-id -'} "
        f"source={card.get('primarySourceRef') or 'none'}\n"
        f"  commands: show={drill_down.get('show') or '-'} explain={drill_down.get('explain') or '-'}\n"
        f"  page: {str(card.get('page') or ((candidate.get('trace') or {}).get('relativePath') or '')).strip() or 'unknown'}\n"
        f"  refs: {_summarize_source_refs(upstream_refs, limit=2)}"
    )


def _render_recent_reviewed_item_compact(card: dict[str, Any]) -> str:
    drill_down = dict(card.get("drillDown") or {})
    return (
        f"- {str(card.get('inboxId') or '').strip()} "
        f"{card.get('scanLabel') or _revisit_scan_label(str(card.get('revisitLevel') or 'quiet'))} "
        f"{_truncate_text(str(card.get('summary') or '').strip(), 52)}\n"
        f"  context={card.get('revisitBasis') or 'related-open=0'} age={card.get('ageLabel') or '-'} "
        f"related={drill_down.get('relatedOpenPreview') or _related_open_preview(card)}\n"
        f"  token={drill_down.get('selectorArg') or '--candidate-id -'} "
        f"source={card.get('primarySourceRef') or 'none'} "
        f"page={card.get('pageShort') or 'unknown'}"
    )


def _render_recent_reviewed_item_readable(card: dict[str, Any]) -> str:
    drill_down = dict(card.get("drillDown") or {})
    return (
        f"- {str(card.get('inboxId') or '').strip()} "
        f"{card.get('scanLabel') or _revisit_scan_label(str(card.get('revisitLevel') or 'quiet'))} "
        f"{_truncate_text(str(card.get('summary') or '').strip(), 60)}\n"
        f"  activity: state={card.get('state') or 'resolved'} age={card.get('ageLabel') or '-'} "
        f"updated={card.get('updatedLabel') or '-'} decision={card.get('decisionSummary') or 'resolved_only_decision_not_persisted'}\n"
        f"  revisit: {card.get('revisitBasis') or 'related-open=0'}\n"
        f"  drill-down: token={drill_down.get('selectorArg') or '--candidate-id -'} "
        f"related-open={drill_down.get('relatedOpenPreview') or _related_open_preview(card)} "
        f"source={card.get('primarySourceRef') or 'none'}\n"
        f"  command: {drill_down.get('projectEvidence') or '-'}\n"
        f"  page: {str(card.get('page') or '').strip() or 'unknown'}\n"
        f"  refs: {card.get('sourceRefPreview') or 'none'}"
    )


def _render_evidence_review_explain(console: Console, *, payload: dict[str, Any]) -> None:
    project = dict(payload.get("project") or {})
    candidate = dict(payload.get("candidate") or {})
    reason = dict(candidate.get("reason") or {})
    item = dict(payload.get("item") or {})
    guidance = dict(payload.get("guidance") or _build_evidence_guidance(project, candidate))
    review = dict(payload.get("review") or {})
    _print_os_evidence_detail(
        label="os evidence review",
        project=project,
        candidate=candidate,
        guidance=guidance,
        review=review,
        candidate_card=dict(payload.get("candidateCard") or {}),
    )
    approve = _review_action_metadata("approve")
    dismiss = _review_action_metadata("dismiss")
    commands = dict(guidance.get("commands") or {})
    console.print(
        f"approve => {approve['semanticMeaning']} | {approve['operatorMeaning']}"
    )
    console.print(
        f"approve follow-up: {approve['followUpExpectation']}"
    )
    console.print(
        f"approve command: {commands.get('approve') or '-'}"
    )
    console.print(
        f"dismiss => {dismiss['semanticMeaning']} | {dismiss['operatorMeaning']}"
    )
    console.print(
        f"dismiss follow-up: {dismiss['followUpExpectation']}"
    )
    console.print(
        f"dismiss command: {commands.get('dismiss') or '-'}"
    )


def _ensure_projection_file(
    *,
    root: Path,
    relative_path: str,
    title: str,
    sections: list[dict[str, Any]],
    adapter,
) -> str:
    target = root / relative_path
    content = adapter.read_text(target)
    if not content.strip():
        content = f"# {title}\n"
    for section in sections:
        key = str(section.get("key") or "").strip()
        body = str(section.get("body") or "").rstrip()
        if not key or not body:
            continue
        content = _upsert_marked_section(content, key, body)
    adapter.write_text(target, content)
    return str(target)


def _render_export_payload(
    *,
    khub,
    payload: dict[str, Any],
    vault_path: str | None,
    backend: str | None,
    cli_binary: str | None,
    vault_name: str | None,
) -> dict[str, Any]:
    config = khub.config
    resolved_vault = str(vault_path or config.vault_path or "").strip()
    if not resolved_vault:
        raise click.ClickException("vault_path not configured")
    resolved_backend = str(
        backend
        or config.get_nested("obsidian", "write_backend", default="filesystem")
    ).strip() or "filesystem"
    resolved_cli_binary = str(
        cli_binary
        or config.get_nested("obsidian", "cli_binary", default="obsidian")
    ).strip() or "obsidian"
    resolved_vault_name = str(
        vault_name
        or config.get_nested("obsidian", "vault_name", default="")
    ).strip()

    adapter = resolve_vault_write_adapter(
        resolved_vault,
        backend=resolved_backend,
        cli_binary=resolved_cli_binary,
        vault_name=resolved_vault_name,
    )
    root = Path(resolved_vault).expanduser().resolve()
    written_files: list[dict[str, Any]] = []
    for projection in list(payload.get("projections") or []):
        relative_path = str(projection.get("relativePath") or "").strip()
        title = str(projection.get("title") or "").strip() or Path(relative_path).stem
        if not relative_path:
            continue
        file_path = _ensure_projection_file(
            root=root,
            relative_path=relative_path,
            title=title,
            sections=list(projection.get("sections") or []),
            adapter=adapter,
        )
        written_files.append(
            {
                "relativePath": relative_path,
                "path": file_path,
                "sectionCount": len(list(projection.get("sections") or [])),
            }
        )

    result = {
        "schema": "knowledge-hub.os.project.export.obsidian.result.v1",
        "status": "ok",
        "project": payload.get("project") or {},
        "goals": payload.get("goals") or [],
        "tasks": payload.get("tasks") or [],
        "inbox": payload.get("inbox") or [],
        "decisions": payload.get("decisions") or [],
        "projectEvidence": payload.get("projectEvidence") or {},
        "nextActionableTasks": payload.get("nextActionableTasks") or [],
        "blockedTasks": payload.get("blockedTasks") or [],
        "recentDecisions": payload.get("recentDecisions") or [],
        "vaultPath": str(root),
        "backend": resolved_backend,
        "files": written_files,
        "projectionCount": len(written_files),
        "sourceRefCount": len(((payload.get("projectEvidence") or {}).get("sourceRefs") or [])),
        "createdAt": payload.get("createdAt") or "",
        "foundrySchema": payload.get("schema") or "",
    }
    return result


@click.group("os")
def os_group():
    """Project OS MVP commands."""


def _source_ref_options(function):
    function = click.option("--source-ref", "source_ref_jsons", multiple=True, default=())(function)
    function = click.option("--paper-id", "paper_ids", multiple=True, default=())(function)
    function = click.option("--url", "urls", multiple=True, default=())(function)
    function = click.option("--note-id", "note_ids", multiple=True, default=())(function)
    function = click.option("--stable-scope-id", "stable_scope_ids", multiple=True, default=())(function)
    function = click.option("--document-scope-id", "document_scope_ids", multiple=True, default=())(function)
    return function


@os_group.command("capture")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--summary", default=None)
@click.option("--kind", default="captured", show_default=True)
@click.option("--severity", default="medium", show_default=True)
@click.option(
    "--from-dinger-json",
    "from_dinger_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Bridge a filed-like dinger result with a Dinger page pointer into OS inbox/evidence candidates",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_capture(
    ctx,
    project_id,
    slug,
    summary,
    kind,
    severity,
    from_dinger_json,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    cli_source_refs = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        cli_source_refs.append(json.loads(raw))
    if from_dinger_json is not None:
        try:
            dinger_payload = json.loads(from_dinger_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise click.ClickException(f"--from-dinger-json is not valid JSON: {error}") from error
        if not isinstance(dinger_payload, dict):
            raise click.ClickException("--from-dinger-json must point to an object payload")
        try:
            payload = bridge_dinger_result_to_os_capture(
                dinger_payload=dinger_payload,
                project_id=project_id,
                slug=slug,
                summary=summary,
                kind=kind,
                severity=str(severity),
                extra_source_refs=cli_source_refs,
                ops_alerts_json=_ops_alerts_json(khub),
                runner=run_foundry_project_cli,
            )
        except DingerOsBridgeError as error:
            raise click.ClickException(str(error)) from error
    else:
        resolved_summary = str(summary or "").strip()
        if not resolved_summary:
            raise click.ClickException("--summary or --from-dinger-json is required")
        ref_args = []
        for ref in cli_source_refs:
            ref_args.extend(["--source-ref", json.dumps(ref, ensure_ascii=False)])
        payload = _run_foundry_os(
            [
                "capture",
                *(["--project-id", str(project_id)] if project_id else []),
                *(["--slug", str(slug)] if slug else []),
                "--summary",
                resolved_summary,
                "--kind",
                str(kind),
                "--severity",
                str(severity),
                *ref_args,
                "--ops-alerts-json",
                _ops_alerts_json(khub),
            ]
        )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.capture.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    item = payload.get("item") or {}
    console.print(
        f"[bold]os capture[/bold] id={item.get('id')} severity={item.get('severity')} "
        f"kind={item.get('kind')} summary={item.get('summary')}"
    )


@os_group.command("decide")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--goal-id", default=None)
@click.option("--task-id", default=None)
@click.option("--kind", required=True)
@click.option("--summary", required=True)
@click.option("--rationale", default=None)
@click.option("--created-by-type", default=None)
@click.option("--created-by-id", default=None)
@click.option("--supersedes-decision-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_decide(
    ctx,
    project_id,
    slug,
    goal_id,
    task_id,
    kind,
    summary,
    rationale,
    created_by_type,
    created_by_id,
    supersedes_decision_id,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    ref_args = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        ref_args.extend(["--source-ref", raw])
    payload = _run_foundry_os(
        [
            "decide",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            *(["--goal-id", str(goal_id)] if goal_id else []),
            *(["--task-id", str(task_id)] if task_id else []),
            "--kind",
            str(kind),
            "--summary",
            str(summary),
            *(["--rationale", str(rationale)] if rationale else []),
            *(["--created-by-type", str(created_by_type)] if created_by_type else []),
            *(["--created-by-id", str(created_by_id)] if created_by_id else []),
            *(["--supersedes-decision-id", str(supersedes_decision_id)] if supersedes_decision_id else []),
            *ref_args,
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.decide.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    decision = payload.get("decision") or {}
    console.print(
        f"[bold]os decide[/bold] id={decision.get('id')} kind={decision.get('kind')} "
        f"summary={decision.get('summary')}"
    )


@os_group.group("project")
def os_project_group():
    """Project records and projections."""


@os_project_group.command("create")
@click.option("--title", required=True)
@click.option("--slug", default=None)
@click.option("--status", default=None)
@click.option("--priority", default=None)
@click.option("--summary", default=None)
@click.option("--owner", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_create(ctx, title, slug, status, priority, summary, owner, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "project",
            "create",
            "--title",
            str(title),
            *(["--slug", str(slug)] if slug else []),
            *(["--status", str(status)] if status else []),
            *(["--priority", str(priority)] if priority else []),
            *(["--summary", str(summary)] if summary else []),
            *(["--owner", str(owner)] if owner else []),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.project.create.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    project = payload.get("project") or {}
    console.print(
        f"[bold]os project create[/bold] created={payload.get('created')} "
        f"id={project.get('id')} slug={project.get('slug')} title={project.get('title')}"
    )


@os_project_group.command("list")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_list(ctx, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(["project", "list"])
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.project.list.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]os project list[/bold] count={payload.get('count')}")
    for item in list(payload.get("items") or [])[:20]:
        console.print(f"- {item.get('id')} [{item.get('status')}] {item.get('title')} ({item.get('slug')})")


@os_project_group.command("update")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--title", default=None)
@click.option("--status", default=None)
@click.option("--priority", default=None)
@click.option("--summary", default=None)
@click.option("--owner", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_update(ctx, project_id, slug, title, status, priority, summary, owner, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "project",
            "update",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            *(["--title", str(title)] if title else []),
            *(["--status", str(status)] if status else []),
            *(["--priority", str(priority)] if priority else []),
            *(["--summary", str(summary)] if summary else []),
            *(["--owner", str(owner)] if owner else []),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.project.update.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    project = payload.get("project") or {}
    console.print(
        f"[bold]os project update[/bold] id={project.get('id')} status={project.get('status')} "
        f"title={project.get('title')}"
    )


@os_project_group.command("show")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_show(ctx, project_id, slug, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "project",
            "show",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.project.show.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    project = payload.get("project") or {}
    console.print(
        f"[bold]os project show[/bold] id={project.get('id')} title={project.get('title')} "
        f"goals={len(list(payload.get('goals') or []))} tasks={len(list(payload.get('tasks') or []))} "
        f"inbox={len(list(payload.get('inbox') or []))} decisions={len(list(payload.get('decisions') or []))}"
    )


@os_project_group.command("evidence")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--replay-action", "replay_actions", multiple=True, default=())
@click.option("--inbox-state", type=click.Choice(["open", "resolved", "all"]), default="open", show_default=True)
@click.option(
    "--source-type",
    "source_types",
    type=click.Choice(["paper", "web", "vault", "scope", "document"]),
    multiple=True,
    default=(),
)
@click.option(
    "--sort",
    "sort_key",
    type=click.Choice(["recency", "attention", "oldest", "replay-action", "supporting-refs", "summary"]),
    default="recency",
    show_default=True,
)
@click.option(
    "--view",
    "view_mode",
    type=click.Choice(["compact", "readable"]),
    default="compact",
    show_default=True,
)
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--recent-reviewed-limit", default=5, type=int, show_default=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_evidence(
    ctx,
    project_id,
    slug,
    replay_actions,
    inbox_state,
    source_types,
    sort_key,
    view_mode,
    limit,
    recent_reviewed_limit,
    as_json,
):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    payload = _project_evidence_payload(khub=khub, project_id=project_id, slug=slug)
    payload["exploration"] = _build_project_evidence_view(
        payload=payload,
        replay_actions=replay_actions,
        inbox_state=inbox_state,
        source_types=source_types,
        sort_key=sort_key,
        view_mode=view_mode,
        limit=max(1, int(limit)),
        recent_reviewed_limit=max(0, int(recent_reviewed_limit)),
    )
    if as_json:
        console.print_json(data=payload)
        return
    project = payload.get("project") or {}
    evidence = payload.get("projectEvidence") or {}
    exploration = dict(payload.get("exploration") or {})
    console.print(
        f"[bold]os project evidence[/bold] project={project.get('slug') or project.get('id')} "
        f"refs={len(list(evidence.get('sourceRefs') or []))} "
        f"taskLinks={len(list(evidence.get('taskLinkedRefs') or []))} "
        f"decisionLinks={len(list(evidence.get('decisionLinkedRefs') or []))} "
        f"candidates={len(list(evidence.get('evidenceCandidates') or []))}"
    )
    console.print(_render_project_evidence_filters(exploration))
    console.print(_render_project_evidence_summary(exploration))
    console.print(
        f"review note: {((exploration.get('summary') or {}).get('reviewDecisionPersistence') or '-')}"
    )
    visible_candidates = [
        dict(item) for item in list(exploration.get("visibleCandidates") or []) if isinstance(item, dict)
    ]
    visible_reviewed = [
        dict(item)
        for item in list(exploration.get("visibleRecentReviewedItems") or [])
        if isinstance(item, dict)
    ]
    if not visible_candidates and not visible_reviewed:
        console.print("- no evidence rows matched the current filters")
        return
    if visible_candidates:
        console.print("open candidates:")
        for card in visible_candidates:
            console.print(
                _render_project_evidence_candidate_readable(card)
                if view_mode == "readable"
                else _render_project_evidence_candidate_compact(card)
            )
    if visible_reviewed:
        console.print("recent reviewed:")
        for card in visible_reviewed:
            console.print(
                _render_recent_reviewed_item_readable(card)
                if view_mode == "readable"
                else _render_recent_reviewed_item_compact(card)
            )


@os_project_group.command("export-obsidian")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--vault-path", default=None)
@click.option("--backend", default=None)
@click.option("--cli-binary", default=None)
@click.option("--vault-name", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_project_export_obsidian(ctx, project_id, slug, vault_path, backend, cli_binary, vault_name, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "project",
            "export",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ],
        timeout_sec=180,
    )
    result = _render_export_payload(
        khub=khub,
        payload=payload,
        vault_path=vault_path,
        backend=backend,
        cli_binary=cli_binary,
        vault_name=vault_name,
    )
    _validate_cli_payload(khub.config, result, "knowledge-hub.os.project.export.obsidian.result.v1")
    if as_json:
        console.print_json(data=result)
        return
    console.print(
        f"[bold]os project export-obsidian[/bold] project={((result.get('project') or {}).get('slug') or '')} "
        f"files={result.get('projectionCount')} vault={result.get('vaultPath')}"
    )
    for item in list(result.get("files") or []):
        console.print(f"- {item.get('relativePath')} -> {item.get('path')}")


@os_group.group("evidence")
def os_evidence_group():
    """Derived Dinger-backed evidence candidates."""


@os_evidence_group.command("show")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option(
    "--candidate-id",
    required=True,
    help="Candidate inbox id or Dinger note relative path from `khub os project evidence`",
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_evidence_show(ctx, project_id, slug, candidate_id, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    project_payload = _project_evidence_payload(khub=khub, project_id=project_id, slug=slug)
    project = dict(project_payload.get("project") or {})
    candidate = _find_evidence_candidate(project_payload, candidate_id)
    candidate_card = _find_candidate_card(project_payload, candidate_id)
    if candidate is None:
        raise click.ClickException(f"evidence candidate not found: {candidate_id}")
    payload = _build_os_evidence_show_payload(project=project, candidate=candidate)
    if candidate_card:
        payload["candidateCard"] = candidate_card
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.evidence.show.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    _print_os_evidence_detail(
        label="os evidence show",
        project=project,
        candidate=candidate,
        guidance=dict(payload.get("guidance") or {}),
        candidate_card=candidate_card,
    )


@os_evidence_group.command("review")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option(
    "--candidate-id",
    required=True,
    help="Candidate inbox id or Dinger note relative path from `khub os project evidence`",
)
@click.option(
    "--action",
    "review_action",
    type=click.Choice(["approve", "dismiss", "explain"]),
    required=True,
)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_evidence_review(ctx, project_id, slug, candidate_id, review_action, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    project_payload = _project_evidence_payload(khub=khub, project_id=project_id, slug=slug)
    project = dict(project_payload.get("project") or {})
    candidate = _find_evidence_candidate(project_payload, candidate_id)
    candidate_card = _find_candidate_card(project_payload, candidate_id)
    if candidate is None:
        raise click.ClickException(f"evidence candidate not found: {candidate_id}")

    if review_action == "approve":
        action_payload = _run_foundry_os(
            [
                "inbox",
                "triage",
                "--item-id",
                str(candidate.get("inboxId") or ""),
                "--resolve-only",
                "--ops-alerts-json",
                _ops_alerts_json(khub),
            ]
        )
        _validate_cli_payload(khub.config, action_payload, "knowledge-hub.os.inbox.triage.result.v1")
        item = dict(action_payload.get("item") or {})
        payload = _build_os_evidence_review_payload(
            action=review_action,
            project=project,
            candidate=candidate,
            item=item,
            command="khub os inbox triage --resolve-only",
            state_changed=True,
            before_state="open",
            after_state=str(item.get("state") or "resolved"),
        )
    elif review_action == "dismiss":
        action_payload = _run_foundry_os(
            [
                "inbox",
                "resolve",
                "--item-id",
                str(candidate.get("inboxId") or ""),
                "--ops-alerts-json",
                _ops_alerts_json(khub),
            ]
        )
        _validate_cli_payload(khub.config, action_payload, "knowledge-hub.os.inbox.resolve.result.v1")
        item = dict(action_payload.get("item") or {})
        payload = _build_os_evidence_review_payload(
            action=review_action,
            project=project,
            candidate=candidate,
            item=item,
            command="khub os inbox resolve",
            state_changed=True,
            before_state="open",
            after_state=str(item.get("state") or "resolved"),
        )
    else:
        payload = _build_os_evidence_review_payload(
            action=review_action,
            project=project,
            candidate=candidate,
            item={
                "id": str(candidate.get("inboxId") or "").strip(),
                "projectId": str(project.get("id") or "").strip(),
                "state": "open",
                "summary": str(candidate.get("summary") or "").strip(),
                "sourceRefs": list(candidate.get("sourceRefs") or []),
                "trace": dict(candidate.get("trace") or {}),
            },
            command="khub os project evidence",
            state_changed=False,
            before_state="open",
            after_state="open",
        )

    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.evidence.review.result.v1")
    if candidate_card:
        payload["candidateCard"] = candidate_card
    if as_json:
        console.print_json(data=payload)
        return
    if review_action == "explain":
        _render_evidence_review_explain(console, payload=payload)
        return
    reason = ((candidate.get("reason") or {}).get("summary") or "").strip()
    item = payload.get("item") or {}
    receipt = ((payload.get("review") or {}).get("receipt") or {})
    console.print(
        f"[bold]os evidence review[/bold] action={review_action} inbox={item.get('id')} "
        f"state={item.get('state')} meaning={receipt.get('semanticMeaning')} reason={reason}"
    )
    console.print(
        f"follow-up={receipt.get('followUpExpectation')} "
        f"operatorMeaning={receipt.get('operatorMeaning')}"
    )


@os_group.group("goal")
def os_goal_group():
    """Goal records."""


@os_goal_group.command("add")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--title", required=True)
@click.option("--status", default=None)
@click.option("--success-criteria", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_goal_add(ctx, project_id, slug, title, status, success_criteria, as_json):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "goal",
            "add",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            "--title",
            str(title),
            *(["--status", str(status)] if status else []),
            *(["--success-criteria", str(success_criteria)] if success_criteria else []),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.goal.add.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    goal = payload.get("goal") or {}
    console.print(f"[bold]os goal add[/bold] created={payload.get('created')} id={goal.get('id')} title={goal.get('title')}")


@os_goal_group.command("update")
@click.option("--goal-id", required=True)
@click.option("--title", default=None)
@click.option("--status", default=None)
@click.option("--success-criteria", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_goal_update(ctx, goal_id, title, status, success_criteria, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "goal",
            "update",
            "--goal-id",
            str(goal_id),
            *(["--title", str(title)] if title else []),
            *(["--status", str(status)] if status else []),
            *(["--success-criteria", str(success_criteria)] if success_criteria else []),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.goal.update.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    goal = payload.get("goal") or {}
    console.print(f"[bold]os goal update[/bold] id={goal.get('id')} status={goal.get('status')} title={goal.get('title')}")


@os_group.group("task")
def os_task_group():
    """Task records."""


@os_task_group.command("add")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--goal-id", default=None)
@click.option("--title", required=True)
@click.option("--kind", default="task", show_default=True)
@click.option("--status", default=None)
@click.option("--priority", default=None)
@click.option("--assignee", default=None)
@click.option("--due-at", default=None)
@click.option("--blocked-by", "blocked_by", multiple=True, default=())
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_task_add(
    ctx,
    project_id,
    slug,
    goal_id,
    title,
    kind,
    status,
    priority,
    assignee,
    due_at,
    blocked_by,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    ref_args = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        ref_args.extend(["--source-ref", raw])
    payload = _run_foundry_os(
        [
            "task",
            "add",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            *(["--goal-id", str(goal_id)] if goal_id else []),
            "--title",
            str(title),
            "--kind",
            str(kind),
            *(["--status", str(status)] if status else []),
            *(["--priority", str(priority)] if priority else []),
            *(["--assignee", str(assignee)] if assignee else []),
            *(["--due-at", str(due_at)] if due_at else []),
            *sum((["--blocked-by", str(item)] for item in blocked_by), []),
            *ref_args,
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.task.add.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    task = payload.get("task") or {}
    console.print(
        f"[bold]os task add[/bold] created={payload.get('created')} id={task.get('id')} "
        f"status={task.get('status')} title={task.get('title')}"
    )


@os_task_group.command("update-status")
@click.option("--task-id", required=True)
@click.option("--status", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_task_update_status(ctx, task_id, status, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "task",
            "update-status",
            "--task-id",
            str(task_id),
            "--status",
            str(status),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.task.update-status.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    task = payload.get("task") or {}
    console.print(
        f"[bold]os task update-status[/bold] id={task.get('id')} status={task.get('status')} "
        f"inbox={len(list(payload.get('inbox') or []))}"
    )


def _run_task_workflow(ctx, *, verb: str, task_id: str, schema_id: str, reason: str | None = None, as_json: bool = False):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "task",
            verb,
            "--task-id",
            str(task_id),
            *(["--reason", str(reason)] if reason else []),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, schema_id)
    if as_json:
        console.print_json(data=payload)
        return
    task = payload.get("task") or {}
    console.print(
        f"[bold]os task {verb}[/bold] id={task.get('id')} status={task.get('status')} "
        f"inbox={len(list(payload.get('inbox') or []))}"
    )


@os_task_group.command("start")
@click.option("--task-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_task_start(ctx, task_id, as_json):
    _run_task_workflow(ctx, verb="start", task_id=task_id, schema_id="knowledge-hub.os.task.start.result.v1", as_json=as_json)


@os_task_group.command("block")
@click.option("--task-id", required=True)
@click.option("--reason", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_task_block(ctx, task_id, reason, as_json):
    _run_task_workflow(
        ctx,
        verb="block",
        task_id=task_id,
        schema_id="knowledge-hub.os.task.block.result.v1",
        reason=reason,
        as_json=as_json,
    )


@os_task_group.command("complete")
@click.option("--task-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_task_complete(ctx, task_id, as_json):
    _run_task_workflow(ctx, verb="complete", task_id=task_id, schema_id="knowledge-hub.os.task.complete.result.v1", as_json=as_json)


@os_task_group.command("cancel")
@click.option("--task-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_task_cancel(ctx, task_id, as_json):
    _run_task_workflow(ctx, verb="cancel", task_id=task_id, schema_id="knowledge-hub.os.task.cancel.result.v1", as_json=as_json)


@os_task_group.command("update")
@click.option("--task-id", required=True)
@click.option("--title", default=None)
@click.option("--kind", default=None)
@click.option("--priority", default=None)
@click.option("--assignee", default=None)
@click.option("--due-at", default=None)
@click.option("--blocked-by", "blocked_by", multiple=True, default=())
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_task_update(
    ctx,
    task_id,
    title,
    kind,
    priority,
    assignee,
    due_at,
    blocked_by,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    khub = ctx.obj["khub"]
    ref_args = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        ref_args.extend(["--source-ref", raw])
    payload = _run_foundry_os(
        [
            "task",
            "update",
            "--task-id",
            str(task_id),
            *(["--title", str(title)] if title else []),
            *(["--kind", str(kind)] if kind else []),
            *(["--priority", str(priority)] if priority else []),
            *(["--assignee", str(assignee)] if assignee else []),
            *(["--due-at", str(due_at)] if due_at else []),
            *sum((["--blocked-by", str(item)] for item in blocked_by), []),
            *ref_args,
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.task.update.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    task = payload.get("task") or {}
    console.print(
        f"[bold]os task update[/bold] id={task.get('id')} priority={task.get('priority')} "
        f"title={task.get('title')}"
    )


@os_group.group("decision")
def os_decision_group():
    """Decision records."""


@os_decision_group.command("add")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--goal-id", default=None)
@click.option("--task-id", default=None)
@click.option("--kind", required=True)
@click.option("--summary", required=True)
@click.option("--rationale", default=None)
@click.option("--created-by-type", default=None)
@click.option("--created-by-id", default=None)
@click.option("--supersedes-decision-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_decision_add(
    ctx,
    project_id,
    slug,
    goal_id,
    task_id,
    kind,
    summary,
    rationale,
    created_by_type,
    created_by_id,
    supersedes_decision_id,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    if not str(project_id or slug or "").strip():
        raise click.ClickException("--project-id or --slug is required")
    khub = ctx.obj["khub"]
    ref_args = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        ref_args.extend(["--source-ref", raw])
    payload = _run_foundry_os(
        [
            "decision",
            "add",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            *(["--goal-id", str(goal_id)] if goal_id else []),
            *(["--task-id", str(task_id)] if task_id else []),
            "--kind",
            str(kind),
            "--summary",
            str(summary),
            *(["--rationale", str(rationale)] if rationale else []),
            *(["--created-by-type", str(created_by_type)] if created_by_type else []),
            *(["--created-by-id", str(created_by_id)] if created_by_id else []),
            *(["--supersedes-decision-id", str(supersedes_decision_id)] if supersedes_decision_id else []),
            *ref_args,
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.decision.add.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    decision = payload.get("decision") or {}
    console.print(
        f"[bold]os decision add[/bold] id={decision.get('id')} kind={decision.get('kind')} "
        f"summary={decision.get('summary')}"
    )


@os_decision_group.command("list")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--goal-id", default=None)
@click.option("--task-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_decision_list(ctx, project_id, slug, goal_id, task_id, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "decision",
            "list",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            *(["--goal-id", str(goal_id)] if goal_id else []),
            *(["--task-id", str(task_id)] if task_id else []),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.decision.list.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]os decision list[/bold] count={payload.get('count')}")
    for item in list(payload.get("items") or [])[:20]:
        console.print(f"- {item.get('id')} [{item.get('kind')}] {item.get('summary')}")


@os_group.group("inbox")
def os_inbox_group():
    """Inbox items."""


@os_inbox_group.command("list")
@click.option("--state", type=click.Choice(["open", "resolved", "all"]), default="open", show_default=True)
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_inbox_list(ctx, state, project_id, slug, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "inbox",
            "list",
            "--state",
            str(state),
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.inbox.list.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    console.print(f"[bold]os inbox list[/bold] count={payload.get('count')} state={state}")
    for item in list(payload.get("items") or [])[:20]:
        console.print(f"- {item.get('id')} [{item.get('severity')}] {item.get('summary')}")


@os_inbox_group.command("resolve")
@click.option("--item-id", required=True)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_inbox_resolve(ctx, item_id, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "inbox",
            "resolve",
            "--item-id",
            str(item_id),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.inbox.resolve.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    item = payload.get("item") or {}
    console.print(f"[bold]os inbox resolve[/bold] id={item.get('id')} state={item.get('state')}")


@os_inbox_group.command("triage")
@click.option("--item-id", required=True)
@click.option("--to-task", is_flag=True, default=False)
@click.option("--to-decision", is_flag=True, default=False)
@click.option("--resolve-only", is_flag=True, default=False)
@click.option("--title", default=None)
@click.option("--kind", default=None)
@click.option("--summary", default=None)
@click.option("--priority", default=None)
@click.option("--assignee", default=None)
@click.option("--due-at", default=None)
@click.option("--goal-id", default=None)
@click.option("--task-id", default=None)
@click.option("--blocked-by", "blocked_by", multiple=True, default=())
@click.option("--rationale", default=None)
@click.option("--created-by-type", default=None)
@click.option("--created-by-id", default=None)
@click.option("--supersedes-decision-id", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@_source_ref_options
@click.pass_context
def os_inbox_triage(
    ctx,
    item_id,
    to_task,
    to_decision,
    resolve_only,
    title,
    kind,
    summary,
    priority,
    assignee,
    due_at,
    goal_id,
    task_id,
    blocked_by,
    rationale,
    created_by_type,
    created_by_id,
    supersedes_decision_id,
    as_json,
    source_ref_jsons,
    paper_ids,
    urls,
    note_ids,
    stable_scope_ids,
    document_scope_ids,
):
    if sum([1 if to_task else 0, 1 if to_decision else 0, 1 if resolve_only else 0]) != 1:
        raise click.ClickException("exactly one of --to-task, --to-decision, or --resolve-only is required")
    if to_task and not str(title or "").strip():
        raise click.ClickException("--title is required with --to-task")
    if to_decision:
        if not str(kind or "").strip():
            raise click.ClickException("--kind is required with --to-decision")
        if not str(summary or "").strip():
            raise click.ClickException("--summary is required with --to-decision")
    khub = ctx.obj["khub"]
    ref_args = []
    for raw in _collect_source_ref_jsons(
        source_ref_jsons=tuple(source_ref_jsons),
        paper_ids=tuple(paper_ids),
        urls=tuple(urls),
        note_ids=tuple(note_ids),
        stable_scope_ids=tuple(stable_scope_ids),
        document_scope_ids=tuple(document_scope_ids),
    ):
        ref_args.extend(["--source-ref", raw])
    payload = _run_foundry_os(
        [
            "inbox",
            "triage",
            "--item-id",
            str(item_id),
            *(["--to-task"] if to_task else []),
            *(["--to-decision"] if to_decision else []),
            *(["--resolve-only"] if resolve_only else []),
            *(["--title", str(title)] if title else []),
            *(["--kind", str(kind)] if kind else []),
            *(["--summary", str(summary)] if summary else []),
            *(["--priority", str(priority)] if priority else []),
            *(["--assignee", str(assignee)] if assignee else []),
            *(["--due-at", str(due_at)] if due_at else []),
            *(["--goal-id", str(goal_id)] if goal_id else []),
            *(["--task-id", str(task_id)] if task_id else []),
            *sum((["--blocked-by", str(item)] for item in blocked_by), []),
            *(["--rationale", str(rationale)] if rationale else []),
            *(["--created-by-type", str(created_by_type)] if created_by_type else []),
            *(["--created-by-id", str(created_by_id)] if created_by_id else []),
            *(["--supersedes-decision-id", str(supersedes_decision_id)] if supersedes_decision_id else []),
            *ref_args,
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.inbox.triage.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    item = payload.get("item") or {}
    created_task = payload.get("createdTask") or {}
    created_decision = payload.get("createdDecision") or {}
    console.print(
        f"[bold]os inbox triage[/bold] id={item.get('id')} state={item.get('state')} "
        f"task={created_task.get('id') or '-'} decision={created_decision.get('id') or '-'}"
    )


@os_group.command("next")
@click.option("--project-id", default=None)
@click.option("--slug", default=None)
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def os_next(ctx, project_id, slug, as_json):
    khub = ctx.obj["khub"]
    payload = _run_foundry_os(
        [
            "next",
            *(["--project-id", str(project_id)] if project_id else []),
            *(["--slug", str(slug)] if slug else []),
            "--ops-alerts-json",
            _ops_alerts_json(khub),
        ]
    )
    _validate_cli_payload(khub.config, payload, "knowledge-hub.os.next.result.v1")
    if as_json:
        console.print_json(data=payload)
        return
    project = payload.get("project") or {}
    console.print(
        f"[bold]os next[/bold] project={project.get('slug') or project.get('id') or 'none'} "
        f"activeProjects={len(list(payload.get('activeProjects') or []))} "
        f"actionable={len(list(payload.get('actionableTasks') or []))} "
        f"blocked={len(list(payload.get('blockedTasks') or []))} "
        f"inbox={len(list(payload.get('openInbox') or []))} "
        f"decisions={len(list(payload.get('recentDecisions') or []))}"
    )


# Keep the default `khub os --help` focused on the preferred capture/review loop.
# Low-level record groups remain directly invokable for compatibility.
for _hidden_command in (os_goal_group, os_decision_group):
    _hidden_command.hidden = True
