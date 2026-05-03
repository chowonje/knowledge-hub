from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_SCHEMA = "knowledge-hub.paper-card-feedback.event.v1"
CARD_QUALITY_ISSUES = (
    "summary_artifact_missing",
    "summary_artifact_unusable",
    "memory_card_missing",
    "memory_card_unusable",
    "concept_links_missing",
    "related_papers_sparse",
    "latex_core",
    "latex_problem_context",
    "text_starts_latex",
    "generic_limitation",
    "empty_problem_context",
    "empty_method",
    "empty_evidence",
    "fallback_used",
    "likely_semantic_mismatch",
    "front_matter_spillover",
    "table_caption_spillover",
    "raw_english_spillover",
    "other",
)
_SOURCE_REPAIR_ISSUES = {
    "latex_core",
    "latex_problem_context",
    "text_starts_latex",
    "likely_semantic_mismatch",
    "front_matter_spillover",
}
_SUMMARY_REBUILD_ISSUES = {
    "summary_artifact_missing",
    "summary_artifact_unusable",
    "fallback_used",
    "front_matter_spillover",
    "table_caption_spillover",
    "raw_english_spillover",
}
_MEMORY_REBUILD_ISSUES = {
    "memory_card_missing",
    "memory_card_unusable",
    "empty_problem_context",
    "empty_method",
    "empty_evidence",
    "generic_limitation",
}
_CONCEPT_REFRESH_ISSUES = {"concept_links_missing"}
_RELATED_REFRESH_ISSUES = {"related_papers_sparse"}
_SAFE_AUTO_ACTIONS = {
    "rebuild_structured_summary",
    "rebuild_paper_memory",
    "refresh_concept_links",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    raw = values if isinstance(values, list) else list(values) if isinstance(values, tuple) else [values]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = _clean_text(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _flag(mapping: dict[str, Any] | None, key: str) -> bool:
    if not isinstance(mapping, dict):
        return False
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    token = _clean_text(value).casefold()
    return token in {"1", "true", "yes", "y"}


def _snapshot_text(snapshot: dict[str, Any] | None, *keys: str) -> str:
    if not isinstance(snapshot, dict):
        return ""
    return " ".join(_clean_text(snapshot.get(key)) for key in keys if _clean_text(snapshot.get(key))).strip()


def _has_summary_snapshot(summary_snapshot: dict[str, Any] | None) -> bool:
    return bool(_snapshot_text(summary_snapshot, "oneLine", "coreIdea"))


def _has_memory_snapshot(memory_snapshot: dict[str, Any] | None) -> bool:
    return bool(_snapshot_text(memory_snapshot, "paperCore", "methodCore", "evidenceCore"))


def _has_concept_links(memory_snapshot: dict[str, Any] | None) -> bool:
    if not isinstance(memory_snapshot, dict):
        return False
    return bool(_clean_list(memory_snapshot.get("conceptLinks")))


def _append_action(
    actions: list[dict[str, Any]],
    *,
    code: str,
    description: str,
    reason: str,
    auto_apply: bool,
) -> None:
    for action in actions:
        if action.get("code") == code:
            if reason and reason not in list(action.get("reasons") or []):
                action["reasons"] = [*(list(action.get("reasons") or [])), reason]
            action["autoApply"] = bool(action.get("autoApply")) and bool(auto_apply)
            return
    actions.append(
        {
            "code": code,
            "description": description,
            "autoApply": bool(auto_apply),
            "reasons": [reason] if reason else [],
        }
    )


def build_card_remediation_plan(
    *,
    issues: list[str] | None = None,
    artifact_flags: dict[str, Any] | None = None,
    summary_snapshot: dict[str, Any] | None = None,
    memory_snapshot: dict[str, Any] | None = None,
    observed_warnings: list[str] | None = None,
) -> dict[str, Any]:
    issue_list = _clean_list(issues)
    warning_list = _clean_list(observed_warnings)
    issue_set = {token.casefold() for token in [*issue_list, *warning_list] if token}
    actions: list[dict[str, Any]] = []
    reasons: list[str] = []

    has_summary = _flag(artifact_flags, "hasSummary") or _has_summary_snapshot(summary_snapshot)
    has_memory = _flag(artifact_flags, "hasMemory") or _has_memory_snapshot(memory_snapshot)
    has_concepts = _has_concept_links(memory_snapshot)
    summary_missing = (not has_summary) or bool(issue_set & _SUMMARY_REBUILD_ISSUES)
    memory_missing = (not has_memory) or "memory_card_missing" in issue_set
    memory_weak = bool(issue_set & (_MEMORY_REBUILD_ISSUES - {"memory_card_missing"}))
    needs_source_repair = bool(issue_set & _SOURCE_REPAIR_ISSUES)
    needs_concept_refresh = ("concept_links_missing" in issue_set) or (has_memory and not has_concepts)
    needs_related_refresh = "related_papers_sparse" in issue_set

    if needs_source_repair:
        reason = "latex-heavy or semantically mismatched cards need source repair before rebuild."
        reasons.append(reason)
        _append_action(
            actions,
            code="repair_source_content",
            description="repair or relink source artifacts before rebuilding summary/memory",
            reason=reason,
            auto_apply=False,
        )

    if summary_missing:
        reason = "structured summary artifact is missing, empty, or degraded."
        reasons.append(reason)
        _append_action(
            actions,
            code="rebuild_structured_summary",
            description="rebuild structured paper summary artifact",
            reason=reason,
            auto_apply=not needs_source_repair,
        )

    if memory_missing or memory_weak or (summary_missing and has_memory):
        reason = "paper memory card is missing or lacks method/evidence/detail coverage."
        reasons.append(reason)
        _append_action(
            actions,
            code="rebuild_paper_memory",
            description="rebuild paper memory card from current artifacts",
            reason=reason,
            auto_apply=not needs_source_repair,
        )

    if needs_concept_refresh:
        reason = "concept links are missing or stale relative to the card snapshot."
        reasons.append(reason)
        _append_action(
            actions,
            code="refresh_concept_links",
            description="rerun keyword/concept synchronization for the paper",
            reason=reason,
            auto_apply=True,
        )

    if needs_related_refresh:
        reason = "related-paper recommendations are sparse and should be recomputed."
        reasons.append(reason)
        _append_action(
            actions,
            code="refresh_related_papers",
            description="refresh related-paper derivation after summary/memory rebuild",
            reason=reason,
            auto_apply=False,
        )

    if not actions:
        reasons.append("no direct remediation action inferred from the current issues/snapshots.")

    auto_apply_actions = [str(action.get("code") or "") for action in actions if action.get("autoApply")]
    primary_action = str(actions[0].get("code") or "") if actions else "none"
    return {
        "policyVersion": "v1",
        "primaryAction": primary_action,
        "actions": actions,
        "autoApplyActions": auto_apply_actions,
        "requiresManualReview": needs_source_repair,
        "canAutoApply": bool(actions) and all(
            str(action.get("code") or "") in _SAFE_AUTO_ACTIONS for action in actions if action.get("autoApply")
        ),
        "reasons": list(dict.fromkeys(reasons)),
    }


class PaperCardFeedbackLogger:
    def __init__(self, config):
        self.config = config
        self.log_path = Path(config.sqlite_path).expanduser().resolve().parent / "paper_card_feedback.jsonl"

    def _append(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.open("a", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def load_feedback_events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if _clean_text(payload.get("event_type")) != "card_quality_feedback":
                continue
            events.append(payload)
        return events

    def log_feedback(
        self,
        *,
        paper_id: str,
        issues: list[str],
        source: str = "manual",
        note: str = "",
        title: str = "",
        extra: dict[str, Any] | None = None,
        artifact_flags: dict[str, Any] | None = None,
        summary_snapshot: dict[str, Any] | None = None,
        memory_snapshot: dict[str, Any] | None = None,
        observed_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        event = {
            "schema": EVENT_SCHEMA,
            "event_type": "card_quality_feedback",
            "recorded_at": _utc_now(),
            "source": _clean_text(source) or "manual",
            "paper_id": _clean_text(paper_id),
            "title": _clean_text(title),
            "issues": _clean_list(issues),
            "note": _clean_text(note),
            "observed_warnings": _clean_list(observed_warnings),
            "artifact_flags": dict(artifact_flags or {}),
            "summary_snapshot": dict(summary_snapshot or {}),
            "memory_snapshot": dict(memory_snapshot or {}),
        }
        if extra:
            event["paper_metadata"] = dict(extra)
        return self._append(event)

    def build_export_queue(
        self,
        *,
        issues: list[str] | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        requested = {token for token in _clean_list(issues) if token}
        events = self.load_feedback_events()
        selected_paper_ids = {
            _clean_text(event.get("paper_id"))
            for event in events
            if _clean_text(event.get("paper_id"))
            and (
                not requested
                or requested & set(_clean_list(event.get("issues")))
            )
        }
        by_paper: dict[str, dict[str, Any]] = {}
        for event in events:
            paper_id = _clean_text(event.get("paper_id"))
            if not paper_id:
                continue
            if paper_id not in selected_paper_ids:
                continue
            event_issues = _clean_list(event.get("issues"))
            row = by_paper.setdefault(
                paper_id,
                {
                    "paperId": paper_id,
                    "title": _clean_text(event.get("title")),
                    "issues": [],
                    "eventCount": 0,
                    "latestRecordedAt": "",
                    "latestNote": "",
                    "observedWarnings": [],
                    "artifactFlags": {},
                    "summarySnapshot": {},
                    "memorySnapshot": {},
                },
            )
            row["eventCount"] += 1
            for issue in event_issues:
                if issue not in row["issues"]:
                    row["issues"].append(issue)
            recorded_at = _clean_text(event.get("recorded_at"))
            if recorded_at >= str(row.get("latestRecordedAt") or ""):
                row["title"] = _clean_text(event.get("title")) or str(row.get("title") or "")
                row["latestRecordedAt"] = recorded_at
                row["latestNote"] = _clean_text(event.get("note"))
                row["observedWarnings"] = _clean_list(event.get("observed_warnings"))
                row["artifactFlags"] = dict(event.get("artifact_flags") or {})
                row["summarySnapshot"] = dict(event.get("summary_snapshot") or {})
                row["memorySnapshot"] = dict(event.get("memory_snapshot") or {})
        for row in by_paper.values():
            row["remediationPlan"] = build_card_remediation_plan(
                issues=list(row.get("issues") or []),
                artifact_flags=dict(row.get("artifactFlags") or {}),
                summary_snapshot=dict(row.get("summarySnapshot") or {}),
                memory_snapshot=dict(row.get("memorySnapshot") or {}),
                observed_warnings=list(row.get("observedWarnings") or []),
            )
        items = sorted(
            by_paper.values(),
            key=lambda item: (
                str(item.get("latestRecordedAt") or ""),
                str(item.get("paperId") or ""),
            ),
            reverse=True,
        )
        if limit > 0:
            items = items[:limit]
        return items


__all__ = [
    "CARD_QUALITY_ISSUES",
    "EVENT_SCHEMA",
    "PaperCardFeedbackLogger",
    "build_card_remediation_plan",
]
