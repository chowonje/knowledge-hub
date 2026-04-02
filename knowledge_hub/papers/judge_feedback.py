from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_SCHEMA = "knowledge-hub.paper-judge.event.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


class PaperJudgeFeedbackLogger:
    def __init__(self, config):
        self.config = config
        self.log_path = Path(config.sqlite_path).expanduser().resolve().parent / "paper_judge_events.jsonl"

    def _append(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.open("a", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def log_judge_decisions(
        self,
        *,
        topic: str,
        items: list[dict[str, Any]],
        backend: str,
        threshold: float,
        degraded: bool,
        allow_external: bool,
        source: str,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for item in items:
            paper_id = _safe_text(item.get("paper_id") or item.get("arxiv_id"))
            if not paper_id:
                continue
            event = {
                "schema": EVENT_SCHEMA,
                "event_type": "judge_decision",
                "recorded_at": _utc_now(),
                "source": source,
                "topic": _safe_text(topic),
                "paper_id": paper_id,
                "title": _safe_text(item.get("title")),
                "judge_decision": _safe_text(item.get("decision")),
                "judge_score": float(item.get("total_score", 0.0) or 0.0),
                "judge_backend": _safe_text(item.get("backend") or backend),
                "judge_threshold": float(threshold or 0.0),
                "judge_degraded": bool(degraded),
                "allow_external": bool(allow_external),
                "dimension_scores": dict(item.get("dimension_scores") or {}),
                "top_reasons": [str(reason).strip() for reason in list(item.get("top_reasons") or []) if str(reason).strip()],
            }
            events.append(self._append(event))
        return events

    def get_latest_judge_decision(self, paper_id: str) -> dict[str, Any] | None:
        token = _safe_text(paper_id)
        if not token or not self.log_path.exists():
            return None
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event_type") != "judge_decision":
                continue
            if _safe_text(payload.get("paper_id")) == token:
                return payload
        return None

    def log_feedback(
        self,
        *,
        paper_id: str,
        label: str,
        source: str = "manual",
        reason: str = "",
        title: str = "",
        topic: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        latest = self.get_latest_judge_decision(paper_id)
        judge_decision = _safe_text((latest or {}).get("judge_decision"))
        event = {
            "schema": EVENT_SCHEMA,
            "event_type": "manual_feedback",
            "recorded_at": _utc_now(),
            "source": _safe_text(source) or "manual",
            "paper_id": _safe_text(paper_id),
            "title": _safe_text(title) or _safe_text((latest or {}).get("title")),
            "topic": _safe_text(topic) or _safe_text((latest or {}).get("topic")),
            "human_label": _safe_text(label),
            "reason": _safe_text(reason),
            "judge_context_found": latest is not None,
            "judge_decision": judge_decision,
            "judge_score": float((latest or {}).get("judge_score", 0.0) or 0.0),
            "judge_backend": _safe_text((latest or {}).get("judge_backend")),
            "judge_threshold": float((latest or {}).get("judge_threshold", 0.0) or 0.0),
            "is_override": bool(judge_decision) and judge_decision != _safe_text(label),
        }
        if extra:
            event["paper_metadata"] = dict(extra)
        return self._append(event)


__all__ = ["EVENT_SCHEMA", "PaperJudgeFeedbackLogger"]
