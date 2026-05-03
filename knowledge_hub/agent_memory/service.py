from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _terms(text: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "how",
        "why",
        "when",
        "where",
        "have",
        "will",
        "into",
        "about",
        "there",
        "their",
        "agent",
        "memory",
        "project",
        "turn",
        "then",
        "보다",
        "이것",
        "저것",
        "그리고",
        "에서",
        "으로",
        "하기",
        "하는",
        "대한",
        "관련",
        "지금",
        "그냥",
    }
    tokens = []
    for part in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").casefold()):
        if len(part) < 2 or part in stopwords:
            continue
        tokens.append(part)
    return tokens


def _top_themes(turns: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for turn in turns:
        for token in _terms(turn.get("content")):
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    return [token for token, _ in ranked[: max(1, int(limit))]]


def _score_overlap(query: str, text: str) -> float:
    q_terms = _terms(query)
    haystack = " ".join(_terms(text))
    if not q_terms or not haystack:
        return 0.0
    score = 0.0
    for term in q_terms:
        if term in haystack:
            score += 1.0
    joined = " ".join(q_terms)
    if joined and joined in haystack:
        score += 1.5
    return round(score, 3)


class EpisodeMemoryService:
    def __init__(self, *, store_path: str | Path):
        self.store_path = Path(store_path)

    def _load_store(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"schema": "knowledge-hub.agent-memory.store.v1", "sessions": {}}
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema": "knowledge-hub.agent-memory.store.v1", "sessions": {}}

    def _save_store(self, payload: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_episode(self, session_id: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        clean_turns = []
        for item in turns:
            role = _clean_text(item.get("role") or item.get("speaker") or "user")
            content = _clean_text(item.get("content") or item.get("text") or "")
            if not content:
                continue
            clean_turns.append({"role": role or "user", "content": content})
        if not clean_turns:
            raise ValueError("episode turns are empty")
        themes = _top_themes(clean_turns)
        user_turns = [item["content"] for item in clean_turns if item["role"] == "user"]
        assistant_turns = [item["content"] for item in clean_turns if item["role"] != "user"]
        semantic_summary = " / ".join((user_turns + assistant_turns)[:3])[:480]
        episode_summary = " ".join(item["content"] for item in clean_turns[:6])[:800]
        episode_id = f"{session_id}:{len(clean_turns)}:{len(themes)}"
        record = {
            "episodeId": episode_id,
            "sessionId": session_id,
            "createdAt": _utc_now(),
            "turnCount": len(clean_turns),
            "themes": themes,
            "semanticSummary": semantic_summary,
            "episodeSummary": episode_summary,
            "turns": clean_turns,
        }
        store = self._load_store()
        sessions = dict(store.get("sessions") or {})
        session_records = list(sessions.get(session_id) or [])
        session_records.append(record)
        sessions[session_id] = session_records
        store["sessions"] = sessions
        self._save_store(store)
        return record

    def search_episode_memory(self, query: str, session_id: str, *, depth: str = "theme") -> list[dict[str, Any]]:
        store = self._load_store()
        records = list((store.get("sessions") or {}).get(session_id) or [])
        scored: list[dict[str, Any]] = []
        for item in records:
            if depth == "theme":
                surface = " ".join(str(token) for token in list(item.get("themes") or []))
            elif depth == "semantic":
                surface = str(item.get("semanticSummary") or "")
            else:
                surface = str(item.get("episodeSummary") or "")
            score = _score_overlap(query, surface)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "episodeId": str(item.get("episodeId") or ""),
                    "score": score,
                    "depth": depth,
                    "summary": surface[:240],
                    "themes": list(item.get("themes") or []),
                    "createdAt": str(item.get("createdAt") or ""),
                }
            )
        scored.sort(key=lambda item: (float(item.get("score") or 0.0), str(item.get("createdAt") or "")), reverse=True)
        return scored[:10]

    def explain_episode_route(self, query: str, session_id: str) -> dict[str, Any]:
        theme_hits = self.search_episode_memory(query, session_id, depth="theme")
        if theme_hits:
            return {"primaryDepth": "theme", "fallbackDepths": ["semantic", "episode"], "reason": "theme_overlap"}
        semantic_hits = self.search_episode_memory(query, session_id, depth="semantic")
        if semantic_hits:
            return {"primaryDepth": "semantic", "fallbackDepths": ["episode"], "reason": "semantic_overlap"}
        return {"primaryDepth": "episode", "fallbackDepths": [], "reason": "episode_scan"}
