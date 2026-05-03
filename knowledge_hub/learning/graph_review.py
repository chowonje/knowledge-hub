"""Pending review helpers for learning graph items."""

from __future__ import annotations

from typing import Any


class LearningGraphReview:
    def __init__(self, store):
        self.store = store

    def list_pending(self, topic: str | None = None, item_type: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        return self.store.list_pending(topic_slug=topic, item_type=item_type, limit=limit)

    def apply(self, pending_id: int) -> dict[str, Any] | None:
        return self.store.get_pending(pending_id)

    def reject(self, pending_id: int) -> dict[str, Any] | None:
        return self.store.get_pending(pending_id)
