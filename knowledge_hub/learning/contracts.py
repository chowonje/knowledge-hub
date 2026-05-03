"""Learning-domain repository contracts."""

from __future__ import annotations

from typing import Protocol

from knowledge_hub.knowledge.contracts import (
    ClaimRepository,
    FeatureRepository,
    NoteRepository,
    OntologyRepository,
)


class LearningGraphDataRepository(OntologyRepository, FeatureRepository, NoteRepository, Protocol):
    """Repository contract for learning graph projection and ranking."""


class LearningServiceRepository(
    OntologyRepository,
    NoteRepository,
    FeatureRepository,
    ClaimRepository,
    Protocol,
):
    def list_learning_sessions(
        self,
        topic_slug: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]: ...
    def get_quality_mode_monthly_spend(self, month_key: str | None = None) -> float: ...
    def record_quality_mode_usage(
        self,
        item_kind: str,
        route: str,
        estimated_cost_usd: float,
        topic_slug: str = "",
    ) -> None: ...


__all__ = [
    "LearningGraphDataRepository",
    "LearningServiceRepository",
]
