"""Runtime helpers for routed paper-memory builder creation."""

from __future__ import annotations

from typing import Any

from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_extraction import PaperMemorySchemaExtractor


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def build_paper_memory_builder(
    sqlite_db: Any,
    *,
    config: Any | None = None,
    allow_external: bool | None = None,
    llm_mode: str | None = None,
    query: str = "",
    context: str = "",
    source_count: int = 1,
) -> PaperMemoryBuilder:
    """Build a paper-memory builder that prefers routed API extraction by default.

    Falls back to the deterministic builder when config/provider routing is unavailable.
    """

    if config is None or not hasattr(config, "get_provider_config"):
        return PaperMemoryBuilder(sqlite_db)

    mode = _clean_text(config.get_nested("paper", "memory", "extraction_mode", default="schema")).lower() or "schema"
    if mode not in {"deterministic", "shadow", "schema"}:
        mode = "schema"
    if mode == "deterministic":
        return PaperMemoryBuilder(sqlite_db)

    allow_external_effective = (
        bool(config.get_nested("paper", "memory", "allow_external", default=True))
        if allow_external is None
        else bool(allow_external)
    )
    forced = _clean_text(llm_mode).lower() or None
    if forced not in {"fallback-only", "local", "mini", "strong", "auto"}:
        forced = None
    timeout_sec = int(config.get_nested("paper", "memory", "extractor_timeout_sec", default=90) or 90)

    llm, decision, _warnings = get_llm_for_task(
        config,
        task_type="paper_memory_extraction",
        allow_external=allow_external_effective,
        query=query,
        context=context,
        source_count=max(1, int(source_count or 1)),
        force_route=forced,  # type: ignore[arg-type]
        timeout_sec=timeout_sec,
    )
    if llm is None:
        return PaperMemoryBuilder(sqlite_db)

    extractor = PaperMemorySchemaExtractor(llm, model=_clean_text(decision.model) or "gpt-5.4")
    return PaperMemoryBuilder(sqlite_db, schema_extractor=extractor, extraction_mode=mode)


__all__ = ["build_paper_memory_builder"]
