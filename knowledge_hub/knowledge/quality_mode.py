"""Quality-first operating mode helpers.

Enforces narrow-topic, cost-capped routing above the generic task router.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from knowledge_hub.infrastructure.config import Config


QualityItemKind = Literal["source", "concept", "claim", "learning"]


@dataclass
class QualityModeDecision:
    topic: str | None
    is_core_topic: bool
    allow_external: bool
    llm_mode: str
    estimated_cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
    usage_key: str | None = None


def _normalize_topic(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9\s-]+", " ", raw)
    raw = re.sub(r"\s+", "-", raw)
    raw = re.sub(r"-+", "-", raw)
    return raw.strip("-")


def core_topics(config: Config) -> set[str]:
    values = config.get_nested("quality_mode", "core_topics", default=[]) or []
    return {_normalize_topic(value) for value in values if str(value).strip()}


def infer_quality_topic(config: Config, explicit_topic: str | None = None, *parts: Any) -> str | None:
    normalized = _normalize_topic(explicit_topic)
    if normalized:
        return normalized
    from knowledge_hub.knowledge.features import topic_matches_text

    for topic in sorted(core_topics(config)):
        if topic_matches_text(topic.replace("-", " "), *parts):
            return topic
    return None


def _allowed_external_route(config: Config, item_kind: QualityItemKind) -> str:
    path_map = {
        "source": ("quality_mode", "routing", "source_external_route"),
        "concept": ("quality_mode", "routing", "concept_external_route"),
        "claim": ("quality_mode", "routing", "claim_external_route"),
        "learning": ("quality_mode", "routing", "learning_external_route"),
    }
    return str(config.get_nested(*path_map[item_kind], default="mini" if item_kind == "source" else "strong") or "").strip() or (
        "mini" if item_kind == "source" else "strong"
    )


def _usage_key_for(item_kind: QualityItemKind, llm_mode: str) -> str | None:
    if item_kind == "source" and llm_mode == "mini":
        return "mini_source_external_used"
    if item_kind == "concept" and llm_mode == "strong":
        return "strong_concept_external_used"
    if item_kind == "claim" and llm_mode == "strong":
        return "strong_claim_external_used"
    if item_kind == "learning" and llm_mode == "strong":
        return "strong_learning_external_used"
    return None


def _cap_key_for(item_kind: QualityItemKind, llm_mode: str) -> str | None:
    if item_kind == "source" and llm_mode == "mini":
        return "mini_max_source_items_per_run"
    if item_kind == "concept" and llm_mode == "strong":
        return "strong_max_concept_items_per_run"
    if item_kind == "claim" and llm_mode == "strong":
        return "strong_max_claim_refinements_per_run"
    if item_kind == "learning" and llm_mode == "strong":
        return "strong_max_learning_refinements_per_run"
    return None


def estimate_quality_mode_cost(config: Config, item_kind: QualityItemKind, llm_mode: str) -> float:
    if llm_mode == "mini" and item_kind == "source":
        return float(
            config.get_nested("quality_mode", "cost_estimates", "mini_source_item_usd", default=0.01) or 0.0
        )
    if llm_mode == "strong" and item_kind == "concept":
        return float(
            config.get_nested("quality_mode", "cost_estimates", "strong_concept_item_usd", default=0.05) or 0.0
        )
    if llm_mode == "strong" and item_kind == "claim":
        return float(
            config.get_nested("quality_mode", "cost_estimates", "strong_claim_item_usd", default=0.03) or 0.0
        )
    if llm_mode == "strong" and item_kind == "learning":
        return float(
            config.get_nested("quality_mode", "cost_estimates", "strong_learning_item_usd", default=0.03) or 0.0
        )
    return 0.0


def resolve_quality_mode_route(
    config: Config,
    *,
    item_kind: QualityItemKind,
    requested_allow_external: bool,
    requested_mode: str,
    topic: str | None,
    counters: dict[str, int] | None = None,
    monthly_spend_usd: float | None = None,
) -> QualityModeDecision:
    mode = str(requested_mode or "auto").strip() or "auto"
    normalized_topic = _normalize_topic(topic)
    core_topic_set = core_topics(config)
    is_core = bool(normalized_topic and normalized_topic in core_topic_set)

    if not bool(config.get_nested("quality_mode", "enabled", default=True)):
        usage_key = _usage_key_for(item_kind, mode) if requested_allow_external else None
        return QualityModeDecision(
            topic=normalized_topic or None,
            is_core_topic=is_core,
            allow_external=bool(requested_allow_external),
            llm_mode=mode,
            estimated_cost_usd=estimate_quality_mode_cost(config, item_kind, mode),
            usage_key=usage_key,
        )

    if not requested_allow_external:
        return QualityModeDecision(
            topic=normalized_topic or None,
            is_core_topic=is_core,
            allow_external=False,
            llm_mode=mode if mode in {"local", "fallback-only"} else "local",
            estimated_cost_usd=0.0,
        )

    warnings: list[str] = []
    if not is_core and not bool(
        config.get_nested("quality_mode", "routing", "non_core_external_allowed", default=False)
    ):
        warnings.append("quality_mode_non_core_external_blocked")
        return QualityModeDecision(
            topic=normalized_topic or None,
            is_core_topic=False,
            allow_external=False,
            llm_mode="local" if mode != "fallback-only" else "fallback-only",
            estimated_cost_usd=0.0,
            warnings=warnings,
        )

    if mode in {"local", "fallback-only"}:
        return QualityModeDecision(
            topic=normalized_topic or None,
            is_core_topic=is_core,
            allow_external=False,
            llm_mode=mode,
            estimated_cost_usd=0.0,
        )

    allowed_mode = _allowed_external_route(config, item_kind)
    if mode not in {"", "auto", allowed_mode}:
        warnings.append(f"quality_mode_route_overridden:{mode}->{allowed_mode}")
    effective_mode = allowed_mode
    usage_key = _usage_key_for(item_kind, effective_mode)
    cap_key = _cap_key_for(item_kind, effective_mode)
    estimated_cost_usd = estimate_quality_mode_cost(config, item_kind, effective_mode)
    if counters is not None and usage_key and cap_key:
        cap = int(config.get_nested("quality_mode", "caps", cap_key, default=0) or 0)
        if cap > 0 and int(counters.get(usage_key, 0)) >= cap:
            warnings.append(f"quality_mode_cap_exceeded:{cap_key}")
            return QualityModeDecision(
                topic=normalized_topic or None,
                is_core_topic=is_core,
                allow_external=False,
                llm_mode="local",
                estimated_cost_usd=0.0,
                warnings=warnings,
            )
    monthly_cap = float(config.get_nested("quality_mode", "caps", "monthly_usd_cap", default=0.0) or 0.0)
    if monthly_cap > 0 and monthly_spend_usd is not None and (float(monthly_spend_usd) + estimated_cost_usd) > monthly_cap:
        warnings.append("quality_mode_budget_cap_exceeded")
        return QualityModeDecision(
            topic=normalized_topic or None,
            is_core_topic=is_core,
            allow_external=False,
            llm_mode="local",
            estimated_cost_usd=0.0,
            warnings=warnings,
        )

    return QualityModeDecision(
        topic=normalized_topic or None,
        is_core_topic=is_core,
        allow_external=True,
        llm_mode=effective_mode,
        estimated_cost_usd=estimated_cost_usd,
        warnings=warnings,
        usage_key=usage_key,
    )
