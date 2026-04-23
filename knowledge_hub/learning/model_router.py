"""Model routing helpers for selective Opus escalation and hybrid local/API usage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.infrastructure.providers import get_llm


@dataclass
class RoutingDecision:
    escalated: bool
    provider: str
    model: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "escalated": self.escalated,
            "provider": self.provider,
            "model": self.model,
            "reasons": self.reasons,
        }


@dataclass
class HybridRoutingDecision:
    route: str  # local | api
    provider: str
    model: str
    reasons: list[str]
    timeout_sec: int = 0
    complexity_score: int = 0
    threshold: int = 0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "provider": self.provider,
            "model": self.model,
            "reasons": self.reasons,
            "timeoutSec": self.timeout_sec,
            "complexityScore": self.complexity_score,
            "threshold": self.threshold,
            "fallbackUsed": self.fallback_used,
        }


def _rag_answer_local_timeout_override(config: Config, current_timeout_sec: int) -> int:
    configured = int(
        config.get_nested(
            "routing",
            "llm",
            "tasks",
            "rag_answer_local_timeout_sec",
            default=max(int(current_timeout_sec or 0), 120),
        )
        or max(int(current_timeout_sec or 0), 120)
    )
    return max(1, configured)


def _resolve_rag_answer_task(
    config: Config,
    *,
    allow_external: bool,
    query: str,
    context: str,
    source_count: int,
    force_route: str | None,
) -> tuple[Any | None, Any, list[str]]:
    forced = None
    if force_route == "local":
        forced = "local"
    elif force_route == "api":
        forced = "mini"
    llm, task_decision, warnings = get_llm_for_task(
        config,
        task_type="rag_answer",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=forced,
    )
    if task_decision.route != "local":
        return llm, task_decision, warnings

    override_timeout_sec = _rag_answer_local_timeout_override(config, int(task_decision.timeout_sec or 0))
    if override_timeout_sec == int(task_decision.timeout_sec or 0):
        return llm, task_decision, warnings

    llm, task_decision, override_warnings = get_llm_for_task(
        config,
        task_type="rag_answer",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=forced,
        timeout_sec=override_timeout_sec,
    )
    return llm, task_decision, override_warnings


def decide_escalation(
    config: Config,
    allow_external: bool,
    *,
    avg_gap_confidence: float = 1.0,
    normalization_failure_rate: float = 0.0,
    contains_essay: bool = False,
    verify_retry_failed: bool = False,
) -> RoutingDecision:
    base_provider = config.summarization_provider
    base_model = config.summarization_model
    reasons: list[str] = []
    if not allow_external:
        return RoutingDecision(False, base_provider, base_model, reasons)

    enabled = bool(config.get_nested("learning", "llm", "escalation", "enabled", default=True))
    if not enabled:
        return RoutingDecision(False, base_provider, base_model, reasons)

    low_conf_threshold = float(
        config.get_nested(
            "learning",
            "llm",
            "escalation",
            "trigger",
            "low_confidence_threshold",
            default=0.7,
        )
    )
    norm_fail_threshold = float(
        config.get_nested(
            "learning",
            "llm",
            "escalation",
            "trigger",
            "normalization_failure_threshold",
            default=0.3,
        )
    )
    allow_verify_retry_trigger = bool(
        config.get_nested(
            "learning",
            "llm",
            "escalation",
            "trigger",
            "verify_retry_failed",
            default=True,
        )
    )

    if avg_gap_confidence < low_conf_threshold:
        reasons.append("low_gap_confidence")
    if normalization_failure_rate > norm_fail_threshold:
        reasons.append("high_normalization_failure")
    if contains_essay:
        reasons.append("essay_grading")
    if allow_verify_retry_trigger and verify_retry_failed:
        reasons.append("verify_retry_failed")

    if not reasons:
        return RoutingDecision(False, base_provider, base_model, reasons)

    provider = str(config.get_nested("learning", "llm", "escalation", "provider", default="anthropic") or "anthropic")
    model = str(
        config.get_nested(
            "learning",
            "llm",
            "escalation",
            "model",
            default="claude-opus-4-20250514",
        )
        or "claude-opus-4-20250514"
    )
    return RoutingDecision(True, provider, model, reasons)


def get_llm_with_routing(
    config: Config,
    allow_external: bool,
    *,
    avg_gap_confidence: float = 1.0,
    normalization_failure_rate: float = 0.0,
    contains_essay: bool = False,
    verify_retry_failed: bool = False,
) -> tuple[Any | None, RoutingDecision, list[str]]:
    decision = decide_escalation(
        config,
        allow_external,
        avg_gap_confidence=avg_gap_confidence,
        normalization_failure_rate=normalization_failure_rate,
        contains_essay=contains_essay,
        verify_retry_failed=verify_retry_failed,
    )
    warnings: list[str] = []

    if not allow_external or not decision.escalated:
        return None, decision, warnings

    provider_cfg = config.get_provider_config(decision.provider)
    try:
        llm = get_llm(decision.provider, model=decision.model, **provider_cfg)
        return llm, decision, warnings
    except Exception as error:
        warnings.append(f"routed model unavailable: {error}")

    base_provider = config.summarization_provider
    base_model = config.summarization_model
    base_cfg = config.get_provider_config(base_provider)
    fallback_decision = RoutingDecision(False, base_provider, base_model, decision.reasons + ["fallback_to_base"])
    try:
        llm = get_llm(base_provider, model=base_model, **base_cfg)
        return llm, fallback_decision, warnings
    except Exception as error:
        warnings.append(f"base model unavailable: {error}")
        return None, fallback_decision, warnings


def _estimate_tokens(text: str) -> int:
    body = (text or "").strip()
    if not body:
        return 0
    # 빠른 근사치: 한글/영문 혼합 텍스트에서 1 token ~= 3.5~4 chars
    return max(1, int(len(body) / 4))


def _has_reasoning_signal(query: str) -> bool:
    body = (query or "").lower()
    if not body:
        return False
    keywords = (
        "why",
        "how",
        "tradeoff",
        "architecture",
        "reason",
        "multi-step",
        "compare",
        "proof",
        "설계",
        "비교",
        "왜",
        "어떻게",
        "추론",
        "전략",
    )
    if any(token in body for token in keywords):
        return True
    # 질문 길이가 길고 문장 구분이 많으면 복잡 질의로 간주
    sentence_markers = len(re.findall(r"[?.!]", body))
    return sentence_markers >= 2 and len(body) >= 120


def decide_hybrid_routing(
    config: Config,
    allow_external: bool,
    *,
    query: str,
    context: str = "",
    source_count: int = 0,
    force_route: str | None = None,
) -> HybridRoutingDecision:
    llm, task_decision, warnings = _resolve_rag_answer_task(
        config,
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=force_route,
    )
    _ = llm
    reasons = list(task_decision.reasons)
    reasons.append("simple_task" if task_decision.route == "local" else "complex_task")
    return HybridRoutingDecision(
        route="local" if task_decision.route == "local" else "api",
        provider=task_decision.provider,
        model=task_decision.model,
        reasons=[*reasons, *warnings],
        timeout_sec=task_decision.timeout_sec,
        complexity_score=task_decision.complexity_score,
        threshold=1500 if task_decision.route == "local" else 3000,
    )


def get_llm_for_hybrid_routing(
    config: Config,
    allow_external: bool,
    *,
    query: str,
    context: str = "",
    source_count: int = 0,
    force_route: str | None = None,
) -> tuple[Any | None, HybridRoutingDecision, list[str]]:
    llm, task_decision, warnings = _resolve_rag_answer_task(
        config,
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=force_route,
    )
    decision = HybridRoutingDecision(
        route="local" if task_decision.route == "local" else "api",
        provider=task_decision.provider,
        model=task_decision.model,
        reasons=[*task_decision.reasons, "simple_task" if task_decision.route == "local" else "complex_task"],
        timeout_sec=task_decision.timeout_sec,
        complexity_score=task_decision.complexity_score,
        threshold=1500 if task_decision.route == "local" else 3000,
        fallback_used="fallback_from_" in " ".join(task_decision.reasons),
    )
    return llm, decision, warnings
