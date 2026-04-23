"""Central task-based LLM routing with explicit external-policy handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import requests

from knowledge_hub.application.runtime_diagnostics import provider_runtime_probe
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.providers import get_llm

TaskType = Literal[
    "title_short_summary",
    "translation",
    "paper_memory_extraction",
    "materialization_summary",
    "materialization_source_enrichment",
    "materialization_concept_enrichment",
    "learning_graph_refinement",
    "rag_query_planning",
    "rag_answer",
    "rag_answer_verification",
    "rag_answer_rewrite",
    "claim_extraction",
    "predicate_reasoning",
    "learning_reinforce",
]
RouteMode = Literal["fallback-only", "local", "mini", "strong", "auto"]
ResolvedRoute = Literal["fallback-only", "local", "mini", "strong"]

_TASK_OVERRIDE_ROUTE: dict[str, ResolvedRoute] = {
    "title_short_summary": "local",
    "translation": "mini",
    "paper_memory_extraction": "strong",
    "materialization_summary": "mini",
    "materialization_source_enrichment": "mini",
    "materialization_concept_enrichment": "strong",
    "learning_graph_refinement": "strong",
    "rag_query_planning": "local",
    "rag_answer_verification": "strong",
    "rag_answer_rewrite": "strong",
    "claim_extraction": "strong",
    "predicate_reasoning": "strong",
}


@dataclass
class TaskRouteDecision:
    task_type: TaskType
    route: ResolvedRoute
    provider: str
    model: str
    timeout_sec: int
    fallback_chain: list[ResolvedRoute]
    reasons: list[str]
    allow_external_effective: bool
    complexity_score: int
    policy_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskType": self.task_type,
            "route": self.route,
            "provider": self.provider,
            "model": self.model,
            "timeoutSec": self.timeout_sec,
            "fallbackChain": list(self.fallback_chain),
            "reasons": list(self.reasons),
            "allowExternal": self.allow_external_effective,
            "complexityScore": self.complexity_score,
            "policyMode": self.policy_mode,
        }


def _estimate_tokens(text: str) -> int:
    body = (text or "").strip()
    if not body:
        return 0
    return max(1, int(len(body) / 4))


def _has_reasoning_signal(query: str, *, context: str = "", source_count: int = 0) -> bool:
    body = f"{query or ''}\n{context or ''}".lower()
    if source_count >= 6:
        return True
    if len((query or "").strip()) >= 160:
        return True
    if len((context or "").strip()) >= 12000:
        return True
    keywords = (
        "why",
        "how",
        "compare",
        "tradeoff",
        "architecture",
        "strategy",
        "reasoning",
        "추론",
        "비교",
        "설계",
        "왜",
        "어떻게",
    )
    return any(token in body for token in keywords)


def _bucket_config(
    config: Config,
    bucket: ResolvedRoute,
) -> tuple[str, str, int]:
    if bucket in {"local", "mini", "strong"}:
        provider = str(
            config.get_nested("routing", "llm", "tasks", bucket, "provider", default="") or ""
        ).strip()
        model = str(
            config.get_nested("routing", "llm", "tasks", bucket, "model", default="") or ""
        ).strip()
        timeout = int(
            config.get_nested("routing", "llm", "tasks", bucket, "timeout_sec", default=45 if bucket == "local" else 60)
            or (45 if bucket == "local" else 60)
        )
        if provider == "ollama":
            model = _resolve_ollama_model(model)
        if provider and model:
            return provider, model, timeout

    if bucket == "local":
        provider = str(
            config.get_nested(
                "routing",
                "llm",
                "hybrid",
                "local",
                "provider",
                default="ollama",
            )
            or "ollama"
        ).strip()
        model = str(
            config.get_nested(
                "routing",
                "llm",
                "hybrid",
                "local",
                "model",
                default="qwen3:14b",
            )
            or "qwen3:14b"
        ).strip()
        model = _resolve_ollama_model(model)
        return provider, model, 45

    if bucket == "mini":
        provider = str(config.translation_provider or "openai").strip()
        model = str(config.translation_model or "gpt-5-mini").strip()
        return provider, model, 60

    provider = str(
        config.get_nested("learning", "llm", "escalation", "provider", default="openai") or "openai"
    ).strip()
    model = str(
        config.get_nested("learning", "llm", "escalation", "model", default="gpt-5.4") or "gpt-5.4"
    ).strip()
    return provider, model, 90


def _resolve_ollama_model(configured_model: str) -> str:
    preferred = [
        str(configured_model or "").strip(),
        "qwen3:14b",
        "exaone3.5:7.8b",
        "mistral:latest",
        "llama3:latest",
    ]
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        response.raise_for_status()
        data = response.json()
        installed = {
            str(item.get("name") or "").strip()
            for item in data.get("models", [])
            if str(item.get("name") or "").strip()
        }
    except Exception:
        return str(configured_model or "").strip()
    for candidate in preferred:
        if candidate and candidate in installed:
            return candidate
    return str(configured_model or "").strip()


def _fallback_chain(route: ResolvedRoute, *, allow_external: bool) -> list[ResolvedRoute]:
    if route == "fallback-only":
        return ["fallback-only"]
    if route == "local":
        return ["local", "mini"] if allow_external else ["local", "fallback-only"]
    if route == "mini":
        return ["mini", "fallback-only"] if allow_external else ["fallback-only"]
    if route == "strong":
        return ["strong", "mini", "fallback-only"] if allow_external else ["fallback-only"]
    return ["fallback-only"]


def decide_task_route(
    config: Config,
    *,
    task_type: TaskType,
    allow_external: bool,
    query: str = "",
    context: str = "",
    source_count: int = 0,
    force_route: RouteMode | None = None,
) -> TaskRouteDecision:
    reasons: list[str] = []
    allow_external_effective = bool(allow_external)
    policy_mode = "external-allowed" if allow_external_effective else "local-only"
    token_est = _estimate_tokens(query) + _estimate_tokens(context)
    complexity_score = token_est + max(0, source_count - 1) * 120
    if _has_reasoning_signal(query, context=context, source_count=source_count):
        complexity_score += 480
        reasons.append("reasoning_signal")

    forced = str(force_route or "").strip().lower()
    if forced:
        reasons.append(f"force_route={forced}")

    if task_type in _TASK_OVERRIDE_ROUTE:
        target_route: ResolvedRoute | str = _TASK_OVERRIDE_ROUTE[task_type]
        reasons.append(f"task_default={target_route}")
    else:
        default_route = str(
            config.get_nested("routing", "llm", "tasks", "defaults", task_type, default="auto") or "auto"
        ).strip().lower()
        target_route = default_route if default_route in {"fallback-only", "local", "mini", "strong", "auto"} else "auto"
        reasons.append(f"task_default={target_route}")

    if forced in {"fallback-only", "local", "mini", "strong"}:
        target_route = forced
    elif forced == "auto":
        reasons.append("force_route=auto-preserve-default")

    if target_route == "auto":
        if complexity_score < 1500 and len((context or "").strip()) < 6000:
            target_route = "local"
            reasons.append("auto_local")
        elif complexity_score < 3000:
            target_route = "mini"
            reasons.append("auto_mini")
        else:
            target_route = "strong"
            reasons.append("auto_strong")

    if target_route in {"mini", "strong"} and not allow_external_effective:
        reasons.append("external_disallowed")
        target_route = "local"

    if target_route == "fallback-only":
        return TaskRouteDecision(
            task_type=task_type,
            route="fallback-only",
            provider="",
            model="",
            timeout_sec=0,
            fallback_chain=["fallback-only"],
            reasons=reasons,
            allow_external_effective=allow_external_effective,
            complexity_score=complexity_score,
            policy_mode=policy_mode,
        )

    provider, model, timeout_sec = _bucket_config(config, target_route)
    return TaskRouteDecision(
        task_type=task_type,
        route=target_route,
        provider=provider,
        model=model,
        timeout_sec=timeout_sec,
        fallback_chain=_fallback_chain(target_route, allow_external=allow_external_effective),
        reasons=reasons,
        allow_external_effective=allow_external_effective,
        complexity_score=complexity_score,
        policy_mode=policy_mode,
    )


def get_llm_for_task(
    config: Config,
    *,
    task_type: TaskType,
    allow_external: bool,
    query: str = "",
    context: str = "",
    source_count: int = 0,
    force_route: RouteMode | None = None,
    timeout_sec: int | None = None,
) -> tuple[Any | None, TaskRouteDecision, list[str]]:
    decision = decide_task_route(
        config,
        task_type=task_type,
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=force_route,
    )
    warnings: list[str] = []
    if decision.route == "fallback-only":
        return None, decision, warnings

    chosen = decision
    for idx, route in enumerate(decision.fallback_chain):
        if route == "fallback-only":
            break
        provider, model, route_timeout = _bucket_config(config, route)
        runtime_probe = provider_runtime_probe(config, provider)
        if runtime_probe and not bool(runtime_probe.get("available", True)):
            summary = str(runtime_probe.get("summary") or runtime_probe.get("detail") or "runtime unavailable").strip()
            warnings.append(f"route unavailable ({provider}/{model}): {summary}")
            continue
        provider_cfg = dict(config.get_provider_config(provider))
        resolved_timeout = float(timeout_sec or route_timeout)
        provider_cfg["timeout"] = resolved_timeout
        provider_cfg["request_timeout"] = resolved_timeout
        try:
            llm = get_llm(provider, model=model, **provider_cfg)
            if idx > 0:
                chosen = TaskRouteDecision(
                    task_type=decision.task_type,
                    route=route,
                    provider=provider,
                    model=model,
                    timeout_sec=int(timeout_sec or route_timeout),
                    fallback_chain=list(decision.fallback_chain),
                    reasons=[*decision.reasons, f"fallback_from_{decision.route}"],
                    allow_external_effective=decision.allow_external_effective,
                    complexity_score=decision.complexity_score,
                    policy_mode=decision.policy_mode,
                )
            elif timeout_sec is not None and timeout_sec != chosen.timeout_sec:
                chosen = TaskRouteDecision(
                    task_type=decision.task_type,
                    route=decision.route,
                    provider=provider,
                    model=model,
                    timeout_sec=int(timeout_sec),
                    fallback_chain=list(decision.fallback_chain),
                    reasons=list(decision.reasons),
                    allow_external_effective=decision.allow_external_effective,
                    complexity_score=decision.complexity_score,
                    policy_mode=decision.policy_mode,
                )
            return llm, chosen, warnings
        except Exception as error:
            warnings.append(f"route unavailable ({provider}/{model}): {error}")

    chosen = TaskRouteDecision(
        task_type=decision.task_type,
        route="fallback-only",
        provider="",
        model="",
        timeout_sec=0,
        fallback_chain=list(decision.fallback_chain),
        reasons=[*decision.reasons, "all_routes_failed"],
        allow_external_effective=decision.allow_external_effective,
        complexity_score=decision.complexity_score,
        policy_mode=decision.policy_mode,
    )
    return None, chosen, warnings
