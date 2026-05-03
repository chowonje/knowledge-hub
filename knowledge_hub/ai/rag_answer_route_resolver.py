from __future__ import annotations

from typing import Any, Callable

from knowledge_hub.application.codex_backend import resolve_preferred_codex_backend
from knowledge_hub.application.runtime_diagnostics import provider_runtime_probe
from knowledge_hub.learning.task_router import decide_task_route, get_llm_for_task


def resolve_llm_for_request(
    *,
    config: Any,
    fixed_llm: Any,
    query: str,
    context: str,
    source_count: int,
    allow_external: bool,
    force_route: str | None = None,
    cached_local_llm: Any = None,
    cached_local_llm_signature: tuple[str, str, int] | None = None,
    get_llm_for_hybrid_routing_fn: Callable[..., tuple[Any, Any, list[str]]],
) -> tuple[Any, dict[str, Any], list[str], Any, tuple[str, str, int] | None]:
    next_cached_local_llm = cached_local_llm
    next_cached_local_llm_signature = cached_local_llm_signature

    codex_llm, codex_decision, codex_warnings = resolve_preferred_codex_backend(
        config=config,
        allow_external=allow_external,
        force_route=force_route,
        task_type="rag_answer",
    )
    if codex_decision is not None:
        return (
            codex_llm,
            codex_decision,
            codex_warnings,
            next_cached_local_llm,
            next_cached_local_llm_signature,
        )

    if not config:
        if not fixed_llm:
            raise ValueError("LLM이 설정되지 않았습니다")
        return (
            fixed_llm,
            {
                "route": "fixed",
                "provider": "",
                "model": "",
                "reasons": ["config_missing"],
                "fallbackUsed": False,
            },
            list(codex_warnings or []),
            next_cached_local_llm,
            next_cached_local_llm_signature,
        )

    llm, decision, warnings = get_llm_for_hybrid_routing_fn(
        config,
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=force_route,
    )
    warnings = [*codex_warnings, *list(warnings or [])]
    if llm is None:
        if fixed_llm is None:
            return (
                None,
                {
                    "route": "fallback-only",
                    "provider": "",
                    "model": "",
                    "reasons": ["routing_failed_no_available_llm"],
                    "fallbackUsed": True,
                },
                warnings,
                next_cached_local_llm,
                next_cached_local_llm_signature,
            )
        fixed_provider = str(getattr(config, "summarization_provider", "") or "").strip()
        fixed_probe = provider_runtime_probe(config, fixed_provider) if fixed_provider else {}
        if fixed_probe and not bool(fixed_probe.get("available", True)):
            unavailable_warnings = list(warnings)
            summary = str(fixed_probe.get("summary") or fixed_probe.get("detail") or "runtime unavailable").strip()
            unavailable_warnings.append(f"fixed llm unavailable ({fixed_provider}): {summary}")
            return (
                None,
                {
                    "route": "fallback-only",
                    "provider": "",
                    "model": "",
                    "reasons": ["routing_failed_no_available_llm"],
                    "fallbackUsed": True,
                },
                unavailable_warnings,
                next_cached_local_llm,
                next_cached_local_llm_signature,
            )
        return (
            fixed_llm,
            {
                "route": "fixed",
                "provider": "",
                "model": "",
                "reasons": ["routing_failed_fallback_to_fixed"],
                "fallbackUsed": True,
            },
            warnings,
            next_cached_local_llm,
            next_cached_local_llm_signature,
        )

    if decision.route == "local":
        local_signature = (
            str(decision.provider or "").strip(),
            str(decision.model or "").strip(),
            int(getattr(decision, "timeout_sec", 0) or 0),
        )
        if cached_local_llm is not None and cached_local_llm_signature == local_signature:
            llm = cached_local_llm
        else:
            next_cached_local_llm = llm
            next_cached_local_llm_signature = local_signature

    return llm, decision.to_dict(), warnings, next_cached_local_llm, next_cached_local_llm_signature


def resolve_llm_for_verification(
    *,
    config: Any,
    query: str,
    context: str,
    source_count: int,
    allow_external: bool,
    verification_route_stub_fn: Callable[..., dict[str, Any]],
) -> tuple[Any | None, dict[str, Any], list[str]]:
    if not config or not hasattr(config, "get_nested"):
        return None, verification_route_stub_fn(reason="config_missing", route="fallback-only"), [
            "answer verification fell back to heuristic: config missing"
        ]

    decision = decide_task_route(
        config,
        task_type="rag_answer_verification",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
    )
    route_meta = decision.to_dict()
    if decision.route == "local":
        return None, route_meta, [
            "answer verification fell back to heuristic: local verifier route is disabled by default"
        ]

    if decision.route not in {"mini", "strong"}:
        return None, route_meta, ["answer verification fell back to heuristic: local route selected"]

    llm, resolved, warnings = get_llm_for_task(
        config,
        task_type="rag_answer_verification",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
    )
    if llm is None:
        return None, resolved.to_dict(), [*warnings, "answer verification fell back to heuristic: verifier unavailable"]
    return llm, resolved.to_dict(), warnings


def resolve_llm_for_rewrite(
    *,
    config: Any,
    query: str,
    context: str,
    source_count: int,
    allow_external: bool,
    rewrite_route_stub_fn: Callable[..., dict[str, Any]],
) -> tuple[Any | None, dict[str, Any], list[str]]:
    if not config or not hasattr(config, "get_nested"):
        return None, rewrite_route_stub_fn(reason="config_missing", route="fallback-only"), [
            "answer rewrite skipped: config missing"
        ]

    decision = decide_task_route(
        config,
        task_type="rag_answer_rewrite",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
    )
    route_meta = decision.to_dict()
    if decision.route == "local":
        return None, route_meta, ["answer rewrite skipped: local rewrite route is disabled by default"]

    if decision.route not in {"local", "mini", "strong"}:
        return None, route_meta, ["answer rewrite skipped: fallback-only route selected"]

    llm, resolved, warnings = get_llm_for_task(
        config,
        task_type="rag_answer_rewrite",
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
    )
    if llm is None:
        return None, resolved.to_dict(), [*warnings, "answer rewrite skipped: rewrite model unavailable"]
    return llm, resolved.to_dict(), warnings
