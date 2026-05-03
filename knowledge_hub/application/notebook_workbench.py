"""Bounded workbench search/chat helpers."""

from __future__ import annotations

import re
from typing import Any

from knowledge_hub.application.claim_signals import build_claim_signal_payload
from knowledge_hub.application.context_pack import build_context_pack, render_context_pack
from knowledge_hub.ai.retrieval_fit import normalize_source_type
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload

CONTEXT_MODE_FULL = "full"
CONTEXT_MODE_SUMMARY = "summary"
CONTEXT_MODE_EXCLUDED = "excluded"
CONTEXT_MODES = {CONTEXT_MODE_FULL, CONTEXT_MODE_SUMMARY, CONTEXT_MODE_EXCLUDED}


def _filter_selected_sources(sources: list[dict[str, Any]], selected_source_ids: list[str] | None) -> list[dict[str, Any]]:
    selected = {str(item).strip() for item in selected_source_ids or [] if str(item).strip()}
    if not selected:
        return list(sources)
    return [item for item in sources if str(item.get("local_source_id") or "") in selected]


def _normalize_context_mode(value: Any, *, default: str = CONTEXT_MODE_FULL) -> str:
    mode = str(value or "").strip().lower()
    if mode in CONTEXT_MODES:
        return mode
    return default if default in CONTEXT_MODES else CONTEXT_MODE_FULL


def _source_context_mode(source: dict[str, Any], context_modes: dict[str, Any] | None = None) -> str:
    modes = context_modes or {}
    for key in (
        str(source.get("local_source_id") or "").strip(),
        str(source.get("stable_scope_id") or "").strip(),
        str(source.get("document_scope_id") or "").strip(),
        str(source.get("section_scope_id") or "").strip(),
        str(source.get("file_path") or "").strip(),
        str(source.get("source_url") or "").strip(),
        str(source.get("title") or "").strip(),
    ):
        if key and key in modes:
            return _normalize_context_mode(modes.get(key))
    default_mode = CONTEXT_MODE_SUMMARY if str(source.get("normalized_source_type") or "") == "summary" else CONTEXT_MODE_FULL
    return _normalize_context_mode(source.get("context_mode"), default=default_mode)


def _apply_context_modes(
    sources: list[dict[str, Any]],
    context_modes: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_sources: list[dict[str, Any]] = []
    excluded_sources: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        mode = _source_context_mode(item, context_modes)
        item["context_mode"] = mode
        item["context_mode_reason"] = "explicit" if mode != CONTEXT_MODE_FULL or item.get("context_mode") else "default"
        if mode == CONTEXT_MODE_EXCLUDED:
            excluded_sources.append(item)
            continue
        if mode == CONTEXT_MODE_SUMMARY:
            item["content"] = str(item.get("snippet") or item.get("content") or "").strip()
            item["selection_reason"] = f"{item.get('selection_reason') or 'selected_source'}:summary"
        active_sources.append(item)
    return active_sources, excluded_sources


def _context_mode_lookup(sources: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for source in sources:
        mode = str(source.get("context_mode") or CONTEXT_MODE_FULL)
        for key in (
            str(source.get("local_source_id") or "").strip(),
            str(source.get("stable_scope_id") or "").strip(),
            str(source.get("document_scope_id") or "").strip(),
            str(source.get("section_scope_id") or "").strip(),
            str(source.get("file_path") or "").strip(),
            str(source.get("source_url") or "").strip(),
            str(source.get("title") or "").strip(),
        ):
            if key:
                lookup.setdefault(key, mode)
    return lookup


def _result_context_mode(result: Any, context_mode_lookup: dict[str, str]) -> str:
    metadata = dict(getattr(result, "metadata", {}) or {})
    for key in (
        str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or "").strip(),
        str(metadata.get("stable_scope_id") or "").strip(),
        str(metadata.get("document_scope_id") or metadata.get("document_id") or "").strip(),
        str(metadata.get("section_scope_id") or "").strip(),
        str(metadata.get("file_path") or "").strip(),
        str(metadata.get("url") or "").strip(),
        str(metadata.get("title") or "").strip(),
    ):
        if key and key in context_mode_lookup:
            return context_mode_lookup[key]
    return CONTEXT_MODE_FULL


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_가-힣]{2,}", str(text or ""))}


def _rank_local_sources(query: str, sources: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    tokens = _tokenize(query)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for source in sources:
        title_tokens = _tokenize(str(source.get("title") or ""))
        content_tokens = _tokenize(str(source.get("content") or source.get("snippet") or ""))
        score = float(len(tokens & title_tokens) * 3 + len(tokens & content_tokens))
        ranked.append((score, source))
    ranked.sort(key=lambda item: (item[0], str(item[1].get("title") or "")), reverse=True)
    return [item for _, item in ranked[: max(1, int(top_k or 5))]]


def _scope_signature(source: dict[str, Any]) -> dict[str, str]:
    metadata = dict(source.get("metadata") or {})
    source_type = str(
        normalize_source_type(source.get("normalized_source_type") or metadata.get("source_type") or "")
        or ""
    ).strip()
    explicit_stable_scope_id = str(metadata.get("stable_scope_id") or "").strip()
    explicit_document_scope_id = str(metadata.get("document_scope_id") or metadata.get("document_id") or "").strip()
    explicit_section_scope_id = str(metadata.get("section_scope_id") or "").strip()
    resolved_parent_id = str(
        metadata.get("resolved_parent_id")
        or metadata.get("parent_id")
        or ""
    ).strip()
    document_scope_id = str(
        explicit_document_scope_id
        or source.get("file_path")
        or metadata.get("file_path")
        or ""
    ).strip()
    section_scope_id = str(explicit_section_scope_id or resolved_parent_id or "").strip()
    stable_scope_id = str(explicit_stable_scope_id or section_scope_id or document_scope_id or resolved_parent_id or "").strip()
    scope_level = str(metadata.get("scope_level") or metadata.get("parent_type") or "").strip()
    arxiv_id = str(metadata.get("arxiv_id") or source.get("local_source_id") or "").strip()
    file_path = str(source.get("file_path") or metadata.get("file_path") or "").strip()
    source_url = str(source.get("source_url") or metadata.get("url") or "").strip()
    return {
        "source_type": source_type,
        "scope_level": scope_level,
        "stable_scope_id": stable_scope_id,
        "explicit_stable_scope_id": explicit_stable_scope_id,
        "document_scope_id": document_scope_id,
        "explicit_document_scope_id": explicit_document_scope_id,
        "section_scope_id": section_scope_id,
        "explicit_section_scope_id": explicit_section_scope_id,
        "parent_id": resolved_parent_id,
        "arxiv_id": arxiv_id if source_type == "paper" else "",
        "file_path": file_path,
        "source_url": source_url,
    }


def _scope_filter_for_source(source: dict[str, Any]) -> dict[str, Any] | None:
    signature = _scope_signature(source)
    for key in (
        "explicit_stable_scope_id",
        "parent_id",
        "explicit_section_scope_id",
        "explicit_document_scope_id",
        "arxiv_id",
        "file_path",
        "source_url",
    ):
        value = signature.get(key, "")
        if not value:
            continue
        metadata_key = "url" if key == "source_url" else key.removeprefix("explicit_")
        return {metadata_key: value}
    return None


def _scope_filter_key(filter_dict: dict[str, Any] | None) -> str:
    if not filter_dict:
        return ""
    for key in ("stable_scope_id", "parent_id", "section_scope_id", "document_scope_id", "arxiv_id", "file_path", "url"):
        value = str(filter_dict.get(key) or "").strip()
        if value:
            return key
    return ""


def _result_matches_scope(result: Any, source: dict[str, Any]) -> bool:
    metadata = dict(getattr(result, "metadata", {}) or {})
    signature = _scope_signature(source)
    checks = (
        ("parent_id", str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or "").strip()),
        ("stable_scope_id", str(metadata.get("stable_scope_id") or "").strip()),
        ("section_scope_id", str(metadata.get("section_scope_id") or "").strip()),
        ("document_scope_id", str(metadata.get("document_scope_id") or metadata.get("document_id") or "").strip()),
        ("arxiv_id", str(metadata.get("arxiv_id") or "").strip()),
        ("file_path", str(metadata.get("file_path") or "").strip()),
        ("source_url", str(metadata.get("url") or "").strip()),
    )
    for key, actual in checks:
        expected = signature.get(key, "")
        if expected:
            return actual == expected
    return False


def _result_scope_id(result: Any) -> str:
    metadata = dict(getattr(result, "metadata", {}) or {})
    return str(
        metadata.get("resolved_parent_id")
        or metadata.get("parent_id")
        or getattr(result, "document_id", "")
        or metadata.get("title")
        or ""
    ).strip()


def _serialize_search_result(
    result: Any,
    *,
    context_mode: str = CONTEXT_MODE_FULL,
    claim_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    extras = dict(getattr(result, "lexical_extras", {}) or {})
    return {
        "local_source_id": _result_scope_id(result),
        "title": str(metadata.get("title") or metadata.get("resolved_parent_label") or "Untitled"),
        "normalized_source_type": str(
            normalize_source_type(metadata.get("source_type") or "") or metadata.get("source_type") or ""
        ),
        "scope_level": str(metadata.get("scope_level") or metadata.get("parent_type") or ""),
        "stable_scope_id": str(metadata.get("stable_scope_id") or ""),
        "document_scope_id": str(metadata.get("document_scope_id") or metadata.get("document_id") or ""),
        "section_scope_id": str(metadata.get("section_scope_id") or ""),
        "preview": str(getattr(result, "document", "") or "")[:280],
        "selection_reason": "scoped_retrieval_hit",
        "context_mode": context_mode,
        "score": round(float(getattr(result, "score", 0.0) or 0.0), 6),
        "top_ranking_signals": list(extras.get("top_ranking_signals") or []),
        "retrieval_mode": str(getattr(result, "retrieval_mode", "") or ""),
        "file_path": str(metadata.get("file_path") or ""),
        "source_url": str(metadata.get("url") or ""),
        "parent_id": str(metadata.get("resolved_parent_id") or metadata.get("parent_id") or ""),
        "claimSignals": dict(claim_signals or {}),
        "claim_signals": dict(claim_signals or {}),
    }


def _dedupe_results(results: list[Any], *, top_k: int) -> list[Any]:
    deduped: dict[str, Any] = {}
    for item in results:
        key = _result_scope_id(item)
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or float(getattr(item, "score", 0.0) or 0.0) > float(getattr(existing, "score", 0.0) or 0.0):
            deduped[key] = item
    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            float(getattr(item, "score", 0.0) or 0.0),
            str((getattr(item, "metadata", {}) or {}).get("title") or ""),
        ),
        reverse=True,
    )
    return ordered[: max(1, int(top_k or 5))]


def _scoped_retrieval(
    searcher: Any,
    *,
    query: str,
    scope_sources: list[dict[str, Any]],
    top_k: int,
    mode: str,
    alpha: float,
) -> tuple[list[Any], dict[str, Any]]:
    search = getattr(searcher, "search", None)
    if not callable(search) or not scope_sources:
        return [], {
            "strategy": "bounded_local_fallback",
            "filters_applied": 0,
            "fallback_sources": len(scope_sources),
            "warnings": ["scoped retrieval unavailable: searcher.search missing or scope empty"],
        }

    hits: list[Any] = []
    filters_applied = 0
    fallback_sources = 0
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    per_scope_top_k = max(3, int(top_k or 5))

    for source in scope_sources:
        filter_dict = _scope_filter_for_source(source)
        diagnostic = {
            "local_source_id": str(source.get("local_source_id") or ""),
            "title": str(source.get("title") or ""),
            "normalized_source_type": str(source.get("normalized_source_type") or ""),
            "context_mode": str(source.get("context_mode") or CONTEXT_MODE_FULL),
            "scoped": False,
            "filter_key": _scope_filter_key(filter_dict),
            "filter_value_present": bool(filter_dict),
            "stable_scope_id": str(_scope_signature(source).get("stable_scope_id") or ""),
            "document_scope_id": str(_scope_signature(source).get("document_scope_id") or ""),
            "section_scope_id": str(_scope_signature(source).get("section_scope_id") or ""),
            "fallback_reason": "",
            "matched_results": 0,
        }
        if not filter_dict:
            fallback_sources += 1
            diagnostic["fallback_reason"] = "missing_stable_filter"
            diagnostics.append(diagnostic)
            warnings.append(
                f"scoped retrieval fallback: no stable metadata filter for {source.get('title') or source.get('local_source_id')}"
            )
            continue
        try:
            scoped = search(
                query,
                top_k=per_scope_top_k,
                source_type="all",
                retrieval_mode=mode,
                alpha=alpha,
                expand_parent_context=True,
                metadata_filter=filter_dict,
            )
        except Exception as error:
            fallback_sources += 1
            diagnostic["fallback_reason"] = "search_failed"
            diagnostics.append(diagnostic)
            warnings.append(
                f"scoped retrieval fallback: search failed for {source.get('title') or source.get('local_source_id')}: {error}"
            )
            continue
        filters_applied += 1
        diagnostic["scoped"] = True
        matched_results = 0
        for item in scoped or []:
            if _result_matches_scope(item, source):
                hits.append(item)
                matched_results += 1
        diagnostic["matched_results"] = matched_results
        if matched_results == 0:
            diagnostic["fallback_reason"] = "zero_match"
            warnings.append(
                f"scoped retrieval zero-match for {source.get('title') or source.get('local_source_id')}"
            )
        diagnostics.append(diagnostic)

    strategy = "scoped_search"
    if not hits or fallback_sources:
        strategy = "hybrid_scoped_with_local_fallback" if hits else "bounded_local_fallback"

    return _dedupe_results(hits, top_k=top_k), {
        "strategy": strategy,
        "filters_applied": filters_applied,
        "fallback_sources": fallback_sources,
        "scope_diagnostics": diagnostics,
        "warnings": warnings,
    }


def _resolve_workbench_scope(
    searcher: Any,
    *,
    sqlite_db: Any,
    topic: str | None,
    query: str,
    selected_source_ids: list[str] | None,
    selected_source_context_modes: dict[str, Any] | None,
    include_vault: bool,
    include_papers: bool,
    include_web: bool,
    top_k: int,
    mode: str,
    alpha: float,
) -> dict[str, Any]:
    scope_seed = str(topic or query or "").strip()
    pack = build_context_pack(
        searcher,
        sqlite_db=sqlite_db,
        query_or_topic=scope_seed,
        target="workbench",
        include_workspace=False,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_items=max(6, top_k * 2),
        max_knowledge_hits=max(6, top_k * 2),
        max_chars=2000,
    )
    warnings = list(pack.get("warnings") or [])
    persistent_sources = list(pack.get("persistent_sources") or [])
    selected = _filter_selected_sources(persistent_sources, selected_source_ids)
    if selected_source_ids and not selected:
        warnings.append("selected_source_ids matched no persistent workbench sources")

    active_sources, excluded_sources = _apply_context_modes(selected, selected_source_context_modes)
    if excluded_sources:
        warnings.append(f"{len(excluded_sources)} selected workbench sources were excluded by context mode")

    scoped_results, scoped_meta = _scoped_retrieval(
        searcher,
        query=query,
        scope_sources=active_sources,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
    )
    warnings.extend(list(scoped_meta.get("warnings") or []))

    if scoped_meta.get("strategy") in {"bounded_local_fallback", "hybrid_scoped_with_local_fallback"}:
        fallback_ranked = _rank_local_sources(query, active_sources, top_k=top_k)
    else:
        fallback_ranked = []

    return {
        "pack": pack,
        "scope_sources": active_sources,
        "excluded_scope_sources": excluded_sources,
        "scoped_results": scoped_results,
        "fallback_ranked_sources": fallback_ranked,
        "strategy": str(scoped_meta.get("strategy") or "bounded_local_fallback"),
        "filters_applied": int(scoped_meta.get("filters_applied") or 0),
        "fallback_sources": int(scoped_meta.get("fallback_sources") or 0),
        "scope_diagnostics": list(scoped_meta.get("scope_diagnostics") or []),
        "excluded_scope_diagnostics": [
            {
                "local_source_id": str(source.get("local_source_id") or ""),
                "title": str(source.get("title") or ""),
                "context_mode": str(source.get("context_mode") or CONTEXT_MODE_EXCLUDED),
                "filter_key": "",
                "filter_value_present": False,
                "stable_scope_id": str(source.get("stable_scope_id") or ""),
                "document_scope_id": str(source.get("document_scope_id") or ""),
                "section_scope_id": str(source.get("section_scope_id") or ""),
                "fallback_reason": "context_mode_excluded",
                "matched_results": 0,
            }
            for source in excluded_sources
        ],
        "warnings": warnings,
    }


def notebook_workbench_search(
    searcher,
    *,
    sqlite_db,
    topic: str | None,
    query: str,
    selected_source_ids: list[str] | None = None,
    selected_source_context_modes: dict[str, Any] | None = None,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    top_k: int = 5,
    mode: str = "hybrid",
    alpha: float = 0.7,
) -> dict[str, Any]:
    resolved = _resolve_workbench_scope(
        searcher,
        sqlite_db=sqlite_db,
        topic=topic,
        query=query,
        selected_source_ids=selected_source_ids,
        selected_source_context_modes=selected_source_context_modes,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
    )
    scope_sources = list(resolved.get("scope_sources") or [])
    excluded_scope_sources = list(resolved.get("excluded_scope_sources") or [])
    scoped_results = list(resolved.get("scoped_results") or [])
    fallback_ranked = list(resolved.get("fallback_ranked_sources") or [])
    strategy = str(resolved.get("strategy") or "bounded_local_fallback")
    warnings = list(resolved.get("warnings") or [])
    context_mode_lookup = _context_mode_lookup(scope_sources)
    claim_payload = build_claim_signal_payload(sqlite_db, scoped_results or fallback_ranked)
    claim_entries = list(claim_payload.get("items") or [])

    if scoped_results:
        results = [
            _serialize_search_result(
                item,
                context_mode=_result_context_mode(item, context_mode_lookup),
                claim_signals=claim_signal,
            )
            for item, claim_signal in zip(scoped_results, claim_entries or [{} for _ in scoped_results], strict=False)
        ]
    else:
        results = [
            {
                "local_source_id": str(item.get("local_source_id") or ""),
                "title": str(item.get("title") or ""),
                "normalized_source_type": str(item.get("normalized_source_type") or ""),
                "scope_level": str(item.get("scope_level") or ""),
                "stable_scope_id": str(item.get("stable_scope_id") or ""),
                "document_scope_id": str(item.get("document_scope_id") or ""),
                "section_scope_id": str(item.get("section_scope_id") or ""),
                "preview": str(item.get("snippet") or item.get("content") or "")[:280],
                "context_mode": str(item.get("context_mode") or CONTEXT_MODE_FULL),
                "selection_reason": "bounded_local_fallback",
                "claimSignals": dict(claim_signal),
                "claim_signals": dict(claim_signal),
            }
            for item, claim_signal in zip(fallback_ranked, claim_entries or [{} for _ in fallback_ranked], strict=False)
        ]
    return {
        "schema": "knowledge-hub.workbench.search.result.v1",
        "status": "ok",
        "topic": str(topic or ""),
        "query": query,
        "mode": mode,
        "alpha": alpha,
        "selected_scope_count": len(scope_sources),
        "excluded_scope_count": len(excluded_scope_sources),
        "retrieval_strategy": strategy,
        "filters_applied": int(resolved.get("filters_applied") or 0),
        "fallback_source_count": int(resolved.get("fallback_sources") or 0),
        "scope_diagnostics": list(resolved.get("scope_diagnostics") or []),
        "excluded_scope_diagnostics": list(resolved.get("excluded_scope_diagnostics") or []),
        "claimSignals": dict(claim_payload.get("summary") or {}),
        "claim_signals": dict(claim_payload.get("summary") or {}),
        "results": results,
        "warnings": warnings,
    }


def notebook_workbench_chat(
    searcher,
    *,
    sqlite_db,
    config,
    topic: str | None,
    message: str,
    intent: str = "qa",
    selected_source_ids: list[str] | None = None,
    selected_source_context_modes: dict[str, Any] | None = None,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    top_k: int = 5,
    mode: str = "hybrid",
    alpha: float = 0.7,
) -> dict[str, Any]:
    resolved = _resolve_workbench_scope(
        searcher,
        sqlite_db=sqlite_db,
        topic=topic,
        query=message,
        selected_source_ids=selected_source_ids,
        selected_source_context_modes=selected_source_context_modes,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        top_k=top_k,
        mode=mode,
        alpha=alpha,
    )
    scoped_results = list(resolved.get("scoped_results") or [])
    excluded_scope_sources = list(resolved.get("excluded_scope_sources") or [])
    fallback_ranked = list(resolved.get("fallback_ranked_sources") or [])
    warnings = list(resolved.get("warnings") or [])
    strategy = str(resolved.get("strategy") or "bounded_local_fallback")
    context_mode_lookup = _context_mode_lookup(list(resolved.get("scope_sources") or []))
    claim_payload = build_claim_signal_payload(sqlite_db, scoped_results or fallback_ranked)
    claim_entries = list(claim_payload.get("items") or [])

    search_payload = {
        "results": (
            [
                _serialize_search_result(
                    item,
                    context_mode=_result_context_mode(item, context_mode_lookup),
                    claim_signals=claim_signal,
                )
                for item, claim_signal in zip(scoped_results, claim_entries or [{} for _ in scoped_results], strict=False)
            ]
            if scoped_results
            else [
                {
                    "local_source_id": str(item.get("local_source_id") or ""),
                    "title": str(item.get("title") or ""),
                    "normalized_source_type": str(item.get("normalized_source_type") or ""),
                    "scope_level": str(item.get("scope_level") or ""),
                    "stable_scope_id": str(item.get("stable_scope_id") or ""),
                    "document_scope_id": str(item.get("document_scope_id") or ""),
                    "section_scope_id": str(item.get("section_scope_id") or ""),
                    "preview": str(item.get("snippet") or item.get("content") or "")[:280],
                    "context_mode": str(item.get("context_mode") or CONTEXT_MODE_FULL),
                    "selection_reason": "bounded_local_fallback",
                    "claimSignals": dict(claim_signal),
                    "claim_signals": dict(claim_signal),
                }
                for item, claim_signal in zip(fallback_ranked, claim_entries or [{} for _ in fallback_ranked], strict=False)
            ]
        )
    }

    if scoped_results:
        context_sources = [
            {
                "normalized_source_type": item["normalized_source_type"],
                "title": item["title"],
                "scope_level": item.get("scope_level", ""),
                "stable_scope_id": item.get("stable_scope_id", ""),
                "document_scope_id": item.get("document_scope_id", ""),
                "section_scope_id": item.get("section_scope_id", ""),
                "content": str(result.document or ""),
            }
            for item, result in zip(search_payload["results"], scoped_results, strict=False)
        ]
    else:
        context_sources = fallback_ranked

    context_text = render_context_pack({"persistent_sources": context_sources}, include_workspace=False)
    allow_external = str(getattr(config, "summarization_provider", "") or "").strip().lower() not in {"ollama", "pplx-local", "pplx_st", "pplx-st"}
    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=[message, context_text],
        mode="workbench",
    )
    prompt = f"Intent: {intent}\nQuestion: {message}\nUse only the provided bounded sources."
    if allow_external and not policy.allowed and policy.classification == "P0":
        prompt = redact_p0(prompt)
        context_text = redact_p0(context_text)
        policy = evaluate_policy_for_payload(allow_external=allow_external, raw_texts=[prompt, context_text], mode="workbench")

    answer = ""
    status = "ok"
    if not policy.allowed:
        status = "blocked"
    elif searcher.llm is not None:
        answer = searcher.llm.generate(prompt, context_text)

    return {
        "schema": "knowledge-hub.workbench.chat.result.v1",
        "status": status,
        "topic": str(topic or ""),
        "intent": intent,
        "message": message,
        "answer": answer,
        "selected_scope_count": len(list(resolved.get("scope_sources") or [])),
        "excluded_scope_count": len(excluded_scope_sources),
        "retrieval_strategy": strategy,
        "filters_applied": int(resolved.get("filters_applied") or 0),
        "fallback_source_count": int(resolved.get("fallback_sources") or 0),
        "scope_diagnostics": list(resolved.get("scope_diagnostics") or []),
        "excluded_scope_diagnostics": list(resolved.get("excluded_scope_diagnostics") or []),
        "claimSignals": dict(claim_payload.get("summary") or {}),
        "claim_signals": dict(claim_payload.get("summary") or {}),
        "sources": search_payload.get("results", []),
        "policy": {
            "allowed": bool(policy.allowed),
            "classification": policy.classification,
            "rule": policy.rule,
            "trace_id": policy.trace_id,
            "warnings": list(policy.warnings or []),
            "blocked_reason": policy.blocked_reason,
        },
        "warnings": warnings,
    }
