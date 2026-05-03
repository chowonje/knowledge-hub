from __future__ import annotations

from typing import Any

from knowledge_hub.application.context_pack import build_context_pack, render_context_pack
from knowledge_hub.core.sanitizer import redact_p0
from knowledge_hub.learning.policy import evaluate_policy_for_payload

from .registry import get_transformation, load_transformations


LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx_st", "pplx-st"}


def _source_preview(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_source_id": str(source.get("local_source_id") or ""),
        "normalized_source_type": str(source.get("normalized_source_type") or source.get("source_type") or ""),
        "title": str(source.get("title") or ""),
        "preview": str(source.get("snippet") or source.get("content") or "")[:280],
        "selection_reason": str(source.get("selection_reason") or ""),
    }


def _provider_allows_external(config) -> bool:
    provider = str(getattr(config, "summarization_provider", "") or "").strip().lower()
    return provider not in LOCAL_PROVIDER_NAMES


def list_transformations() -> dict[str, Any]:
    items = load_transformations()
    return {
        "schema": "knowledge-hub.transform.list.result.v1",
        "status": "ok",
        "count": len(items),
        "items": [
            {
                "id": item.id,
                "version": item.version,
                "title": item.title,
                "description": item.description,
                "declared_inputs": item.declared_inputs,
                "declared_output": item.declared_output,
            }
            for item in items
        ],
        "warnings": [],
    }


def preview_transformation(
    searcher,
    *,
    sqlite_db,
    transformation_id: str,
    query: str,
    repo_path: str | None = None,
    include_workspace: bool = False,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    max_sources: int = 6,
    max_chars: int = 4000,
) -> dict[str, Any]:
    definition = get_transformation(transformation_id)
    if definition is None:
        return {
            "schema": "knowledge-hub.transform.preview.result.v1",
            "status": "failed",
            "transformation": {"id": transformation_id, "version": "", "title": transformation_id},
            "query": query,
            "selected_sources": [],
            "prompt_preview": "",
            "warnings": [f"unknown transformation: {transformation_id}"],
        }

    pack = build_context_pack(
        searcher,
        sqlite_db=sqlite_db,
        query_or_topic=query,
        target="transform",
        repo_path=repo_path,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_items=max_sources,
        max_chars=max_chars,
        max_knowledge_hits=max_sources,
        max_workspace_files=min(4, max_sources),
    )
    prompt_preview = definition.prompt_template.format(
        query=query,
        transformation_title=definition.title,
        source_count=len(pack.get("sources") or []),
    )
    return {
        "schema": "knowledge-hub.transform.preview.result.v1",
        "status": "ok",
        "transformation": {
            "id": definition.id,
            "version": definition.version,
            "title": definition.title,
            "description": definition.description,
        },
        "query": query,
        "context_pack": pack,
        "selected_sources": [_source_preview(item) for item in pack.get("sources") or []],
        "prompt_preview": prompt_preview,
        "warnings": list(pack.get("warnings") or []),
    }


def run_transformation(
    searcher,
    *,
    sqlite_db,
    llm,
    config,
    transformation_id: str,
    query: str,
    repo_path: str | None = None,
    include_workspace: bool = False,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    max_sources: int = 6,
    max_chars: int = 4000,
    dry_run: bool = False,
) -> dict[str, Any]:
    preview = preview_transformation(
        searcher,
        sqlite_db=sqlite_db,
        transformation_id=transformation_id,
        query=query,
        repo_path=repo_path,
        include_workspace=include_workspace,
        include_vault=include_vault,
        include_papers=include_papers,
        include_web=include_web,
        max_sources=max_sources,
        max_chars=max_chars,
    )
    if preview["status"] != "ok":
        preview["schema"] = "knowledge-hub.transform.run.result.v1"
        return preview

    pack = dict(preview.get("context_pack") or {})
    context_text = render_context_pack(pack, include_workspace=include_workspace)
    prompt = str(preview.get("prompt_preview") or "")
    allow_external = _provider_allows_external(config)
    policy = evaluate_policy_for_payload(
        allow_external=allow_external,
        raw_texts=[prompt, context_text],
        mode="transform",
    )

    effective_prompt = prompt
    effective_context = context_text
    redacted = False
    warnings = list(preview.get("warnings") or [])
    if allow_external and not policy.allowed and policy.classification == "P0":
        effective_prompt = redact_p0(prompt)
        effective_context = redact_p0(context_text)
        policy = evaluate_policy_for_payload(
            allow_external=allow_external,
            raw_texts=[effective_prompt, effective_context],
            mode="transform",
        )
        redacted = True
        warnings.append("transformation context was redacted before external execution")

    output = ""
    status = "ok"
    if not policy.allowed:
        status = "blocked"
        warnings.extend(list(policy.warnings or []))
    elif not dry_run and llm is not None:
        output = llm.generate(effective_prompt, effective_context)

    return {
        "schema": "knowledge-hub.transform.run.result.v1",
        "status": status,
        "transformation": preview["transformation"],
        "query": query,
        "selected_sources": preview["selected_sources"],
        "output": output,
        "dry_run": bool(dry_run),
        "redacted": redacted,
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
