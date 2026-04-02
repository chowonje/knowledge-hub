"""Shared paper CLI runtime and helper functions.

This module holds the lowest-risk shared helpers extracted from `paper_cmd.py`
without changing the facade command surface.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import click
import requests
from rich.console import Console

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.learning.task_router import TaskRouteDecision, get_llm_for_task
from knowledge_hub.papers.memory_payloads import card_payload
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.memory_runtime import build_paper_memory_builder
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService

console = Console()
log = logging.getLogger("khub.paper")

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

API_MAX_RETRIES = 3
API_RETRY_BASE_SEC = 2.0
MAX_SUMMARIZE_CHARS = 30000
_LOCAL_PROVIDER_NAMES = {"ollama", "pplx-local", "pplx-st"}


def _sqlite_db(config, *, khub=None):
    if khub is not None and hasattr(khub, "sqlite_db"):
        return khub.sqlite_db()
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    return SQLiteDatabase(config.sqlite_path)


def _vector_db(config, *, khub=None):
    if khub is not None and hasattr(khub, "vector_db"):
        return khub.vector_db()
    from knowledge_hub.infrastructure.persistence import VectorDatabase

    return VectorDatabase(config.vector_db_path, config.collection_name)


def _build_llm(config, provider: str, model: str | None = None, *, khub=None):
    if khub is not None and hasattr(khub, "build_llm"):
        return khub.build_llm(provider, model)
    from knowledge_hub.infrastructure.providers import get_llm

    return get_llm(provider, model=model, **config.get_provider_config(provider))


def _build_embedder(config, *, khub=None):
    if khub is not None and hasattr(khub, "build_embedder"):
        return khub.build_embedder(config.embedding_provider, config.embedding_model)
    from knowledge_hub.infrastructure.providers import get_embedder

    embed_cfg = config.get_provider_config(config.embedding_provider)
    return get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)


def _paper_summary_parser(config) -> str:
    return str(getattr(config, "paper_summary_parser", "") or config.get_nested("paper", "summary", "parser", default="auto"))


def _derive_user_card_from_payload(payload: dict) -> dict:
    evidence_summaries = dict(payload.get("evidenceSummaries") or {})
    claim_coverage = dict(payload.get("claimCoverage") or {})
    normalized_claims = int(claim_coverage.get("normalizedClaims") or claim_coverage.get("usedClaimHints") or 0)
    claim_coverage.setdefault("status", "good" if normalized_claims > 0 else "low")
    return {
        "schema": "knowledge-hub.paper-summary.user-card.result.v1",
        "status": str(payload.get("status") or "failed"),
        "paperId": str(payload.get("paperId") or ""),
        "paperTitle": str(payload.get("paperTitle") or ""),
        "parserUsed": str(payload.get("parserUsed") or ""),
        "fallbackUsed": bool(payload.get("fallbackUsed")),
        "llmRoute": str(payload.get("llmRoute") or ""),
        "summary": dict(payload.get("summary") or {}),
        "evidenceSummary": {
            "keyResults": list(((evidence_summaries.get("keyResults") or {}).get("summaryLines") or [])),
            "limitations": list(((evidence_summaries.get("limitations") or {}).get("summaryLines") or [])),
            "whatIsNew": list(((evidence_summaries.get("whatIsNew") or {}).get("summaryLines") or [])),
        },
        "evidenceMap": list(payload.get("evidenceMap") or []),
        "claimCoverage": claim_coverage,
        "warnings": list(payload.get("warnings") or []),
        "artifactPaths": {"summaryMdPath": str((payload.get("artifactPaths") or {}).get("summaryMdPath") or "")},
    }


def _load_user_card(payload: dict) -> dict:
    card_path = Path(str((payload.get("artifactPaths") or {}).get("cardJsonPath") or "")).expanduser()
    if card_path.exists():
        try:
            loaded = json.loads(card_path.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            claim_coverage = dict(loaded.get("claimCoverage") or {})
            normalized_claims = int(claim_coverage.get("normalizedClaims") or claim_coverage.get("usedClaimHints") or 0)
            if "status" not in claim_coverage:
                claim_coverage["status"] = "good" if normalized_claims > 0 else "low"
            loaded["claimCoverage"] = claim_coverage
            return loaded
    return _derive_user_card_from_payload(payload)


def _compact_evidence_summary(payload: dict, *, include_refs: bool = False) -> dict:
    evidence_summaries = dict(payload.get("evidenceSummaries") or {})
    out: dict[str, dict[str, object]] = {}
    for field, packet in evidence_summaries.items():
        summary_lines = [str(item) for item in list((packet or {}).get("summaryLines") or []) if str(item).strip()]
        entry: dict[str, object] = {
            "summaryLines": summary_lines,
            "claimHintsUsed": int((packet or {}).get("claimHintsUsed") or 0),
            "unitCount": int((packet or {}).get("unitCount") or 0),
        }
        if include_refs:
            entry["evidenceRefs"] = list((packet or {}).get("evidenceRefs") or [])
        out[str(field)] = entry
    return out


def _ensure_public_paper_summary(khub, paper_id: str) -> tuple[dict, dict]:
    service = StructuredPaperSummaryService(_sqlite_db(khub.config, khub=khub), khub.config)
    payload = service.load_artifact(paper_id=paper_id) or {}
    if not payload:
        payload = service.build(
            paper_id=paper_id,
            paper_parser=_paper_summary_parser(khub.config),
            refresh_parse=False,
            quick=False,
        )
    if str(payload.get("status") or "") == "blocked":
        detail = "; ".join(str(item) for item in (payload.get("warnings") or [])[:3]) or f"paper summary blocked: {paper_id}"
        raise click.ClickException(detail)
    return payload, _load_user_card(payload)


def _public_paper_memory(sqlite_db, *, config=None, paper_id: str) -> dict:
    retriever = PaperMemoryRetriever(sqlite_db)
    card = retriever.get(paper_id, include_refs=True)
    if card is not None:
        return dict(card)
    row = build_paper_memory_builder(sqlite_db, config=config).build_and_store(paper_id=paper_id)
    if row is None:
        return {}
    return card_payload(row)


def _public_related_knowledge(khub, *, paper_id: str, paper_title: str, top_k: int) -> dict:
    results = khub.searcher().search(
        paper_title or paper_id,
        top_k=max(3, int(top_k) * 2),
        retrieval_mode="hybrid",
        alpha=0.7,
        expand_parent_context=True,
    )
    grouped = {"papers": [], "concepts": [], "notes": [], "web": []}
    for result in results:
        source_type = str(result.metadata.get("source_type") or "")
        title = str(result.metadata.get("title") or "Untitled").strip()
        document_id = str(result.document_id or "")
        if document_id == f"paper:{paper_id}" or title == paper_title:
            continue
        item = {
            "title": title,
            "sourceType": source_type,
            "score": float(result.score),
            "documentId": document_id,
        }
        if source_type == "paper":
            grouped["papers"].append(item)
        elif source_type == "concept":
            grouped["concepts"].append(item)
        elif source_type == "vault":
            grouped["notes"].append(item)
        elif source_type == "web":
            grouped["web"].append(item)
    return {key: value[:top_k] for key, value in grouped.items()}


def _validate_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise click.BadParameter(
            f"유효하지 않은 arXiv ID: '{arxiv_id}' (예: 2501.06322)",
            param_hint="arxiv_id",
        )
    return arxiv_id


def _api_call_with_retry(fn, *args, **kwargs):
    last_err: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as e:
            last_err = e
            status = getattr(e.response, "status_code", 0)
            if status == 429 or status >= 500:
                wait = API_RETRY_BASE_SEC * (2 ** (attempt - 1))
                log.warning("API %d 에러, %d/%d 재시도 (%.1fs 대기)", status, attempt, API_MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            wait = API_RETRY_BASE_SEC * (2 ** (attempt - 1))
            log.warning("네트워크 오류, %d/%d 재시도 (%.1fs 대기)", attempt, API_MAX_RETRIES, wait)
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _provider_is_local(name: str) -> bool:
    token = str(name or "").strip().lower()
    if token in _LOCAL_PROVIDER_NAMES:
        return True
    try:
        from knowledge_hub.infrastructure.providers import get_provider_info

        info = get_provider_info(token)
        return bool(info and info.is_local)
    except Exception:
        return False


def _resolve_routed_llm(
    config,
    *,
    task_type: str,
    allow_external: bool,
    llm_mode: str = "auto",
    query: str = "",
    context: str = "",
    source_count: int = 0,
    provider_override: str | None = None,
    model_override: str | None = None,
    timeout_sec: int | None = None,
):
    if provider_override:
        provider = str(provider_override).strip()
        model = str(model_override or "").strip()
        if not _provider_is_local(provider) and not allow_external:
            raise click.ClickException("외부 프로바이더 사용은 --allow-external 이 필요합니다.")
        provider_cfg = dict(config.get_provider_config(provider))
        if timeout_sec is not None:
            provider_cfg.setdefault("timeout", float(timeout_sec))
            provider_cfg.setdefault("request_timeout", float(timeout_sec))
        from knowledge_hub.infrastructure.providers import get_llm

        llm = get_llm(provider, model=model or None, **provider_cfg)
        route = "local" if _provider_is_local(provider) else "mini"
        decision = TaskRouteDecision(
            task_type=task_type,  # type: ignore[arg-type]
            route=route,  # type: ignore[arg-type]
            provider=provider,
            model=model or getattr(llm, "model", ""),
            timeout_sec=int(timeout_sec or 0),
            fallback_chain=[route],  # type: ignore[list-item]
            reasons=["provider_override"],
            allow_external_effective=allow_external,
            complexity_score=0,
            policy_mode="external-allowed" if allow_external else "local-only",
        )
        return llm, decision, []

    force_route = str(llm_mode or "auto").strip().lower() or "auto"
    return get_llm_for_task(
        config,
        task_type=task_type,  # type: ignore[arg-type]
        allow_external=allow_external,
        query=query,
        context=context,
        source_count=source_count,
        force_route=force_route,  # type: ignore[arg-type]
        timeout_sec=timeout_sec,
    )


def _fallback_to_mini_llm(
    config,
    *,
    task_type: str,
    allow_external: bool,
    query: str,
    context: str,
):
    if not allow_external:
        return None, None, []
    return _resolve_routed_llm(
        config,
        task_type=task_type,
        allow_external=True,
        llm_mode="mini",
        query=query,
        context=context,
        source_count=1,
    )


def _resolve_vault_papers_dir(vault_path: str) -> Path | None:
    candidates = [
        Path(vault_path) / "AI" / "AI_Papers" / "Papers",
        Path(vault_path) / "AI" / "AI_Papers",
        Path(vault_path) / "Papers",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers" / "Papers",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers",
        Path(vault_path) / "papers",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(vault_path) / "Papers"


def _resolve_vault_concepts_dir(vault_path: str) -> Path:
    papers_dir = _resolve_vault_papers_dir(vault_path)
    if papers_dir:
        sibling_concepts = papers_dir.parent / "Concepts"
        if papers_dir.name.lower() == "papers" and sibling_concepts.exists():
            return sibling_concepts
        concepts = papers_dir / "Concepts"
        if concepts.exists():
            return concepts
    candidates = [
        Path(vault_path) / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Papers" / "Concepts",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Concepts",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (papers_dir or Path(vault_path) / "Papers") / "Concepts"


def _normalize_wikilink_target(raw: str) -> str:
    token = str(raw or "").strip()
    if not token:
        return ""
    token = token.split("|", 1)[0].strip()
    if token.endswith(".md"):
        token = token[:-3].strip()
    return token


def _extract_note_concepts(content: str) -> list[str]:
    body = str(content or "")
    sections: list[str] = []
    for heading in ("내가 배워야 할 개념", "관련 개념"):
        marker = body.find(heading)
        if marker < 0:
            continue
        section = body[marker:]
        next_heading = re.search(r"\n#{1,3}\s+", section[1:])
        if next_heading:
            section = section[: next_heading.start() + 1]
        sections.append(section)

    concepts: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for raw in re.findall(r"\[\[([^\]]+)\]\]", section):
            name = _normalize_wikilink_target(raw)
            lowered = name.lower()
            if (
                not name
                or lowered == "00_concept_index"
                or lowered == "00_concept_index.md"
                or "/" in name
                or lowered.startswith("learninghub/")
                or lowered.startswith("papers/")
                or lowered.startswith("projects/")
                or lowered.startswith("archives/")
            ):
                continue
            if lowered not in seen:
                seen.add(lowered)
                concepts.append(name)
    return concepts


def _collect_paper_text(paper: dict, config) -> str:
    translated = paper.get("translated_path")
    if translated and Path(translated).exists():
        text = Path(translated).read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    parsed_markdown = Path(config.papers_dir) / "parsed" / str(paper.get("arxiv_id") or "").strip() / "document.md"
    if parsed_markdown.exists():
        text = parsed_markdown.read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    text_path = paper.get("text_path")
    if text_path and Path(text_path).exists():
        text = Path(text_path).read_text(encoding="utf-8")
        if len(text) > 200:
            return text[:MAX_SUMMARIZE_CHARS]

    papers_dir = Path(config.papers_dir)
    for pattern in [f"*{paper['arxiv_id']}*.txt", f"*{paper['title'][:30]}*.txt"]:
        for path in papers_dir.glob(pattern):
            text = path.read_text(encoding="utf-8")
            if len(text) > 200:
                return text[:MAX_SUMMARIZE_CHARS]

    title = paper.get("title", "")
    authors = paper.get("authors", "")
    field = paper.get("field", "")
    notes = paper.get("notes", "")
    return f"제목: {title}\n저자: {authors}\n분야: {field}\n{notes}"


def _update_obsidian_summary(paper: dict, summary: str, config):
    if not getattr(config, "vault_path", None):
        return
    vault = Path(config.vault_path)
    safe_title = re.sub(r'[\\/:*?"<>|]', "", paper["title"]).strip()
    safe_title = re.sub(r"\s+", " ", safe_title)[:100].strip()

    papers_dir = _resolve_vault_papers_dir(str(vault))
    if papers_dir:
        note_path = papers_dir / f"{safe_title}.md"
        if not note_path.exists():
            return
        content = note_path.read_text(encoding="utf-8")
        placeholder = "요약본/번역본이 아직 등록되지 않았습니다"

        if placeholder in content:
            content = content.replace(placeholder, summary)
            note_path.write_text(content, encoding="utf-8")
            console.print(f"[dim]Obsidian 노트 업데이트: {note_path.name}[/dim]")
            return

        if "## 요약" in content:
            lines = content.split("\n")
            start = None
            end = None
            for index, line in enumerate(lines):
                if line.strip() == "## 요약":
                    start = index
                elif start is not None and line.startswith("## ") and index > start:
                    end = index
                    break
            if start is not None:
                if end is None:
                    end = len(lines)
                new_lines = lines[:start] + ["## 요약", "", summary, ""] + lines[end:]
                note_path.write_text("\n".join(new_lines), encoding="utf-8")
                console.print(f"[dim]Obsidian 노트 업데이트: {note_path.name}[/dim]")


def _validate_cli_payload(config, payload: dict, schema_id: str) -> None:
    strict = bool(config.get_nested("validation", "schema", "strict", default=False))
    result = annotate_schema_errors(payload, schema_id, strict=strict)
    if not result.ok:
        problems = ", ".join(result.errors[:5]) or "unknown schema validation error"
        raise click.ClickException(f"schema validation failed for {schema_id}: {problems}")


__all__ = [
    "MAX_SUMMARIZE_CHARS",
    "_api_call_with_retry",
    "_build_embedder",
    "_build_llm",
    "_collect_paper_text",
    "_compact_evidence_summary",
    "_derive_user_card_from_payload",
    "_ensure_public_paper_summary",
    "_extract_note_concepts",
    "_fallback_to_mini_llm",
    "_load_user_card",
    "_normalize_wikilink_target",
    "_paper_summary_parser",
    "_provider_is_local",
    "_public_paper_memory",
    "_public_related_knowledge",
    "_resolve_routed_llm",
    "_resolve_vault_concepts_dir",
    "_resolve_vault_papers_dir",
    "_sqlite_db",
    "_update_obsidian_summary",
    "_validate_arxiv_id",
    "_validate_cli_payload",
    "_vector_db",
]
