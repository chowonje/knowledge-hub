"""Shared bounded context selection for task, transform, notebook, and workbench flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any

from knowledge_hub.core.chunking import snippet_for_path


CONTEXT_PACK_SCHEMA = "knowledge-hub.context-pack.result.v1"
ALLOWED_WORKSPACE_SUFFIXES = {
    ".json",
    ".js",
    ".jsx",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
EXCLUDED_WORKSPACE_DIRS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "venv",
}
EXCLUDED_WORKSPACE_DIR_PATTERN = re.compile(r"^(?:\.?venv|env|virtualenv)(?:$|[\d._-].*)$", re.IGNORECASE)
EXCLUDED_WORKSPACE_PATH_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "dist-packages",
    "vendor",
    "third_party",
}
PRIORITY_PROJECT_DOCS = (
    "AGENTS.md",
    "README.md",
    "docs/PROJECT_STATE.md",
    "docs/ARCHITECTURE.md",
)
TASK_MODE_KEYWORDS = {
    "debug": {
        "bug",
        "debug",
        "error",
        "exception",
        "fail",
        "failure",
        "failing",
        "fix",
        "regression",
        "traceback",
        "문제",
        "버그",
        "오류",
        "실패",
    },
    "design": {
        "architecture",
        "design",
        "interface",
        "migration",
        "pattern",
        "plan",
        "refactor",
        "spec",
        "구조",
        "리팩터",
        "설계",
    },
    "coding": {
        "change",
        "code",
        "command",
        "implement",
        "patch",
        "script",
        "test",
        "update",
        "write",
        "함수",
        "코드",
        "구현",
        "수정",
        "테스트",
    },
}

DEFAULT_MAX_VAULT_SOURCES = 6
DEFAULT_MAX_PAPER_SOURCES = 4
DEFAULT_MAX_WEB_SOURCES = 4
DEFAULT_MAX_SOURCE_CHARS = 6000


def normalize_source_type(source_type: str | None) -> str:
    source = str(source_type or "").strip().lower()
    if source in {"", "all", "*"}:
        return ""
    if source == "note":
        return "vault"
    if source in {"repo", "repository", "workspace"}:
        return "project"
    if source == "youtube":
        return "web"
    return source


@dataclass(frozen=True)
class WorkspaceCandidate:
    path: Path
    relative_path: str
    role: str
    reason: str
    score: int
    snippet: str


def _truncate_text(text: str, limit: int) -> str:
    body = str(text or "").strip()
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 1)].rstrip() + "…"


def classify_task_mode(goal: str) -> str:
    text = str(goal or "").strip().lower()
    if not text:
        return "knowledge"
    explicit_paths = _extract_explicit_path_mentions(text)
    if explicit_paths:
        return "coding"
    for mode in ("debug", "design", "coding"):
        keywords = TASK_MODE_KEYWORDS[mode]
        if any(token in text for token in keywords):
            return mode
    return "knowledge"


def build_context_pack(
    searcher: Any | None,
    *,
    sqlite_db: Any | None = None,
    query_or_topic: str,
    target: str,
    repo_path: str | None = None,
    include_workspace: bool = True,
    include_vault: bool = True,
    include_papers: bool = True,
    include_web: bool = True,
    max_items: int | None = None,
    max_chars: int | None = None,
    max_workspace_files: int = 8,
    max_project_docs: int = 6,
    max_knowledge_hits: int = 5,
    max_vault_sources: int = DEFAULT_MAX_VAULT_SOURCES,
    max_paper_sources: int = DEFAULT_MAX_PAPER_SOURCES,
    max_web_sources: int = DEFAULT_MAX_WEB_SOURCES,
) -> dict[str, Any]:
    text = str(query_or_topic or "").strip()
    resolved_target = str(target or "task").strip().lower() or "task"
    max_chars = max(200, int(max_chars or DEFAULT_MAX_SOURCE_CHARS))
    warnings: list[str] = []
    resolved_repo_path = _resolve_repo_path(repo_path) if include_workspace else None

    persistent_sources: list[dict[str, Any]]
    workspace_sources: list[dict[str, Any]] = []
    mode = classify_task_mode(text)

    if resolved_target == "notebook":
        if sqlite_db is None:
            raise ValueError("sqlite_db is required for notebook context packs")
        persistent_sources = _collect_notebook_sources(
            sqlite_db,
            topic=text,
            include_vault=include_vault,
            include_papers=include_papers,
            include_web=include_web,
            max_vault_sources=max_vault_sources,
            max_paper_sources=max_paper_sources,
            max_web_sources=max_web_sources,
            max_source_chars=max_chars,
        )
        if not persistent_sources:
            warnings.append("no local topic sources matched the query")
    else:
        persistent_sources = _collect_search_sources(
            searcher,
            goal=text,
            include_vault=include_vault,
            include_papers=include_papers,
            include_web=include_web,
            max_knowledge_hits=max_items or max_knowledge_hits,
            max_chars=min(800, max_chars),
        )
        if not persistent_sources:
            warnings.append("knowledge retrieval returned no matching persistent evidence")

        if include_workspace and mode in {"coding", "design", "debug"}:
            if resolved_repo_path is None:
                warnings.append("workspace context skipped: repo_path unavailable")
            elif not resolved_repo_path.exists():
                warnings.append(f"workspace context skipped: repo_path not found: {resolved_repo_path}")
            elif not resolved_repo_path.is_dir():
                warnings.append(f"workspace context skipped: repo_path is not a directory: {resolved_repo_path}")
            else:
                workspace_payload = _collect_workspace_context(
                    goal=text,
                    repo_path=resolved_repo_path,
                    max_workspace_files=max_workspace_files,
                    max_project_docs=max_project_docs,
                    max_excerpt_chars=max_chars,
                )
                workspace_sources = workspace_payload["workspace_files"]
                warnings.extend(workspace_payload["warnings"])
        elif include_workspace and mode == "knowledge":
            warnings.append("workspace context skipped: knowledge mode defaults to persistent sources only")

    pack = {
        "schema": CONTEXT_PACK_SCHEMA,
        "status": "ok",
        "target": resolved_target,
        "query": text,
        "mode": mode,
        "repo_path": str(resolved_repo_path) if resolved_repo_path else "",
        "persistent_sources": persistent_sources,
        "workspace_sources": workspace_sources,
        "sources": [*persistent_sources, *workspace_sources],
        "counts": {
            "persistent": len(persistent_sources),
            "workspace": len(workspace_sources),
            "total": len(persistent_sources) + len(workspace_sources),
        },
        "budget": {
            "max_items": max_items or max_knowledge_hits,
            "max_chars": max_chars,
            "repo_context_ephemeral": True,
        },
        "warnings": warnings,
    }

    if resolved_target == "notebook":
        notebook_title = f"Topic: {text}"
        notebook_description = _topic_description(text, persistent_sources)
        summary_block = _build_notebook_summary(text, notebook_description, persistent_sources)
        pack["notebook"] = {
            "topic": text,
            "topic_slug": slugify_topic(text),
            "title": notebook_title,
            "description": notebook_description,
            "summary_block": summary_block,
        }
    else:
        conventions = _extract_project_conventions(
            [
                WorkspaceCandidate(
                    path=Path(item["path"]),
                    relative_path=item["relative_path"],
                    role=item["role"],
                    reason=item["reason"],
                    score=0,
                    snippet=item["snippet"],
                )
                for item in workspace_sources
            ]
        )
        pack["project_conventions"] = conventions
        pack["prompt_context"] = _build_prompt_context(
            goal=text,
            mode=mode,
            persistent_sources=persistent_sources,
            workspace_sources=workspace_sources,
            project_conventions=conventions,
            warnings=warnings,
        )
    return pack


def render_context_pack(pack: dict[str, Any], *, include_workspace: bool = True) -> str:
    lines: list[str] = []
    for source in pack.get("persistent_sources") or []:
        header = f"[{source.get('normalized_source_type', source.get('source_type', 'source'))}] {source.get('title', 'Untitled')}"
        lines.extend([f"## {header}", str(source.get("content") or source.get("snippet") or "").strip(), ""])
    if include_workspace:
        for source in pack.get("workspace_sources") or []:
            lines.extend(
                [
                    f"## [project] {source.get('relative_path') or source.get('title') or 'workspace'}",
                    str(source.get("snippet") or "").strip(),
                    "",
                ]
            )
    return "\n".join(line for line in lines if line is not None).strip()


def slugify_topic(topic: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "topic"


def _resolve_repo_path(repo_path: str | None) -> Path | None:
    raw = str(repo_path or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    try:
        return Path.cwd().resolve()
    except Exception:
        return None


def _extract_goal_tokens(goal: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_./\\-]+|[가-힣]{2,}", str(goal or ""))
    normalized: set[str] = set()
    for token in tokens:
        lowered = token.strip().lower()
        if not lowered:
            continue
        normalized.add(lowered)
        for part in re.split(r"[/_.\\-]+", lowered):
            if len(part) >= 2:
                normalized.add(part)
    return normalized


def _extract_explicit_path_mentions(goal: str) -> set[str]:
    tokens = _extract_goal_tokens(goal)
    matches: set[str] = set()
    for token in tokens:
        if "/" in token or "\\" in token:
            matches.add(token.replace("\\", "/").strip("/"))
            continue
        if token in {name.lower() for name in PRIORITY_PROJECT_DOCS}:
            matches.add(token)
            continue
        if any(token.endswith(suffix) for suffix in ALLOWED_WORKSPACE_SUFFIXES):
            matches.add(token)
    return matches


def _allowed_source_types(include_vault: bool, include_papers: bool, include_web: bool) -> set[str]:
    allowed: set[str] = set()
    if include_vault:
        allowed.update({"note", "vault", "concept", "person", "organization", "event"})
    if include_papers:
        allowed.add("paper")
    if include_web:
        allowed.add("web")
    return allowed


def _collect_search_sources(
    searcher: Any | None,
    *,
    goal: str,
    include_vault: bool,
    include_papers: bool,
    include_web: bool,
    max_knowledge_hits: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if max_knowledge_hits <= 0 or searcher is None:
        return []
    allowed = _allowed_source_types(include_vault, include_papers, include_web)
    if not allowed:
        return []
    search = getattr(searcher, "search", None)
    if not callable(search):
        return []
    try:
        raw_results = search(
            goal,
            top_k=max(max_knowledge_hits * 3, max_knowledge_hits),
            source_type="all",
            retrieval_mode="hybrid",
            alpha=0.7,
            expand_parent_context=True,
        )
    except Exception:
        return []

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_results or []:
        metadata = dict(getattr(item, "metadata", {}) or {})
        source = normalize_source_type(metadata.get("source_type")) or "vault"
        if source not in allowed:
            continue
        title = str(metadata.get("title", "") or metadata.get("resolved_parent_label", "") or "Untitled").strip()
        snippet = _truncate_text(str(getattr(item, "document", "") or ""), max_chars)
        key = (source, title or snippet[:80])
        if key in seen:
            continue
        seen.add(key)
        stable_scope_id = str(metadata.get("stable_scope_id") or "").strip()
        document_scope_id = str(metadata.get("document_scope_id") or metadata.get("document_id") or "").strip()
        section_scope_id = str(metadata.get("section_scope_id") or "").strip()
        scope_level = str(metadata.get("scope_level") or metadata.get("parent_type") or "").strip()
        sources.append(
            {
                "local_source_id": str(
                    stable_scope_id
                    or section_scope_id
                    or document_scope_id
                    or metadata.get("resolved_parent_id")
                    or metadata.get("document_id")
                    or title
                    or snippet[:40]
                ),
                "source_type": str(metadata.get("source_type") or source),
                "normalized_source_type": source,
                "title": title or "Untitled",
                "snippet": snippet,
                "content": snippet,
                "score": round(float(getattr(item, "score", 0.0) or 0.0), 6),
                "file_path": str(metadata.get("file_path") or ""),
                "source_url": str(metadata.get("url") or ""),
                "scope_level": scope_level,
                "stable_scope_id": stable_scope_id,
                "document_scope_id": document_scope_id,
                "section_scope_id": section_scope_id,
                "metadata": metadata,
                "ephemeral": False,
                "selection_reason": "retrieval_hit",
            }
        )
        if len(sources) >= max_knowledge_hits:
            break
    return sources


def _collect_workspace_context(
    *,
    goal: str,
    repo_path: Path,
    max_workspace_files: int,
    max_project_docs: int,
    max_excerpt_chars: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    max_workspace_files = max(1, int(max_workspace_files or 8))
    max_project_docs = max(0, int(max_project_docs or 6))
    max_excerpt_chars = max(200, int(max_excerpt_chars or 1500))

    explicit_paths = _extract_explicit_path_mentions(goal)
    goal_tokens = _extract_goal_tokens(goal)
    priority_candidates: list[WorkspaceCandidate] = []
    ranked_candidates: list[WorkspaceCandidate] = []
    seen_rel_paths: set[str] = set()
    priority_rel_paths = {path.lower(): index for index, path in enumerate(PRIORITY_PROJECT_DOCS)}

    for path in _iter_workspace_files(repo_path):
        rel_path = path.relative_to(repo_path).as_posix()
        lowered_rel = rel_path.lower()
        snippet = snippet_for_path(path, max_chars=max_excerpt_chars)
        score, reason = _score_workspace_file(
            path=path,
            relative_path=rel_path,
            explicit_paths=explicit_paths,
            goal_tokens=goal_tokens,
        )
        if lowered_rel in priority_rel_paths:
            priority_candidates.append(
                WorkspaceCandidate(
                    path=path,
                    relative_path=rel_path,
                    role="project_doc",
                    reason=f"priority project document: {rel_path}",
                    score=10_000 - priority_rel_paths[lowered_rel],
                    snippet=snippet,
                )
            )
            continue
        if score <= 0:
            continue
        ranked_candidates.append(
            WorkspaceCandidate(
                path=path,
                relative_path=rel_path,
                role="workspace_file",
                reason=reason,
                score=score,
                snippet=snippet,
            )
        )

    priority_candidates.sort(key=lambda item: (-item.score, item.relative_path))
    ranked_candidates.sort(key=lambda item: (-item.score, item.relative_path))

    selected: list[WorkspaceCandidate] = []
    for candidate in priority_candidates[:max_project_docs]:
        selected.append(candidate)
        seen_rel_paths.add(candidate.relative_path)

    if len(priority_candidates) > max_project_docs:
        warnings.append(
            f"workspace priority documents truncated: kept {max_project_docs} of {len(priority_candidates)} high-signal docs"
        )

    for candidate in ranked_candidates:
        if len(selected) >= max_workspace_files:
            break
        if candidate.relative_path in seen_rel_paths:
            continue
        selected.append(candidate)
        seen_rel_paths.add(candidate.relative_path)

    if len(selected) < len(priority_candidates) + len(ranked_candidates):
        warnings.append(
            f"workspace snippets truncated: kept {len(selected)} files out of {len(priority_candidates) + len(ranked_candidates)} candidates"
        )
    if not selected:
        warnings.append("workspace context found no relevant project files")

    return {
        "workspace_files": [
            {
                "path": str(candidate.path),
                "relative_path": candidate.relative_path,
                "role": candidate.role,
                "snippet": candidate.snippet,
                "reason": candidate.reason,
                "source_type": "project",
                "normalized_source_type": "project",
                "ephemeral": True,
                "title": candidate.relative_path,
            }
            for candidate in selected
        ],
        "warnings": warnings,
    }


def _iter_workspace_files(repo_path: Path):
    for root, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_excluded_workspace_dir(name)
            and not _path_contains_excluded_workspace_part(Path(root) / name)
        ]
        base_path = Path(root)
        for filename in filenames:
            path = base_path / filename
            if _path_contains_excluded_workspace_part(path):
                continue
            if path.suffix.lower() not in ALLOWED_WORKSPACE_SUFFIXES:
                continue
            yield path


def _is_excluded_workspace_dir(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return False
    if lowered in EXCLUDED_WORKSPACE_DIRS:
        return True
    return bool(EXCLUDED_WORKSPACE_DIR_PATTERN.match(lowered))


def _path_contains_excluded_workspace_part(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return any(part in EXCLUDED_WORKSPACE_PATH_PARTS for part in parts)


def _score_workspace_file(
    *,
    path: Path,
    relative_path: str,
    explicit_paths: set[str],
    goal_tokens: set[str],
) -> tuple[int, str]:
    rel_lower = relative_path.lower()
    name_lower = path.name.lower()
    path_parts = {part.lower() for part in path.parts}
    token_parts = set()
    for token in path_parts | {name_lower, rel_lower}:
        token_parts.update(part for part in re.split(r"[/_.\\-]+", token) if len(part) >= 2)

    for explicit in explicit_paths:
        if explicit and (rel_lower == explicit or rel_lower.endswith(explicit) or name_lower == explicit):
            return 9_000, f"explicit file/path mention in goal: {explicit}"
    if name_lower in goal_tokens:
        return 6_000, f"filename mentioned in goal: {name_lower}"
    overlap = sorted(token_parts & goal_tokens)
    if overlap:
        return 1_000 + len(overlap) * 20, f"keyword overlap with goal: {', '.join(overlap[:5])}"
    return 0, ""


def _extract_project_conventions(selected: list[WorkspaceCandidate]) -> list[str]:
    conventions: list[str] = []
    seen: set[str] = set()
    for candidate in selected:
        if candidate.role != "project_doc":
            continue
        lines = candidate.snippet.splitlines()
        per_file = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "##", "###")):
                continue
            if not (
                stripped.startswith(("- ", "* ", "1. ", "2. ", "3. "))
                or "must" in stripped.lower()
                or "do not" in stripped.lower()
                or "prefer" in stripped.lower()
                or "should" in stripped.lower()
            ):
                continue
            entry = f"{candidate.relative_path}: {stripped.lstrip('-* ').strip()}"
            if entry in seen:
                continue
            seen.add(entry)
            conventions.append(entry)
            per_file += 1
            if per_file >= 4 or len(conventions) >= 12:
                break
        if len(conventions) >= 12:
            break
    return conventions


def _build_prompt_context(
    *,
    goal: str,
    mode: str,
    persistent_sources: list[dict[str, Any]],
    workspace_sources: list[dict[str, Any]],
    project_conventions: list[str],
    warnings: list[str],
) -> str:
    lines = [
        "You are helping with a read-only task context.",
        f"Goal: {goal}",
        f"Mode: {mode}",
    ]
    if project_conventions:
        lines.append("Project conventions:")
        for item in project_conventions[:8]:
            lines.append(f"- {item}")
    if persistent_sources:
        lines.append("Persistent knowledge evidence:")
        for hit in persistent_sources:
            lines.append(
                f"- [{hit['normalized_source_type']}] {hit['title']}: {str(hit.get('snippet') or hit.get('content') or '')}"
            )
    if workspace_sources:
        lines.append("Ephemeral workspace evidence:")
        for item in workspace_sources:
            snippet = _truncate_text(str(item.get("snippet", "")), 240)
            lines.append(f"- {item.get('relative_path')}: {item.get('reason')}")
            if snippet:
                lines.append(f"  {snippet}")
    if warnings:
        lines.append("Warnings:")
        for warning in warnings[:6]:
            lines.append(f"- {warning}")
    return "\n".join(lines).strip()


def _parse_json_field(value: Any, fallback: Any) -> Any:
    import json

    if isinstance(value, (dict, list)):
        return value
    if not value:
        return fallback
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def _hash_text(*parts: Any) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for part in parts:
        if part is None:
            continue
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _note_candidates(sqlite_db, topic: str, *, source_types: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in sqlite_db.search_notes(topic, limit=max(limit * 4, limit))
        if str(row.get("source_type") or "") in source_types
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("updated_at") or ""),
            str(row.get("title") or ""),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    return rows[:limit]


def _paper_candidates(sqlite_db, topic: str, *, limit: int) -> list[dict[str, Any]]:
    rows = sqlite_db.search_papers(topic, limit=max(limit * 4, limit))
    rows.sort(
        key=lambda row: (
            int(row.get("importance") or 0),
            int(row.get("year") or 0),
            str(row.get("title") or ""),
            str(row.get("arxiv_id") or ""),
        ),
        reverse=True,
    )
    return rows[:limit]


def _paper_source_url(arxiv_id: str) -> str | None:
    token = str(arxiv_id or "").strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", token):
        return f"https://arxiv.org/abs/{token}"
    return None


def _collect_notebook_sources(
    sqlite_db,
    *,
    topic: str,
    include_vault: bool,
    include_papers: bool,
    include_web: bool,
    max_vault_sources: int,
    max_paper_sources: int,
    max_web_sources: int,
    max_source_chars: int,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if include_vault:
        for row in _note_candidates(sqlite_db, topic, source_types=("vault", "note"), limit=max_vault_sources):
            metadata = _parse_json_field(row.get("metadata"), {})
            content = _truncate_text(str(row.get("content") or ""), max_source_chars)
            sources.append(
                {
                    "source_key": f"note:{row['id']}",
                    "local_source_id": str(row["id"]),
                    "source_type": str(row.get("source_type") or "vault"),
                    "normalized_source_type": "note",
                    "local_source_type": "note",
                    "title": str(row.get("title") or row["id"]),
                    "content": content,
                    "snippet": _truncate_text(content, 280),
                    "updated_at": str(row.get("updated_at") or ""),
                    "source_url": str(metadata.get("url") or ""),
                    "file_path": str(row.get("file_path") or ""),
                    "metadata": metadata,
                    "content_hash": _hash_text(row["id"], row.get("updated_at"), content),
                    "ephemeral": False,
                    "selection_reason": "topic_vault_match",
                }
            )
    if include_papers:
        for row in _paper_candidates(sqlite_db, topic, limit=max_paper_sources):
            summary = _truncate_text(str(row.get("notes") or ""), max_source_chars)
            metadata = {
                "authors": row.get("authors") or "",
                "year": row.get("year"),
                "field": row.get("field") or "",
                "pdf_path": row.get("pdf_path") or "",
                "text_path": row.get("text_path") or "",
                "translated_path": row.get("translated_path") or "",
            }
            sources.append(
                {
                    "source_key": f"paper:{row['arxiv_id']}",
                    "local_source_id": str(row["arxiv_id"]),
                    "source_type": "paper",
                    "normalized_source_type": "paper",
                    "local_source_type": "paper",
                    "title": str(row.get("title") or row["arxiv_id"]),
                    "content": summary or "No local summary available.",
                    "snippet": _truncate_text(summary or "No local summary available.", 280),
                    "updated_at": str(row.get("created_at") or ""),
                    "source_url": _paper_source_url(str(row.get("arxiv_id") or "")) or "",
                    "file_path": str(row.get("text_path") or row.get("pdf_path") or ""),
                    "metadata": metadata,
                    "content_hash": _hash_text(row["arxiv_id"], row.get("created_at"), summary),
                    "ephemeral": False,
                    "selection_reason": "topic_paper_match",
                }
            )
    if include_web:
        for row in _note_candidates(sqlite_db, topic, source_types=("web",), limit=max_web_sources):
            metadata = _parse_json_field(row.get("metadata"), {})
            content = _truncate_text(str(row.get("content") or ""), max_source_chars)
            sources.append(
                {
                    "source_key": f"web:{row['id']}",
                    "local_source_id": str(row["id"]),
                    "source_type": "web",
                    "normalized_source_type": "web",
                    "local_source_type": "web",
                    "title": str(row.get("title") or row["id"]),
                    "content": content,
                    "snippet": _truncate_text(content, 280),
                    "updated_at": str(row.get("updated_at") or ""),
                    "source_url": str(metadata.get("url") or ""),
                    "file_path": str(row.get("file_path") or ""),
                    "metadata": metadata,
                    "content_hash": _hash_text(row["id"], row.get("updated_at"), content),
                    "ephemeral": False,
                    "selection_reason": "topic_web_match",
                }
            )
    summary_source = _build_summary_source(topic, sources)
    return [summary_source, *sources]


def _build_summary_source(topic: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    topic_slug = slugify_topic(topic)
    description = _topic_description(topic, sources)
    summary_content = _build_notebook_summary(topic, description, sources)
    return {
        "source_key": f"summary:{topic_slug}",
        "local_source_id": topic_slug,
        "source_type": "summary",
        "normalized_source_type": "summary",
        "local_source_type": "summary",
        "title": f"Topic Summary: {topic}",
        "content": summary_content,
        "snippet": _truncate_text(summary_content, 280),
        "updated_at": "",
        "source_url": "",
        "file_path": "",
        "metadata": {"topic": topic},
        "content_hash": _hash_text(topic_slug, summary_content),
        "ephemeral": False,
        "selection_reason": "topic_summary",
    }


def _topic_description(topic: str, bundle_sources: list[dict[str, Any]]) -> str:
    note_count = len([item for item in bundle_sources if item.get("local_source_type") == "note"])
    paper_count = len([item for item in bundle_sources if item.get("local_source_type") == "paper"])
    web_count = len([item for item in bundle_sources if item.get("local_source_type") == "web"])
    return (
        f"KnowledgeOS topic notebook for '{topic}'. "
        f"Includes {note_count} Obsidian notes, {paper_count} paper summaries, and {web_count} web references."
    )


def _build_notebook_summary(topic: str, description: str, bundle_sources: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"# Topic Summary: {topic}",
            "",
            description,
            "",
            "## Included Sources",
            *[
                f"- [{item.get('local_source_type') or item.get('normalized_source_type')}] {item.get('title')}"
                for item in bundle_sources
                if item.get("local_source_type") != "summary"
            ],
        ]
    ).strip()
