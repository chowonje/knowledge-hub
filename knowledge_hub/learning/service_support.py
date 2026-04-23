"""Support helpers for `LearningCoachService`.

This module holds runtime/dependency/support logic while keeping
`LearningCoachService` itself as the stable facade/entrypoint.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from knowledge_hub.core.schema_validator import annotate_schema_errors
from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.learning.contracts import LearningServiceRepository
from knowledge_hub.learning.assessor import parse_edges_from_session, parse_frontmatter
from knowledge_hub.learning.obsidian_writeback import build_paths, resolve_vault_write_adapter


def runtime_provider(service) -> Any | None:
    provider = service._app_context
    nested = getattr(provider, "app_context", None) if provider is not None else None
    return nested or provider


def get_injected_sqlite_db(service) -> LearningServiceRepository | None:
    if service._sqlite_db_resolved:
        return service._resolved_sqlite_db
    db = service._sqlite_db
    provider = service._runtime_provider()
    if db is None and provider is not None:
        db = getattr(provider, "sqlite_db", None)
        if db is None:
            getter = getattr(provider, "get_sqlite_db", None)
            if callable(getter):
                db = getter()
    service._resolved_sqlite_db = db
    service._sqlite_db_resolved = True
    return db


def get_injected_vector_db(service) -> Any | None:
    if service._vector_db_resolved:
        return service._resolved_vector_db
    vector_db = service._vector_db
    provider = service._runtime_provider()
    if vector_db is None and provider is not None:
        vector_db = getattr(provider, "vector_db", None)
        if vector_db is None:
            getter = getattr(provider, "get_vector_db", None)
            if callable(getter):
                vector_db = getter()
    service._resolved_vector_db = vector_db
    service._vector_db_resolved = True
    return vector_db


def get_injected_embedder(service) -> Any | None:
    if service._embedder_resolved:
        return service._resolved_embedder
    embedder = service._embedder
    provider = service._runtime_provider()
    if embedder is None and provider is not None:
        getter = getattr(provider, "get_embedder", None)
        if callable(getter):
            embedder = getter()
    service._resolved_embedder = embedder
    service._embedder_resolved = True
    return embedder


def create_sqlite_db(service) -> LearningServiceRepository:
    return SQLiteDatabase(service.config.sqlite_path)


def create_vector_db(service):
    return VectorDatabase(service.config.vector_db_path, service.config.collection_name)


def create_embedder(service):
    from knowledge_hub.infrastructure.providers import get_embedder

    embedder_cfg = service.config.get_provider_config(service.config.embedding_provider)
    return get_embedder(
        service.config.embedding_provider,
        model=service.config.embedding_model,
        **embedder_cfg,
    )


def open_db(service) -> LearningServiceRepository:
    db = service._get_injected_sqlite_db()
    if db is not None:
        return db
    provider = service._runtime_provider()
    if service._sqlite_db_factory is not None:
        return service._sqlite_db_factory()
    creator = getattr(provider, "create_sqlite_db", None) if provider is not None else None
    if callable(creator):
        return creator()
    return service._create_sqlite_db()


def close_db(service, db: LearningServiceRepository | None) -> None:
    if db is None or db is service._get_injected_sqlite_db():
        return
    close = getattr(db, "close", None)
    if callable(close):
        close()


def get_vector_db(service):
    vector_db = service._get_injected_vector_db()
    if vector_db is not None:
        return vector_db
    provider = service._runtime_provider()
    if service._vector_db_factory is not None:
        return service._vector_db_factory()
    creator = getattr(provider, "create_vector_db", None) if provider is not None else None
    if callable(creator):
        return creator()
    return service._create_vector_db()


def get_embedder(service):
    embedder = service._get_injected_embedder()
    if embedder is not None:
        return embedder
    provider = service._runtime_provider()
    if service._embedder_factory is not None:
        return service._embedder_factory()
    builder = getattr(provider, "build_embedder", None) if provider is not None else None
    if callable(builder):
        return builder(service.config.embedding_provider, model=service.config.embedding_model)
    return service._create_embedder()


def build_rag_searcher(service, *, sqlite_db: LearningServiceRepository):
    from knowledge_hub.ai.rag import RAGSearcher

    return RAGSearcher(
        embedder=service._get_embedder(),
        database=service._get_vector_db(),
        sqlite_db=sqlite_db,
        config=service.config,
    )


def record_schema_validation(service, payload: dict) -> dict:
    schema_id = payload.get("schema")
    if isinstance(schema_id, str) and schema_id:
        strict_schema = bool(service.config.get_nested("validation", "schema", "strict", default=False))
        try:
            result = annotate_schema_errors(payload, schema_id, strict=strict_schema)
            if strict_schema and not result.ok:
                verify = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
                verify["allowed"] = False
                verify["schemaValid"] = False
                verify["schemaErrors"] = list(result.errors or [])
                payload["verify"] = verify
                payload["status"] = "blocked"
                writeback = payload.get("writeback")
                if isinstance(writeback, dict):
                    writeback["ok"] = False
                    writeback.setdefault("detail", "verify blocked by schema validation")
        except Exception as error:
            if strict_schema:
                message = f"schema validation failed ({schema_id}): {error}"
                payload.setdefault("schemaErrors", [])
                if isinstance(payload.get("schemaErrors"), list):
                    payload["schemaErrors"].append(message)
                verify = payload.get("verify") if isinstance(payload.get("verify"), dict) else {}
                verify["allowed"] = False
                verify["schemaValid"] = False
                verify["schemaErrors"] = list(payload.get("schemaErrors") or [])
                payload["verify"] = verify
                payload["status"] = "blocked"
                writeback = payload.get("writeback")
                if isinstance(writeback, dict):
                    writeback["ok"] = False
                    writeback.setdefault("detail", "verify blocked by schema validation")
                return payload
    return payload


def resolve_dynamic_dir(service) -> Path:
    repo_dynamic = Path(__file__).resolve().parents[2] / "data" / "dynamic"
    if repo_dynamic.exists():
        return repo_dynamic
    return Path(service.config.sqlite_path).expanduser().resolve().parent / "dynamic"


def resolve_vault_adapter(service):
    backend = str(service.config.get_nested("obsidian", "write_backend", default="filesystem") or "filesystem")
    cli_binary = str(service.config.get_nested("obsidian", "cli_binary", default="obsidian") or "obsidian")
    vault_name = str(service.config.get_nested("obsidian", "vault_name", default="") or "")
    return resolve_vault_write_adapter(
        vault_path=service.config.vault_path,
        backend=backend,
        cli_binary=cli_binary,
        vault_name=vault_name,
    )


def obsidian_backend_kwargs(service) -> dict:
    return {
        "backend": str(service.config.get_nested("obsidian", "write_backend", default="filesystem") or "filesystem"),
        "cli_binary": str(service.config.get_nested("obsidian", "cli_binary", default="obsidian") or "obsidian"),
        "vault_name": str(service.config.get_nested("obsidian", "vault_name", default="") or ""),
    }


def resolve_session_context(service, topic: str, session_id: str) -> tuple[str, str, list[str]]:
    if not service.config.vault_path:
        return "", "", []
    paths = build_paths(service.config.vault_path, topic)
    session_path = paths.session_file(session_id)
    if not session_path.exists():
        return "", str(session_path), []
    content = session_path.read_text(encoding="utf-8")
    _, body = parse_frontmatter(content)
    _, _, raw_evidence_texts = parse_edges_from_session(body, session_note_path=str(session_path))
    return content, str(session_path), raw_evidence_texts


def extract_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    for pattern in (r"```json\s*(\{.*?\})\s*```", r"(\{.*\})"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def llm_json_generate(service, llm, prompt: str, context_payload: dict, max_tokens: int = 1400) -> tuple[dict, str | None]:
    if llm is None:
        return {}, "llm unavailable"
    try:
        context = json.dumps(context_payload, ensure_ascii=False)
        raw = llm.generate(prompt=prompt, context=context, max_tokens=max_tokens)
        parsed = service._extract_json_object(raw)
        if parsed:
            return parsed, None
        return {}, "llm output parse failed"
    except Exception as error:
        return {}, f"llm call failed: {error}"


__all__ = [
    "build_rag_searcher",
    "close_db",
    "create_embedder",
    "create_sqlite_db",
    "create_vector_db",
    "extract_json_object",
    "get_embedder",
    "get_injected_embedder",
    "get_injected_sqlite_db",
    "get_injected_vector_db",
    "get_vector_db",
    "llm_json_generate",
    "obsidian_backend_kwargs",
    "open_db",
    "record_schema_validation",
    "resolve_dynamic_dir",
    "resolve_session_context",
    "resolve_vault_adapter",
    "runtime_provider",
]
