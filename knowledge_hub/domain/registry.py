from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class DomainRegistration:
    name: str
    module_path: str
    sources: tuple[str, ...]
    active: bool


_REGISTRY: tuple[DomainRegistration, ...] = (
    DomainRegistration(
        name="ai_papers",
        module_path="knowledge_hub.domain.ai_papers",
        sources=("paper",),
        active=True,
    ),
    DomainRegistration(
        name="web_knowledge",
        module_path="knowledge_hub.domain.web_knowledge",
        sources=("web",),
        active=True,
    ),
    DomainRegistration(
        name="youtube_knowledge",
        module_path="knowledge_hub.domain.youtube_knowledge",
        sources=("youtube",),
        active=True,
    ),
    DomainRegistration(
        name="vault_knowledge",
        module_path="knowledge_hub.domain.vault_knowledge",
        sources=("vault",),
        active=True,
    ),
)


def normalize_domain_source(source_type: str | None) -> str:
    source = str(source_type or "").strip().lower()
    if source in {"", "all", "*"}:
        return ""
    if source == "note":
        return "vault"
    if source in {"repo", "repository", "workspace"}:
        return "project"
    return source


def list_domain_registrations() -> list[DomainRegistration]:
    return list(_REGISTRY)


def resolve_domain_name(source_type: str | None) -> str:
    normalized = normalize_domain_source(source_type)
    for registration in _REGISTRY:
        if normalized and normalized in registration.sources and registration.active:
            return registration.name
    return ""


def get_domain_registration(name: str) -> DomainRegistration | None:
    normalized = str(name or "").strip().lower()
    for registration in _REGISTRY:
        if registration.name == normalized:
            return registration
    return None


def get_domain_pack(*, source_type: str | None = None, name: str | None = None) -> Any | None:
    registration = get_domain_registration(name or resolve_domain_name(source_type))
    if registration is None or not registration.active:
        return None
    return import_module(registration.module_path)


__all__ = [
    "DomainRegistration",
    "get_domain_pack",
    "get_domain_registration",
    "list_domain_registrations",
    "normalize_domain_source",
    "resolve_domain_name",
]
