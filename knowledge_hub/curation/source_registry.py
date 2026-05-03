from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceIdentity:
    source_vendor: str
    source_channel: str
    source_channel_type: str
    source_item_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryProvenance:
    method: str
    origin_url: str
    entry_ref: str
    discovered_at: str
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredSourceItem:
    identity: SourceIdentity
    source_name: str
    source_type: str
    url: str
    canonical_url: str = ""
    title_hint: str = ""
    published_at: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    provenance: DiscoveryProvenance = field(
        default_factory=lambda: DiscoveryProvenance(
            method="unknown",
            origin_url="",
            entry_ref="",
            discovered_at=datetime.now(timezone.utc).isoformat(),
            rank=0,
        )
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "source_name": self.source_name,
            "source_type": self.source_type,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title_hint": self.title_hint,
            "published_at": self.published_at,
            "author": self.author,
            "tags": list(self.tags),
            "provenance": self.provenance.to_dict(),
            "metadata": dict(self.metadata),
        }


class SourceExtractor(Protocol):
    source_name: str

    def matches(self, source: dict[str, Any]) -> bool:
        ...

    def discover_latest(self, source: dict[str, Any], limit: int) -> list[DiscoveredSourceItem]:
        ...

    def resolve_identity(self, item: DiscoveredSourceItem) -> SourceIdentity:
        ...

    def metadata_precedence(self) -> list[str]:
        ...


class SourceRegistry:
    def __init__(self, extractors: list[SourceExtractor] | None = None):
        self._extractors: list[SourceExtractor] = list(extractors or [])

    def register(self, extractor: SourceExtractor) -> None:
        self._extractors.append(extractor)

    def get_extractor(self, source: dict[str, Any]) -> SourceExtractor | None:
        for extractor in self._extractors:
            try:
                if extractor.matches(source):
                    return extractor
            except Exception:
                continue
        return None

    def discover_latest(self, source: dict[str, Any], limit: int) -> tuple[list[DiscoveredSourceItem], list[str]]:
        extractor = self.get_extractor(source)
        source_name = str(source.get("source_name") or "").strip() or "unknown"
        if extractor is None:
            return [], [f"{source_name}: no extractor registered"]
        try:
            items = extractor.discover_latest(source, limit)
            for item in items:
                item.identity = extractor.resolve_identity(item)
                item.metadata.setdefault("metadata_precedence", extractor.metadata_precedence())
            return items, []
        except Exception as error:
            return [], [f"{source_name}: extractor_failed: {error}"]

    def all_extractors(self) -> list[SourceExtractor]:
        return list(self._extractors)


def build_source_registry() -> SourceRegistry:
    from knowledge_hub.curation.source_extractors import build_builtin_extractors

    return SourceRegistry(build_builtin_extractors())
