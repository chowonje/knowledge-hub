from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RAGContext:
    embedder: Any
    database: Any
    llm: Any = None
    sqlite_db: Any = None
    config: Any = None
    parent_fetch_limit: int = 256
    parent_window_chunks: int = 3

    @classmethod
    def from_searcher(cls, searcher: Any) -> "RAGContext":
        return cls(
            embedder=getattr(searcher, "embedder", None),
            database=getattr(searcher, "database", None),
            llm=getattr(searcher, "llm", None),
            sqlite_db=getattr(searcher, "sqlite_db", None),
            config=getattr(searcher, "config", None),
            parent_fetch_limit=int(getattr(searcher, "parent_fetch_limit", 256)),
            parent_window_chunks=int(getattr(searcher, "parent_window_chunks", 3)),
        )


@dataclass
class RAGCaches:
    cached_local_llm: Any = None
    cached_local_llm_signature: tuple[str, str, int] | None = None
    topology_cache: dict[str, Any] | None = None
    profile_cache: dict[str, Any] | None = None
    active_request_llm: Any = None

    @classmethod
    def from_searcher(cls, searcher: Any) -> "RAGCaches":
        return cls(
            cached_local_llm=getattr(searcher, "_cached_local_llm", None),
            cached_local_llm_signature=getattr(searcher, "_cached_local_llm_signature", None),
            topology_cache=getattr(searcher, "_topology_cache", None),
            profile_cache=getattr(searcher, "_profile_cache", None),
            active_request_llm=getattr(searcher, "_active_request_llm", None),
        )

    def refresh_route_llm_cache_from_searcher(self, searcher: Any) -> None:
        self.cached_local_llm = getattr(searcher, "_cached_local_llm", None)
        self.cached_local_llm_signature = getattr(searcher, "_cached_local_llm_signature", None)

    def route_llm_cache(self) -> tuple[Any, tuple[str, str, int] | None]:
        return self.cached_local_llm, self.cached_local_llm_signature

    def writeback_route_llm_cache(
        self,
        searcher: Any,
        *,
        cached_local_llm: Any,
        cached_local_llm_signature: tuple[str, str, int] | None,
    ) -> None:
        self.cached_local_llm = cached_local_llm
        self.cached_local_llm_signature = cached_local_llm_signature
        setattr(searcher, "_cached_local_llm", cached_local_llm)
        setattr(searcher, "_cached_local_llm_signature", cached_local_llm_signature)

    def active_request_llm_value(self) -> Any:
        return self.active_request_llm

    def sync_active_request_llm_from_searcher(self, searcher: Any) -> None:
        self.active_request_llm = getattr(searcher, "_active_request_llm", None)

    def write_active_request_llm(self, searcher: Any, llm: Any) -> None:
        self.active_request_llm = llm
        setattr(searcher, "_active_request_llm", llm)
