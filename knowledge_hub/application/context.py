"""Shared composition root for CLI and MCP surfaces."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from knowledge_hub.infrastructure.config import Config, DEFAULT_CONFIG_PATH
from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.infrastructure.providers import get_embedder, get_llm


def resolve_config_path(
    explicit_path: str | None = None,
    *,
    project_root: Path | None = None,
    environ: dict[str, str] | None = None,
) -> str | None:
    env = environ or os.environ
    if explicit_path:
        return str(Path(explicit_path).expanduser())

    env_path = str(env.get("KHUB_CONFIG", "")).strip()
    if env_path:
        return str(Path(env_path).expanduser())

    root = project_root or Path(__file__).resolve().parents[2]
    repo_config = root / "config.yaml"
    if repo_config.exists():
        return str(repo_config)

    if DEFAULT_CONFIG_PATH.exists():
        return str(DEFAULT_CONFIG_PATH)

    return None


@dataclass
class AppContext:
    """Lazy runtime container shared by CLI and MCP entrypoints."""

    project_root: Path
    config_path: str | None = None
    _config: Config | None = None
    _sqlite_db: SQLiteDatabase | None = None
    _vector_db: VectorDatabase | None = None
    _embedder: Any | None = None
    _summarizer: Any | None = None
    _translator: Any | None = None
    _searcher: Any | None = None
    _learning_service: Any | None = None
    _web_ingest_service: Any | None = None

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = Config(self.config_path)
        return self._config

    @property
    def sqlite_db(self) -> SQLiteDatabase:
        if self._sqlite_db is None:
            self._sqlite_db = SQLiteDatabase(self.config.sqlite_path)
        return self._sqlite_db

    @property
    def vector_db(self) -> VectorDatabase:
        if self._vector_db is None:
            self._vector_db = VectorDatabase(
                self.config.vector_db_path,
                self.config.collection_name,
                repair_on_init=False,
            )
        return self._vector_db

    def get_embedder(self) -> Any:
        if self._embedder is None:
            embed_cfg = self.config.get_provider_config(self.config.embedding_provider)
            self._embedder = get_embedder(
                self.config.embedding_provider,
                model=self.config.embedding_model,
                **embed_cfg,
            )
        return self._embedder

    def get_summarizer(self) -> Any:
        if self._summarizer is None:
            summ_cfg = self.config.get_provider_config(self.config.summarization_provider)
            self._summarizer = get_llm(
                self.config.summarization_provider,
                model=self.config.summarization_model,
                **summ_cfg,
            )
        return self._summarizer

    def get_translator(self) -> Any:
        if self._translator is None:
            trans_cfg = self.config.get_provider_config(self.config.translation_provider)
            self._translator = get_llm(
                self.config.translation_provider,
                model=self.config.translation_model,
                **trans_cfg,
            )
        return self._translator

    @property
    def searcher(self):  # noqa: ANN201
        if self._searcher is None:
            from knowledge_hub.ai.rag import RAGSearcher

            signature = inspect.signature(RAGSearcher)
            kwargs: dict[str, Any] = {}
            if "sqlite_db" in signature.parameters:
                kwargs["sqlite_db"] = self.sqlite_db
            if "config" in signature.parameters:
                kwargs["config"] = self.config

            self._searcher = RAGSearcher(
                self.get_embedder(),
                self.vector_db,
                self.get_summarizer(),
                **kwargs,
            )
        return self._searcher

    def get_searcher(self):  # noqa: ANN201
        return self.searcher

    @property
    def learning_service(self):  # noqa: ANN201
        if self._learning_service is None:
            from knowledge_hub.learning import LearningCoachService

            self._learning_service = LearningCoachService(
                self.config,
                sqlite_db_factory=self.create_sqlite_db,
                vector_db_factory=self.create_vector_db,
                embedder_factory=self.get_embedder,
            )
        return self._learning_service

    @property
    def web_ingest_service(self):  # noqa: ANN201
        if self._web_ingest_service is None:
            from knowledge_hub.web import WebIngestService

            self._web_ingest_service = WebIngestService(
                self.config,
                sqlite_db_factory=self.create_sqlite_db,
                vector_db_factory=self.create_vector_db,
                embedder_factory=self.get_embedder,
            )
        return self._web_ingest_service

    def build_discover_runtime(self, *, need_translator: bool = False):  # noqa: ANN201
        return SimpleNamespace(
            config=self.config,
            summarizer=self.get_summarizer(),
            translator=self.get_translator() if need_translator else None,
            sqlite_db=self.sqlite_db,
        )


@dataclass
class AppContextFactory:
    """Factory that applies the shared config precedence rules."""

    config_path: str | None = None
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    _app_context: AppContext | None = field(default=None, init=False, repr=False)

    @classmethod
    def resolve_config_path(
        cls,
        override: str | None = None,
        *,
        project_root: Path | None = None,
        environ: dict[str, str] | None = None,
    ) -> str | None:
        return resolve_config_path(
            override,
            project_root=project_root,
            environ=environ,
        )

    def build(
        self,
        config_path: str | None = None,
        *,
        require_search: bool = False,
        core_only: bool = False,
    ) -> AppContext:
        if config_path is None:
            if self._app_context is None:
                self._app_context = AppContext(
                    project_root=self.project_root,
                    config_path=self.__class__.resolve_config_path(
                        self.config_path,
                        project_root=self.project_root,
                    ),
                )
            app = self._app_context
        else:
            app = AppContext(
                project_root=self.project_root,
                config_path=self.__class__.resolve_config_path(
                    config_path,
                    project_root=self.project_root,
                ),
            )
        if require_search:
            _ = app.searcher
        elif core_only:
            _ = app.sqlite_db
        return app

    @property
    def app_context(self) -> AppContext:
        return self.build()

    @property
    def config(self) -> Config:
        return self.app_context.config

    def get_sqlite_db(self) -> SQLiteDatabase:
        return self.app_context.sqlite_db

    def get_vector_db(self) -> VectorDatabase:
        return self.app_context.vector_db

    def create_sqlite_db(
        self,
        *,
        enable_event_store: bool = True,
        bootstrap: bool = True,
        read_only: bool = False,
    ) -> SQLiteDatabase:
        return SQLiteDatabase(
            self.config.sqlite_path,
            enable_event_store=enable_event_store,
            bootstrap=bootstrap,
            read_only=read_only,
        )

    def create_vector_db(self, *, repair_on_init: bool = True) -> VectorDatabase:
        return VectorDatabase(
            self.config.vector_db_path,
            self.config.collection_name,
            repair_on_init=repair_on_init,
        )

    def get_searcher(self):  # noqa: ANN201
        return self.build(require_search=True).searcher

    def build_llm(self, provider: str, model: str | None = None):  # noqa: ANN201
        cfg = self.config.get_provider_config(provider)
        return get_llm(provider, model=model, **cfg)

    def build_embedder(self, provider: str, model: str | None = None):  # noqa: ANN201
        cfg = self.config.get_provider_config(provider)
        return get_embedder(provider, model=model, **cfg)

    def get_embedder(self):  # noqa: ANN201
        return self.app_context.get_embedder()

    def create_learning_service(self):  # noqa: ANN201
        from knowledge_hub.learning import LearningCoachService

        return LearningCoachService(
            self.config,
            sqlite_db_factory=self.create_sqlite_db,
            vector_db_factory=self.create_vector_db,
            embedder_factory=self.get_embedder,
        )

    def create_web_ingest_service(self):  # noqa: ANN201
        from knowledge_hub.web import WebIngestService

        return WebIngestService(
            self.config,
            sqlite_db_factory=self.create_sqlite_db,
            vector_db_factory=self.create_vector_db,
            embedder_factory=self.get_embedder,
        )


def get_app_context(value: Any) -> AppContext:
    if isinstance(value, AppContext):
        return value
    app_context = getattr(value, "app_context", None)
    if isinstance(app_context, AppContext):
        return app_context
    raise TypeError(f"Unsupported app context provider: {type(value)!r}")
