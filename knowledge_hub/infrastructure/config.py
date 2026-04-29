"""Canonical runtime configuration surface."""

from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

from knowledge_hub.core.vault_paths import resolve_vault_exclude_folders

log = logging.getLogger("khub.config")


class ConfigError(Exception):
    """설정 관련 오류 — 복구 불가, 즉시 종료 대상"""


def mask_secret(value: str, visible: int = 4) -> str:
    """API 키 등 민감 값을 마스킹하여 로그 안전하게 출력"""
    if not value or len(value) <= visible:
        return "***"
    return value[:visible] + "*" * (len(value) - visible)


def resolve_api_key(provider: str, config_value: str = "") -> str:
    """프로바이더별 API 키를 환경변수 → config 순으로 해석"""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = env_map.get(provider, "")
    key = os.environ.get(env_var, "") or config_value
    key = key.strip()
    if key.startswith("${"):
        return ""
    return key

DEFAULT_CONFIG_DIR = Path.home() / ".khub"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"

PUBLIC_PARSER_DEFAULT = "auto"
PUBLIC_DEFAULT_TRANSLATION_PROVIDER = "openai"
PUBLIC_DEFAULT_TRANSLATION_MODEL = "gpt-5-nano"
PUBLIC_DEFAULT_SUMMARIZATION_PROVIDER = "ollama"
PUBLIC_DEFAULT_SUMMARIZATION_MODEL = "qwen3:14b"
PUBLIC_DEFAULT_EMBEDDING_PROVIDER = "ollama"
PUBLIC_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
PUBLIC_SETUP_PROFILE_CHOICES = ("local", "hybrid", "custom")
PUBLIC_OPENDATALOADER_READING_ORDER_DEFAULT = "auto"
PUBLIC_OPENDATALOADER_USE_STRUCT_TREE_DEFAULT = False
PUBLIC_OPENDATALOADER_TABLE_METHOD_DEFAULT = "auto"

PUBLIC_PROMPT_DEFAULTS: dict[str, dict[str, str]] = {
    "translation": {
        "provider": PUBLIC_DEFAULT_TRANSLATION_PROVIDER,
        "model": PUBLIC_DEFAULT_TRANSLATION_MODEL,
    },
    "summarization": {
        "provider": PUBLIC_DEFAULT_SUMMARIZATION_PROVIDER,
        "model": PUBLIC_DEFAULT_SUMMARIZATION_MODEL,
    },
    "embedding": {
        "provider": PUBLIC_DEFAULT_EMBEDDING_PROVIDER,
        "model": PUBLIC_DEFAULT_EMBEDDING_MODEL,
    },
}

PUBLIC_SETUP_PROFILES: dict[str, dict[str, Any]] = {
    "local": {
        "translation": {"provider": "ollama", "model": "qwen3:14b"},
        "summarization": {"provider": "ollama", "model": "qwen3:14b"},
        "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
        "providers": {
            "ollama": {"base_url": "http://localhost:11434"},
        },
        "storage": {
            "papers_dir": str(DEFAULT_CONFIG_DIR / "papers"),
            "vector_db": str(DEFAULT_CONFIG_DIR / "chroma_db"),
            "sqlite": str(DEFAULT_CONFIG_DIR / "knowledge.db"),
        },
        "obsidian": {"enabled": False, "vault_path": ""},
        "paper": {
            "summary": {
                "parser": PUBLIC_PARSER_DEFAULT,
                "opendataloader": {
                    "reading_order": PUBLIC_OPENDATALOADER_READING_ORDER_DEFAULT,
                    "use_struct_tree": PUBLIC_OPENDATALOADER_USE_STRUCT_TREE_DEFAULT,
                    "table_method": PUBLIC_OPENDATALOADER_TABLE_METHOD_DEFAULT,
                },
            },
        },
    },
    "hybrid": {
        "translation": {
            "provider": PUBLIC_DEFAULT_TRANSLATION_PROVIDER,
            "model": PUBLIC_DEFAULT_TRANSLATION_MODEL,
        },
        "summarization": {
            "provider": PUBLIC_DEFAULT_TRANSLATION_PROVIDER,
            "model": PUBLIC_DEFAULT_TRANSLATION_MODEL,
        },
        "embedding": {
            "provider": PUBLIC_DEFAULT_EMBEDDING_PROVIDER,
            "model": PUBLIC_DEFAULT_EMBEDDING_MODEL,
            "fallback_provider": "openai",
            "fallback_model": "text-embedding-3-small",
        },
        "providers": {
            "ollama": {"base_url": "http://localhost:11434"},
        },
        "storage": {
            "papers_dir": str(DEFAULT_CONFIG_DIR / "papers"),
            "vector_db": str(DEFAULT_CONFIG_DIR / "chroma_db"),
            "sqlite": str(DEFAULT_CONFIG_DIR / "knowledge.db"),
        },
        "obsidian": {"enabled": False, "vault_path": ""},
        "paper": {
            "summary": {
                "parser": PUBLIC_PARSER_DEFAULT,
                "opendataloader": {
                    "reading_order": PUBLIC_OPENDATALOADER_READING_ORDER_DEFAULT,
                    "use_struct_tree": PUBLIC_OPENDATALOADER_USE_STRUCT_TREE_DEFAULT,
                    "table_method": PUBLIC_OPENDATALOADER_TABLE_METHOD_DEFAULT,
                },
            },
        },
    },
}

DEFAULT_CONFIG = {
    "translation": {
        "provider": "openai",
        "model": "gpt-5-nano",
    },
    "summarization": {
        "provider": "ollama",
        "model": "qwen3:14b",
    },
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
    },
    "storage": {
        "papers_dir": str(DEFAULT_CONFIG_DIR / "papers"),
        "vector_db": str(DEFAULT_CONFIG_DIR / "chroma_db"),
        "sqlite": str(DEFAULT_CONFIG_DIR / "knowledge.db"),
    },
    "paper": {
        "summary": {
            "parser": PUBLIC_PARSER_DEFAULT,
            "opendataloader": {
                "reading_order": PUBLIC_OPENDATALOADER_READING_ORDER_DEFAULT,
                "use_struct_tree": PUBLIC_OPENDATALOADER_USE_STRUCT_TREE_DEFAULT,
                "table_method": PUBLIC_OPENDATALOADER_TABLE_METHOD_DEFAULT,
            },
        },
        "memory": {
            "extraction_mode": "schema",
            "allow_external": True,
            "extractor_timeout_sec": 90,
        },
    },
    "indexing": {
        "failure_report_dir": str(DEFAULT_CONFIG_DIR / "runs"),
        "embed_batch_size": 24,
        "embed_pause_ms": 50,
    },
    "pipeline": {
        "storage": {
            "root": "/Volumes/T9/knowledge_os",
        },
        "profile": "safe",
        "source_policy": "hybrid",
        "allowlist_domains": [],
        "resource": {
            "memory_high_watermark": 0.82,
            "cpu_pause_threshold": 0.90,
            "backoff_base_sec": 1.0,
            "backoff_max_sec": 30.0,
        },
        "profiles": {
            "safe": {
                "workers": 2,
                "download_concurrency": 2,
                "normalize_concurrency": 2,
                "embed_batch_size": 4,
            },
            "balanced": {
                "workers": 4,
                "download_concurrency": 4,
                "normalize_concurrency": 3,
                "embed_batch_size": 8,
            },
            "fast": {
                "workers": 8,
                "download_concurrency": 8,
                "normalize_concurrency": 6,
                "embed_batch_size": 16,
            },
        },
        "retry": {
            "max_retries": 3,
        },
        "health": {
            "failed_ratio_threshold": 0.4,
        },
    },
    "materialization": {
        "enabled": True,
        "source_score_threshold": 0.66,
        "source_min_entities": 2,
        "source_min_relations": 0,
        "summary_fallback_source_score_threshold": 0.46,
        "concept_score_threshold": 0.68,
        "concept_support_source_score_threshold": 0.60,
        "key_excerpt_threshold": 0.82,
        "max_source_notes_per_run": 50,
        "max_concept_notes_per_run": 30,
        "glossary_path": str(DEFAULT_CONFIG_DIR / "ko_glossary.yaml"),
        "run_stale_after_sec": 1800,
        "enrichment": {
            "enabled": True,
            "source_context_chars": 14000,
            "source_excerpt_count": 4,
            "concept_support_docs": 8,
            "existing_top_source_limit": 120,
            "existing_top_concept_limit": 80,
            "minimum_source_bullets_per_section": 3,
            "minimum_concept_support_docs": 4,
            "version": "v1",
        },
    },
    "validation": {
        "schema": {
            "strict": True,
        },
    },
    "obsidian": {
        "enabled": False,
        "vault_path": "",
        "vault_name": "",
        "exclude_folders": [".obsidian", ".trash", "templates"],
        "notes_folder": "Papers",
        "ko_notes_staging_folder": "LearningHub/ai/ko_notes",
        "web_sources_folder": "Projects/AI/AI_Papers/Web_Sources",
        "concepts_folder": "Projects/AI/AI_Papers/Concepts",
        "write_backend": "filesystem",
        "cli_binary": "obsidian",
    },
    "learning": {
        "gap": {
            "min_confidence": 0.7,
        },
        "quiz": {
            "mix": "mixed",
        },
        "llm": {
            "escalation": {
                "enabled": True,
                "provider": "anthropic",
                "model": "claude-opus-4-20250514",
                "trigger": {
                    "low_confidence_threshold": 0.7,
                    "normalization_failure_threshold": 0.3,
                    "verify_retry_failed": True,
                },
            },
        },
    },
    "routing": {
        "llm": {
            "tasks": {
                "local": {
                    "provider": "ollama",
                    "model": "qwen3:14b",
                    "timeout_sec": 45,
                },
                "mini": {
                    "provider": "openai",
                    "model": "gpt-5-nano",
                    "timeout_sec": 60,
                },
                "strong": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "timeout_sec": 90,
                },
                "defaults": {
                    "title_short_summary": "local",
                    "translation": "mini",
                    "paper_memory_extraction": "strong",
                    "materialization_summary": "mini",
                    "materialization_source_enrichment": "mini",
                    "materialization_concept_enrichment": "strong",
                    "learning_graph_refinement": "strong",
                    "rag_answer": "auto",
                    "claim_extraction": "strong",
                    "predicate_reasoning": "strong",
                    "learning_reinforce": "auto",
                },
            },
            "hybrid": {
                "enabled": True,
                "fallback_enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "qwen3:14b",
                },
                "api": {
                    "provider": "openai-compat",
                    "model": "sonar-pro",
                },
                "complexity_threshold": 2200,
                "query_char_threshold": 140,
                "context_char_threshold": 8000,
                "source_weight": 120,
                "reasoning_boost": 480,
            },
        },
    },
    "quality_mode": {
        "enabled": True,
        "core_topics": [
            "large language models",
            "rag",
            "agents",
            "multimodal",
            "safety",
        ],
        "routing": {
            "non_core_external_allowed": False,
            "source_external_route": "mini",
            "concept_external_route": "strong",
            "claim_external_route": "strong",
            "learning_external_route": "strong",
        },
        "caps": {
            "mini_max_source_items_per_run": 50,
            "strong_max_concept_items_per_run": 10,
            "strong_max_claim_refinements_per_run": 20,
            "strong_max_learning_refinements_per_run": 10,
            "monthly_usd_cap": 30.0,
        },
        "cost_estimates": {
            "mini_source_item_usd": 0.01,
            "strong_concept_item_usd": 0.05,
            "strong_claim_item_usd": 0.03,
            "strong_learning_item_usd": 0.03,
        },
    },
    "labs": {
        "answer_readiness": {
            "paper_short_citation_first": {
                "enabled": False,
                "budget_v2": {
                    "enabled": False,
                    "max_bullets": 2,
                    "output_max_tokens": 256,
                    "context_items": 2,
                    "context_excerpt_chars": 160,
                },
            },
        },
    },
    "providers": {
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY}",
        },
        "google": {
            "api_key": "${GOOGLE_API_KEY}",
        },
        "ollama": {
            "base_url": "http://localhost:11434",
        },
        "pplx-local": {
            "base_url": "http://localhost:8080",
            "timeout": 60,
        },
        "pplx-st": {
            "batch_size": 8,
            "device": "auto",
            "torch_num_threads": 1,
            "disable_tokenizers_parallelism": True,
            "max_chars_per_chunk": 1000,
            "chunk_overlap_chars": 200,
            "normalize_embeddings": True,
            "trust_remote_code": True,
            "load_timeout_sec": 600,
            "encode_timeout_sec": 180,
            "auto_batch_backoff": True,
            "min_batch_size": 1,
        },
    },
}


def _expand_env_vars(value: Any) -> Any:
    """${VAR_NAME} 패턴을 환경 변수 값으로 치환"""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return re.sub(r"\$\{(\w+)\}", replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """딥 머지: override 값이 base를 덮어씀"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_public_setup_profile(name: str) -> dict[str, Any]:
    token = str(name or "").strip().lower()
    if token == "custom":
        return {}
    profile = PUBLIC_SETUP_PROFILES.get(token)
    if profile is None:
        raise ConfigError(f"unknown public setup profile: {name}")
    return deepcopy(profile)


def get_public_prompt_defaults() -> dict[str, dict[str, str]]:
    return deepcopy(PUBLIC_PROMPT_DEFAULTS)


def apply_public_setup_profile(config: Any, name: str) -> dict[str, Any]:
    profile = get_public_setup_profile(name)

    def _apply_tree(prefix: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                _apply_tree(prefix + (key,), nested_value)
            return
        config.set_nested(*prefix, value)

    for key, value in profile.items():
        _apply_tree((key,), value)
    return profile


class Config:
    """통합 설정 클래스 - ~/.khub/config.yaml 기반"""

    _instance: Optional[Config] = None

    def __init__(self, config_path: Optional[str] = None):
        self._data: dict = {}
        self._raw_data: dict = {}
        self._path: Optional[Path] = None

        if config_path:
            self.load(config_path)
        elif DEFAULT_CONFIG_PATH.exists():
            self.load(str(DEFAULT_CONFIG_PATH))
        else:
            self._data = dict(DEFAULT_CONFIG)
            self._raw_data = {}

    @classmethod
    def get(cls, config_path: Optional[str] = None) -> Config:
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """싱글톤 초기화 (테스트용)"""
        cls._instance = None

    def load(self, config_path: str):
        """YAML 설정 파일 로드 (기본값과 머지)"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
        self._raw_data = dict(user_data)
        self._data = _deep_merge(DEFAULT_CONFIG, user_data)

    def save(self, config_path: Optional[str] = None):
        """현재 설정을 YAML 파일로 저장"""
        path = Path(config_path) if config_path else (self._path or DEFAULT_CONFIG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._path = path

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get_nested(self, *keys, default=None) -> Any:
        """중첩 키로 값 가져오기: config.get_nested('embedding', 'model')"""
        d = self._data
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return _expand_env_vars(d)

    def get_first_nested(self, *key_paths: tuple[str, ...], default=None) -> Any:
        """여러 중첩 키 후보 중 첫 번째로 존재하는 값을 반환"""
        for key_path in key_paths:
            d = self._raw_data
            found = True
            for key in key_path:
                if isinstance(d, dict) and key in d:
                    d = d[key]
                else:
                    found = False
                    break
            if found:
                return _expand_env_vars(d)
        return default

    def set_nested(self, *keys_and_value):
        """중첩 키에 값 설정: config.set_nested('translation', 'provider', 'openai')"""
        if len(keys_and_value) < 2:
            raise ValueError("최소 키 1개와 값 1개가 필요합니다")
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        d = self._data
        raw = self._raw_data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
            if k not in raw or not isinstance(raw[k], dict):
                raw[k] = {}
            raw = raw[k]
        d[keys[-1]] = value
        raw[keys[-1]] = value

    def get_provider_config(self, provider_name: str) -> dict:
        """특정 프로바이더의 설정 반환 (env var 확장 적용)"""
        raw = self.get_nested("providers", provider_name, default={})
        return _expand_env_vars(raw) if isinstance(raw, dict) else {}

    # --- 편의 프로퍼티 ---

    @property
    def translation_provider(self) -> str:
        return self.get_nested("translation", "provider", default="openai")

    @property
    def translation_model(self) -> str:
        return self.get_nested("translation", "model", default="gpt-5-nano")

    @property
    def summarization_provider(self) -> str:
        return self.get_nested("summarization", "provider", default="ollama")

    @property
    def summarization_model(self) -> str:
        return self.get_nested("summarization", "model", default="qwen3:14b")

    @property
    def embedding_provider(self) -> str:
        return self.get_nested("embedding", "provider", default="ollama")

    @property
    def embedding_model(self) -> str:
        return self.get_nested("embedding", "model", default="nomic-embed-text")

    @property
    def paper_summary_parser(self) -> str:
        return self.get_nested("paper", "summary", "parser", default=PUBLIC_PARSER_DEFAULT)

    @property
    def papers_dir(self) -> str:
        return str(
            Path(
                self.get_first_nested(
                    ("storage", "papers_dir"),
                    ("papers", "download_dir"),
                    default=str(DEFAULT_CONFIG_DIR / "papers"),
                )
            ).expanduser()
        )

    @property
    def vector_db_path(self) -> str:
        return str(
            Path(
                self.get_first_nested(
                    ("storage", "vector_db"),
                    ("database", "vector_db_path"),
                    default=str(DEFAULT_CONFIG_DIR / "chroma_db"),
                )
            ).expanduser()
        )

    @property
    def sqlite_path(self) -> str:
        return str(
            Path(
                self.get_first_nested(
                    ("storage", "sqlite"),
                    ("database", "sqlite_path"),
                    default=str(DEFAULT_CONFIG_DIR / "knowledge.db"),
                )
            ).expanduser()
        )

    @property
    def vault_path(self) -> str:
        return self.get_first_nested(
            ("obsidian", "vault_path"),
            ("vault", "path"),
            default="",
        )

    @property
    def vault_enabled(self) -> bool:
        explicit = self.get_first_nested(("obsidian", "enabled"), default=None)
        if explicit is not None:
            return bool(explicit)
        return bool(self.vault_path)

    @property
    def vault_excludes(self) -> list:
        configured = self.get_nested("obsidian", "exclude_folders", default=[".obsidian", ".trash"])
        return resolve_vault_exclude_folders(configured)

    @property
    def obsidian_notes_folder(self) -> str:
        return self.get_nested("obsidian", "notes_folder", default="Papers")

    @property
    def obsidian_ko_notes_staging_folder(self) -> str:
        return self.get_nested("obsidian", "ko_notes_staging_folder", default="LearningHub/ai/ko_notes")

    @property
    def obsidian_web_sources_folder(self) -> str:
        return self.get_nested("obsidian", "web_sources_folder", default="Projects/AI/AI_Papers/Web_Sources")

    @property
    def obsidian_concepts_folder(self) -> str:
        return self.get_nested("obsidian", "concepts_folder", default="Projects/AI/AI_Papers/Concepts")

    @property
    def collection_name(self) -> str:
        return self.get_first_nested(
            ("storage", "collection_name"),
            ("storage", "collection"),
            ("database", "collection_name"),
            ("database", "collection"),
            default="knowledge_hub",
        )

    @property
    def indexing_failure_report_dir(self) -> str:
        return str(
            Path(
                self.get_nested("indexing", "failure_report_dir", default=str(DEFAULT_CONFIG_DIR / "runs"))
            ).expanduser()
        )

    # --- 이전 API 호환 ---

    @property
    def embed_model(self) -> str:
        return self.embedding_model

    @property
    def embed_base_url(self) -> str:
        return self.get_provider_config(self.embedding_provider).get("base_url", "http://localhost:11434")

    @property
    def llm_model(self) -> str:
        return self.summarization_model

    @property
    def llm_base_url(self) -> str:
        return self.get_provider_config(self.summarization_provider).get("base_url", "http://localhost:11434")

    @property
    def db_path(self) -> str:
        return self.vector_db_path

    @property
    def translate_model(self) -> str:
        return self.translation_model

    @property
    def papers_csv(self) -> str:
        return self.get_nested("papers", "csv_path", default="")

    @property
    def chunk_size(self) -> int:
        return self.get_nested("embedding", "chunk_size", default=1000)

    @property
    def chunk_overlap(self) -> int:
        return self.get_nested("embedding", "chunk_overlap", default=200)

    @property
    def top_k(self) -> int:
        return self.get_nested("storage", "top_k", default=5)

    @property
    def config_path(self) -> Optional[str]:
        return str(self._path) if self._path else None

    @property
    def data(self) -> dict:
        return self._data

    # --- 검증 ---

    def validate(self, require_providers: list[str] | None = None):
        """설정 무결성 검증. 실패 시 ConfigError 발생.

        require_providers: 반드시 API 키가 있어야 하는 프로바이더 목록
                          (예: ["openai"])
        """
        errors: list[str] = []

        storage_dirs = [
            ("storage.papers_dir", self.papers_dir),
            ("storage.vector_db", self.vector_db_path),
        ]
        for label, dir_path in storage_dirs:
            parent = Path(dir_path).parent
            if not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    errors.append(f"{label}: 디렉토리 생성 불가 ({e})")

        for provider in (require_providers or []):
            from knowledge_hub.providers import registry

            info = registry.get_provider_info(provider)
            if info is not None and not bool(info.requires_api_key):
                log.debug("Skipping API key validation for local provider [%s]", provider)
                continue
            cfg = self.get_provider_config(provider)
            key = resolve_api_key(provider, cfg.get("api_key", ""))
            if not key:
                errors.append(
                    f"providers.{provider}.api_key 누락 — "
                    f"환경변수 또는 config.yaml에 설정하세요"
                )
            else:
                log.debug("API key [%s]: %s", provider, mask_secret(key))

        if errors:
            msg = "설정 검증 실패:\n  " + "\n  ".join(errors)
            raise ConfigError(msg)

    def require_api_key(self, provider: str) -> str:
        """특정 프로바이더의 API 키를 반환. 없으면 ConfigError."""
        cfg = self.get_provider_config(provider)
        key = resolve_api_key(provider, cfg.get("api_key", ""))
        if not key:
            raise ConfigError(
                f"{provider} API 키가 설정되지 않았습니다. "
                f"OPENAI_API_KEY 등 환경변수를 설정하세요."
            )
        return key
