"""
통합 설정 관리 모듈

~/.khub/config.yaml 또는 프로젝트 로컬 config.yaml을 지원합니다.
환경 변수 치환 (${VAR_NAME}) 및 프로바이더별 설정을 관리합니다.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

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

DEFAULT_CONFIG = {
    "translation": {
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "summarization": {
        "provider": "ollama",
        "model": "qwen2.5:7b",
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
    "obsidian": {
        "enabled": False,
        "vault_path": "",
        "exclude_folders": [".obsidian", ".trash", "templates"],
    },
    "notebooklm": {
        "project_number": "${GOOGLE_CLOUD_PROJECT}",
        "location": "global",
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


class Config:
    """통합 설정 클래스 - ~/.khub/config.yaml 기반"""

    _instance: Optional[Config] = None

    def __init__(self, config_path: Optional[str] = None):
        self._data: dict = {}
        self._path: Optional[Path] = None

        if config_path:
            self.load(config_path)
        elif DEFAULT_CONFIG_PATH.exists():
            self.load(str(DEFAULT_CONFIG_PATH))
        else:
            self._data = dict(DEFAULT_CONFIG)

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

    def set_nested(self, *keys_and_value):
        """중첩 키에 값 설정: config.set_nested('translation', 'provider', 'openai')"""
        if len(keys_and_value) < 2:
            raise ValueError("최소 키 1개와 값 1개가 필요합니다")
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

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
        return self.get_nested("translation", "model", default="gpt-4o-mini")

    @property
    def summarization_provider(self) -> str:
        return self.get_nested("summarization", "provider", default="ollama")

    @property
    def summarization_model(self) -> str:
        return self.get_nested("summarization", "model", default="qwen2.5:7b")

    @property
    def embedding_provider(self) -> str:
        return self.get_nested("embedding", "provider", default="ollama")

    @property
    def embedding_model(self) -> str:
        return self.get_nested("embedding", "model", default="nomic-embed-text")

    @property
    def papers_dir(self) -> str:
        return str(Path(self.get_nested("storage", "papers_dir", default=str(DEFAULT_CONFIG_DIR / "papers"))).expanduser())

    @property
    def vector_db_path(self) -> str:
        return str(Path(self.get_nested("storage", "vector_db", default=str(DEFAULT_CONFIG_DIR / "chroma_db"))).expanduser())

    @property
    def sqlite_path(self) -> str:
        return str(Path(self.get_nested("storage", "sqlite", default=str(DEFAULT_CONFIG_DIR / "knowledge.db"))).expanduser())

    @property
    def vault_path(self) -> str:
        return self.get_nested("obsidian", "vault_path", default="")

    @property
    def vault_enabled(self) -> bool:
        return bool(self.get_nested("obsidian", "enabled", default=False))

    @property
    def vault_excludes(self) -> list:
        return self.get_nested("obsidian", "exclude_folders", default=[".obsidian", ".trash"])

    @property
    def collection_name(self) -> str:
        return self.get_nested("storage", "collection_name", default="knowledge_hub")

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
