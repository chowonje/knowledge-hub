"""config.py 핵심 실패 경로 테스트"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from knowledge_hub.core.config import (
    Config,
    ConfigError,
    mask_secret,
    resolve_api_key,
)


class TestMaskSecret:
    def test_short_value(self):
        assert mask_secret("ab") == "***"

    def test_normal_key(self):
        result = mask_secret("sk-abcdefghijklmnop")
        assert result.startswith("sk-a")
        assert "*" in result
        assert len(result) == len("sk-abcdefghijklmnop")

    def test_empty(self):
        assert mask_secret("") == "***"

    def test_none_like(self):
        assert mask_secret("") == "***"


class TestResolveApiKey:
    def test_env_var_takes_priority(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            assert resolve_api_key("openai", "sk-from-config") == "sk-from-env"

    def test_config_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            assert resolve_api_key("openai", "sk-from-config") == "sk-from-config"

    def test_unresolved_template_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            assert resolve_api_key("openai", "${OPENAI_API_KEY}") == ""

    def test_unknown_provider(self):
        assert resolve_api_key("unknown_provider", "some-key") == "some-key"


class TestConfigValidation:
    def test_missing_api_key_raises(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
            "providers": {"openai": {"api_key": ""}},
        }))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = Config(str(cfg_path))
            with pytest.raises(ConfigError, match="api_key 누락"):
                config.validate(require_providers=["openai"])

    def test_valid_config_passes(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "embedding": {"provider": "openai"},
        }))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = Config(str(cfg_path))
            config.validate(require_providers=["openai"])

    def test_local_provider_does_not_require_api_key(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "embedding": {"provider": "ollama"},
        }))

        class _ProviderInfo:
            requires_api_key = False

        monkeypatch.setattr("knowledge_hub.providers.registry.get_provider_info", lambda _provider: _ProviderInfo())
        config = Config(str(cfg_path))
        config.validate(require_providers=["ollama"])

    def test_require_api_key_missing(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = Config(str(cfg_path))
            with pytest.raises(ConfigError):
                config.require_api_key("openai")

    def test_require_api_key_present(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real"}):
            config = Config(str(cfg_path))
            key = config.require_api_key("openai")
            assert key == "sk-real"


class TestConfigFileNotFound:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")


class TestConfigCompatibility:
    def test_storage_collection_alias_is_supported(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "storage": {
                        "collection": "legacy_collection",
                    }
                }
            ),
            encoding="utf-8",
        )

        config = Config(str(cfg_path))

        assert config.collection_name == "legacy_collection"

    def test_legacy_example_paths_still_resolve(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "vault": {"path": str(tmp_path / "vault")},
                    "database": {
                        "vector_db_path": str(tmp_path / "db" / "chroma"),
                        "sqlite_path": str(tmp_path / "db" / "knowledge.db"),
                        "collection_name": "legacy_db_collection",
                    },
                    "papers": {
                        "download_dir": str(tmp_path / "papers"),
                    },
                }
            ),
            encoding="utf-8",
        )

        config = Config(str(cfg_path))

        assert config.vault_path == str(tmp_path / "vault")
        assert config.vault_enabled is True
        assert config.vector_db_path == str(tmp_path / "db" / "chroma")
        assert config.sqlite_path == str(tmp_path / "db" / "knowledge.db")
        assert config.collection_name == "legacy_db_collection"
        assert config.papers_dir == str(tmp_path / "papers")

    def test_paper_answer_readiness_p1_flag_defaults_off(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}), encoding="utf-8")

        config = Config(str(cfg_path))

        assert config.get_nested("labs", "answer_readiness", "paper_short_citation_first", "enabled") is False
        assert (
            config.get_nested("labs", "answer_readiness", "paper_short_citation_first", "budget_v2", "enabled")
            is False
        )
        assert config.get_nested("labs", "answer_readiness", "paper_short_citation_first", "budget_v2", "max_bullets") == 2
        assert (
            config.get_nested("labs", "answer_readiness", "paper_short_citation_first", "budget_v2", "output_max_tokens")
            == 256
        )
        assert config.get_nested("labs", "answer_readiness", "paper_short_citation_first", "budget_v2", "context_items") == 2
        assert (
            config.get_nested(
                "labs",
                "answer_readiness",
                "paper_short_citation_first",
                "budget_v2",
                "context_excerpt_chars",
            )
            == 160
        )
