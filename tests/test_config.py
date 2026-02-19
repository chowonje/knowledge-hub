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
