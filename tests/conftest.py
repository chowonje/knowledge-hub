"""공통 pytest fixtures"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_dir(tmp_path):
    """깨끗한 임시 디렉토리"""
    return tmp_path


@pytest.fixture()
def fake_env(tmp_path):
    """테스트용 환경변수 세트"""
    env = {
        "OPENAI_API_KEY": "sk-test-fake-key-for-unit-tests",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture()
def config_dir(tmp_path):
    """격리된 config 디렉토리"""
    cfg_dir = tmp_path / ".khub"
    cfg_dir.mkdir()
    (cfg_dir / "papers").mkdir()
    return cfg_dir
