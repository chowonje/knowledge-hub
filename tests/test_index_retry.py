"""index_cmd.py 임베딩 재시도 로직 테스트"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import requests

from knowledge_hub.cli.index_cmd import _embed_with_retry


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class TestEmbedWithRetry:
    def test_success_on_first_try(self):
        fake_embs = [[0.1, 0.2, 0.3]]
        with patch("knowledge_hub.cli.index_cmd._embed_batch_openai", return_value=fake_embs):
            result = _embed_with_retry(
                ["test text"], "openai", "text-embedding-3-small", api_key="sk-test",
            )
        assert result == fake_embs

    def test_retries_on_429(self):
        http_err = requests.HTTPError(response=_FakeResponse(429))
        fake_embs = [[0.1, 0.2]]

        call_count = {"n": 0}
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise http_err
            return fake_embs

        with patch("knowledge_hub.cli.index_cmd._embed_batch_openai", side_effect=side_effect):
            with patch("knowledge_hub.cli.index_cmd.EMBED_RETRY_BASE_SEC", 0.01):
                result = _embed_with_retry(
                    ["a"], "openai", "model", api_key="sk-test",
                )
        assert result == fake_embs
        assert call_count["n"] == 3

    def test_raises_on_401_no_retry(self):
        http_err = requests.HTTPError(response=_FakeResponse(401))
        with patch("knowledge_hub.cli.index_cmd._embed_batch_openai", side_effect=http_err):
            with pytest.raises(requests.HTTPError):
                _embed_with_retry(
                    ["a"], "openai", "model", api_key="bad-key",
                )

    def test_exhausts_retries_raises(self):
        http_err = requests.HTTPError(response=_FakeResponse(500))
        with patch("knowledge_hub.cli.index_cmd._embed_batch_openai", side_effect=http_err):
            with patch("knowledge_hub.cli.index_cmd.EMBED_RETRY_BASE_SEC", 0.01):
                with pytest.raises(requests.HTTPError):
                    _embed_with_retry(
                        ["a"], "openai", "model", api_key="sk-test",
                    )
