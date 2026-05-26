"""index_cmd.py 임베딩 재시도 로직 테스트"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import requests

from knowledge_hub.interfaces.cli.commands.index_cmd import _embed_batch_via_provider, _embed_with_retry, _run_index_batches
from knowledge_hub.providers.ollama import OllamaEmbedder


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _Config:
    def get_nested(self, *path, default=None):
        values = {
            ("indexing", "embed_batch_size"): 1,
            ("indexing", "min_batch_size"): 1,
            ("indexing", "max_batch_retries"): 0,
            ("indexing", "embed_pause_ms"): 0,
            ("indexing", "auto_batch_backoff"): True,
        }
        return values.get(tuple(path), default)


class _ContextLimitEmbedder:
    def __init__(self, *, max_chars: int = 10):
        self.max_chars = max_chars
        self._last_status = {"retries": 0, "failures": []}

    def embed_batch(self, texts, show_progress=False):
        _ = show_progress
        failures = []
        results = []
        for index, text in enumerate(texts):
            if len(text) > self.max_chars:
                failures.append(
                    {
                        "stage": "embed_batch",
                        "errorCode": "ResponseError",
                        "message": "the input length exceeds the context length",
                        "itemIndex": index,
                    }
                )
                results.append(None)
            else:
                results.append([0.1, 0.2])
        self._last_status = {"retries": 0, "failures": failures}
        return results

    def get_last_status(self):
        return self._last_status


class _VectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []

    def add_documents(self, *, documents, embeddings, metadatas, ids):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)


class TestEmbedWithRetry:
    def test_success_on_first_try(self):
        fake_embs = [[0.1, 0.2, 0.3]]
        with patch("knowledge_hub.interfaces.cli.commands.index_cmd._embed_batch_openai", return_value=fake_embs):
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

        with patch("knowledge_hub.interfaces.cli.commands.index_cmd._embed_batch_openai", side_effect=side_effect):
            with patch("knowledge_hub.interfaces.cli.commands.index_cmd.EMBED_RETRY_BASE_SEC", 0.01):
                result = _embed_with_retry(
                    ["a"], "openai", "model", api_key="sk-test",
                )
        assert result == fake_embs
        assert call_count["n"] == 3

    def test_raises_on_401_no_retry(self):
        http_err = requests.HTTPError(response=_FakeResponse(401))
        with patch("knowledge_hub.interfaces.cli.commands.index_cmd._embed_batch_openai", side_effect=http_err):
            with pytest.raises(requests.HTTPError):
                _embed_with_retry(
                    ["a"], "openai", "model", api_key="bad-key",
                )

    def test_exhausts_retries_raises(self):
        http_err = requests.HTTPError(response=_FakeResponse(500))
        with patch("knowledge_hub.interfaces.cli.commands.index_cmd._embed_batch_openai", side_effect=http_err):
            with patch("knowledge_hub.interfaces.cli.commands.index_cmd.EMBED_RETRY_BASE_SEC", 0.01):
                with pytest.raises(requests.HTTPError):
                    _embed_with_retry(
                        ["a"], "openai", "model", api_key="sk-test",
                    )


def test_embed_batch_error_includes_provider_context_message():
    embedder = _ContextLimitEmbedder(max_chars=4)

    with pytest.raises(RuntimeError) as exc_info:
        _embed_batch_via_provider(embedder, ["too long"])

    message = str(exc_info.value)
    assert "1개 텍스트 임베딩 실패" in message
    assert "the input length exceeds the context length" in message


def test_run_index_batches_context_safe_retry_truncates_isolated_text():
    item = {"id": "paper-1", "title": "Long Paper"}
    vector_db = _VectorDB()
    marked = []
    original = "x" * 30

    result = _run_index_batches(
        config=_Config(),
        items=[item],
        embedder=_ContextLimitEmbedder(max_chars=10),
        vector_db=vector_db,
        build_text=lambda _item: original,
        build_metadata=lambda _item: {"source_type": "paper"},
        build_id=lambda _item: "paper_paper-1_0",
        failure_builder=lambda _item, message: {"id": _item["id"], "error": message},
        warning_builder=lambda _item, message: {"id": _item["id"] if _item else "", "error": message},
        context_retry_texts=lambda _item, text: [text[:8]],
        mark_indexed=lambda batch: marked.extend(item["id"] for item in batch),
    )

    assert result["succeeded"] == 1
    assert result["failures"] == []
    assert any(warning["error"].startswith("CONTEXT_SAFE_EMBED_RETRY") for warning in result["warnings"])
    assert vector_db.documents == [original[:8]]
    assert marked == ["paper-1"]


def test_ollama_embedder_records_batch_failures(monkeypatch):
    embedder = OllamaEmbedder()

    def fake_embed_text(text):
        if text == "bad":
            raise RuntimeError("the input length exceeds the context length")
        return [0.1, 0.2]

    monkeypatch.setattr(embedder, "embed_text", fake_embed_text)

    assert embedder.embed_batch(["ok", "bad"], show_progress=False) == [[0.1, 0.2], None]
    status = embedder.get_last_status()
    assert status["retries"] == 0
    assert status["failures"][0]["errorCode"] == "RuntimeError"
    assert status["failures"][0]["itemIndex"] == 1
    assert "context length" in status["failures"][0]["message"]
