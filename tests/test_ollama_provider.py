from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.providers.ollama import OllamaLLM


def test_ollama_llm_maps_per_call_max_tokens_to_num_predict(monkeypatch):
    calls: list[dict[str, object]] = []

    class _FakeClient:
        def generate(self, **kwargs):  # noqa: ANN001
            calls.append(dict(kwargs))
            return {"response": "ok"}

    fake_client = _FakeClient()
    monkeypatch.setitem(
        __import__("sys").modules,
        "ollama",
        SimpleNamespace(Client=lambda host, timeout: fake_client),
    )

    llm = OllamaLLM(model="qwen3:8b", base_url="http://localhost:11434", max_tokens=2000, timeout=60)
    answer = llm.generate("prompt", "context", max_tokens=192)

    assert answer == "ok"
    assert calls[0]["model"] == "qwen3:8b"
    assert calls[0]["options"]["num_predict"] == 192
    assert llm.max_tokens == 2000
