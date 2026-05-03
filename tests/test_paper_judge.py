from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_hub.core.config import Config
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.judge import PaperJudgeService


@dataclass
class _Paper:
    arxiv_id: str
    title: str
    authors: str = ""
    year: int = 2026
    abstract: str = ""
    citation_count: int = 0
    fields_of_study: list[str] = field(default_factory=list)


class _FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = (prompt, context, max_tokens)
        self.calls += 1
        return '{"read_value_score": 0.81, "top_reasons": ["matches the topic well", "worth reading now"]}'


def _config(provider: str) -> Config:
    config = Config()
    config.set_nested("summarization", "provider", provider)
    config.set_nested("summarization", "model", "test-model")
    return config


def test_paper_judge_returns_schema_valid_ranked_payload():
    service = PaperJudgeService(_config("ollama"), llm=_FakeLLM(), allow_external=False)
    papers = [
        _Paper(
            arxiv_id="2501.00001",
            title="Agentic RAG for Code Retrieval",
            abstract="This paper studies retrieval grounded agent systems for code tasks.",
            citation_count=34,
            fields_of_study=["Artificial Intelligence"],
        ),
        _Paper(
            arxiv_id="2501.00002",
            title="Biology Survey",
            abstract="Protein folding survey.",
            citation_count=2,
            fields_of_study=["Biology"],
        ),
    ]

    payload = service.evaluate_candidates(papers, topic="agent retrieval for code")

    assert payload["items"][0]["paper_id"] == "2501.00001"
    assert payload["items"][0]["decision"] == "keep"
    assert payload["candidateCount"] == 2
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_paper_judge_degrades_to_rule_only_when_external_is_disallowed():
    llm = _FakeLLM()
    service = PaperJudgeService(_config("openai"), llm=llm, allow_external=False)
    papers = [
        _Paper(
            arxiv_id="2501.00003",
            title="Retrieval Safety Notes",
            abstract="Safety-aware retrieval for agents.",
            citation_count=8,
            fields_of_study=["Computer Science"],
        )
    ]

    payload = service.evaluate_candidates(papers, topic="retrieval safety")

    assert payload["degraded"] is True
    assert "external not allowed" in " ".join(payload["warnings"])
    assert payload["items"][0]["llm_used"] is False
    assert llm.calls == 0
