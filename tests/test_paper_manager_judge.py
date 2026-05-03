from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.discoverer import DiscoveredPaper
from knowledge_hub.papers.manager import PaperManager


class _FakeVectorDB:
    def has_metadata(self, metadata):  # noqa: ANN001
        _ = metadata
        return False

    def add_documents(self, documents, embeddings, metadatas, ids):  # noqa: ANN001
        _ = (documents, embeddings, metadatas, ids)


class _FakeEmbedder:
    def embed_batch(self, texts, show_progress=False):  # noqa: ANN001
        _ = show_progress
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeLLM:
    def generate(self, prompt: str, context: str = "", max_tokens: int | None = None) -> str:
        _ = (prompt, context, max_tokens)
        return "간단 요약"


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    config.set_nested("obsidian", "vault_path", str(tmp_path / "vault"))
    config.set_nested("obsidian", "notes_folder", "Papers")
    config.set_nested("summarization", "provider", "ollama")
    config.set_nested("summarization", "model", "qwen")
    return config


def test_manager_discover_and_ingest_only_writes_selected_judged_notes(monkeypatch, tmp_path):
    config = _config(tmp_path)
    vault = Path(config.vault_path)
    vault.mkdir(parents=True, exist_ok=True)
    db = SQLiteDatabase(config.sqlite_path)
    manager = PaperManager(
        config=config,
        vector_db=_FakeVectorDB(),
        sqlite_db=db,
        embedder=_FakeEmbedder(),
    )

    papers = [
        DiscoveredPaper(
            arxiv_id="2501.00001",
            title="Agent Retrieval",
            authors="A",
            year=2025,
            abstract="Retrieval agents for code tasks.",
            citation_count=12,
            fields_of_study=["AI"],
        ),
        DiscoveredPaper(
            arxiv_id="2501.00002",
            title="Unrelated Biology",
            authors="B",
            year=2025,
            abstract="Biology abstract.",
            citation_count=1,
            fields_of_study=["Biology"],
        ),
    ]

    monkeypatch.setattr("knowledge_hub.papers.manager.discover_papers", lambda **kwargs: papers)
    monkeypatch.setattr("knowledge_hub.papers.manager.PaperDownloader.download_single", lambda self, arxiv_id, title: {"pdf": None, "text": None})  # noqa: ARG005
    monkeypatch.setattr("knowledge_hub.papers.manager.PaperManager._find_related_vault_notes", lambda self, paper, topic, top_k=5: [])  # noqa: ARG005

    def _fake_select(self, candidates, *, topic, threshold=None, top_k=None, user_goal=""):  # noqa: ANN001
        _ = (self, topic, threshold, top_k, user_goal)
        return [candidates[0]], {
            "backend": "rule_llm_v1",
            "threshold": 0.62,
            "candidateCount": 2,
            "selectedCount": 1,
            "degraded": False,
            "warnings": [],
            "items": [
                {
                    "paper_id": "2501.00001",
                    "decision": "keep",
                    "total_score": 0.82,
                    "backend": "rule_llm_v1",
                    "dimension_scores": {
                        "relevance_score": 0.9,
                        "novelty_score": 0.7,
                        "read_value_score": 0.8,
                        "citation_signal_score": 0.4,
                    },
                    "top_reasons": ["good topic match", "worth reading now"],
                },
                {
                    "paper_id": "2501.00002",
                    "decision": "skip",
                    "total_score": 0.2,
                    "backend": "rule_llm_v1",
                    "dimension_scores": {},
                    "top_reasons": ["bad match"],
                },
            ],
        }

    monkeypatch.setattr("knowledge_hub.papers.judge.PaperJudgeService.select_candidates", _fake_select)

    try:
        result = manager.discover_and_ingest(
            topic="agent retrieval",
            max_papers=2,
            create_obsidian_note=True,
            generate_summary=True,
            llm=_FakeLLM(),
            judge_enabled=True,
        )
    finally:
        db.close()

    note_path = vault / "Papers" / "Agent Retrieval.md"
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "judge_enabled: true" in content
    assert "## Judge Assessment" in content
    assert "good topic match" in content
    assert not (vault / "Papers" / "Unrelated Biology.md").exists()
    assert result["judge"]["selectedCount"] == 1
    assert validate_payload(result, result["schema"], strict=True).ok
    log_path = Path(config.sqlite_path).parent / "paper_judge_events.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert events[0]["source"] == "discover_and_ingest"
    assert events[0]["event_type"] == "judge_decision"
