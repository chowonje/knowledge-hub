from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands.paper_cmd import _extract_note_concepts
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.interfaces.cli.commands.paper_shared_runtime import _collect_paper_text
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.core.models import ClaimCandidate
from knowledge_hub.learning.task_router import TaskRouteDecision


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


class _FakeLLM:
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
        _ = (source_lang, target_lang)
        return f"번역:{text[:20]}"

    def summarize(self, text: str, language: str = "ko", max_sentences: int = 5) -> str:
        _ = (text, language, max_sentences)
        return "요약"

    def summarize_paper(self, text: str, title: str = "", language: str = "ko") -> str:
        _ = (text, title, language)
        return "### 한줄 요약\n요약"


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    config.set_nested("translation", "provider", "openai")
    config.set_nested("translation", "model", "gpt-5-mini")
    config.set_nested("summarization", "provider", "openai")
    config.set_nested("summarization", "model", "gpt-5-mini")
    return config


def _seed_paper(config: Config, tmp_path: Path) -> None:
    text_path = tmp_path / "paper.txt"
    text_path.write_text("Transformer uses attention. " * 40, encoding="utf-8")
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Transformer Test Paper",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": None,
            "text_path": str(text_path),
            "translated_path": None,
        }
    )
    db.close()


def test_collect_paper_text_prefers_parsed_markdown_over_raw_text(tmp_path: Path):
    config = _config(tmp_path)
    papers_dir = Path(config.papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_path / "paper.txt"
    raw_path.write_text("\\documentclass{article}\n\\hypersetup{colorlinks=true}\n", encoding="utf-8")
    parsed_dir = papers_dir / "parsed" / "2501.00002"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.joinpath("document.md").write_text(
        "# Parsed Paper\n\n## Abstract\n\n"
        + ("This parsed markdown contains the real paper summary and method. " * 8)
        + "\n",
        encoding="utf-8",
    )
    paper = {
        "arxiv_id": "2501.00002",
        "title": "Parsed Runtime Paper",
        "authors": "A",
        "field": "AI",
        "notes": "",
        "text_path": str(raw_path),
        "translated_path": None,
    }

    text = _collect_paper_text(paper, config)

    assert "parsed markdown contains the real paper summary" in text
    assert "\\documentclass" not in text


def test_paper_translate_defaults_to_no_external(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config, tmp_path)
    runner = CliRunner()
    seen: dict[str, object] = {}

    def _fake_resolve(config_obj, **kwargs):  # noqa: ANN003
        _ = config_obj
        seen.update(kwargs)
        return (
            _FakeLLM(),
            TaskRouteDecision(
                task_type="translation",
                route="local",
                provider="ollama",
                model="qwen2.5:7b",
                timeout_sec=45,
                fallback_chain=["local", "fallback-only"],
                reasons=["test"],
                allow_external_effective=False,
                complexity_score=0,
                policy_mode="local-only",
            ),
            [],
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.paper_cmd._resolve_routed_llm", _fake_resolve)
    result = runner.invoke(paper_group, ["translate", "2501.00001"], obj={"khub": _StubKhub(config)})
    assert result.exit_code == 0
    assert seen["allow_external"] is False
    assert seen["llm_mode"] == "auto"


def test_paper_summarize_allow_external_mini(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    _seed_paper(config, tmp_path)
    runner = CliRunner()
    seen: dict[str, object] = {}

    def _fake_resolve(config_obj, **kwargs):  # noqa: ANN003
        _ = config_obj
        seen.update(kwargs)
        return (
            _FakeLLM(),
            TaskRouteDecision(
                task_type="rag_answer",
                route="mini",
                provider="openai",
                model="gpt-5-mini",
                timeout_sec=60,
                fallback_chain=["mini", "fallback-only"],
                reasons=["test"],
                allow_external_effective=True,
                complexity_score=2200,
                policy_mode="external-allowed",
            ),
            [],
        )

    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.paper_cmd._resolve_routed_llm", _fake_resolve)
    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.paper_cmd._update_obsidian_summary", lambda *a, **k: None)
    result = runner.invoke(
        paper_group,
        ["summarize", "2501.00001", "--allow-external", "--llm-mode", "mini"],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0
    assert seen["allow_external"] is True
    assert seen["llm_mode"] == "mini"


def test_paper_sync_keywords_claims_store_breakdown(monkeypatch, tmp_path: Path):
    config = _config(tmp_path)
    vault_path = tmp_path / "vault"
    papers_dir = vault_path / "Papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    config.set_nested("obsidian", "vault_path", str(vault_path))

    note_path = papers_dir / "Transformer Benchmark.md"
    note_path.write_text(
        (
            'arxiv_id: "2501.00001"\n\n'
            "## 요약\n"
            "Transformer benchmark study shows the model reduces latency by 32% on EvalSet while "
            "preserving answer quality across multiple tasks.\n"
        ),
        encoding="utf-8",
    )

    candidates = [
        ClaimCandidate(
            claim_text="Transformer reduces latency by 32% on EvalSet.",
            subject="Transformer",
            predicate="reduces",
            object_value="latency",
            evidence="Controlled experiments show Transformer reduces latency by 32% on EvalSet.",
            llm_confidence=0.98,
        ),
        ClaimCandidate(
            claim_text="Transformer improves results for several tasks.",
            subject="Transformer",
            predicate="improves",
            object_value="results",
            evidence="Experiments often show Transformer improves results for several tasks.",
            llm_confidence=0.99,
        ),
    ]

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._resolve_routed_llm",
        lambda *args, **kwargs: (
            object(),
            TaskRouteDecision(
                task_type="claim_extraction",
                route="local",
                provider="ollama",
                model="qwen2.5:7b",
                timeout_sec=45,
                fallback_chain=["local", "fallback-only"],
                reasons=["test"],
                allow_external_effective=False,
                complexity_score=0,
                policy_mode="local-only",
            ),
            [],
        ),
    )
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_cmd._extract_keywords_with_evidence",
        lambda *args, **kwargs: [{"concept": "latency", "evidence": "EvalSet latency measurements", "confidence": 0.9}],
    )
    monkeypatch.setattr("knowledge_hub.papers.claim_extractor.extract_claim_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.paper_cmd._update_note_concepts", lambda content, concepts: content)
    monkeypatch.setattr("knowledge_hub.interfaces.cli.commands.paper_cmd._regenerate_concept_index", lambda *args, **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["sync-keywords", "--force", "--limit", "1", "--claims"],
        obj={"khub": _StubKhub(config)},
    )
    assert result.exit_code == 0

    db = SQLiteDatabase(config.sqlite_path)
    accepted_claims = db.list_claims(limit=10)
    assert len(accepted_claims) == 1
    accepted_ptr = accepted_claims[0]["evidence_ptrs"][0]
    assert accepted_ptr["claim_decision"] == "accepted"
    assert set(accepted_ptr["score_breakdown"]) >= {
        "evidence_quality",
        "entity_resolution_confidence",
        "generic_claim_penalty",
        "contradiction_hint",
        "final_score",
    }

    pending_items = db.list_ontology_pending(pending_type="claim", topic_slug="paper", limit=10)
    assert len(pending_items) == 1
    pending_reason = pending_items[0]["reason_json"]
    assert pending_reason["generic_claim_penalty"] > 0.0
    assert "score_breakdown" in pending_reason
    assert pending_items[0]["evidence_ptrs_json"][0]["claim_decision"] == "pending"
    db.close()


def test_extract_note_concepts_only_reads_concept_sections():
    content = (
        "# Sample\n\n"
        "본문 링크 [[Projects/AI/AI_Papers/Web_Sources/Some Paper|Some Paper]]\n\n"
        "# 🧩 내가 배워야 할 개념\n"
        "- [[00_Concept_Index]]\n"
        "- [[Retrieval-Augmented Generation]]\n"
        "- [[Multi-Agent Systems|MAS]]\n\n"
        "## 관련 개념\n"
        "- [[Tool Calling]]\n"
        "- [[LearningHub/Cluster_Views/cluster-0001-ai-projects|cluster]]\n"
    )

    assert _extract_note_concepts(content) == [
        "Retrieval-Augmented Generation",
        "Multi-Agent Systems",
        "Tool Calling",
    ]
