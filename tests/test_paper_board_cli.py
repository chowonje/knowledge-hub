from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubConfig:
    translation_provider = ""
    translation_model = ""
    paper_summary_parser = "auto"

    def __init__(self, *, papers_dir: str = ""):
        self._papers_dir = papers_dir

    def get_nested(self, *args, default=None):  # noqa: ANN002, ANN003
        _ = args
        return default

    def get_provider_config(self, provider):  # noqa: ANN001
        _ = provider
        return {}

    @property
    def papers_dir(self) -> str:
        return self._papers_dir


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, *, papers_dir: str):
        self._db = db
        self.config = _StubConfig(papers_dir=papers_dir)

    def sqlite_db(self):
        return self._db


def _seed(db: SQLiteDatabase) -> None:
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 5,
            "notes": "",
            "pdf_path": "/tmp/2603.13017.pdf",
            "text_path": "",
            "translated_path": "/tmp/2603.13017.ko.md",
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "2603.13018",
            "title": "Memory Cards for Agents",
            "authors": "B. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Diffusion Systems Overview",
            "authors": "C. Researcher",
            "year": 2025,
            "field": "Vision",
            "importance": 3,
            "notes": "",
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2603.13017",
            "paper_id": "2603.13017",
            "source_note_id": "",
            "title": "Personalized Agent Memory",
            "paper_core": "장기 에이전트 세션을 메모리 카드로 재사용한다.",
            "problem_context": "긴 세션에서 과거 문맥 재호출이 어렵다.",
            "method_core": "구조화된 memory card를 저장하고 검색한다.",
            "evidence_core": "developer-agent benchmark에서 성능 향상.",
            "limitations": "recall 저하 시 성능 하락.",
            "concept_links": ["agent memory", "memory card"],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "agent memory card long running sessions benchmark",
            "quality_flag": "ok",
        }
    )


def _write_summary_artifact(papers_dir: Path, *, paper_id: str, title: str) -> None:
    target = papers_dir / "summaries" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "knowledge-hub.paper-summary.build.result.v1",
        "status": "ok",
        "paperId": paper_id,
        "paperTitle": title,
        "parserUsed": "raw",
        "summary": {
            "oneLine": "장기 에이전트 세션을 메모리 카드로 압축해 재사용한다.",
            "coreIdea": "중요한 상호작용을 카드로 저장하고 후속 작업에서 검색한다.",
        },
        "contextStats": {
            "claimCoverage": {
                "totalClaims": 2,
                "normalizedClaims": 2,
            }
        },
        "warnings": [],
    }
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_paper_board_export_is_schema_backed_and_read_only(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    papers_dir = tmp_path / "papers"
    _write_summary_artifact(papers_dir, paper_id="2603.13017", title="Personalized Agent Memory")
    khub = _StubKhub(db, papers_dir=str(papers_dir))
    runner = CliRunner()

    monkeypatch.setattr(
        structured_summary_module.StructuredPaperSummaryService,
        "build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("summary build should not run")),  # noqa: ARG005
    )

    result = runner.invoke(paper_group, ["board-export", "--json", "--limit", "2"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.paper.board-export.v1"
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["stats"]["returnedPapers"] == 2
    assert payload["papers"][0]["paperId"] == "2603.13017"
    assert payload["papers"][0]["artifactFlags"]["hasSummary"] is True
    assert payload["papers"][0]["artifactFlags"]["hasMemory"] is True
    assert payload["papers"][0]["quality"]["band"] in {"strong", "usable"}
    assert payload["papers"][0]["quality"]["slotStatus"]["methodCore"] == "ok"
    assert payload["papers"][0]["conceptsDetailed"][0]["source"] == "title_fallback"
    assert payload["papers"][0]["conceptsDetailed"][0]["band"] == "heuristic"
    assert payload["stats"]["qualityBands"]["strong"] + payload["stats"]["qualityBands"]["usable"] >= 1
    assert payload["papers"][0]["paths"]["pdfPath"].endswith(".pdf")
    assert payload["papers"][0]["relatedPapers"][0]["paperId"] == "2603.13018"


def test_paper_board_export_tolerates_missing_artifacts(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()

    result = runner.invoke(
        paper_group,
        ["board-export", "--json", "--field", "Vision", "--limit", "5"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert payload["stats"]["returnedPapers"] == 1
    paper = payload["papers"][0]
    assert paper["paperId"] == "2501.00001"
    assert paper["artifactFlags"]["hasPdf"] is False
    assert paper["artifactFlags"]["hasSummary"] is False
    assert paper["artifactFlags"]["hasMemory"] is False
    assert paper["quality"]["band"] == "degraded"
    assert paper["quality"]["slotStatus"]["paperCore"] == "missing"
    assert "summary_artifact_missing" in paper["warnings"]
    assert "memory_card_missing" in paper["warnings"]


def test_paper_board_export_marks_unusable_summary_and_memory(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    papers_dir = tmp_path / "papers"
    target = papers_dir / "summaries" / "2603.13017"
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.paper-summary.build.result.v1",
                "status": "ok",
                "paperId": "2603.13017",
                "paperTitle": "Personalized Agent Memory",
                "parserUsed": "raw",
                "summary": {
                    "oneLine": "요청하신 요약을 만들려면 논문 원문(PDF)이 필요합니다.",
                    "coreIdea": "요청하신 요약을 만들려면 논문 원문(PDF)이 필요합니다.",
                },
                "warnings": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2603.13017",
            "paper_id": "2603.13017",
            "source_note_id": "",
            "title": "Personalized Agent Memory",
            "paper_core": "\\pdfoutput=1 \\includepdf[pages=1-last]{paper.pdf}",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": [],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "latex wrapper only",
            "quality_flag": "needs_review",
        }
    )
    khub = _StubKhub(db, papers_dir=str(papers_dir))
    result = CliRunner().invoke(paper_group, ["board-export", "--json", "--field", "AI", "--limit", "5"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    paper = next(item for item in payload["papers"] if item["paperId"] == "2603.13017")
    assert paper["artifactFlags"]["hasSummary"] is False
    assert paper["artifactFlags"]["hasMemory"] is False
    assert paper["quality"]["band"] == "degraded"
    assert "summary_artifact_unusable" in paper["quality"]["reasons"]
    assert paper["conceptsDetailed"] == []
    assert "summary_artifact_unusable" in paper["warnings"]
    assert "memory_card_unusable" in paper["warnings"]


def test_paper_board_export_prefers_stronger_concept_source_over_memory_fallback(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    db.upsert_ontology_entity(
        entity_id="ai_agent",
        entity_type="concept",
        canonical_name="AI Agent",
        source="paper_title_seed",
    )
    db.add_relation(
        source_type="paper",
        source_id="2603.13017",
        relation="uses",
        target_type="concept",
        target_id="ai_agent",
        evidence_text=json.dumps({"source": "paper_title_seed", "relation_norm": "uses"}, ensure_ascii=False),
        confidence=0.74,
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2603.13017:seed",
            "paper_id": "2603.13017",
            "source_note_id": "",
            "title": "Personalized Agent Memory",
            "paper_core": "Long-running agent sessions are summarized into cards.",
            "problem_context": "Long sessions are hard to revisit.",
            "method_core": "Store and retrieve cards.",
            "evidence_core": "Benchmark gains.",
            "limitations": "",
            "concept_links": ["AI Agent"],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "ai agent memory cards",
            "quality_flag": "ok",
        }
    )
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))

    result = CliRunner().invoke(paper_group, ["board-export", "--json", "--field", "AI", "--limit", "5"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    paper = next(item for item in payload["papers"] if item["paperId"] == "2603.13017")
    ai_agent = next(item for item in paper["conceptsDetailed"] if item["name"] == "AI Agent")
    assert ai_agent["source"] == "ontology"
    assert ai_agent["band"] == "verified"


def test_paper_board_export_canonicalizes_memory_fallback_and_drops_generic_junk(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2603.13017:cleanup",
            "paper_id": "2603.13017",
            "source_note_id": "",
            "title": "Personalized Agent Memory",
            "paper_core": "Long-running agent sessions are summarized into cards.",
            "problem_context": "Long sessions are hard to revisit.",
            "method_core": "Store and retrieve cards.",
            "evidence_core": "Benchmark gains.",
            "limitations": "",
            "concept_links": ["LLM-based agents", "Benchmark", "Language Models"],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "llm-based agents benchmark language models",
            "quality_flag": "ok",
        }
    )
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))

    result = CliRunner().invoke(paper_group, ["board-export", "--json", "--field", "AI", "--limit", "5"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    paper = next(item for item in payload["papers"] if item["paperId"] == "2603.13017")
    concept_names = [item["name"] for item in paper["conceptsDetailed"]]
    assert "Benchmark" not in concept_names
    assert "LLM Agent" in concept_names
    assert "Language Model" in concept_names
