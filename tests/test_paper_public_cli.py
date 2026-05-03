from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

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


class _FakeSearchResult:
    def __init__(self, *, title: str, source_type: str, score: float, document_id: str):
        self.metadata = {"title": title, "source_type": source_type}
        self.score = score
        self.document_id = document_id


class _FakeSearcher:
    def search(self, query: str, **kwargs):  # noqa: ANN003
        _ = (query, kwargs)
        return [
            _FakeSearchResult(title="Memory Cards for Agents", source_type="paper", score=0.91, document_id="paper:2603.13018"),
            _FakeSearchResult(title="Agent Memory", source_type="concept", score=0.88, document_id="concept:agent-memory"),
            _FakeSearchResult(title="Related Vault Note", source_type="vault", score=0.71, document_id="vault:note-1"),
        ]


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, *, papers_dir: str):
        self._db = db
        self.config = _StubConfig(papers_dir=papers_dir)

    def sqlite_db(self):
        return self._db

    def searcher(self):
        return _FakeSearcher()


def _seed(db: SQLiteDatabase):
    db.upsert_paper(
        {
            "arxiv_id": "2603.13017",
            "title": "Personalized Agent Memory",
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 4,
            "notes": (
                "### 한줄 요약\n\n기존 요약 경로 샘플.\n\n"
                "### 핵심 기여\n\n- 세션 압축\n\n"
                "### 방법론\n\n- 메모리 카드\n"
            ),
            "pdf_path": "",
            "text_path": "",
            "translated_path": "",
        }
    )
    db.upsert_note(
        note_id="paper:2603.13017",
        title="[논문] Personalized Agent Memory",
        content=(
            "# Personalized Agent Memory\n\n"
            "## Abstract\n\n"
            "This paper compresses long-running agent sessions into searchable memory cards.\n\n"
            "## Method\n\n"
            "The system builds memory cards, retrieves them for future tasks, and reranks them with context signals.\n\n"
            "## Results\n\n"
            "On a developer-agent benchmark, success rate improves by 8.2 points over a baseline memory buffer.\n\n"
            "## Limitations\n\n"
            "Performance drops when retrieval recall is low or when domain shift changes the task distribution.\n"
        ),
        source_type="paper",
        metadata={"arxiv_id": "2603.13017"},
    )
    db.upsert_ontology_entity(
        entity_id="paper:2603.13017",
        entity_type="paper",
        canonical_name="Personalized Agent Memory",
        source="test",
    )
    db.upsert_claim(
        claim_id="claim:2603.13017:1",
        claim_text="Success rate improves by 8.2 points on a developer-agent benchmark over a baseline memory buffer.",
        subject_entity_id="paper:2603.13017",
        predicate="improves",
        object_entity_id=None,
        object_literal="baseline memory buffer",
        confidence=0.91,
        evidence_ptrs=[{"note_id": "paper:2603.13017", "claim_decision": "accepted"}],
        source="test",
    )
    db.upsert_claim(
        claim_id="claim:2603.13017:2",
        claim_text="Performance drops when retrieval recall is low under domain shift.",
        subject_entity_id="paper:2603.13017",
        predicate="limits",
        object_entity_id=None,
        object_literal="domain shift",
        confidence=0.81,
        evidence_ptrs=[{"note_id": "paper:2603.13017", "claim_decision": "accepted"}],
        source="test",
    )


def _write_summary_artifact(papers_dir: Path, *, paper_id: str, title: str, built_at: str = "2026-04-03T00:00:00+00:00") -> None:
    target = papers_dir / "summaries" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "knowledge-hub.paper-summary.build.result.v1",
        "status": "ok",
        "paperId": paper_id,
        "paperTitle": title,
        "parserUsed": "raw",
        "fallbackUsed": False,
        "llmRoute": "local",
        "summary": {
            "oneLine": "장기 에이전트 세션을 메모리 카드로 압축해 재사용하는 구조를 제안한다.",
            "problem": "긴 세션에서 과거 문맥을 다시 찾기 어렵다는 문제를 다룬다.",
            "coreIdea": "세션을 구조화된 메모리 카드로 저장하고 후속 작업에서 검색해 활용한다.",
            "methodSteps": [
                "세션에서 중요한 이벤트를 카드로 압축한다.",
                "새 작업에서 관련 카드를 검색하고 재랭킹한다.",
            ],
            "keyResults": [
                "developer-agent benchmark에서 baseline memory buffer 대비 성공률이 8.2포인트 향상된다."
            ],
            "limitations": [
                "retrieval recall이 낮거나 domain shift가 크면 성능이 떨어진다."
            ],
        },
        "evidenceSummaries": {
            "keyResults": {"summaryLines": ["baseline 대비 8.2포인트 향상"], "claimHintsUsed": 1, "unitCount": 1},
            "limitations": {"summaryLines": ["recall 저하와 domain shift에서 성능 저하"], "claimHintsUsed": 1, "unitCount": 1},
        },
        "evidenceMap": [{"field": "keyResults", "page": 1, "excerpt": "success rate improves by 8.2 points"}],
        "contextStats": {"claimCoverage": {"totalClaims": 2, "normalizedClaims": 2, "status": "good"}},
        "claimCoverage": {"totalClaims": 2, "normalizedClaims": 2, "status": "good"},
        "warnings": [],
    }
    manifest = {"paper_id": paper_id, "paper_title": title, "built_at": built_at}
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_memory_card(db: SQLiteDatabase, *, paper_id: str) -> None:
    db.upsert_paper_memory_card(
        card={
            "memory_id": f"pm:{paper_id}",
            "paper_id": paper_id,
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


def _write_document_memory_summary(db: SQLiteDatabase, *, paper_id: str) -> None:
    db.replace_document_memory_units(
        document_id=f"paper:{paper_id}",
        units=[
            {
                "unit_id": f"document-summary:{paper_id}",
                "document_title": "Personalized Agent Memory",
                "source_type": "paper",
                "source_ref": paper_id,
                "unit_type": "document_summary",
                "title": "Document Summary",
                "section_path": "Abstract",
                "contextual_summary": "document memory summary is newer than the structured summary artifact",
                "source_excerpt": "fresh document memory",
                "search_text": "fresh document memory",
                "order_index": 0,
            }
        ],
    )


def test_public_paper_surfaces_are_user_facing(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    papers_dir = tmp_path / "papers"
    _write_summary_artifact(papers_dir, paper_id="2603.13017", title="Personalized Agent Memory")
    _write_memory_card(db, paper_id="2603.13017")
    khub = _StubKhub(db, papers_dir=str(papers_dir))
    runner = CliRunner()

    monkeypatch.setattr(
        structured_summary_module.StructuredPaperSummaryService,
        "build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("summary build should not run")),  # noqa: ARG005
    )

    summary = runner.invoke(paper_group, ["summary", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert summary.exit_code == 0
    summary_payload = json.loads(summary.output)
    assert summary_payload["schema"] == "knowledge-hub.paper.public.summary.v1"
    assert summary_payload["summary"]["oneLine"]
    assert summary_payload["quality"]["band"] in {"strong", "usable"}
    assert summary_payload["conceptsDetailed"]
    assert "memoryRoute" not in summary_payload
    assert "readingCore" not in summary_payload

    evidence = runner.invoke(paper_group, ["evidence", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert evidence.exit_code == 0
    evidence_payload = json.loads(evidence.output)
    assert evidence_payload["schema"] == "knowledge-hub.paper.public.evidence.v1"
    assert evidence_payload["evidenceMap"]
    assert evidence_payload["evidenceSummary"]["keyResults"]
    assert evidence_payload["quality"]["summaryStatus"] in {"ok", "stale"}

    memory = runner.invoke(paper_group, ["memory", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert memory.exit_code == 0
    memory_payload = json.loads(memory.output)
    assert memory_payload["schema"] == "knowledge-hub.paper.public.memory.v1"
    assert memory_payload["claimCoverage"]["normalizedClaims"] >= 1
    assert memory_payload["quality"]["slotStatus"]["methodCore"] == "ok"
    assert memory_payload["memoryCard"]["paperCore"] or memory_payload["memoryCard"]["methodCore"]

    related = runner.invoke(paper_group, ["related", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert related.exit_code == 0
    related_payload = json.loads(related.output)
    assert related_payload["schema"] == "knowledge-hub.paper.public.related.v1"
    assert related_payload["relatedKnowledge"]
    assert related_payload["quality"]["band"] in {"strong", "usable"}
    assert any(item.get("sourceType") == "paper" for item in related_payload["relatedKnowledge"])


def test_public_paper_surfaces_report_missing_artifacts_without_build(tmp_path, monkeypatch):
    from knowledge_hub.papers import structured_summary as structured_summary_module

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    khub = _StubKhub(db, papers_dir=str(tmp_path / "papers"))
    runner = CliRunner()

    monkeypatch.setattr(
        structured_summary_module.StructuredPaperSummaryService,
        "build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("summary build should not run")),  # noqa: ARG005
    )

    summary = runner.invoke(paper_group, ["summary", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert summary.exit_code == 0
    summary_payload = json.loads(summary.output)
    assert summary_payload["status"] == "missing"
    assert summary_payload["quality"]["band"] == "degraded"
    assert "summary_artifact_missing" in summary_payload["warnings"]

    memory = runner.invoke(paper_group, ["memory", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})
    assert memory.exit_code == 0
    memory_payload = json.loads(memory.output)
    assert memory_payload["status"] == "missing"
    assert memory_payload["quality"]["slotStatus"]["paperCore"] == "missing"
    assert "memory_card_missing" in memory_payload["warnings"]


def test_public_summary_reports_stale_when_document_memory_is_newer(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    papers_dir = tmp_path / "papers"
    _write_summary_artifact(
        papers_dir,
        paper_id="2603.13017",
        title="Personalized Agent Memory",
        built_at="2026-04-01T00:00:00+00:00",
    )
    _write_document_memory_summary(db, paper_id="2603.13017")
    khub = _StubKhub(db, papers_dir=str(papers_dir))

    result = CliRunner().invoke(paper_group, ["summary", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "stale"
    assert "summary_artifact_stale" in payload["warnings"]


def test_public_summary_is_not_stale_when_only_memory_card_is_newer(tmp_path):
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed(db)
    papers_dir = tmp_path / "papers"
    _write_summary_artifact(
        papers_dir,
        paper_id="2603.13017",
        title="Personalized Agent Memory",
        built_at="2026-04-01T00:00:00+00:00",
    )
    _write_memory_card(db, paper_id="2603.13017")
    khub = _StubKhub(db, papers_dir=str(papers_dir))

    result = CliRunner().invoke(paper_group, ["summary", "--paper-id", "2603.13017", "--json"], obj={"khub": khub})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert "summary_artifact_stale" not in payload["warnings"]
