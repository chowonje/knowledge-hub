from __future__ import annotations

import json
from pathlib import Path

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_eval import PaperMemoryEvalCase, PaperMemoryEvalHarness
from tests.test_paper_memory import _seed_paper_with_note


FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "paper_memory_eval"


def _load_cases() -> list[PaperMemoryEvalCase]:
    raw = json.loads((FIXTURES_ROOT / "cases.json").read_text(encoding="utf-8"))
    return [PaperMemoryEvalCase(**item) for item in raw]


def _seed_eval_database(db: SQLiteDatabase, tmp_path: Path) -> None:
    _seed_paper_with_note(db, tmp_path, paper_id="2603.13017")
    translated_path = tmp_path / "2603.13019_translated.md"
    translated_path.write_text(
        "This paper proposes a compact memory card for paper retrieval. It improves search efficiency.",
        encoding="utf-8",
    )
    db.upsert_paper(
        {
            "arxiv_id": "2603.13019",
            "title": "Metadata Only Paper",
            "authors": "C. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "metadata fallback note",
            "pdf_path": "",
            "text_path": "",
            "translated_path": str(translated_path),
        }
    )
    PaperMemoryBuilder(db).build_and_store(paper_id="2603.13017")
    PaperMemoryBuilder(db).build_and_store(paper_id="2603.13019")


def test_paper_memory_eval_harness_fixture_matrix(tmp_path: Path) -> None:
    db = SQLiteDatabase(str(tmp_path / "knowledge.db"))
    _seed_eval_database(db, tmp_path)

    report = PaperMemoryEvalHarness(db).evaluate_cases(_load_cases())

    assert report["summary"]["caseCount"] == 3
    assert report["summary"]["paperMemory"]["top1MatchCount"] >= 2
    assert report["summary"]["paperMemory"]["noResultRate"] <= (1 / 3)
    assert report["summary"]["searchPapers"]["top1MatchRate"] < report["summary"]["paperMemory"]["top1MatchRate"]
    assert report["summary"]["paperLookupAndSummarize"]["top1MatchRate"] < report["summary"]["paperMemory"]["top1MatchRate"]
    assert report["summary"]["paperMemory"]["top1LiftVsSearchPapers"] > 0
    assert report["summary"]["paperMemory"]["top1LiftVsLookup"] > 0

    thematic_case = next(case for case in report["cases"] if case["caseId"] == "thematic_korean")
    assert thematic_case["surfaces"]["paper_memory_search"]["top1Match"] is True
    assert thematic_case["surfaces"]["search_papers"]["noResult"] is True

    exact_case = next(case for case in report["cases"] if case["caseId"] == "exact_title")
    assert exact_case["surfaces"]["paper_lookup_and_summarize"]["drillDownUseful"] is True
