from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import knowledge_hub.interfaces.cli.commands.paper_cmd as paper_cmd_module
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.papers.card_feedback import PaperCardFeedbackLogger
from knowledge_hub.papers.judge_feedback import PaperJudgeFeedbackLogger


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    return config


def test_paper_feedback_records_override_against_latest_judge_decision(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00001",
            "title": "Agent Retrieval",
            "authors": "A",
            "year": 2025,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": None,
            "text_path": None,
            "translated_path": None,
        }
    )
    db.close()

    logger = PaperJudgeFeedbackLogger(config)
    logger.log_judge_decisions(
        topic="agent retrieval",
        items=[
            {
                "paper_id": "2501.00001",
                "title": "Agent Retrieval",
                "decision": "skip",
                "total_score": 0.31,
                "backend": "rule_llm_v1",
                "dimension_scores": {"relevance_score": 0.5},
                "top_reasons": ["low confidence"],
            }
        ],
        backend="rule_llm_v1",
        threshold=0.62,
        degraded=False,
        allow_external=False,
        source="discover_cli",
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["feedback", "2501.00001", "--label", "keep", "--reason", "읽을 가치가 있음", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event_type"] == "manual_feedback"
    assert payload["paper_id"] == "2501.00001"
    assert payload["judge_context_found"] is True
    assert payload["judge_decision"] == "skip"
    assert payload["human_label"] == "keep"
    assert payload["is_override"] is True

    log_path = Path(config.sqlite_path).parent / "paper_judge_events.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert events[-1]["event_type"] == "manual_feedback"


def test_paper_review_card_records_card_quality_feedback(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00002",
            "title": "Sparse Memory Card",
            "authors": "B",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(tmp_path / "papers" / "2501.00002.pdf"),
            "text_path": None,
            "translated_path": None,
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2501.00002",
            "paper_id": "2501.00002",
            "source_note_id": "",
            "title": "Sparse Memory Card",
            "paper_core": "에이전트 메모리의 핵심만 짧게 설명한다.",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": ["agent memory"],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "agent memory sparse card",
            "quality_flag": "needs_review",
        }
    )
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "review-card",
            "2501.00002",
            "--issue",
            "empty_method",
            "--issue",
            "empty_evidence",
            "--note",
            "보드 카드가 너무 빈약해서 재빌드 필요",
            "--json",
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["event_type"] == "card_quality_feedback"
    assert payload["paper_id"] == "2501.00002"
    assert payload["issues"] == ["empty_method", "empty_evidence"]
    assert payload["artifact_flags"]["hasMemory"] is True
    assert payload["artifact_flags"]["hasSummary"] is False
    assert payload["memory_snapshot"]["qualityFlag"] == "needs_review"
    assert payload["memory_snapshot"]["paperCore"] == "에이전트 메모리의 핵심만 짧게 설명한다."
    assert payload["observed_warnings"] == ["summary_artifact_missing"]
    assert payload["remediationPlan"]["primaryAction"] == "rebuild_structured_summary"
    assert payload["remediationPlan"]["autoApplyActions"] == [
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]

    log_path = Path(config.sqlite_path).parent / "paper_card_feedback.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["event_type"] == "card_quality_feedback"


def test_paper_review_card_export_writes_unique_filtered_paper_ids(tmp_path: Path):
    config = _config(tmp_path)
    logger = PaperCardFeedbackLogger(config)
    logger.log_feedback(
        paper_id="2501.00002",
        issues=["empty_method"],
        note="first pass",
        title="Sparse Memory Card",
    )
    logger.log_feedback(
        paper_id="2501.00002",
        issues=["empty_evidence"],
        note="second pass",
        title="Sparse Memory Card",
    )
    logger.log_feedback(
        paper_id="2501.00003",
        issues=["likely_semantic_mismatch"],
        note="wrong topic",
        title="Wrong Topic Card",
    )

    output_path = tmp_path / "exports" / "paper_ids.txt"
    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "review-card-export",
            "--issue",
            "empty_method",
            "--output",
            str(output_path),
            "--json",
        ],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["paperIds"] == ["2501.00002"]
    assert payload["items"][0]["paperId"] == "2501.00002"
    assert payload["items"][0]["issues"] == ["empty_method", "empty_evidence"]
    assert payload["items"][0]["eventCount"] == 2
    assert payload["items"][0]["remediationPlan"]["primaryAction"] == "rebuild_structured_summary"
    assert "rebuild_paper_memory" in payload["items"][0]["remediationPlan"]["autoApplyActions"]
    assert payload["outputPath"] == str(output_path)
    assert output_path.read_text(encoding="utf-8") == "2501.00002\n"


def test_paper_review_card_plan_uses_logged_issues_and_current_snapshots(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00004",
            "title": "Plan Me",
            "authors": "C",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(tmp_path / "papers" / "2501.00004.pdf"),
            "text_path": None,
            "translated_path": None,
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2501.00004",
            "paper_id": "2501.00004",
            "source_note_id": "",
            "title": "Plan Me",
            "paper_core": "대략적인 핵심만 있다.",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": [],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "plan me sparse memory",
            "quality_flag": "needs_review",
        }
    )
    db.close()

    logger = PaperCardFeedbackLogger(config)
    logger.log_feedback(
        paper_id="2501.00004",
        issues=["empty_method"],
        note="logged issue",
        title="Plan Me",
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-plan", "2501.00004", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["paperId"] == "2501.00004"
    assert payload["issues"] == ["empty_method"]
    assert payload["observedWarnings"] == ["summary_artifact_missing", "concept_links_missing"]
    assert payload["remediationPlan"]["primaryAction"] == "rebuild_structured_summary"
    assert payload["remediationPlan"]["autoApplyActions"] == [
        "rebuild_structured_summary",
        "rebuild_paper_memory",
        "refresh_concept_links",
    ]


def test_paper_review_card_apply_dry_run_only_returns_plan(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00005",
            "title": "Dry Run Card",
            "authors": "D",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(tmp_path / "papers" / "2501.00005.pdf"),
            "text_path": None,
            "translated_path": None,
        }
    )
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply", "2501.00005", "--issue", "summary_artifact_missing", "--dry-run", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["dryRun"] is True
    assert payload["executedActions"] == []
    assert payload["remediationPlan"]["primaryAction"] == "rebuild_structured_summary"


def test_paper_review_card_apply_executes_safe_actions(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00006",
            "title": "Apply Me",
            "authors": "E",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "summary seed",
            "pdf_path": str(tmp_path / "papers" / "2501.00006.pdf"),
            "text_path": None,
            "translated_path": None,
        }
    )
    db.upsert_paper_memory_card(
        card={
            "memory_id": "pm:2501.00006",
            "paper_id": "2501.00006",
            "source_note_id": "",
            "title": "Apply Me",
            "paper_core": "짧은 핵심만 있다.",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "concept_links": [],
            "claim_refs": [],
            "published_at": "2026-03-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": "apply me",
            "quality_flag": "needs_review",
        }
    )
    db.close()

    calls: list[tuple[str, str]] = []

    def _fake_run_paper_summarize(**kwargs):
        calls.append(("summary", kwargs["arxiv_id"]))

    class _FakeBuilder:
        def build_and_store(self, *, paper_id: str):
            calls.append(("memory", paper_id))
            return {"paperId": paper_id, "qualityFlag": "ok"}

    def _fake_build_paper_memory_builder(*args, **kwargs):
        return _FakeBuilder()

    monkeypatch.setattr(paper_cmd_module, "run_paper_summarize", _fake_run_paper_summarize)
    monkeypatch.setattr(paper_cmd_module, "build_paper_memory_builder", _fake_build_paper_memory_builder)

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply", "2501.00006", "--issue", "empty_method", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert [item["code"] for item in payload["executedActions"]] == [
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
    assert payload["skippedActions"] == [
        {
            "code": "refresh_concept_links",
            "reason": "paper memory rebuild already refreshed concept links in this run",
        }
    ]
    assert calls == [("summary", "2501.00006"), ("memory", "2501.00006")]


def test_paper_review_card_apply_blocks_manual_review_plans(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    db.upsert_paper(
        {
            "arxiv_id": "2501.00007",
            "title": "Repair First",
            "authors": "F",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": str(tmp_path / "papers" / "2501.00007.pdf"),
            "text_path": None,
            "translated_path": None,
        }
    )
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply", "2501.00007", "--issue", "likely_semantic_mismatch", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "blocked"
    assert payload["executedActions"] == []
    assert payload["blockedActions"][0]["code"] == "repair_source_content"
    assert "manual source repair" in payload["blockedActions"][0]["reason"]


def test_paper_review_card_apply_batch_selects_targets_from_issue_filter(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    for paper_id, title in (("2501.00008", "Batch One"), ("2501.00009", "Batch Two")):
        db.upsert_paper(
            {
                "arxiv_id": paper_id,
                "title": title,
                "authors": "G",
                "year": 2026,
                "field": "AI",
                "importance": 3,
                "notes": "",
                "pdf_path": str(tmp_path / "papers" / f"{paper_id}.pdf"),
                "text_path": None,
                "translated_path": None,
            }
        )
    db.close()

    logger = PaperCardFeedbackLogger(config)
    logger.log_feedback(paper_id="2501.00008", issues=["empty_method"], title="Batch One")
    logger.log_feedback(paper_id="2501.00009", issues=["empty_method"], title="Batch Two")

    calls: list[tuple[str, str]] = []

    def _fake_run_paper_summarize(**kwargs):
        calls.append(("summary", kwargs["arxiv_id"]))

    class _FakeBuilder:
        def build_and_store(self, *, paper_id: str):
            calls.append(("memory", paper_id))
            return {"paperId": paper_id, "qualityFlag": "ok"}

    monkeypatch.setattr(paper_cmd_module, "run_paper_summarize", _fake_run_paper_summarize)
    monkeypatch.setattr(paper_cmd_module, "build_paper_memory_builder", lambda *args, **kwargs: _FakeBuilder())

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply-batch", "--issue", "empty_method", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["processedCount"] == 2
    assert payload["counts"]["ok"] == 2
    assert [item["paperId"] for item in payload["items"]] == ["2501.00009", "2501.00008"]
    assert calls == [
        ("summary", "2501.00009"),
        ("memory", "2501.00009"),
        ("summary", "2501.00008"),
        ("memory", "2501.00008"),
    ]


def test_paper_review_card_apply_batch_supports_paper_id_file_dry_run(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    for paper_id, title in (("2501.00010", "File One"), ("2501.00011", "File Two")):
        db.upsert_paper(
            {
                "arxiv_id": paper_id,
                "title": title,
                "authors": "H",
                "year": 2026,
                "field": "AI",
                "importance": 3,
                "notes": "",
                "pdf_path": str(tmp_path / "papers" / f"{paper_id}.pdf"),
                "text_path": None,
                "translated_path": None,
            }
        )
    db.close()

    ids_path = tmp_path / "paper_ids.txt"
    ids_path.write_text("2501.00010\n2501.00011\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply-batch", "--paper-id-file", str(ids_path), "--dry-run", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dryRun"] is True
    assert payload["processedCount"] == 2
    assert payload["counts"]["ok"] == 2
    assert [item["paperId"] for item in payload["items"]] == ["2501.00010", "2501.00011"]
    assert all(item["executedActions"] == [] for item in payload["items"])


def test_paper_review_card_apply_batch_returns_empty_ok_when_selector_matches_no_targets(tmp_path: Path):
    config = _config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["review-card-apply-batch", "--issue", "empty_method", "--dry-run", "--json"],
        obj={"khub": _StubKhub(config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["targetCount"] == 0
    assert payload["processedCount"] == 0
    assert payload["items"] == []
