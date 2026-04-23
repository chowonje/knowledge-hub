from __future__ import annotations

import csv
import json
from pathlib import Path

from click.testing import CliRunner

import knowledge_hub.interfaces.cli.commands.paper_cmd as paper_cmd_module
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.paper_cmd import paper_group


class _StubKhub:
    def __init__(self, db: SQLiteDatabase, config: Config):
        self._db = db
        self.config = config

    def sqlite_db(self):
        return self._db


def _config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    return config


def _upsert_paper(db: SQLiteDatabase, *, paper_id: str, title: str) -> None:
    db.upsert_paper(
        {
            "arxiv_id": paper_id,
            "title": title,
            "authors": "A. Researcher",
            "year": 2026,
            "field": "AI",
            "importance": 3,
            "notes": "",
            "pdf_path": f"/tmp/{paper_id}.pdf",
            "text_path": f"/tmp/{paper_id}.txt",
            "translated_path": None,
        }
    )


def _write_summary_artifact(
    papers_dir: Path,
    *,
    paper_id: str,
    title: str,
    summary: dict[str, object],
    fallback_used: bool = False,
    warnings: list[str] | None = None,
) -> None:
    target = papers_dir / "summaries" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "knowledge-hub.paper-summary.build.result.v1",
        "status": "ok",
        "paperId": paper_id,
        "paperTitle": title,
        "parserUsed": "raw",
        "fallbackUsed": bool(fallback_used),
        "llmRoute": "local",
        "summary": summary,
        "evidenceSummaries": {},
        "evidenceMap": [],
        "contextStats": {"claimCoverage": {"totalClaims": 0, "normalizedClaims": 0, "status": "none"}},
        "claimCoverage": {"totalClaims": 0, "normalizedClaims": 0, "status": "none"},
        "warnings": list(warnings or []),
    }
    manifest = {"paper_id": paper_id, "paper_title": title, "built_at": "2026-04-19T00:00:00+00:00"}
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_memory_card(
    db: SQLiteDatabase,
    *,
    paper_id: str,
    title: str,
    paper_core: str,
    problem_context: str,
    method_core: str,
    evidence_core: str,
    limitations: str,
    concept_links: list[str] | None = None,
    quality_flag: str = "ok",
) -> None:
    db.upsert_paper_memory_card(
        card={
            "memory_id": f"pm:{paper_id}",
            "paper_id": paper_id,
            "source_note_id": "",
            "title": title,
            "paper_core": paper_core,
            "problem_context": problem_context,
            "method_core": method_core,
            "evidence_core": evidence_core,
            "limitations": limitations,
            "concept_links": list(concept_links or []),
            "claim_refs": [],
            "published_at": "2026-04-01T00:00:00+00:00",
            "evidence_window": "",
            "search_text": f"{title} canon audit",
            "quality_flag": quality_flag,
        }
    )


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "title",
                "year",
                "target_primary_lane",
                "target_secondary_tags",
                "tranche",
                "source_status",
                "card_quality",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_canon_quality_audit_writes_report_and_selector_without_feedback_log(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10001", title="Canon Good")
    _upsert_paper(db, paper_id="2501.10002", title="Canon Bad")
    _write_memory_card(
        db,
        paper_id="2501.10001",
        title="Canon Good",
        paper_core="이 논문은 추론 구조를 안정적으로 정리한다.",
        problem_context="기존 모델은 긴 추론에서 오류가 누적된다.",
        method_core="단계별 자기검증 경로를 추가한다.",
        evidence_core="대표 벤치마크에서 정확도가 향상된다.",
        limitations="도메인 이동에서는 성능이 흔들린다.",
        concept_links=["reasoning model"],
    )
    _write_memory_card(
        db,
        paper_id="2501.10002",
        title="Canon Bad",
        paper_core="논문의 핵심만 아주 짧게 적었다.",
        problem_context="author@example.com",
        method_core="",
        evidence_core="",
        limitations="limitations not explicit in visible excerpt",
        concept_links=["reasoning model"],
        quality_flag="needs_review",
    )
    db.close()

    papers_dir = Path(config.papers_dir)
    _write_summary_artifact(
        papers_dir,
        paper_id="2501.10001",
        title="Canon Good",
        summary={
            "oneLine": "추론 경로를 구조화해 안정성을 높인다.",
            "problem": "긴 추론에서 누적 오차가 커지는 문제를 다룬다.",
            "coreIdea": "단계별 자기검증을 추가한다.",
            "methodSteps": ["생성 후 자기검증 경로를 적용한다."],
            "keyResults": ["대표 벤치마크에서 정확도가 향상된다."],
            "limitations": ["도메인 이동에서는 성능이 떨어진다."],
        },
    )
    _write_summary_artifact(
        papers_dir,
        paper_id="2501.10002",
        title="Canon Bad",
        summary={
            "oneLine": "문제와 방법을 설명한다.",
            "problem": "Contact author at author@example.com for materials.",
            "coreIdea": "핵심 아이디어를 간단히 적었다.",
            "methodSteps": ["Table 1 shows the best results across settings."],
            "keyResults": ["결과 요약이 비어 있다."],
            "limitations": ["한계 설명이 충분하지 않다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10001",
                "title": "Canon Good",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "ok",
                "notes": "green",
            },
            {
                "paper_id": "2501.10002",
                "title": "Canon Bad",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "needs_review",
                "notes": "bad",
            },
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["canon-quality-audit", "--manifest", str(manifest_path), "--output-dir", str(output_dir), "--json"],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.paper.canon-quality-audit.result.v1"
    assert payload["targetCount"] == 2
    assert payload["needsReviewCount"] == 1
    bad_item = next(item for item in payload["items"] if item["paperId"] == "2501.10002")
    assert "front_matter_spillover" in bad_item["issues"]
    assert "table_caption_spillover" in bad_item["issues"]
    assert (output_dir / "canon_quality_report.json").exists()
    assert (output_dir / "canon_needs_review.txt").read_text(encoding="utf-8") == "2501.10002\n"
    assert not (Path(config.sqlite_path).parent / "paper_card_feedback.jsonl").exists()


def test_canon_quality_audit_green_state_keeps_empty_selector_and_allows_metric_tokens(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10003", title="Canon Green")
    _write_memory_card(
        db,
        paper_id="2501.10003",
        title="Canon Green",
        paper_core="이 논문은 멀티모달 추론 모델을 정리한다.",
        problem_context="긴 컨텍스트와 멀티모달 정렬 문제를 다룬다.",
        method_core="GRPO, Arena-Hard, TextCaps 같은 이름은 등장하지만 설명은 한국어로 유지한다.",
        evidence_core="TextCaps와 Arena-Hard에서 지표가 개선된다.",
        limitations="데이터 편향에는 여전히 민감하다.",
        concept_links=["multimodal llm"],
    )
    db.close()

    _write_summary_artifact(
        Path(config.papers_dir),
        paper_id="2501.10003",
        title="Canon Green",
        summary={
            "oneLine": "멀티모달 추론 경로를 안정화한다.",
            "problem": "멀티모달 정렬과 긴 추론 경로의 불안정을 줄이려 한다.",
            "coreIdea": "단계별 정렬과 검증을 묶어 학습한다.",
            "methodSteps": ["GRPO와 TextCaps 같은 토큰이 있어도 설명은 한국어다."],
            "keyResults": ["Arena-Hard와 TextCaps 계열 평가에서 향상된다."],
            "limitations": ["데이터 편향에는 여전히 민감하다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10003",
                "title": "Canon Green",
                "year": "2026",
                "target_primary_lane": "multimodal",
                "target_secondary_tags": "ai_canon,canon_t2",
                "tranche": "t2",
                "source_status": "ready",
                "card_quality": "ok",
                "notes": "green",
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        ["canon-quality-audit", "--manifest", str(manifest_path), "--output-dir", str(output_dir), "--json"],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["needsReviewCount"] == 0
    item = payload["items"][0]
    assert "raw_english_spillover" not in item["issues"]
    assert (output_dir / "canon_needs_review.txt").read_text(encoding="utf-8") == ""


def test_canon_quality_audit_apply_dry_run_plans_without_feedback_log(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10004", title="Canon Apply Dry Run")
    _write_memory_card(
        db,
        paper_id="2501.10004",
        title="Canon Apply Dry Run",
        paper_core="짧은 설명",
        problem_context="author@example.com",
        method_core="",
        evidence_core="",
        limitations="limitations not explicit in visible excerpt",
        concept_links=[],
        quality_flag="needs_review",
    )
    db.close()

    _write_summary_artifact(
        Path(config.papers_dir),
        paper_id="2501.10004",
        title="Canon Apply Dry Run",
        summary={
            "oneLine": "요약",
            "problem": "Contact author at author@example.com for details.",
            "coreIdea": "핵심은 짧다.",
            "methodSteps": ["Table 2 reports the final ablation."],
            "keyResults": ["결과가 충분히 정리되지 않았다."],
            "limitations": ["한계 설명이 부족하다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10004",
                "title": "Canon Apply Dry Run",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "needs_review",
                "notes": "dry-run",
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "canon-quality-audit",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--apply",
            "--dry-run",
            "--json",
        ],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["apply"] is True
    assert payload["dryRun"] is True
    remediation = payload["remediation"]["items"][0]
    assert remediation["status"] == "planned"
    assert remediation["plannedActions"][:3] == [
        "repair_source_content",
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
    assert not (Path(config.sqlite_path).parent / "paper_card_feedback.jsonl").exists()


def test_canon_quality_audit_table_caption_only_rebuilds_without_source_repair(tmp_path: Path):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10004b", title="Canon Caption Only")
    _write_memory_card(
        db,
        paper_id="2501.10004b",
        title="Canon Caption Only",
        paper_core="짧은 설명",
        problem_context="문제 설명은 한국어다.",
        method_core="방법 설명도 한국어다.",
        evidence_core="Table 4 reports the final benchmark scores across settings.",
        limitations="한계 설명은 있다.",
        concept_links=["reasoning model"],
        quality_flag="needs_review",
    )
    db.close()

    _write_summary_artifact(
        Path(config.papers_dir),
        paper_id="2501.10004b",
        title="Canon Caption Only",
        summary={
            "oneLine": "요약",
            "problem": "문제 설명은 한국어다.",
            "coreIdea": "핵심은 한국어다.",
            "methodSteps": ["방법 설명도 한국어다."],
            "keyResults": ["Table 4 reports the final benchmark scores across settings."],
            "limitations": ["한계 설명은 있다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10004b",
                "title": "Canon Caption Only",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "needs_review",
                "notes": "caption only",
            }
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "canon-quality-audit",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--apply",
            "--dry-run",
            "--json",
        ],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    remediation = payload["remediation"]["items"][0]
    assert remediation["plannedActions"] == [
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]


def test_canon_quality_audit_apply_executes_in_fixed_order(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10005", title="Canon Apply Execute")
    _write_memory_card(
        db,
        paper_id="2501.10005",
        title="Canon Apply Execute",
        paper_core="짧은 설명",
        problem_context="author@example.com",
        method_core="",
        evidence_core="",
        limitations="limitations not explicit in visible excerpt",
        concept_links=[],
        quality_flag="needs_review",
    )
    db.close()

    _write_summary_artifact(
        Path(config.papers_dir),
        paper_id="2501.10005",
        title="Canon Apply Execute",
        summary={
            "oneLine": "요약",
            "problem": "Contact author at author@example.com for details.",
            "coreIdea": "핵심은 짧다.",
            "methodSteps": ["Table 3 reports the final ablation."],
            "keyResults": ["결과가 충분히 정리되지 않았다."],
            "limitations": ["한계 설명이 부족하다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10005",
                "title": "Canon Apply Execute",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "needs_review",
                "notes": "execute",
            }
        ],
    )

    calls: list[str] = []

    def _fake_repair_paper_sources(**kwargs):
        calls.append("repair_source_content")
        return {
            "schema": "knowledge-hub.paper.source-repair.result.v1",
            "status": "ok",
            "items": [
                {
                    "paperId": kwargs["paper_ids"][0],
                    "repairStatus": "ok",
                    "action": "keep_current_source",
                    "canonicalPaperId": "",
                    "resolutionReason": "",
                }
            ],
        }

    def _fake_run_paper_summarize(**kwargs):
        calls.append("rebuild_structured_summary")

    class _FakeBuilder:
        def build_and_store(self, *, paper_id: str):
            calls.append("rebuild_paper_memory")
            return {"paperId": paper_id, "qualityFlag": "ok"}

    monkeypatch.setattr(paper_cmd_module, "repair_paper_sources", _fake_repair_paper_sources)
    monkeypatch.setattr(paper_cmd_module, "run_paper_summarize", _fake_run_paper_summarize)
    monkeypatch.setattr(paper_cmd_module, "build_paper_memory_builder", lambda *args, **kwargs: _FakeBuilder())

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "canon-quality-audit",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--apply",
            "--json",
        ],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    remediation = payload["remediation"]["items"][0]
    assert remediation["status"] == "ok"
    assert [item["code"] for item in remediation["executedActions"]] == [
        "repair_source_content",
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
    assert calls == [
        "repair_source_content",
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
    assert not (Path(config.sqlite_path).parent / "paper_card_feedback.jsonl").exists()


def test_canon_quality_audit_soft_source_repair_block_still_rebuilds_summary_and_memory(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    db = SQLiteDatabase(config.sqlite_path)
    _upsert_paper(db, paper_id="2501.10005b", title="Canon Soft Block")
    _write_memory_card(
        db,
        paper_id="2501.10005b",
        title="Canon Soft Block",
        paper_core="짧은 설명",
        problem_context="author@example.com",
        method_core="",
        evidence_core="",
        limitations="limitations not explicit in visible excerpt",
        concept_links=[],
        quality_flag="needs_review",
    )
    db.close()

    _write_summary_artifact(
        Path(config.papers_dir),
        paper_id="2501.10005b",
        title="Canon Soft Block",
        summary={
            "oneLine": "요약",
            "problem": "Contact author at author@example.com for details.",
            "coreIdea": "핵심은 짧다.",
            "methodSteps": ["Table 3 reports the final ablation."],
            "keyResults": ["결과가 충분히 정리되지 않았다."],
            "limitations": ["한계 설명이 부족하다."],
        },
    )

    manifest_path = tmp_path / "artifacts" / "ai_canon" / "ai_canon_manifest.csv"
    output_dir = tmp_path / "artifacts" / "ai_canon"
    _write_manifest(
        manifest_path,
        [
            {
                "paper_id": "2501.10005b",
                "title": "Canon Soft Block",
                "year": "2026",
                "target_primary_lane": "architecture",
                "target_secondary_tags": "ai_canon,canon_t1",
                "tranche": "t1",
                "source_status": "ready",
                "card_quality": "needs_review",
                "notes": "soft-block",
            }
        ],
    )

    calls: list[str] = []

    def _fake_repair_paper_sources(**kwargs):
        calls.append("repair_source_content")
        return {
            "schema": "knowledge-hub.paper.source-repair.result.v1",
            "status": "ok",
            "items": [
                {
                    "paperId": kwargs["paper_ids"][0],
                    "repairStatus": "blocked",
                    "action": "",
                    "canonicalPaperId": "",
                    "resolutionReason": "no cleanup rule available",
                }
            ],
        }

    def _fake_run_paper_summarize(**kwargs):
        calls.append("rebuild_structured_summary")

    class _FakeBuilder:
        def build_and_store(self, *, paper_id: str):
            calls.append("rebuild_paper_memory")
            return {"paperId": paper_id, "qualityFlag": "ok"}

    monkeypatch.setattr(paper_cmd_module, "repair_paper_sources", _fake_repair_paper_sources)
    monkeypatch.setattr(paper_cmd_module, "run_paper_summarize", _fake_run_paper_summarize)
    monkeypatch.setattr(paper_cmd_module, "build_paper_memory_builder", lambda *args, **kwargs: _FakeBuilder())

    runner = CliRunner()
    result = runner.invoke(
        paper_group,
        [
            "canon-quality-audit",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--apply",
            "--json",
        ],
        obj={"khub": _StubKhub(SQLiteDatabase(config.sqlite_path), config)},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    remediation = payload["remediation"]["items"][0]
    assert remediation["status"] == "ok"
    assert remediation["blockedActions"] == [
        {
            "code": "repair_source_content",
            "reason": "no cleanup rule available",
        }
    ]
    assert [item["code"] for item in remediation["executedActions"]] == [
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
    assert calls == [
        "repair_source_content",
        "rebuild_structured_summary",
        "rebuild_paper_memory",
    ]
