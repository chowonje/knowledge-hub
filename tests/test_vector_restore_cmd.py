from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.application.vector_restore import compare_vector_backup
from knowledge_hub.application.vector_restore import assess_vector_restore
from knowledge_hub.application.vector_restore import restore_vector_backup
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase, inspect_vector_store
from knowledge_hub.interfaces.cli.commands import vector_compare_cmd
from knowledge_hub.interfaces.cli.commands import vector_cmd


def _config(vector_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vector_db_path=str(vector_root),
        collection_name="knowledge_hub",
    )


def _seed_vector_store(root: Path, count: int) -> None:
    db = VectorDatabase(str(root), "knowledge_hub")
    db.add_documents(
        [f"document {idx}" for idx in range(count)],
        [[float(idx)] for idx in range(count)],
        [{"title": f"Doc {idx}", "source_type": "vault", "file_path": f"doc-{idx}.md"} for idx in range(count)],
        ids=[f"doc-{idx}" for idx in range(count)],
    )


def _seed_lexical_only_store(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lexical = sqlite3.connect(root / "_lexical.sqlite3")
    lexical.execute(
        """
        CREATE VIRTUAL TABLE lexical_documents_fts
        USING fts5(
            doc_id UNINDEXED,
            title,
            section_title,
            contextual_summary,
            keywords,
            field,
            document,
            searchable_text
        )
        """
    )
    lexical.execute(
        """
        INSERT INTO lexical_documents_fts
        (doc_id, title, section_title, contextual_summary, keywords, field, document, searchable_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("doc-1", "Doc 1", "", "", "", "", "body", "doc 1 body"),
    )
    lexical.commit()
    lexical.close()


def test_assess_vector_restore_uses_latest_backup_preview(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    payload = assess_vector_restore(config=_config(active_root), use_latest_backup=True)

    assert payload["status"] == "ok"
    assert payload["dryRun"] is True
    assert payload["selection"]["selectedPath"] == str(backup_root)
    assert payload["activeVector"]["total_documents"] == 1
    assert payload["backupVector"]["total_documents"] == 2
    assert payload["backupVector"]["openable"] is True
    assert payload["backupVector"]["restorable"] is True
    assert payload["action"]["canRestore"] is True


def test_restore_vector_backup_requires_confirm_for_apply(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    payload = restore_vector_backup(
        config=_config(active_root),
        use_latest_backup=True,
        apply=True,
        confirm=False,
    )

    assert payload["status"] == "failed"
    assert payload["applied"] is False
    assert "--confirm is required with --apply" in payload["errors"][-1]
    assert inspect_vector_store(active_root, "knowledge_hub")["total_documents"] == 1


def test_restore_vector_backup_apply_replaces_active_and_preserves_previous_active(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    payload = restore_vector_backup(
        config=_config(active_root),
        use_latest_backup=True,
        apply=True,
        confirm=True,
    )

    assert payload["status"] == "ok"
    assert payload["applied"] is True
    assert payload["activeBackupPath"]
    assert inspect_vector_store(active_root, "knowledge_hub")["total_documents"] == 2
    assert inspect_vector_store(Path(payload["activeBackupPath"]), "knowledge_hub")["total_documents"] == 1
    assert inspect_vector_store(backup_root, "knowledge_hub")["total_documents"] == 2


def test_assess_vector_restore_blocks_backup_when_chroma_open_probe_fails(tmp_path: Path, monkeypatch):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    def _probe(path, collection_name):  # noqa: ANN001
        if Path(path) == backup_root:
            return {"openable": False, "error": "probe failed"}
        return {"openable": True, "error": ""}

    monkeypatch.setattr("knowledge_hub.application.vector_restore.probe_vector_store_openability", _probe)

    payload = assess_vector_restore(config=_config(active_root), use_latest_backup=True)

    assert payload["status"] == "blocked"
    assert payload["backupVector"]["openable"] is False
    assert payload["backupVector"]["restorable"] is False
    assert "selected backup cannot open Chroma without repair: probe failed" in payload["errors"]
    assert payload["action"]["canRestore"] is False


def test_restore_vector_backup_rolls_back_when_restored_copy_fails_open_probe(tmp_path: Path, monkeypatch):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)
    probe_calls = {"count": 0}

    def _probe(path, collection_name):  # noqa: ANN001
        probe_calls["count"] += 1
        if probe_calls["count"] == 1:
            return {"openable": True, "error": ""}
        return {"openable": False, "error": "post-restore probe failed"}

    monkeypatch.setattr("knowledge_hub.application.vector_restore.probe_vector_store_openability", _probe)

    payload = restore_vector_backup(
        config=_config(active_root),
        use_latest_backup=True,
        apply=True,
        confirm=True,
    )

    assert payload["status"] == "failed"
    assert payload["applied"] is False
    assert "restored vector store cannot open Chroma without repair: post-restore probe failed" in payload["errors"]
    assert inspect_vector_store(active_root, "knowledge_hub")["total_documents"] == 1
    assert inspect_vector_store(backup_root, "knowledge_hub")["total_documents"] == 2


def test_assess_vector_restore_blocks_lexical_only_backup(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_lexical_only_store(backup_root)

    payload = assess_vector_restore(config=_config(active_root), use_latest_backup=True)

    assert payload["status"] == "blocked"
    assert payload["backupVector"]["lexical_documents"] == 1
    assert payload["backupVector"]["chroma_embeddings"] == 0
    assert "selected backup has no Chroma embeddings" in payload["errors"][-1]


def test_vector_restore_cmd_json_preview(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    runner = CliRunner()
    result = runner.invoke(
        vector_cmd.vector_restore_cmd,
        ["--latest-backup", "--json"],
        obj={"khub": type("Ctx", (), {"config": _config(active_root)})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.vector-restore.result.v1"
    assert payload["status"] == "ok"
    assert payload["selection"]["selectedPath"] == str(backup_root)


def test_compare_vector_backup_reports_document_diff(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    active_db = VectorDatabase(str(active_root), "knowledge_hub")
    active_db.add_documents(
        ["active only", "shared active"],
        [[0.0], [1.0]],
        [
            {"title": "Active Only", "source_type": "vault", "file_path": "active-only.md"},
            {"title": "Shared Active", "source_type": "vault", "file_path": "shared.md"},
        ],
        ids=["active-only", "shared"],
    )
    backup_db = VectorDatabase(str(backup_root), "knowledge_hub")
    backup_db.add_documents(
        ["backup only", "shared backup"],
        [[0.0], [1.0]],
        [
            {"title": "Backup Only", "source_type": "vault", "file_path": "backup-only.md"},
            {"title": "Shared Backup", "source_type": "vault", "file_path": "shared.md"},
        ],
        ids=["backup-only", "shared"],
    )

    payload = compare_vector_backup(config=_config(active_root), use_latest_backup=True, sample_limit=5)

    assert payload["status"] == "ok"
    assert payload["diff"]["activeOnlyCount"] == 1
    assert payload["diff"]["backupOnlyCount"] == 1
    assert payload["diff"]["changedSharedCount"] == 1
    assert payload["diff"]["activeOnlySample"][0]["doc_id"] == "active-only"
    assert payload["diff"]["backupOnlySample"][0]["doc_id"] == "backup-only"
    assert payload["diff"]["provenance"]["activeOnly"]["sourceTypeCounts"] == {"vault": 1}
    assert payload["diff"]["provenance"]["backupOnly"]["sourceTypeCounts"] == {"vault": 1}
    assert payload["action"]["recommendedRestoreCommand"] == "khub vector-restore --latest-backup --apply --confirm"


def test_compare_vector_backup_adds_provenance_decision_hint_for_web_heavy_backup(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    web_dir = tmp_path / "web_docs"
    web_dir.mkdir(parents=True, exist_ok=True)
    web_file = web_dir / "web_abc123-measuring-ai-agent-autonomy-in-practice-anthropic.md"
    web_file.write_text("alpha retrieval web source", encoding="utf-8")

    active_db = VectorDatabase(str(active_root), "knowledge_hub")
    active_db.add_documents(
        ["shared active"],
        [[1.0]],
        [{"title": "Shared Active", "source_type": "paper", "file_path": "shared.md"}],
        ids=["shared"],
    )
    backup_db = VectorDatabase(str(backup_root), "knowledge_hub")
    backup_db.add_documents(
        ["shared active", "web only"],
        [[1.0], [2.0]],
        [
            {"title": "Shared Active", "source_type": "paper", "file_path": "shared.md"},
            {"title": "Web Only", "source_type": "web", "file_path": str(web_file)},
        ],
        ids=["shared", "web-only"],
    )

    payload = compare_vector_backup(config=_config(active_root), use_latest_backup=True, sample_limit=5)

    provenance = payload["diff"]["provenance"]
    assert provenance["backupOnly"]["sourceTypeCounts"] == {"web": 1}
    assert provenance["backupOnly"]["publisherCounts"] == {"anthropic": 1}
    assert provenance["backupOnly"]["uniqueFileCount"] == 1
    assert provenance["backupOnly"]["fileMtimeRange"]["oldest"]
    assert provenance["decisionHint"]["recommendedAction"] == "review_before_restore"
    assert "backup_only_all_web" in provenance["decisionHint"]["reasonCodes"]


def test_vector_compare_cmd_json_preview(tmp_path: Path):
    active_root = tmp_path / "vector"
    backup_root = tmp_path / "vector.corrupt.20260416_150415"
    _seed_vector_store(active_root, 1)
    _seed_vector_store(backup_root, 2)

    runner = CliRunner()
    result = runner.invoke(
        vector_compare_cmd.vector_compare_cmd,
        ["--latest-backup", "--json"],
        obj={"khub": type("Ctx", (), {"config": _config(active_root)})()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.vector-compare.result.v1"
    assert payload["status"] == "ok"
    assert payload["selection"]["selectedPath"] == str(backup_root)
    assert "diff" in payload
