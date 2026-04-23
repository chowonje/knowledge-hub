from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_hub.core.vector_db import VectorDatabase
from knowledge_hub.infrastructure.persistence.vector import VectorDatabaseInitError, inspect_vector_store


def test_build_fts5_match_query_quotes_and_normalizes_tokens():
    query = VectorDatabase._build_fts5_match_query('RAG -hallucination "attention" transformer-based')
    assert query == '"rag" AND "hallucination" AND "attention" AND "transformer" AND "based"'


def test_build_fts5_match_query_returns_empty_for_control_only_input():
    assert VectorDatabase._build_fts5_match_query('--- ::: """') == ""


def test_vector_db_does_not_repair_backup_paths_on_init_failure(monkeypatch, tmp_path: Path):
    backup_path = tmp_path / "chroma_db.corrupt.20260416_150352"
    backup_path.mkdir()
    marker = backup_path / "_lexical.sqlite3"
    marker.write_text("backup marker", encoding="utf-8")

    def _raise_client(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("Could not connect to tenant default_tenant")

    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.vector.chromadb.PersistentClient", _raise_client)

    with pytest.raises(VectorDatabaseInitError):
        VectorDatabase(str(backup_path), "knowledge_hub_bge_m3_v1", repair_on_init=True)

    assert backup_path.exists()
    assert marker.exists()


def test_vector_db_repairs_active_path_after_tenant_init_failure(monkeypatch, tmp_path: Path):
    active_path = tmp_path / "chroma_db"
    active_path.mkdir()
    marker = active_path / "marker.txt"
    marker.write_text("active marker", encoding="utf-8")
    calls: list[str] = []

    class _StubClient:
        def get_or_create_collection(self, **kwargs):  # noqa: ANN003
            return object()

    def _client(path, settings):  # noqa: ANN001
        calls.append(str(path))
        if len(calls) == 1:
            raise RuntimeError("Could not connect to tenant default_tenant")
        Path(path).mkdir(parents=True, exist_ok=True)
        return _StubClient()

    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.vector.chromadb.PersistentClient", _client)

    db = VectorDatabase(
        str(active_path),
        "knowledge_hub_bge_m3_v1",
        repair_on_init=True,
        init_lexical=False,
    )

    backups = sorted(tmp_path.glob("chroma_db.corrupt.*"))
    assert len(backups) == 1
    assert (backups[0] / "marker.txt").read_text(encoding="utf-8") == "active marker"
    assert active_path.exists()
    assert calls == [str(active_path), str(active_path)]
    assert db.collection is not None


def test_inspect_vector_store_marks_latest_backup_non_restorable_when_open_probe_fails(monkeypatch, tmp_path: Path):
    active_path = tmp_path / "chroma_db"
    backup_path = tmp_path / "chroma_db.corrupt.20260418_150000"

    active_db = VectorDatabase(str(active_path), "knowledge_hub_bge_m3_v1")
    active_db.add_documents(["active"], [[0.0]], [{"title": "Active", "source_type": "vault", "file_path": "a.md"}], ids=["a"])
    backup_db = VectorDatabase(str(backup_path), "knowledge_hub_bge_m3_v1")
    backup_db.add_documents(["backup"], [[0.0]], [{"title": "Backup", "source_type": "vault", "file_path": "b.md"}], ids=["b"])

    monkeypatch.setattr(
        "knowledge_hub.infrastructure.persistence.vector.probe_vector_store_openability",
        lambda db_path, collection_name="": {"openable": False, "error": "probe failed"} if str(db_path) == str(backup_path) else {"openable": True, "error": ""},
    )

    payload = inspect_vector_store(active_path, "knowledge_hub_bge_m3_v1")

    assert payload["recovery_backup"]["path"] == str(backup_path)
    assert payload["recovery_backup"]["openable"] is False
    assert payload["recovery_backup"]["restorable"] is False
    assert payload["recovery_backup"]["openProbeError"] == "probe failed"


def test_lexical_search_normalizes_negative_bm25_scores(tmp_path: Path):
    db = VectorDatabase(str(tmp_path / "chroma"), "test")
    db.add_documents(
        ["강화 학습 핵심 요약", "강화 학습 용어집", "무관한 문서"],
        [[0.0], [0.0], [0.0]],
        [
            {"title": "강화 학습 핵심 요약", "source_type": "vault", "file_path": "a.md"},
            {"title": "강화 학습 용어집", "source_type": "vault", "file_path": "b.md"},
            {"title": "무관한 문서", "source_type": "vault", "file_path": "c.md"},
        ],
        ids=["a", "b", "c"],
    )

    results = db.lexical_search("강화 학습", top_k=2)

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert results[0]["score"] > 0.0
    assert results[0]["score"] <= 1.0
    assert results[0]["score"] != results[1]["score"]


def test_get_documents_supports_multi_key_metadata_filter(tmp_path: Path):
    db = VectorDatabase(str(tmp_path / "chroma"), "test_multi_filter")
    db.add_documents(
        ["A", "B"],
        [[0.0], [0.0]],
        [
            {"title": "Doc A", "source_type": "vault", "file_path": "a.md"},
            {"title": "Doc B", "source_type": "paper", "arxiv_id": "2501.00001"},
        ],
        ids=["a", "b"],
    )

    results = db.get_documents(filter_dict={"source_type": "vault", "file_path": "a.md"}, limit=5)

    assert results["ids"] == ["a"]
    assert results["metadatas"][0]["file_path"] == "a.md"


def test_search_supports_multi_key_metadata_filter(tmp_path: Path):
    db = VectorDatabase(str(tmp_path / "chroma"), "test_multi_query_filter")
    db.add_documents(
        ["vault memory route", "paper memory route"],
        [[0.0], [1.0]],
        [
            {"title": "Vault Doc", "source_type": "vault", "file_path": "vault.md"},
            {"title": "Paper Doc", "source_type": "paper", "arxiv_id": "2501.00001"},
        ],
        ids=["vault", "paper"],
    )

    results = db.search([0.0], top_k=3, filter_dict={"source_type": "vault", "file_path": "vault.md"})

    assert results["ids"][0] == ["vault"]
    assert results["metadatas"][0][0]["file_path"] == "vault.md"


def test_search_and_lexical_search_exclude_explicit_stale_metadata(tmp_path: Path):
    db = VectorDatabase(str(tmp_path / "chroma"), "test_stale_filter")
    db.add_documents(
        ["alpha stale evidence", "alpha fresh evidence"],
        [[0.0], [0.1]],
        [
            {"title": "Stale", "source_type": "vault", "file_path": "stale.md", "stale": 1},
            {"title": "Fresh", "source_type": "vault", "file_path": "fresh.md", "stale": 0},
        ],
        ids=["stale", "fresh"],
    )

    semantic = db.search([0.0], top_k=1)
    lexical = db.lexical_search("alpha evidence", top_k=2)
    semantic_with_stale = db.search([0.0], top_k=2, include_stale=True)

    assert semantic["ids"][0] == ["fresh"]
    assert [item["id"] for item in lexical] == ["fresh"]
    assert "stale" in semantic_with_stale["ids"][0]


def test_delete_by_metadata_removes_chroma_and_lexical_rows(tmp_path: Path):
    db = VectorDatabase(str(tmp_path / "chroma"), "test_delete_by_metadata")
    db.add_documents(
        ["alpha old chunk", "beta chunk"],
        [[0.0], [1.0]],
        [
            {"title": "Alpha", "source_type": "web", "document_id": "web:alpha"},
            {"title": "Beta", "source_type": "web", "document_id": "web:beta"},
        ],
        ids=["alpha-0", "beta-0"],
    )

    deleted = db.delete_by_metadata({"source_type": "web", "document_id": "web:alpha"})

    assert deleted == 1
    assert db.get_documents(filter_dict={"document_id": "web:alpha"}, limit=5)["ids"] == []
    assert [item["id"] for item in db.lexical_search("alpha", top_k=5, include_stale=True)] == []
    assert db.get_documents(filter_dict={"document_id": "web:beta"}, limit=5)["ids"] == ["beta-0"]
