from __future__ import annotations

from click.testing import CliRunner

from knowledge_hub.interfaces.cli.commands.health_cmd import (
    _event_integrity_rows,
    _paper_source_integrity_rows,
    _pipeline_integrity_rows,
    health_cmd,
)
from knowledge_hub.core.config import Config


class _StubKhub:
    def __init__(self, config: Config):
        self.config = config


def _make_config(tmp_path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector_db"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    return config


def test_health_check_events_success(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._event_integrity_rows",
        lambda _cfg: ([("이벤트 로그 수 (JSONL vs SQLite)", "OK", "1 vs 1")], True),
    )

    result = runner.invoke(health_cmd, ["--check-events"], obj={"khub": _StubKhub(config)})
    assert result.exit_code == 0


def test_health_check_events_mismatch_returns_nonzero(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._event_integrity_rows",
        lambda _cfg: ([("이벤트 로그 수 (JSONL vs SQLite)", "WARN", "1 vs 2")], False),
    )

    result = runner.invoke(health_cmd, ["--check-events"], obj={"khub": _StubKhub(config)})
    assert result.exit_code != 0
    assert "health check failed" in result.output


def test_health_check_pipeline_success(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._pipeline_integrity_rows",
        lambda _cfg: ([("indexed 레코드 vs indexed manifest", "OK", "1 vs 1")], True),
    )

    result = runner.invoke(health_cmd, ["--check-pipeline"], obj={"khub": _StubKhub(config)})
    assert result.exit_code == 0


def test_health_check_pipeline_mismatch_returns_nonzero(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._pipeline_integrity_rows",
        lambda _cfg: ([("indexed 레코드 vs indexed manifest", "WARN", "2 vs 1")], False),
    )

    result = runner.invoke(health_cmd, ["--check-pipeline"], obj={"khub": _StubKhub(config)})
    assert result.exit_code != 0
    assert "pipeline integrity mismatch" in result.output


def test_health_check_paper_sources_success(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._paper_source_integrity_rows",
        lambda _cfg: (
            [
                ("known cleanup rules", "OK", "6 tracked rules"),
                ("pending canonical relinks", "OK", "0"),
            ],
            True,
        ),
    )

    result = runner.invoke(health_cmd, ["--check-paper-sources"], obj={"khub": _StubKhub(config)})
    assert result.exit_code == 0


def test_health_check_paper_sources_mismatch_returns_nonzero(monkeypatch, tmp_path):
    runner = CliRunner()
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd._paper_source_integrity_rows",
        lambda _cfg: (
            [
                ("pending canonical relinks", "WARN", "1"),
                ("manual source fixes required", "WARN", "1"),
            ],
            False,
        ),
    )

    result = runner.invoke(health_cmd, ["--check-paper-sources"], obj={"khub": _StubKhub(config)})
    assert result.exit_code != 0
    assert "paper source integrity mismatch" in result.output


class _FakeConn:
    def __init__(self, rows_by_query):
        self.rows_by_query = rows_by_query

    def execute(self, query, params=()):  # noqa: ARG002
        for marker, value in self.rows_by_query.items():
            if marker in query:
                return _FakeCursor(value)
        raise AssertionError(f"unexpected query: {query}")


class _FakeCursor:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        if isinstance(self.value, list):
            return self.value[0] if self.value else None
        return self.value

    def fetchall(self):
        if isinstance(self.value, list):
            return self.value
        return [self.value] if self.value is not None else []


class _FakeEventStore:
    def __init__(self, entity_count: int):
        self.entity_count = entity_count

    def snapshot_at(self, _timestamp):
        return {"entity_count": self.entity_count}


class _FakeSQLiteDB:
    def __init__(self, *, conn=None, latest=None, counts=None, indexed_rows=None, checkpoints=None, records=None, event_store=None):
        self.conn = conn
        self._latest = latest or {}
        self._counts = counts or {}
        self._indexed_rows = indexed_rows or []
        self._checkpoints = checkpoints or []
        self._records = records or {}
        self.event_store = event_store

    def get_latest_crawl_pipeline_job(self):
        return self._latest

    def count_crawl_pipeline_records(self, _job_id):
        return self._counts

    def list_crawl_pipeline_records(self, _job_id, state=None, limit=0):  # noqa: ARG002
        return self._indexed_rows

    def list_crawl_pipeline_checkpoints(self, _job_id):
        return self._checkpoints

    def get_crawl_pipeline_record(self, _job_id, record_id):
        return self._records.get(record_id)

    def close(self):
        return None


def test_pipeline_integrity_skips_manifest_check_when_storage_root_unavailable(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    fake_db = _FakeSQLiteDB(
        latest={"job_id": "job-1", "storage_root": str(tmp_path / "missing-root")},
        counts={"total": 1, "indexed": 1, "failed": 0},
        indexed_rows=[{"indexed_path": str(tmp_path / "missing-root" / "artifact.json")}],
        checkpoints=[],
        records={},
    )
    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.SQLiteDatabase", lambda _path: fake_db)

    rows, ok = _pipeline_integrity_rows(config)

    assert ok is True
    assert ("스토리지 루트", "SKIP", f"job=job-1 root unavailable: {tmp_path / 'missing-root'}") in rows
    assert (
        "indexed 레코드 vs indexed manifest",
        "SKIP",
        f"storage_root unavailable: {tmp_path / 'missing-root'}",
    ) in rows


def test_pipeline_integrity_fails_when_storage_root_exists_but_manifest_missing(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    storage_root = tmp_path / "mounted-root"
    storage_root.mkdir(parents=True)
    fake_db = _FakeSQLiteDB(
        latest={"job_id": "job-2", "storage_root": str(storage_root)},
        counts={"total": 1, "indexed": 1, "failed": 0},
        indexed_rows=[{"indexed_path": str(storage_root / "artifact.json")}],
        checkpoints=[],
        records={},
    )
    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.SQLiteDatabase", lambda _path: fake_db)

    rows, ok = _pipeline_integrity_rows(config)

    assert ok is False
    assert ("스토리지 루트", "OK", f"job=job-2 root={storage_root}") in rows
    assert ("indexed 레코드 vs indexed manifest", "WARN", "1 vs 0 (missing=1)") in rows


def test_event_integrity_reports_orphan_entity_samples(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    fake_conn = _FakeConn(
        {
            "SELECT COUNT(*) AS cnt FROM ontology_events": {"cnt": 2},
            "SELECT COUNT(*) AS cnt FROM ontology_entities": {"cnt": 3},
            "SELECT COUNT(*) AS cnt\n                FROM ontology_entities e": {"cnt": 1},
            "SELECT e.entity_id\n                FROM ontology_entities e": [{"entity_id": "note:web_123"}],
        }
    )
    fake_db = _FakeSQLiteDB(conn=fake_conn, event_store=_FakeEventStore(entity_count=2))
    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.SQLiteDatabase", lambda _path: fake_db)
    jsonl_path = tmp_path / "ontology_events.jsonl"
    jsonl_path.write_text('{"event_id":"evt1"}\n{"event_id":"evt2"}\n', encoding="utf-8")
    config.set_nested("storage", "sqlite", str(tmp_path / "knowledge.db"))

    rows, ok = _event_integrity_rows(config)

    assert ok is False
    assert ("엔티티 수 (snapshot vs ontology_entities)", "WARN", "2 vs 3") in rows
    assert ("이벤트 없이 남은 엔티티", "WARN", "1 sample=note:web_123") in rows


def test_paper_source_integrity_rows_reports_pending_and_manual(monkeypatch, tmp_path):
    config = _make_config(tmp_path)

    class _FakeSQLiteDB:
        def close(self):
            return None

    monkeypatch.setattr("knowledge_hub.infrastructure.persistence.SQLiteDatabase", lambda _path: _FakeSQLiteDB())
    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.health_cmd.build_paper_source_ops_report",
        lambda sqlite_db, limit=20: {
            "counts": {
                "knownRuleCount": 6,
                "presentInStore": 5,
                "missingFromStore": 1,
                "repairablePending": 2,
                "blockedManual": 1,
                "blockedMissingCanonical": 0,
                "keepCurrentReviewed": 1,
                "alreadyAligned": 1,
            }
        },
    )

    rows, ok = _paper_source_integrity_rows(config)

    assert ok is False
    assert ("pending canonical relinks", "WARN", "2") in rows
    assert ("manual source fixes required", "WARN", "1") in rows
    assert ("missing canonical rows", "OK", "0") in rows
