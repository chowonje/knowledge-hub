"""JudgmentStore: persistent user-judgment ledger (jsonl ground truth + sqlite mirror)."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from knowledge_hub.infrastructure.persistence.stores.judgment_store import JudgmentStore


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _store(tmp_path, *, ensure_schema: bool = True) -> JudgmentStore:
    db_path = tmp_path / "knowledge.db"
    store = JudgmentStore(_connect(db_path), db_path=db_path)
    if ensure_schema:
        store.ensure_schema()
    return store


def _base_fields(**overrides):
    fields = {
        "target_type": "answer",
        "target_id": "answer_001",
        "decision": "accept",
        "reviewer": "won",
        "reason": "spans verified against quoted source text",
    }
    fields.update(overrides)
    return fields


def _read_jsonl(store: JudgmentStore) -> list[dict]:
    assert store.jsonl_path.exists()
    lines = [line for line in store.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def test_record_judgment_appends_jsonl_line_and_mirror_row(tmp_path):
    store = _store(tmp_path)
    record = store.record_judgment(**_base_fields())

    assert record["judgment_id"].startswith("judgment_")
    assert len(record["judgment_id"]) == len("judgment_") + 12
    assert record["created_at"]
    assert record["reviewed_at"]
    assert record["confidence"] == "medium"
    assert record["decision"] == "accept"

    rows = _read_jsonl(store)
    assert len(rows) == 1
    assert rows[0]["judgment_id"] == record["judgment_id"]
    assert rows[0]["target_id"] == "answer_001"

    mirror = store.conn.execute(
        "SELECT * FROM judgments_v1 WHERE judgment_id = ?", (record["judgment_id"],)
    ).fetchone()
    assert mirror is not None
    assert mirror["target_type"] == "answer"
    assert mirror["decision"] == "accept"
    assert mirror["reviewer"] == "won"


def test_record_judgment_returned_record_round_trips_via_get(tmp_path):
    store = _store(tmp_path)
    record = store.record_judgment(
        **_base_fields(
            snippet_hashes=["abc123"],
            evidence_span_ids=["span_1", "span_2"],
            source_ids=["2203.15556"],
            query_hash="deadbeef",
            query_digest="where does performance come from",
            rag_answer_log_id=7,
            runtime_used="ask_v2",
        )
    )
    fetched = store.get_judgment(record["judgment_id"])
    assert fetched is not None
    assert fetched["snippet_hashes"] == ["abc123"]
    assert fetched["evidence_span_ids"] == ["span_1", "span_2"]
    assert fetched["source_ids"] == ["2203.15556"]
    assert fetched["query_hash"] == "deadbeef"
    assert fetched["rag_answer_log_id"] == 7
    assert fetched["runtime_used"] == "ask_v2"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_type": "card"},
        {"decision": "approve"},
        {"reviewer": ""},
        {"reviewer": "   "},
        {"reason": ""},
        {"target_id": ""},
        {"confidence": "certain"},
        {"decision_source": "api"},
        {"reviewed_at": "not-a-timestamp"},
    ],
)
def test_record_judgment_rejects_invalid_fields(tmp_path, overrides):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.record_judgment(**_base_fields(**overrides))
    # Nothing may be persisted on validation failure.
    assert not store.jsonl_path.exists()
    assert store.conn.execute("SELECT COUNT(*) AS cnt FROM judgments_v1").fetchone()["cnt"] == 0


def test_target_text_is_hashed_and_never_persisted(tmp_path):
    store = _store(tmp_path)
    raw_text = "the chinchilla scaling recipe dominates at fixed compute"
    record = store.record_judgment(**_base_fields(target_text=raw_text))

    expected = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:16]
    assert record["target_text_hash"] == expected
    assert "target_text" not in record

    jsonl_body = store.jsonl_path.read_text(encoding="utf-8")
    assert raw_text not in jsonl_body
    mirror = store.conn.execute(
        "SELECT * FROM judgments_v1 WHERE judgment_id = ?", (record["judgment_id"],)
    ).fetchone()
    assert mirror["target_text_hash"] == expected
    assert raw_text not in json.dumps(dict(mirror), ensure_ascii=False)


def test_supersede_lifecycle_is_append_only(tmp_path):
    store = _store(tmp_path)
    old = store.record_judgment(**_base_fields(decision="thin"))
    new = store.supersede_judgment(
        old["judgment_id"],
        **_base_fields(decision="accept", reason="re-reviewed with full span set"),
    )

    assert new["supersedes"] == old["judgment_id"]
    assert new["judgment_id"] != old["judgment_id"]

    old_row = store.get_judgment(old["judgment_id"])
    assert old_row["superseded_by"] == new["judgment_id"]

    rows = _read_jsonl(store)
    assert len(rows) == 2
    assert rows[0]["judgment_id"] == old["judgment_id"]
    assert rows[1]["judgment_id"] == new["judgment_id"]
    # jsonl ground truth is append-only: the old line is never rewritten.
    assert "superseded_by" not in rows[0] or not rows[0].get("superseded_by")


def test_supersede_unknown_judgment_raises(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.supersede_judgment("judgment_missing0000", **_base_fields())


def test_list_judgments_filters_and_limit(tmp_path):
    store = _store(tmp_path)
    store.record_judgment(**_base_fields(target_type="claim", target_id="c1", decision="accept"))
    store.record_judgment(**_base_fields(target_type="answer", target_id="a1", decision="reject"))
    store.record_judgment(
        **_base_fields(target_type="brief", target_id="b1", decision="thin", query_hash="qh1")
    )

    assert len(store.list_judgments()) == 3
    assert [item["target_id"] for item in store.list_judgments(target_type="claim")] == ["c1"]
    assert [item["target_id"] for item in store.list_judgments(decision="reject")] == ["a1"]
    assert [item["target_id"] for item in store.list_judgments(query_hash="qh1")] == ["b1"]
    assert len(store.list_judgments(limit=2)) == 2


def test_read_only_missing_table_returns_empty(tmp_path):
    store = _store(tmp_path, ensure_schema=False)
    assert store.list_judgments() == []
    assert store.get_judgment("judgment_abc123def456") is None


def test_store_registry_facade_bootstraps_and_delegates(tmp_path):
    from knowledge_hub.infrastructure.persistence import SQLiteDatabase

    db = SQLiteDatabase(str(tmp_path / "knowledge.db"), enable_event_store=False)
    try:
        record = db.record_judgment(**_base_fields(target_type="brief", target_id="brief_1"))
        assert record["judgment_id"].startswith("judgment_")
        items = db.list_judgments(target_type="brief")
        assert [item["judgment_id"] for item in items] == [record["judgment_id"]]
        assert db.get_judgment(record["judgment_id"]) is not None
        assert (tmp_path / "judgments.jsonl").exists()
    finally:
        db.close()
