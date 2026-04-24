from __future__ import annotations

from types import SimpleNamespace
import sqlite3

from knowledge_hub.application.index_freshness import (
    build_index_freshness_check,
    build_index_freshness_report,
)
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase


def test_index_freshness_report_flags_source_type_with_no_vectors():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 2, "vault": 3, "web": 4},
        vector_source_counts={"paper": 2, "vault": 1, "web": 0},
    )

    assert report["status"] == "degraded"
    assert report["missingVectorSourceTypes"] == ["web"]
    assert report["partialVectorSourceTypes"] == ["vault"]
    assert report["fixCommand"] == "khub index --all"


def test_index_freshness_check_reads_sqlite_and_vector_metadata(tmp_path):
    sqlite_path = tmp_path / "khub.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO papers VALUES ('1706.03762', 'Attention Is All You Need')")
    conn.execute("CREATE TABLE web_cards_v2 (document_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO web_cards_v2 VALUES ('web_alpha')")
    conn.commit()
    conn.close()

    vector_path = tmp_path / "vector"
    vector = VectorDatabase(str(vector_path), "knowledge_hub_test")
    vector.add_documents(
        ["Transformer attention"],
        [[0.0]],
        [{"source_type": "paper", "source_id": "paper:1706.03762#0", "title": "Attention"}],
        ids=["paper:1706.03762#0"],
    )

    check = build_index_freshness_check(
        SimpleNamespace(sqlite_path=str(sqlite_path), vector_db_path=str(vector_path)),
    )

    assert check["area"] == "index freshness"
    assert check["status"] == "degraded"
    assert check["diagnostics"]["sqliteSourceCounts"]["web"] == 1
    assert check["diagnostics"]["vectorSourceCounts"]["paper"] == 1
    assert check["diagnostics"]["missingVectorSourceTypes"] == ["web"]
