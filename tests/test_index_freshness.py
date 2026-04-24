from __future__ import annotations

from types import SimpleNamespace
import sqlite3

from knowledge_hub.application.index_freshness import (
    build_index_freshness_check,
    build_index_freshness_report,
    collect_sqlite_source_ids,
    collect_vector_source_ids,
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


def test_index_freshness_report_keeps_partial_counts_informational_without_source_id_metadata():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 3, "web": 0},
        vector_source_counts={"paper": 0, "vault": 1, "web": 0},
    )

    assert report["status"] == "ok"
    assert report["partialVectorSourceTypes"] == ["vault"]
    assert report["sourceIdCoverage"]["status"] == "unavailable"
    assert report["fixCommand"] == ""


def test_index_freshness_report_source_id_audit_detects_missing_ids_when_counts_match():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 2, "vault": 0, "web": 0},
        vector_source_counts={"paper": 2, "vault": 0, "web": 0},
        sqlite_source_ids={"paper": ["paper:1706.03762", "paper:9999.00001"]},
        vector_source_ids={"paper": ["paper:1706.03762"]},
    )

    assert report["status"] == "degraded"
    assert report["missingVectorSourceTypes"] == []
    assert report["partialVectorSourceTypes"] == []
    assert report["sourceIdCoverage"]["scope"] == "source_id"
    assert report["sourceIdCoverage"]["missingVectorSourceIdTypes"] == ["paper"]
    assert report["sourceIdCoverage"]["missingVectorSourceIds"]["paper"] == ["paper:9999.00001"]
    assert report["sourceIdCoverage"]["missingVectorSourceIdCounts"]["paper"] == 1
    assert report["fixCommand"] == "khub index --all"


def test_index_freshness_report_source_id_not_span_audit_documents_limitation():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 1, "vault": 0, "web": 0},
        vector_source_counts={"paper": 1, "vault": 0, "web": 0},
        sqlite_source_ids={"paper": ["1706.03762"]},
        vector_source_ids={"paper": ["paper:1706.03762#0"]},
    )

    assert report["status"] == "ok"
    assert report["sourceIdCoverage"]["spanCoverageAvailable"] is False
    assert "compares canonical source IDs" in report["sourceIdCoverage"]["limitation"]


def test_index_freshness_report_source_id_audit_skips_failure_when_vector_identity_metadata_unavailable():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 1, "web": 0},
        vector_source_counts={"paper": 0, "vault": 1, "web": 0},
        sqlite_source_ids={"vault": ["vault:notes/rag.md"]},
        vector_source_ids={"vault": []},
    )

    assert report["status"] == "ok"
    assert report["sourceIdCoverage"]["status"] == "unavailable"
    assert report["sourceIdCoverage"]["missingVectorSourceIds"]["vault"] == []
    assert report["sourceIdCoverage"]["sourceIdMetadataUnavailableTypes"] == ["vault"]


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


def test_index_freshness_check_reads_source_ids_from_sqlite_and_vector_sidecar(tmp_path):
    sqlite_path = tmp_path / "khub.sqlite3"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO papers VALUES ('1706.03762', 'Attention Is All You Need')")
    conn.execute("INSERT INTO papers VALUES ('9999.00001', 'Missing Paper')")
    conn.commit()
    conn.close()

    vector_path = tmp_path / "vector"
    vector = VectorDatabase(str(vector_path), "knowledge_hub_source_id_test")
    vector.add_documents(
        ["Transformer attention", "Duplicate chunk"],
        [[0.0], [0.0]],
        [
            {"source_type": "paper", "arxiv_id": "1706.03762", "title": "Attention"},
            {"source_type": "paper", "source_id": "paper:1706.03762#chunk-1", "title": "Attention"},
        ],
        ids=["paper_1706.03762_0", "paper_1706.03762_1"],
    )

    assert collect_sqlite_source_ids(sqlite_path)["paper"] == ["paper:1706.03762", "paper:9999.00001"]
    assert collect_vector_source_ids(vector_path)["paper"] == ["paper:1706.03762"]

    check = build_index_freshness_check(
        SimpleNamespace(sqlite_path=str(sqlite_path), vector_db_path=str(vector_path)),
    )

    assert check["status"] == "degraded"
    assert check["diagnostics"]["missingVectorSourceTypes"] == []
    assert check["diagnostics"]["vectorSourceCounts"]["paper"] == 2
    assert check["diagnostics"]["sourceIdCoverage"]["missingVectorSourceIds"]["paper"] == ["paper:9999.00001"]
    assert check["diagnostics"]["sourceIdCoverage"]["missingVectorSourceIdCounts"]["paper"] == 1
