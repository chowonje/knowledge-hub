from __future__ import annotations

import json
from types import SimpleNamespace
import sqlite3

from knowledge_hub.application.index_freshness import (
    build_index_freshness_check,
    build_index_freshness_report,
    collect_sqlite_source_ids,
    collect_vector_prepared_counts,
    collect_vector_source_counts,
    collect_vector_source_ids,
)
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase


def _create_lexical_sidecar(vector_path):
    vector_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(vector_path / "_lexical.sqlite3")
    conn.execute(
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
    conn.execute(
        """
        CREATE TABLE lexical_documents_meta (
            doc_id TEXT PRIMARY KEY,
            metadata_json TEXT
        )
        """
    )
    return conn


def _insert_lexical_row(conn, *, doc_id: str, metadata: dict[str, object]):
    document = str(metadata.get("document") or doc_id)
    conn.execute(
        """
        INSERT INTO lexical_documents_fts(
            doc_id, title, section_title, contextual_summary, keywords, field, document, searchable_text
        )
        VALUES (?, ?, '', '', '', '', ?, ?)
        """,
        (doc_id, str(metadata.get("title") or doc_id), document, document),
    )
    conn.execute(
        "INSERT INTO lexical_documents_meta(doc_id, metadata_json) VALUES (?, ?)",
        (doc_id, json.dumps(metadata, ensure_ascii=False)),
    )


def _write_prepared_record(path, *, stale: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "knowledge-hub.prepared-source-record.v1",
                "quality": {"passed": True},
                "lifecycle": {"stale": bool(stale)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_index_freshness_report_flags_source_type_with_no_vectors():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 2, "vault": 3, "web": 4},
        vector_source_counts={"paper": 2, "vault": 1, "web": 0},
    )

    assert report["status"] == "degraded"
    assert report["missingVectorSourceTypes"] == ["web"]
    assert report["partialVectorSourceTypes"] == ["vault"]
    assert report["fixCommand"] == "khub crawl reindex-approved --include-unrated --json"
    assert report["fixCommands"] == ["khub crawl reindex-approved --include-unrated --json"]
    assert report["sourceRepairCommands"] == {
        "web": "khub crawl reindex-approved --include-unrated --json",
    }


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
    assert report["sourceIdCoverage"]["missingVectorSourceIds"]["paper"][0].startswith("paper-redacted-")
    assert report["sourceIdCoverage"]["missingVectorSourceIdCounts"]["paper"] == 1
    assert report["sourceIdCoverage"]["sourceIdCoverageRatios"]["paper"] == 0.5
    assert report["sourceIdCoverage"]["missingVectorSourceIdSampleMode"] == "redacted_sha256"
    assert report["fixCommand"] == "khub index --all"
    assert report["fixCommands"] == ["khub index --all"]
    assert report["sourceRepairCommands"] == {"paper": "khub index --all"}


def test_index_freshness_report_uses_source_specific_fix_commands():
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 1, "vault": 1, "web": 1},
        vector_source_counts={"paper": 0, "vault": 0, "web": 0},
    )

    assert report["status"] == "degraded"
    assert report["fixCommand"] == (
        "khub index --all --vault-all && khub crawl reindex-approved --include-unrated --json"
    )
    assert report["fixCommands"] == [
        "khub index --all --vault-all",
        "khub crawl reindex-approved --include-unrated --json",
    ]
    assert report["sourceRepairCommands"] == {
        "paper": "khub index --all",
        "vault": "khub index --vault-all",
        "web": "khub crawl reindex-approved --include-unrated --json",
    }

    vault_report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 1, "web": 0},
        vector_source_counts={"paper": 0, "vault": 0, "web": 0},
    )

    assert vault_report["fixCommand"] == "khub index --vault-all"
    assert vault_report["sourceRepairCommands"] == {"vault": "khub index --vault-all"}


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
    assert check["diagnostics"]["sourceIdCoverage"]["missingVectorSourceIds"]["paper"][0].startswith("paper-redacted-")
    assert check["diagnostics"]["sourceIdCoverage"]["missingVectorSourceIdCounts"]["paper"] == 1


def test_index_freshness_reports_prepared_record_vector_coverage(tmp_path):
    vector_path = tmp_path / "vector"
    prepared_alpha_path = tmp_path / "prepared_sources" / "web" / "prepared-web-alpha.json"
    _write_prepared_record(prepared_alpha_path)
    vector = VectorDatabase(str(vector_path), "knowledge_hub_prepared_coverage_test")
    vector.add_documents(
        ["Prepared web chunk", "Legacy web chunk"],
        [[0.0], [0.0]],
        [
            {
                "source_type": "web",
                "document_id": "web:alpha",
                "prepared_record_id": "prepared:web:alpha",
                "prepared_record_path": str(prepared_alpha_path),
            },
            {
                "source_type": "web",
                "document_id": "web:beta",
            },
        ],
        ids=["web_alpha_0", "web_beta_0"],
    )

    prepared_counts = collect_vector_prepared_counts(vector_path)
    assert prepared_counts["web"] == {"vectorRows": 2, "preparedRows": 1}

    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 0, "web": 2},
        vector_source_counts={"paper": 0, "vault": 0, "web": 2},
        vector_prepared_counts=prepared_counts,
    )

    coverage = report["preparedRecordCoverage"]
    assert report["status"] == "degraded"
    assert report["fixCommand"] == "khub crawl reindex-approved --include-unrated --prepared-metadata-only --json"
    assert report["fixCommands"] == ["khub crawl reindex-approved --include-unrated --prepared-metadata-only --json"]
    assert report["sourceRepairCommands"] == {
        "web": "khub crawl reindex-approved --include-unrated --prepared-metadata-only --json"
    }
    assert coverage["status"] == "degraded"
    assert coverage["preparedRowCoverageRatios"]["web"] == 0.5
    assert coverage["missingPreparedRecordTypes"] == ["web"]
    assert coverage["preparedRepairCommands"] == {
        "web": "khub crawl reindex-approved --include-unrated --prepared-metadata-only --json"
    }


def test_index_freshness_counts_web_rows_beyond_default_lexical_prefix(tmp_path):
    vector_path = tmp_path / "vector"
    prepared_late_path = tmp_path / "prepared_sources" / "web" / "prepared-web-late-source.json"
    _write_prepared_record(prepared_late_path)
    conn = _create_lexical_sidecar(vector_path)
    for index in range(20005):
        _insert_lexical_row(
            conn,
            doc_id=f"paper_{index:05d}_0",
            metadata={
                "source_type": "paper",
                "source_id": f"paper:{index:05d}",
                "title": "Paper",
            },
        )
    _insert_lexical_row(
        conn,
        doc_id="web_z_late_0",
        metadata={
            "source_type": "web",
            "document_id": "web:late-source",
            "prepared_record_id": "prepared:web:late-source",
            "prepared_record_path": str(prepared_late_path),
            "title": "Late web source",
        },
    )
    conn.commit()
    conn.close()

    counts = collect_vector_source_counts(vector_path)
    source_ids = collect_vector_source_ids(vector_path)
    prepared_counts = collect_vector_prepared_counts(vector_path)
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 0, "web": 1},
        vector_source_counts=counts,
        sqlite_source_ids={"web": ["web:late-source"]},
        vector_source_ids=source_ids,
        vector_prepared_counts=prepared_counts,
    )

    assert counts["web"] == 1
    assert len(source_ids["paper"]) == 20005
    assert source_ids["web"] == ["web:late-source"]
    assert prepared_counts["paper"] == {"vectorRows": 20005, "preparedRows": 0}
    assert prepared_counts["web"] == {"vectorRows": 1, "preparedRows": 1}
    assert report["status"] == "ok"
    assert report["fixCommand"] == ""
    assert report["fixCommands"] == []
    assert report["sourceRepairCommands"] == {}
    assert "web" not in report["missingVectorSourceTypes"]
    assert report["sourceIdCoverage"]["missingVectorSourceIdCounts"]["web"] == 0
    assert "paper" not in report["preparedRecordCoverage"]["missingPreparedRecordTypes"]


def test_index_freshness_prepared_metadata_repairs_only_web_scope(tmp_path):
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 1, "vault": 1, "web": 0},
        vector_source_counts={"paper": 1, "vault": 1, "web": 0},
        sqlite_source_ids={"paper": ["paper:1706.03762"], "vault": ["vault:notes/a.md"]},
        vector_source_ids={"paper": ["paper:1706.03762"], "vault": ["vault:notes/a.md"]},
        vector_prepared_counts={
            "paper": {"vectorRows": 1, "preparedRows": 0},
            "vault": {"vectorRows": 1, "preparedRows": 0},
            "web": {"vectorRows": 0, "preparedRows": 0},
        },
    )

    assert report["status"] == "ok"
    assert report["fixCommand"] == ""
    assert report["fixCommands"] == []
    assert report["sourceRepairCommands"] == {}
    assert report["preparedRecordCoverage"]["missingPreparedRecordTypes"] == []


def test_index_freshness_marks_prepared_coverage_unavailable_without_lexical_metadata(tmp_path):
    vector_path = tmp_path / "vector_without_lexical"
    vector_path.mkdir(parents=True)

    prepared_counts = collect_vector_prepared_counts(vector_path)
    assert prepared_counts is None

    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 0, "web": 1},
        vector_source_counts={"paper": 0, "vault": 0, "web": 1},
        vector_prepared_counts=prepared_counts,
    )

    coverage = report["preparedRecordCoverage"]
    assert coverage["status"] == "unavailable"
    assert coverage["metadataUnavailableTypes"] == ["web"]
    assert coverage["preparedRowCoverageRatios"]["web"] is None
    assert coverage["missingPreparedRecordTypes"] == []
    assert coverage["preparedRepairCommands"] == {}


def test_index_freshness_missing_source_id_samples_are_redacted():
    macos_user_root = "/Users" + "/example"
    report = build_index_freshness_report(
        sqlite_source_counts={"paper": 0, "vault": 2, "web": 1},
        vector_source_counts={"paper": 0, "vault": 1, "web": 1},
        sqlite_source_ids={
            "vault": [
                f"vault:{macos_user_root}/Library/Mobile Documents/private/Client Strategy.md",
                "vault:AI/Private/Research Notes.md",
            ],
            "web": [f"web:{macos_user_root}/private-cache/source.html"],
        },
        vector_source_ids={"vault": ["vault:AI/Public/Indexed.md"], "web": ["web:web_indexed"]},
    )

    payload = json.dumps(report, ensure_ascii=False)
    assert report["sourceIdCoverage"]["missingVectorSourceIdCounts"] == {"paper": 0, "vault": 2, "web": 1}
    assert report["sourceIdCoverage"]["missingVectorSourceIdTypes"] == ["vault", "web"]
    assert report["sourceIdCoverage"]["missingVectorSourceIds"]["vault"]
    assert all(item.startswith("vault-redacted-") for item in report["sourceIdCoverage"]["missingVectorSourceIds"]["vault"])
    assert all(item.startswith("web-redacted-") for item in report["sourceIdCoverage"]["missingVectorSourceIds"]["web"])
    assert macos_user_root not in payload
    assert "Mobile Documents" not in payload
    assert "Client Strategy" not in payload
    assert "Research Notes" not in payload
    assert "private-cache" not in payload
