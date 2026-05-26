from __future__ import annotations

import copy
import json
from pathlib import Path

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.core.config import Config
from knowledge_hub.core.prepared_source_record import (
    build_prepared_source_record_from_text,
    prepared_vector_metadata,
)
from knowledge_hub.core.source_ledger_record import (
    DEFAULT_SOURCE_LEDGER_POLICY,
    build_source_ledger_record,
)
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.web.crawl4ai_adapter import CrawlDocument
from knowledge_hub.web.ingest import WebIngestService, make_web_note_id
from knowledge_hub.web.prepared_source import (
    PREPARED_SOURCE_RECORD_SCHEMA,
    build_prepared_source_record_from_quality_doc,
)
from knowledge_hub.web.quality import evaluate_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "docs" / "schemas" / "fixtures" / "prepared-source-record.v1.fixture.json"
SOURCE_LEDGER_FIXTURE_PATH = PROJECT_ROOT / "docs" / "schemas" / "fixtures" / "source-ledger-record.v1.fixture.json"
SCHEMA_ID = "knowledge-hub.prepared-source-record.v1"


class _IngestConfig:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = str(sqlite_path)
        self.vault_path = ""
        self.embedding_provider = "local"
        self.embedding_model = "test-embedding"

    def get_nested(self, *_keys: str, default: object = None) -> object:
        return default


class _FakeEmbedder:
    def embed_batch(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class _FailingEmbedder:
    def embed_batch(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        raise AssertionError("embedding should not be called")


class _FakeVectorDatabase:
    def __init__(self) -> None:
        self.deleted: list[dict[str, object]] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, object]] = []
        self.ids: list[str] = []

    def delete_by_metadata(self, filter_dict: dict[str, object]) -> None:
        self.deleted.append(dict(filter_dict))

    def add_documents(
        self,
        *,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
        ids: list[str],
    ) -> None:
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def count(self) -> int:
        return len(self.documents)

    def get_documents(
        self,
        filter_dict: dict[str, object] | None = None,
        limit: int = 500,
        offset: int = 0,
        include_ids: bool = True,
        include_documents: bool = True,
        include_metadatas: bool = True,
        include_embeddings: bool = False,
    ) -> dict[str, object]:
        matches: list[int] = []
        for index, metadata in enumerate(self.metadatas):
            if all(metadata.get(key) == value for key, value in dict(filter_dict or {}).items()):
                matches.append(index)
        matches = matches[offset : offset + limit]
        payload: dict[str, object] = {}
        if include_ids:
            payload["ids"] = [self.ids[index] for index in matches]
        if include_documents:
            payload["documents"] = [self.documents[index] for index in matches]
        if include_metadatas:
            payload["metadatas"] = [dict(self.metadatas[index]) for index in matches]
        return payload

    def update_metadata_by_id(self, metadata_by_id: dict[str, dict[str, object]]) -> int:
        updated = 0
        for doc_id, metadata in metadata_by_id.items():
            if doc_id not in self.ids:
                continue
            index = self.ids.index(doc_id)
            self.metadatas[index] = dict(metadata)
            updated += 1
        return updated


def _runtime_config(tmp_path: Path) -> Config:
    config = Config()
    config.set_nested("storage", "sqlite", str(tmp_path / "khub.sqlite"))
    config.set_nested("storage", "papers_dir", str(tmp_path / "papers"))
    config.set_nested("storage", "vector_db", str(tmp_path / "vector"))
    config.set_nested("obsidian", "vault_path", str(tmp_path / "vault"))
    config.set_nested("chunking", "chunk_size", 1000)
    config.set_nested("chunking", "chunk_overlap", 100)
    return config


def _load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _load_source_ledger_fixture() -> dict[str, object]:
    return json.loads(SOURCE_LEDGER_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_prepared_source_record_fixture_validates() -> None:
    payload = _load_fixture()

    result = validate_payload(payload, SCHEMA_ID, strict=True)

    assert result.ok, result.errors


def test_source_ledger_record_fixture_validates() -> None:
    payload = _load_source_ledger_fixture()

    result = validate_payload(payload, "knowledge-hub.source-ledger-record.v1", strict=True)

    assert result.ok, result.errors


def test_source_ledger_default_policy_is_unknown_and_fail_closed() -> None:
    record = build_source_ledger_record(
        ledger_id="ledger-1",
        source_id="web:ledger-1",
        source_type="web",
        canonical_uri="https://example.com/ledger-1",
        source_content_hash="sha256:ledger-source",
    )

    assert record["policy"] == DEFAULT_SOURCE_LEDGER_POLICY
    assert record["policy"]["classification"] == "UNKNOWN"
    assert record["policy"]["external_allowed"] is False


def test_prepared_source_record_requires_source_content_hash() -> None:
    payload = copy.deepcopy(_load_fixture())
    payload.pop("source_content_hash")

    result = validate_payload(payload, SCHEMA_ID, strict=True)

    assert not result.ok
    assert any("source_content_hash" in error for error in result.errors)


def test_prepared_source_record_requires_segment_locator() -> None:
    payload = copy.deepcopy(_load_fixture())
    prepared = payload["prepared"]
    assert isinstance(prepared, dict)
    segments = prepared["segments"]
    assert isinstance(segments, list)
    first_segment = segments[0]
    assert isinstance(first_segment, dict)
    first_segment.pop("locator")

    result = validate_payload(payload, SCHEMA_ID, strict=True)

    assert not result.ok
    assert any("/prepared/segments/0" in error and "locator" in error for error in result.errors)


def test_youtube_quality_doc_adapter_emits_prepared_source_record() -> None:
    content = (
        "# Example Video\n\n"
        "## Transcript\n"
        "[00:00:12] This segment explains retrieval augmented generation.\n"
        "[00:00:18] The next segment discusses vector search evidence."
    )
    doc = CrawlDocument(
        url="https://www.youtube.com/watch?v=abc123xyz89",
        title="Example Video",
        content=content,
        markdown=content,
        author="Example Channel",
        source_metadata={
            "media_platform": "youtube",
            "media_type": "video",
            "language": "en",
            "video_id": "abc123xyz89",
            "channel_name": "Example Channel",
            "transcript_source": "caption",
            "transcript_segments": [
                {
                    "start_sec": 12.0,
                    "end_sec": 18.0,
                    "text": "This segment explains retrieval augmented generation.",
                },
                {
                    "start_sec": 18.0,
                    "end_sec": 24.0,
                    "text": "The next segment discusses vector search evidence.",
                },
            ],
        },
        fetched_at="2026-05-04T00:00:00+00:00",
        engine="youtube",
    )
    quality_batch = evaluate_batch([doc], threshold=0.0, min_tokens=1)
    quality_doc = quality_batch.items[0]

    record = build_prepared_source_record_from_quality_doc(
        quality_doc,
        source_id="web_abc123",
        topic="retrieval",
        run_id="run_1",
        created_at="2026-05-04T00:00:00Z",
    )

    assert record["schema"] == PREPARED_SOURCE_RECORD_SCHEMA
    assert record["source_type"] == "youtube"
    assert record["source_content_hash"] == quality_doc.content_hash
    assert record["processing"]["parser"] == "caption"
    assert record["prepared"]["segments"][0]["locator"]["timestamp_start_sec"] == 12.0
    result = validate_payload(record, SCHEMA_ID, strict=True)
    assert result.ok, result.errors


def test_generic_prepared_source_record_supports_paper_and_vault_sources() -> None:
    paper_record = build_prepared_source_record_from_text(
        source_id="paper:1706.03762",
        source_type="paper",
        canonical_uri="arxiv:1706.03762",
        text="Attention mechanisms connect tokens through weighted context.",
        title="Attention Is All You Need",
        source_content_hash="sha256:paper-source",
        ledger_id="1706.03762",
        raw_ref="papers/1706.03762.txt",
        metadata={"source_vendor": "arxiv", "source_item_id": "1706.03762"},
        processor="paper_text_preparer",
        parser="plain_text",
    )
    vault_record = build_prepared_source_record_from_text(
        source_id="vault:AI/Note.md",
        source_type="vault",
        canonical_uri="vault:AI/Note.md",
        text="Local vault note text stays local.",
        title="Note",
        source_content_hash="sha256:vault-source",
        ledger_id="AI/Note.md",
        raw_ref="AI/Note.md",
        metadata={"source_name": "vault"},
        processor="vault_markdown_preparer",
        parser="obsidian_markdown",
    )

    for record in (paper_record, vault_record):
        result = validate_payload(record, SCHEMA_ID, strict=True)
        assert result.ok, result.errors
        metadata = prepared_vector_metadata(record)
        assert metadata["prepared_record_id"] == record["record_id"]
        assert metadata["prepared_text_hash"] == record["prepared"]["text_hash"]
        assert "prepared_raw_ref" not in metadata


def test_web_ingest_persists_prepared_source_record_with_storage_ref(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    service = WebIngestService(config, sqlite_db=sqlite_db)
    content = (
        "# Prepared Source Storage\n\n"
        "This web document has enough words to pass the local preparation quality gate. "
        "It should be stored as cleaned markdown and as a prepared source JSON artifact."
    )
    doc = CrawlDocument(
        url="https://example.com/prepared-source-storage",
        title="Prepared Source Storage",
        content=content,
        markdown=content,
        source_metadata={
            "source_name": "Example",
            "source_type": "web",
            "source_vendor": "example",
            "source_channel": "example",
            "language": "en",
        },
        fetched_at="2026-05-04T00:00:00+00:00",
        engine="test",
    )

    try:
        summary = service.ingest_documents(
            [doc],
            topic="retrieval",
            index=False,
            extract_concepts=False,
            incremental=False,
            quality_threshold=0.0,
            quality_min_tokens=10,
        )
        payload = summary.to_dict()

        assert payload["quality"]["preparedRecordCount"] == 1
        assert payload["quality"]["preparedRecordIds"]
        assert "preparedRecordDir" not in payload["quality"]
        assert "preparedRecordPaths" not in payload["quality"]

        note = sqlite_db.get_note(make_web_note_id(doc.url))
        assert note is not None
        metadata = json.loads(note["metadata"])
        prepared_path = Path(str(metadata["prepared_record_path"]))
        assert prepared_path.exists()
        assert prepared_path.parent.name == "web"

        record = json.loads(prepared_path.read_text(encoding="utf-8"))
        assert record["schema"] == PREPARED_SOURCE_RECORD_SCHEMA
        assert record["storage_ref"] == str(prepared_path)
        assert record["raw_ref"].endswith(".md")
        assert record["ledger_id"] == record["source_id"]

        validation = validate_payload(record, SCHEMA_ID, strict=True)
        assert validation.ok, validation.errors

        assert metadata["prepared_record_path"] == str(prepared_path)
        assert metadata["prepared_record_schema"] == PREPARED_SOURCE_RECORD_SCHEMA
        ledger_path = next((tmp_path / "source_ledger" / "web").glob("*.json"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["policy"]["classification"] == "UNKNOWN"
        assert ledger["policy"]["external_allowed"] is False
    finally:
        sqlite_db.close()


def test_web_ingest_keeps_legacy_storage_when_prepared_save_fails(tmp_path: Path, monkeypatch) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(config, sqlite_db=sqlite_db, vector_db=vector_db, embedder=_FakeEmbedder())
    content = (
        "# Prepared Source Fallback\n\n"
        "This web document has enough words to pass quality while the prepared-source ledger "
        "write is forced to fail. The cleaned legacy row should still be stored and indexed."
    )
    doc = CrawlDocument(
        url="https://example.com/prepared-source-fallback",
        title="Prepared Source Fallback",
        content=content,
        markdown=content,
        source_metadata={
            "source_name": "Example",
            "source_type": "web",
            "source_vendor": "example",
            "source_channel": "example",
            "language": "en",
        },
        fetched_at="2026-05-04T00:00:00+00:00",
        engine="test",
    )

    def fail_ledger_write(*_args: object, **_kwargs: object) -> Path:
        raise OSError("ledger write blocked")

    monkeypatch.setattr("knowledge_hub.web.ingest.write_source_ledger_record", fail_ledger_write)

    try:
        summary = service.ingest_documents(
            [doc],
            topic="retrieval",
            index=True,
            extract_concepts=False,
            incremental=False,
            quality_threshold=0.0,
            quality_min_tokens=10,
        )
        payload = summary.to_dict()

        assert payload["stored"] == 1
        assert payload["failed"] == []
        assert payload["quality"]["preparedRecordCount"] == 0
        assert "preparedRecordDir" not in payload["quality"]
        assert "preparedRecordPaths" not in payload["quality"]
        assert any("prepared source save failed" in warning for warning in payload["warnings"])
        note = sqlite_db.get_note(make_web_note_id(doc.url))
        assert note is not None
        metadata = json.loads(note["metadata"])
        assert "prepared_record_path" not in metadata
        assert vector_db.documents
        assert "Prepared Source Fallback" in vector_db.documents[0]
        assert "prepared_record_id" not in vector_db.metadatas[0]
    finally:
        sqlite_db.close()


def test_youtube_ingest_writes_matching_prepared_and_ledger_source_family(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    service = WebIngestService(config, sqlite_db=sqlite_db, vector_db=_FakeVectorDatabase(), embedder=_FakeEmbedder())
    content = (
        "First transcript segment explains prepared-source ledger family handling. "
        "Second transcript segment keeps enough words for quality approval."
    )
    doc = CrawlDocument(
        url="https://www.youtube.com/watch?v=ledgerFamily",
        title="Prepared Ledger Family",
        content=content,
        markdown=content,
        source_metadata={
            "source_name": "YouTube",
            "source_type": "web",
            "source_vendor": "youtube",
            "source_channel": "video",
            "media_platform": "youtube",
            "media_type": "video",
            "video_id": "ledgerFamily",
            "transcript_segments": [
                {"text": "First transcript segment explains prepared-source ledger family handling.", "start_sec": 0.0, "end_sec": 8.0},
                {"text": "Second transcript segment keeps enough words for quality approval.", "start_sec": 8.0, "end_sec": 16.0},
            ],
            "language": "en",
        },
        fetched_at="2026-05-04T00:00:00+00:00",
        engine="youtube",
    )

    try:
        summary = service.ingest_documents(
            [doc],
            topic="retrieval",
            index=False,
            extract_concepts=False,
            incremental=False,
            quality_threshold=0.0,
            quality_min_tokens=10,
        )
        payload = summary.to_dict()

        assert payload["quality"]["preparedRecordCount"] == 1
        note = sqlite_db.get_note(make_web_note_id(doc.url))
        assert note is not None
        metadata = json.loads(note["metadata"])
        prepared_path = Path(str(metadata["prepared_record_path"]))
        assert prepared_path.parent.name == "youtube"
        prepared_record = json.loads(prepared_path.read_text(encoding="utf-8"))
        assert prepared_record["source_type"] == "youtube"

        ledger_path = next((tmp_path / "source_ledger" / "youtube").glob("*.json"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["source_type"] == "youtube"
        assert ledger["artifacts"]["prepared_ref"] == str(prepared_path)
        assert not list((tmp_path / "source_ledger" / "web").glob("*.json"))
    finally:
        sqlite_db.close()


def test_web_indexing_reads_persisted_prepared_source_record(tmp_path: Path) -> None:
    record = copy.deepcopy(_load_fixture())
    record.update(
        {
            "record_id": "prepared:web:index-handoff",
            "source_id": "web_index_handoff",
            "source_type": "web",
            "canonical_uri": "https://example.com/index-handoff",
            "source_content_hash": "sha256:prepared-index-source",
            "raw_ref": str(tmp_path / "web_docs" / "web_index_handoff.md"),
            "title": "Prepared Index Handoff",
        }
    )
    prepared = record["prepared"]
    assert isinstance(prepared, dict)
    prepared["text"] = "Prepared canonical text should be embedded from the saved preparation artifact."
    prepared["text_hash"] = "sha256:prepared-index-text"
    prepared_path = tmp_path / "prepared_sources" / "web" / "prepared-web-index-handoff.json"
    prepared_path.parent.mkdir(parents=True)
    record["storage_ref"] = str(prepared_path)
    prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        _IngestConfig(tmp_path / "khub.sqlite"),
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    indexed, _ = service._index_web_records(
        [
            {
                "note_id": "web_index_handoff",
                "title": "Legacy Row Title",
                "content": "stale row content must not be embedded",
                "url": "https://example.com/legacy",
                "source_content_hash": "sha256:legacy-row-hash",
                "prepared_source_record_path": str(prepared_path),
            }
        ],
        topic="retrieval",
        archive=tmp_path / "web_docs",
    )

    assert indexed == 1
    assert len(vector_db.documents) == 1
    assert "Prepared canonical text should be embedded" in vector_db.documents[0]
    assert "stale row content" not in vector_db.documents[0]
    metadata = vector_db.metadatas[0]
    assert metadata["url"] == "https://example.com/index-handoff"
    assert metadata["source_content_hash"] == "sha256:prepared-index-source"
    assert metadata["prepared_record_id"] == "prepared:web:index-handoff"
    assert metadata["prepared_record_path"] == str(prepared_path)
    assert metadata["prepared_text_hash"] == "sha256:prepared-index-text"
    assert "prepared_raw_ref" not in metadata


def test_web_reindex_approved_creates_prepared_record_for_legacy_note(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    sqlite_db.upsert_note(
        "web_legacy_prepared_backfill",
        title="Legacy Prepared Backfill",
        content="Legacy web note content should receive a prepared source record during approved reindex.",
        file_path=str(tmp_path / "web_docs" / "legacy.md"),
        source_type="web",
        metadata={
            "url": "https://example.com/legacy-prepared-backfill",
            "topic": "retrieval",
            "source_name": "Example",
            "content_sha1": "legacy-content-sha1",
            "quality": {"approved": True, "score": 0.91},
        },
    )

    try:
        payload = service.reindex_approved(
            topic="retrieval",
            limit=1,
        )

        assert payload["status"] == "ok"
        assert payload["selected"] == 1
        assert payload["indexedChunks"] == 1
        assert len(vector_db.metadatas) == 1
        metadata = vector_db.metadatas[0]
        assert metadata["prepared_record_schema"] == PREPARED_SOURCE_RECORD_SCHEMA
        assert metadata["prepared_record_id"].startswith("prepared:web:")
        prepared_path = Path(str(metadata["prepared_record_path"]))
        assert prepared_path.exists()

        record = json.loads(prepared_path.read_text(encoding="utf-8"))
        assert record["source_id"] == "web_legacy_prepared_backfill"
        assert record["source_type"] == "web"
        assert record["canonical_uri"] == "https://example.com/legacy-prepared-backfill"
        assert record["processing"]["processor"] == "web_reindex_preparer"
        assert record["quality"]["passed"] is True
        assert "Legacy web note content" in record["prepared"]["text"]

        note = sqlite_db.get_note("web_legacy_prepared_backfill")
        assert note is not None
        note_metadata = json.loads(note["metadata"])
        assert note_metadata["prepared_record_path"] == str(prepared_path)
        assert note_metadata["prepared_record_schema"] == PREPARED_SOURCE_RECORD_SCHEMA
        ledger_path = next((tmp_path / "source_ledger" / "web").glob("*.json"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["artifacts"]["prepared_ref"] == str(prepared_path)
    finally:
        sqlite_db.close()


def test_web_reindex_approved_requires_source_ledger_for_prepared_backfill(
    tmp_path: Path, monkeypatch
) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    sqlite_db.upsert_note(
        "web_legacy_ledger_required",
        title="Legacy Ledger Required",
        content="Legacy web note content must not report prepared-source success when ledger persistence fails.",
        file_path=str(tmp_path / "web_docs" / "ledger-required.md"),
        source_type="web",
        metadata={
            "url": "https://example.com/legacy-ledger-required",
            "topic": "retrieval",
            "source_name": "Example",
            "quality": {"approved": True, "score": 0.93},
        },
    )

    def fail_ledger_write(*_args: object, **_kwargs: object) -> Path:
        raise OSError("ledger write blocked")

    monkeypatch.setattr("knowledge_hub.web.ingest.write_source_ledger_record", fail_ledger_write)

    try:
        payload = service.reindex_approved(topic="retrieval", limit=1)

        assert payload["status"] == "partial"
        assert payload["selected"] == 1
        assert payload["preparedRecordsCreated"] == 0
        assert "ledger write blocked" in payload["failed"][0]["error"]
        note = sqlite_db.get_note("web_legacy_ledger_required")
        assert note is not None
        note_metadata = json.loads(note["metadata"])
        assert "prepared_record_path" not in note_metadata
        assert not list((tmp_path / "prepared_sources").glob("**/*.json"))
        assert not list((tmp_path / "source_ledger").glob("**/*.json"))
        assert len(vector_db.metadatas) == 1
        assert "prepared_record_id" not in vector_db.metadatas[0]
    finally:
        sqlite_db.close()


def test_web_reindex_prepared_metadata_only_backfills_existing_vector_without_embedding(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    vector_db.documents.append("Existing vector document text should not be re-embedded.")
    vector_db.ids.append("web_legacy_metadata_only_0")
    vector_db.metadatas.append(
        {
            "source_type": "web",
            "document_id": "web:web_legacy_metadata_only",
            "title": "Legacy Metadata Only",
            "url": "https://example.com/legacy-metadata-only",
        }
    )
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FailingEmbedder(),
    )

    sqlite_db.upsert_note(
        "web_legacy_metadata_only",
        title="Legacy Metadata Only",
        content="Legacy web note content can backfill prepared metadata without embedding.",
        file_path=str(tmp_path / "web_docs" / "metadata-only.md"),
        source_type="web",
        metadata={
            "url": "https://example.com/legacy-metadata-only",
            "topic": "retrieval",
            "source_name": "Example",
            "quality": {"approved": True, "score": 0.88},
        },
    )

    try:
        payload = service.reindex_approved(
            topic="retrieval",
            limit=1,
            prepared_metadata_only=True,
        )

        assert payload["status"] == "ok"
        assert payload["selected"] == 1
        assert payload["indexedChunks"] == 0
        assert payload["preparedMetadataOnly"] is True
        assert payload["preparedRecordsCreated"] == 1
        assert payload["vectorMetadataUpdated"] == 1

        metadata = vector_db.metadatas[0]
        assert metadata["prepared_record_schema"] == PREPARED_SOURCE_RECORD_SCHEMA
        assert metadata["prepared_record_id"].startswith("prepared:web:")
        prepared_path = Path(str(metadata["prepared_record_path"]))
        assert prepared_path.exists()
        assert vector_db.documents == ["Existing vector document text should not be re-embedded."]
    finally:
        sqlite_db.close()


def test_web_reindex_prepared_metadata_only_requires_existing_vector_rows(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FailingEmbedder(),
    )

    sqlite_db.upsert_note(
        "web_metadata_only_without_vector",
        title="Metadata Only Without Vector",
        content="Approved legacy web note must not create prepared artifacts when no vector row can be patched.",
        file_path=str(tmp_path / "web_docs" / "metadata-only-without-vector.md"),
        source_type="web",
        metadata={
            "url": "https://example.com/metadata-only-without-vector",
            "topic": "retrieval",
            "source_name": "Example",
            "quality": {"approved": True, "score": 0.9},
        },
    )

    try:
        payload = service.reindex_approved(
            topic="retrieval",
            limit=1,
            prepared_metadata_only=True,
        )

        assert payload["status"] == "partial"
        assert payload["selected"] == 1
        assert payload["preparedRecordsCreated"] == 0
        assert payload["vectorMetadataUpdated"] == 0
        assert payload["failed"][0]["error"] == "no matching web vector rows for prepared metadata backfill"
        note = sqlite_db.get_note("web_metadata_only_without_vector")
        assert note is not None
        note_metadata = json.loads(note["metadata"])
        assert "prepared_record_path" not in note_metadata
        assert not list((tmp_path / "prepared_sources").glob("**/*.json"))
        assert not list((tmp_path / "source_ledger").glob("**/*.json"))
    finally:
        sqlite_db.close()


def test_web_reindex_include_unrated_keeps_prepared_source_non_authoritative(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    sqlite_db.upsert_note(
        "web_unrated_legacy",
        title="Unrated Legacy",
        content="Unrated legacy web note can still reindex through the legacy path.",
        file_path=str(tmp_path / "web_docs" / "unrated.md"),
        source_type="web",
        metadata={
            "url": "https://example.com/unrated-legacy",
            "topic": "retrieval",
            "source_name": "Example",
        },
    )

    try:
        payload = service.reindex_approved(
            topic="retrieval",
            limit=1,
            include_unrated=True,
        )

        assert payload["status"] == "ok"
        assert payload["selected"] == 1
        assert payload["indexedChunks"] == 1
        assert payload["preparedRecordsCreated"] == 0
        assert len(vector_db.metadatas) == 1
        assert "prepared_record_id" not in vector_db.metadatas[0]
        assert "prepared_record_path" not in vector_db.metadatas[0]
        note = sqlite_db.get_note("web_unrated_legacy")
        assert note is not None
        note_metadata = json.loads(note["metadata"])
        assert "prepared_record_path" not in note_metadata
    finally:
        sqlite_db.close()


def test_web_reindex_approved_repairs_broken_prepared_record_paths(tmp_path: Path) -> None:
    config = _IngestConfig(tmp_path / "khub.sqlite")
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        config,
        sqlite_db=sqlite_db,
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )
    broken_dir = tmp_path / "broken_prepared"
    broken_dir.mkdir()
    invalid_path = broken_dir / "invalid.json"
    invalid_path.write_text(json.dumps({"schema": "knowledge-hub.prepared-source-record.v0"}), encoding="utf-8")
    missing_path = broken_dir / "missing.json"
    stale_record = copy.deepcopy(_load_fixture())
    stale_record.update(
        {
            "record_id": "prepared:web:stale-broken",
            "source_id": "web_broken_stale",
            "source_type": "web",
            "canonical_uri": "https://example.com/web_broken_stale",
            "source_content_hash": "sha256:stale-broken",
        }
    )
    lifecycle = stale_record["lifecycle"]
    assert isinstance(lifecycle, dict)
    lifecycle["stale"] = True
    lifecycle["stale_reason"] = "source_content_changed"
    stale_path = broken_dir / "stale.json"
    stale_record["storage_ref"] = str(stale_path)
    stale_path.write_text(json.dumps(stale_record, ensure_ascii=False, indent=2), encoding="utf-8")
    cases = (
        ("web_broken_missing", "Broken Missing", missing_path),
        ("web_broken_invalid", "Broken Invalid", invalid_path),
        ("web_broken_stale", "Broken Stale", stale_path),
    )
    for note_id, title, path in cases:
        sqlite_db.upsert_note(
            note_id,
            title=title,
            content=f"{title} legacy content should regenerate a prepared source record.",
            file_path=str(tmp_path / "web_docs" / f"{note_id}.md"),
            source_type="web",
            metadata={
                "url": f"https://example.com/{note_id}",
                "topic": "retrieval",
                "source_name": "Example",
                "quality": {"approved": True, "score": 0.93},
                "prepared_record_path": str(path),
            },
        )

    try:
        payload = service.reindex_approved(topic="retrieval", limit=3)

        assert payload["status"] == "ok"
        assert payload["selected"] == 3
        assert payload["preparedRecordsCreated"] == 3
        assert payload["indexedChunks"] == 3
        assert len(vector_db.metadatas) == 3
        repaired_paths = {str(metadata["prepared_record_path"]) for metadata in vector_db.metadatas}
        assert str(missing_path) not in repaired_paths
        assert str(invalid_path) not in repaired_paths
        assert str(stale_path) not in repaired_paths
        for prepared_path in repaired_paths:
            payload = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
            assert payload["quality"]["passed"] is True
            assert payload["lifecycle"]["stale"] is False
            assert payload["processing"]["processor"] == "web_reindex_preparer"
    finally:
        sqlite_db.close()


def test_web_indexing_ignores_failed_or_stale_prepared_source_record(tmp_path: Path) -> None:
    for key, value in (("quality", {"passed": False}), ("lifecycle", {"stale": True})):
        record = copy.deepcopy(_load_fixture())
        record.update(
            {
                "record_id": f"prepared:web:blocked-{key}",
                "source_id": f"web_blocked_{key}",
                "source_type": "web",
                "canonical_uri": f"https://example.com/blocked-{key}",
                "source_content_hash": f"sha256:blocked-{key}",
                "title": f"Blocked {key}",
            }
        )
        node = record[key]
        assert isinstance(node, dict)
        node.update(value)
        prepared = record["prepared"]
        assert isinstance(prepared, dict)
        prepared["text"] = f"Blocked prepared text from {key} must not be embedded."
        prepared_path = tmp_path / "prepared_sources" / "web" / f"blocked-{key}.json"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        record["storage_ref"] = str(prepared_path)
        prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        vector_db = _FakeVectorDatabase()
        service = WebIngestService(
            _IngestConfig(tmp_path / f"{key}.sqlite"),
            vector_db=vector_db,
            embedder=_FakeEmbedder(),
        )

        indexed, _ = service._index_web_records(
            [
                {
                    "note_id": f"web_blocked_{key}",
                    "title": f"Legacy {key}",
                    "content": f"Legacy fallback content for {key}.",
                    "url": f"https://example.com/legacy-{key}",
                    "source_content_hash": f"sha256:legacy-{key}",
                    "prepared_source_record_path": str(prepared_path),
                }
            ],
            topic="retrieval",
            archive=tmp_path / "web_docs",
        )

        assert indexed == 1
        assert f"Legacy fallback content for {key}." in vector_db.documents[0]
        assert "Blocked prepared text" not in vector_db.documents[0]
        assert vector_db.metadatas[0]["source_content_hash"] == f"sha256:legacy-{key}"
        assert "prepared_record_id" not in vector_db.metadatas[0]


def test_web_indexing_requires_explicit_prepared_quality_and_freshness(tmp_path: Path) -> None:
    cases = (
        ("missing_quality", "quality"),
        ("missing_lifecycle", "lifecycle"),
    )
    for key, removed_field in cases:
        record = copy.deepcopy(_load_fixture())
        record.update(
            {
                "record_id": f"prepared:web:strict-{key}",
                "source_id": f"web_strict_{key}",
                "source_type": "web",
                "canonical_uri": f"https://example.com/strict-{key}",
                "source_content_hash": f"sha256:strict-{key}",
                "title": f"Strict {key}",
            }
        )
        record.pop(removed_field)
        prepared = record["prepared"]
        assert isinstance(prepared, dict)
        prepared["text"] = f"Prepared text missing {removed_field} must not be embedded."
        prepared_path = tmp_path / "prepared_sources" / "web" / f"strict-{key}.json"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        record["storage_ref"] = str(prepared_path)
        prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        vector_db = _FakeVectorDatabase()
        service = WebIngestService(
            _IngestConfig(tmp_path / f"{key}.sqlite"),
            vector_db=vector_db,
            embedder=_FakeEmbedder(),
        )

        indexed, _ = service._index_web_records(
            [
                {
                    "note_id": f"web_strict_{key}",
                    "title": f"Legacy {key}",
                    "content": f"Legacy fallback content for {key}.",
                    "url": f"https://example.com/legacy-{key}",
                    "source_content_hash": f"sha256:legacy-{key}",
                    "prepared_source_record_path": str(prepared_path),
                }
            ],
            topic="retrieval",
            archive=tmp_path / "web_docs",
        )

        assert indexed == 1
        assert f"Legacy fallback content for {key}." in vector_db.documents[0]
        assert "Prepared text missing" not in vector_db.documents[0]
        assert vector_db.metadatas[0]["source_content_hash"] == f"sha256:legacy-{key}"
        assert "prepared_record_id" not in vector_db.metadatas[0]


def test_web_indexing_ignores_unknown_prepared_source_schema(tmp_path: Path) -> None:
    record = copy.deepcopy(_load_fixture())
    record["schema"] = "knowledge-hub.prepared-source-record.v0"
    prepared = record["prepared"]
    assert isinstance(prepared, dict)
    prepared["text"] = "Unknown schema prepared text must not be embedded."
    prepared_path = tmp_path / "prepared_sources" / "web" / "unknown-schema.json"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    record["storage_ref"] = str(prepared_path)
    prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        _IngestConfig(tmp_path / "khub.sqlite"),
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    indexed, _ = service._index_web_records(
        [
            {
                "note_id": "web_unknown_schema",
                "title": "Legacy Schema",
                "content": "Legacy schema fallback content.",
                "url": "https://example.com/legacy-schema",
                "source_content_hash": "sha256:legacy-schema",
                "prepared_source_record_path": str(prepared_path),
            }
        ],
        topic="retrieval",
        archive=tmp_path / "web_docs",
    )

    assert indexed == 1
    assert "Legacy schema fallback content." in vector_db.documents[0]
    assert "Unknown schema prepared text" not in vector_db.documents[0]
    assert "prepared_record_id" not in vector_db.metadatas[0]


def test_pipeline_indexing_accepts_note_prepared_record_path_key(tmp_path: Path) -> None:
    record = copy.deepcopy(_load_fixture())
    record.update(
        {
            "record_id": "prepared:web:pipeline-handoff",
            "source_id": "web_pipeline_handoff",
            "source_type": "web",
            "canonical_uri": "https://example.com/pipeline-handoff",
            "source_content_hash": "sha256:prepared-pipeline-source",
            "raw_ref": str(tmp_path / "normalized" / "web_pipeline_handoff.json"),
            "title": "Prepared Pipeline Handoff",
        }
    )
    prepared = record["prepared"]
    assert isinstance(prepared, dict)
    prepared["text"] = "Prepared pipeline text should be embedded from the note metadata path."
    prepared["text_hash"] = "sha256:prepared-pipeline-text"
    prepared_path = tmp_path / "prepared_sources" / "web" / "prepared-web-pipeline-handoff.json"
    prepared_path.parent.mkdir(parents=True)
    record["storage_ref"] = str(prepared_path)
    prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        _IngestConfig(tmp_path / "khub.sqlite"),
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    result = service._index_records_with_meta(
        [
            {
                "note_id": "web_pipeline_handoff",
                "title": "Legacy Pipeline Title",
                "content": "legacy pipeline content",
                "url": "https://example.com/legacy-pipeline",
                "prepared_record_path": str(prepared_path),
            }
        ],
        topic="retrieval",
        archive=tmp_path / "web_docs",
        embed_batch_size=4,
    )

    assert result["indexed_chunks"] == 1
    assert "Prepared pipeline text should be embedded" in vector_db.documents[0]
    metadata = vector_db.metadatas[0]
    assert metadata["url"] == "https://example.com/pipeline-handoff"
    assert metadata["source_content_hash"] == "sha256:prepared-pipeline-source"
    assert metadata["prepared_record_id"] == "prepared:web:pipeline-handoff"
    assert metadata["prepared_record_path"] == str(prepared_path)


def test_youtube_indexing_rehydrates_prepared_segments_for_timestamp_chunks(tmp_path: Path) -> None:
    record = copy.deepcopy(_load_fixture())
    record.update(
        {
            "record_id": "prepared:youtube:segment-handoff",
            "source_id": "web_youtube_handoff",
            "source_type": "youtube",
            "canonical_uri": "https://www.youtube.com/watch?v=segment12345",
            "source_content_hash": "sha256:prepared-youtube-source",
            "raw_ref": str(tmp_path / "web_docs" / "web_youtube_handoff.md"),
            "title": "Prepared YouTube Handoff",
        }
    )
    prepared = record["prepared"]
    assert isinstance(prepared, dict)
    prepared["text"] = "\n".join(
        [
            "First timestamped prepared segment.",
            "Second timestamped prepared segment.",
            "Third timestamped prepared segment.",
        ]
    )
    prepared["text_hash"] = "sha256:prepared-youtube-text"
    prepared["segments"] = [
        {
            "segment_id": "web_youtube_handoff:seg:0000",
            "text": "First timestamped prepared segment.",
            "char_start": 0,
            "char_end": 35,
            "locator": {
                "section": "Transcript",
                "timestamp_start_sec": 12.0,
                "timestamp_end_sec": 18.0,
                "source_ref": "https://www.youtube.com/watch?v=segment12345#t=12.000",
            },
        },
        {
            "segment_id": "web_youtube_handoff:seg:0001",
            "text": "Second timestamped prepared segment.",
            "char_start": 36,
            "char_end": 72,
            "locator": {
                "section": "Transcript",
                "timestamp_start_sec": 80.0,
                "timestamp_end_sec": 86.0,
                "source_ref": "https://www.youtube.com/watch?v=segment12345#t=80.000",
            },
        },
        {
            "segment_id": "web_youtube_handoff:seg:0002",
            "text": "Third timestamped prepared segment.",
            "char_start": 73,
            "char_end": 108,
            "locator": {
                "section": "Transcript",
                "timestamp_start_sec": 150.0,
                "timestamp_end_sec": 156.0,
                "source_ref": "https://www.youtube.com/watch?v=segment12345#t=150.000",
            },
        },
    ]
    prepared_path = tmp_path / "prepared_sources" / "youtube" / "prepared-youtube-segment-handoff.json"
    prepared_path.parent.mkdir(parents=True)
    record["storage_ref"] = str(prepared_path)
    prepared_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    vector_db = _FakeVectorDatabase()
    service = WebIngestService(
        _IngestConfig(tmp_path / "khub.sqlite"),
        vector_db=vector_db,
        embedder=_FakeEmbedder(),
    )

    indexed, _ = service._index_web_records(
        [
            {
                "note_id": "web_youtube_handoff",
                "title": "Legacy YouTube Title",
                "content": "",
                "url": "https://example.com/legacy-youtube",
                "prepared_source_record_path": str(prepared_path),
            }
        ],
        topic="retrieval",
        archive=tmp_path / "web_docs",
    )

    assert indexed >= 1
    metadata = vector_db.metadatas[0]
    assert metadata["media_platform"] == "youtube"
    assert metadata["start_sec"] == 12.0
    assert metadata["end_sec"] > 12.0
    assert metadata["timestamp_label"]
    assert metadata["prepared_segment_count"] == 3
