from __future__ import annotations

import pytest

from knowledge_hub.application.evidence_registry import (
    register_answer_trace,
    register_context_pack,
    register_packet,
    resolve_registry_lookup,
)
from knowledge_hub.application.evidence_substrate import build_trace_payload
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase


def _db(tmp_path):
    return SQLiteDatabase(str(tmp_path / "knowledge.db"))


def test_registry_lookup_returns_exact_packet_by_id(tmp_path):
    db = _db(tmp_path)
    packet = {
        "schema": "knowledge-hub.evidence-packet.v1",
        "packet_id": "epkt_1",
        "query": "registry",
        "spans": [
            {
                "span_id": "span_1",
                "source_id": "src_1",
                "content_hash": "sha256:aaa",
                "char_start": 0,
                "char_end": 12,
                "text": "source text",
            }
        ],
    }

    record = register_packet(db, packet)
    lookup = resolve_registry_lookup(db, "packet", "epkt_1")

    assert record["registryId"] == "epkt_1"
    assert lookup["status"] == "ok"
    assert lookup["storedStaleness"]["status"] == "fresh"
    assert lookup["currentStaleness"]["status"] == "unchecked"
    assert lookup["payload"] == packet
    assert lookup["registryRecord"]["payloadHash"].startswith("sha256:")
    assert validate_payload(record, record["schema"], strict=True).ok
    assert validate_payload(lookup, lookup["schema"], strict=True).ok


def test_registry_missing_context_returns_stable_not_found(tmp_path):
    db = _db(tmp_path)

    lookup = resolve_registry_lookup(db, "context", "ctx_missing")

    assert lookup["schema"] == "knowledge-hub.evidence-registry.lookup.result.v1"
    assert lookup["status"] == "not_found"
    assert lookup["resourceKind"] == "context"
    assert lookup["identifier"] == "ctx_missing"
    assert validate_payload(lookup, lookup["schema"], strict=True).ok


def test_registry_source_hash_mismatch_marks_record_stale(tmp_path):
    db = _db(tmp_path)
    packet = {
        "schema": "knowledge-hub.evidence-packet.v1",
        "packet_id": "epkt_stale",
        "spans": [{"span_id": "span_1", "source_id": "src_1", "content_hash": "sha256:old"}],
    }
    register_packet(db, packet)

    lookup = resolve_registry_lookup(
        db,
        "packet",
        "epkt_stale",
        current_source_refs=[{"sourceId": "src_1", "sourceContentHash": "sha256:new"}],
    )

    assert lookup["status"] == "stale"
    assert lookup["storedStaleness"]["status"] == "fresh"
    assert lookup["currentStaleness"]["status"] == "stale"
    assert lookup["currentStaleness"]["mismatchedSourceIds"] == ["src_1"]
    assert lookup["registryRecord"]["status"] == "ok"
    assert lookup["registryRecord"]["staleReason"] == ""


def test_registry_source_hash_match_marks_current_staleness_fresh(tmp_path):
    db = _db(tmp_path)
    packet = {
        "schema": "knowledge-hub.evidence-packet.v1",
        "packet_id": "epkt_fresh",
        "spans": [{"span_id": "span_1", "source_id": "src_1", "content_hash": "sha256:aaa"}],
    }
    register_packet(db, packet)

    lookup = resolve_registry_lookup(
        db,
        "packet",
        "epkt_fresh",
        current_source_refs=[{"sourceId": "src_1", "sourceContentHash": "sha256:aaa"}],
    )

    assert lookup["status"] == "ok"
    assert lookup["currentStaleness"]["status"] == "fresh"
    assert lookup["currentStaleness"]["matchedSourceIds"] == ["src_1"]


def test_answer_trace_registry_keeps_citations_linked_to_evidence_spans(tmp_path):
    db = _db(tmp_path)
    answer_payload = {
        "question": "trace",
        "answer": "Grounded answer.",
        "answerContract": {"answer_id": "ans_1", "evidence_packet_id": "epkt_1"},
        "evidencePacketContract": {
            "schema": "knowledge-hub.evidence-packet.v1",
            "packet_id": "epkt_1",
            "spans": [{"span_id": "span_1", "source_id": "src_1", "content_hash": "sha256:aaa"}],
        },
        "citations": [{"label": "S1", "source_id": "src_1", "span_id": "span_1"}],
        "sources": [{"source_id": "src_1", "title": "Source 1"}],
    }
    trace = build_trace_payload(answer_payload)

    write = register_answer_trace(db, answer_payload=answer_payload, trace_payload=trace)
    lookup = resolve_registry_lookup(db, "trace", "ans_1")

    assert write["status"] == "ok"
    assert lookup["status"] == "ok"
    assert lookup["lineage"]["sourceIds"] == ["src_1"]
    assert lookup["lineage"]["evidenceSpanIds"] == ["span_1"]
    assert lookup["lineage"]["citationLabels"] == ["S1"]


def test_context_pack_registry_enforces_token_budget(tmp_path):
    db = _db(tmp_path)
    context_pack = {
        "schema": "knowledge-hub.context-pack.result.v1",
        "contextPackId": "ctx_1",
        "status": "ok",
        "target": "task",
        "query": "budget",
        "sources": [{"source_id": "src_1", "text": "x" * 200}],
        "counts": {"total": 1},
        "warnings": [],
    }

    with pytest.raises(ValueError, match="context pack exceeds token budget"):
        register_context_pack(db, context_pack, max_tokens=1)


def test_context_pack_registry_lookup_returns_payload_when_present(tmp_path):
    db = _db(tmp_path)
    context_pack = {
        "schema": "knowledge-hub.context-pack.result.v1",
        "contextPackId": "ctx_1",
        "status": "ok",
        "target": "task",
        "query": "lookup",
        "sources": [{"source_id": "src_1", "source_content_hash": "sha256:aaa"}],
        "counts": {"total": 1},
        "warnings": [],
    }

    record = register_context_pack(db, context_pack, max_tokens=1000)
    lookup = resolve_registry_lookup(db, "context", "ctx_1")

    assert record["recordKind"] == "context"
    assert lookup["status"] == "ok"
    assert lookup["payload"] == context_pack
