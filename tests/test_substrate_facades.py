from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands import substrate_cmd
from knowledge_hub.interfaces.cli.main import cli


class _Config:
    def __init__(self, root):
        self.vector_db_path = str(root / "chroma_db")
        self.sqlite_path = str(root / "knowledge.db")
        self.collection_name = "knowledge_hub_test"


class _Khub:
    def __init__(self, root):
        self.config = _Config(root)

    def sqlite_db(self, **_kwargs):
        return SimpleNamespace(get_stats=lambda: {"notes": 1, "papers": 2})


class _RegistryKhub(_Khub):
    def __init__(self, root):
        super().__init__(root)
        self._db = SQLiteDatabase(str(root / "knowledge.db"))

    def sqlite_db(self, **_kwargs):
        return self._db


def test_inspect_contract_json_reports_index_boundary(tmp_path):
    result = CliRunner().invoke(
        substrate_cmd.inspect_cmd,
        ["contract", "--json"],
        obj={"khub": _Khub(tmp_path)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.inspect.result.v1"
    assert payload["contract"]["canonicalFlow"][:3] == [
        "SourceLedgerRecord",
        "PreparedSourceRecord",
        "RetrievalUnit",
    ]
    assert "claim_cards" in payload["contract"]["indexContract"]["khubIndexDoesNotBuildByDefault"]
    assert validate_payload(payload, payload["schema"], strict=True).ok
    assert validate_payload(payload["contract"], payload["contract"]["schema"], strict=True).ok


def test_trace_from_json_exposes_answer_to_citation_links(tmp_path):
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps(
            {
                "question": "compare tools and resources",
                "answer": "Use both when warranted.",
                "answerContract": {"answer_id": "ans_1", "evidence_packet_id": "epkt_1"},
                "evidencePacketContract": {
                    "packet_id": "epkt_1",
                    "spans": [{"span_id": "span_1", "source_id": "src_1"}],
                    "gaps": [],
                },
                "citations": [{"label": "S1", "title": "MCP design", "kind": "web"}],
                "sources": [{"title": "MCP design", "source_type": "web"}],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(substrate_cmd.trace_cmd, ["--from-json", str(answer_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.answer-trace.result.v1"
    assert payload["status"] == "ok"
    assert payload["answerId"] == "ans_1"
    assert payload["evidencePacketId"] == "epkt_1"
    assert payload["citations"][0]["label"] == "S1"
    assert payload["lineage"]["sourceIds"] == ["src_1"]
    assert payload["lineage"]["evidenceSpanIds"] == ["span_1"]
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_trace_marks_missing_citations_as_insufficient(tmp_path):
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps(
            {
                "question": "unsupported",
                "answer": "Unsupported answer.",
                "sources": [{"title": "Source only", "source_type": "web"}],
                "evidencePacketContract": {"packet_id": "epkt_1", "spans": []},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(substrate_cmd.trace_cmd, ["--from-json", str(answer_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "insufficient_evidence"
    assert "trace has no citation entries" in payload["warnings"]


def test_trace_from_json_save_registry_persists_lookup_records(tmp_path):
    answer_path = tmp_path / "answer.json"
    answer_path.write_text(
        json.dumps(
            {
                "question": "trace",
                "answer": "Use cited evidence.",
                "answerContract": {"answer_id": "ans_1", "evidence_packet_id": "epkt_1"},
                "evidencePacketContract": {
                    "schema": "knowledge-hub.evidence-packet.v1",
                    "packet_id": "epkt_1",
                    "spans": [{"span_id": "span_1", "source_id": "src_1", "content_hash": "sha256:aaa"}],
                },
                "citations": [{"label": "S1", "source_id": "src_1", "span_id": "span_1"}],
                "sources": [{"source_id": "src_1", "title": "Source 1"}],
            }
        ),
        encoding="utf-8",
    )
    khub = _RegistryKhub(tmp_path)

    result = CliRunner().invoke(
        substrate_cmd.trace_cmd,
        ["--from-json", str(answer_path), "--save-registry", "--json"],
        obj={"khub": khub},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["registry"]["status"] == "ok"
    assert {item["recordKind"] for item in payload["registry"]["records"]} == {"packet", "trace"}
    assert khub.sqlite_db().get_evidence_registry_record("packet", "epkt_1")["payload"]["packet_id"] == "epkt_1"


def test_compare_json_reuses_existing_answer_payload(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "answer": "A and B differ in evidence stance.",
            "comparePacketContract": {
                "schema": "knowledge-hub.compare-packet.v1",
                "packet_id": "cmp_1",
                "dimensions": [{"name": "stance", "status": "conflict"}],
                "coverage": {"answerable": True},
            },
            "citations": [{"label": "S1", "title": "A"}],
            "sources": [{"title": "A", "source_type": "paper"}],
            "evidencePacketContract": {
                "packet_id": "epkt_1",
                "spans": [{"span_id": "span_1", "source_id": "src_1"}],
            },
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["compare source A and source B", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "knowledge-hub.compare.result.v1"
    assert payload["status"] == "ok"
    assert payload["comparePacket"]["packet_id"] == "cmp_1"
    assert payload["trace"]["evidencePacketId"] == "epkt_1"
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_compare_json_builds_source_fallback_packet_when_claim_packet_missing(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "answer": "근거가 부족합니다.",
            "citations": [{"label": "S1", "target": "2005.11401", "kind": "arxiv"}],
            "sources": [
                {
                    "source_id": "2005.11401",
                    "source_type": "paper",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "excerpt": "RAG combines retrieval and generation.",
                }
            ],
            "evidencePacketContract": {"packet_id": "epkt_1", "spans": []},
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["compare RAG retrieval generation", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "insufficient_evidence"
    assert payload["comparePacket"]["schema"] == "knowledge-hub.compare-packet.v1"
    assert payload["comparePacket"]["coverage"]["supportingSpanCount"] == 1
    assert payload["comparePacket"]["coverage"]["answerable"] is False
    assert payload["comparePacket"]["dimensions"][0]["supportingSpans"][0]["sourceId"] == "2005.11401"
    assert payload["comparePacket"]["dimensions"][0]["supportingSpans"][0]["strictSpanBacked"] is False
    assert payload["comparePacket"]["dimensions"][0]["supportingSpans"][0]["fallbackSpan"] is True
    assert any("compare packet built from retrieved source spans" in warning for warning in payload["warnings"])
    assert payload["trace"]["comparePacket"]["packet_id"] == payload["comparePacket"]["packet_id"]
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_compare_json_enriches_existing_packet_with_strict_evidence_spans(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "answer": "A and B conflict on accuracy.",
            "comparePacketContract": {
                "schema": "knowledge-hub.compare-packet.v1",
                "packet_id": "cmp_strict",
                "dimensions": [
                    {
                        "dimensionId": "accuracy",
                        "label": "accuracy",
                        "comparisonStatus": "conflict",
                        "supportingSpans": [],
                    }
                ],
                "coverage": {"answerable": False},
            },
            "answerContract": {
                "citations": [
                    {
                        "spanRef": "span:1",
                        "source_id": "paper:a#0",
                        "content_hash": "sha256:a",
                        "char_start": 10,
                        "char_end": 40,
                        "label": "S1",
                    },
                    {
                        "spanRef": "span:2",
                        "source_id": "paper:b#0",
                        "content_hash": "sha256:b",
                        "char_start": 50,
                        "char_end": 90,
                        "label": "S2",
                    },
                ]
            },
            "citations": [{"label": "S1", "target": "paper:a#0"}, {"label": "S2", "target": "paper:b#0"}],
            "sources": [
                {"source_id": "paper:a#0", "source_type": "paper", "title": "A", "excerpt": "A accuracy"},
                {"source_id": "paper:b#0", "source_type": "paper", "title": "B", "excerpt": "B accuracy"},
            ],
            "evidencePacketContract": {
                "packet_id": "epkt_1",
                "spans": [
                    {
                        "spanRef": "span:1",
                        "sourceId": "paper:a#0",
                        "source_type": "paper",
                        "sourceContentHash": "sha256:a",
                        "spanLocator": "chars:10-40",
                        "text": "A accuracy",
                    },
                    {
                        "spanRef": "span:2",
                        "sourceId": "paper:b#0",
                        "source_type": "paper",
                        "sourceContentHash": "sha256:b",
                        "spanLocator": "chars:50-90",
                        "text": "B accuracy",
                    },
                ],
            },
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["compare strict evidence", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["comparePacket"]["packet_id"] == "cmp_strict"
    assert payload["comparePacket"]["coverage"]["answerable"] is True
    assert payload["comparePacket"]["coverage"]["strictSpanBackedCount"] == 2
    assert payload["comparePacket"]["coverage"]["fallbackSpanCount"] == 0
    spans = payload["comparePacket"]["dimensions"][0]["supportingSpans"]
    assert {span["sourceId"] for span in spans} == {"paper:a#0", "paper:b#0"}
    assert {span["strictSpanBacked"] for span in spans} == {True}
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_top_level_help_exposes_substrate_facades():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "compare" in result.output
    assert "trace" in result.output
