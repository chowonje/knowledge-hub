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


def test_compare_json_builds_slot_packet_from_ask_v2_no_result_payload(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "status": "no_result",
            "answer": "근거 span으로만 비교합니다.",
            "sourceType": "paper",
            "paperFamily": "paper_compare",
            "queryFrame": {
                "source_type": "paper",
                "family": "paper_compare",
                "resolved_source_ids": ["1706.03762", "2312.00752"],
            },
            "v2": {
                "runtimeExecution": {"used": "ask_v2"},
                "claimCards": [
                    {
                        "claimCardId": "claim-rag-metric",
                        "sourceId": "2005.11401",
                        "claimType": "empirical",
                        "summaryText": "RAG reports retrieval-generation QA metrics.",
                        "evidenceAnchors": [
                            {
                                "anchorId": "claim-rag-metric-anchor",
                                "sourceId": "2005.11401",
                                "sourceType": "paper",
                                "evidenceRole": "metric",
                                "spanLocator": "chars:10-80",
                                "sourceContentHash": "sha256:rag",
                                "quote": "RAG reports retrieval-generation QA metrics.",
                            }
                        ],
                    },
                    {
                        "claimCardId": "claim-fid-metric",
                        "sourceId": "2007.01282",
                        "claimType": "empirical",
                        "summaryText": "FiD metric claim from a parsed preamble.",
                        "evidenceAnchors": [
                            {
                                "anchorId": "claim-fid-metric-anchor",
                                "sourceId": "2007.01282",
                                "sourceType": "paper",
                                "evidenceRole": "metric",
                                "spanLocator": "chars:20-100",
                                "sourceContentHash": "sha256:fid",
                                "quote": "[Block 1] %%%%% NEW MATH DEFINITIONS %%%%% \\newcommand{\\figleft}{x}",
                            }
                        ],
                    },
                ],
                "claimAlignment": {"groups": []},
                "paperKnowledgeSlots": [
                    {
                        "paperId": "1706.03762",
                        "title": "Attention Is All You Need",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "The Transformer uses attention for sequence modeling.",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-transformer-method",
                                        "sourceId": "1706.03762",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:10-80",
                                        "sourceContentHash": "sha256:transformer",
                                        "quote": "Transformer method.",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "paperId": "2312.00752",
                        "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "Mamba uses selective state spaces for sequence modeling.",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-mamba-method",
                                        "sourceId": "2312.00752",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:20-100",
                                        "sourceContentHash": "sha256:mamba",
                                        "quote": "Mamba method.",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
            "citations": [
                {"label": "S1", "target": "1706.03762", "kind": "paper"},
                {"label": "S2", "target": "2312.00752", "kind": "paper"},
            ],
            "sources": [
                {"source_id": "1706.03762", "source_type": "paper", "title": "Attention Is All You Need"},
                {
                    "source_id": "2312.00752",
                    "source_type": "paper",
                    "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                },
            ],
            "evidencePacketContract": {
                "packet_id": "epkt_1",
                "spans": [
                    {"span_id": "span_1", "source_id": "1706.03762"},
                    {"span_id": "span_2", "source_id": "2312.00752"},
                ],
            },
            "evidencePolicy": {"policyKey": "paper_compare_policy"},
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["Transformer와 Mamba를 논문 기준으로 비교해줘", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["comparePacket"]["coverage"]["answerable"] is True
    assert payload["comparePacket"]["coverage"]["strictSupportedDimensionCount"] == 1
    assert payload["comparePacket"]["coverage"]["fallbackSpanCount"] == 0
    assert payload["comparePacket"]["dimensions"][0]["dimensionId"] == "paper-slot:method"
    assert any("paper knowledge slots" in warning for warning in payload["warnings"])
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_compare_json_keeps_generic_unknown_compare_slot_payload_insufficient(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "status": "no_result",
            "answer": "근거가 부족합니다.",
            "sourceType": "paper",
            "paperFamily": "paper_compare",
            "queryFrame": {
                "source_type": "paper",
                "family": "paper_compare",
                "resolved_source_ids": ["2312.10997", "2512.13564"],
            },
            "v2": {
                "runtimeExecution": {"used": "ask_v2"},
                "claimCards": [],
                "claimAlignment": {"groups": []},
                "paperKnowledgeSlots": [
                    {
                        "paperId": "2312.10997",
                        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "A retrieval survey method summary.",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-rag-survey-method",
                                        "sourceId": "2312.10997",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:10-80",
                                        "sourceContentHash": "sha256:rag-survey",
                                        "quote": "RAG survey method.",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "paperId": "2512.13564",
                        "title": "Memory in the Age of AI Agents",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "A memory survey method summary.",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-memory-survey-method",
                                        "sourceId": "2512.13564",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:20-100",
                                        "sourceContentHash": "sha256:memory-survey",
                                        "quote": "Memory survey method.",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
            "citations": [
                {"label": "S1", "target": "2312.10997", "kind": "paper"},
                {"label": "S2", "target": "2512.13564", "kind": "paper"},
            ],
            "sources": [
                {"source_id": "2312.10997", "source_type": "paper", "title": "RAG survey"},
                {"source_id": "2512.13564", "source_type": "paper", "title": "Memory survey"},
            ],
            "evidencePacketContract": {
                "packet_id": "epkt_1",
                "spans": [
                    {
                        "spanRef": "span:1",
                        "sourceId": "2312.10997",
                        "sourceContentHash": "sha256:rag-survey",
                        "spanLocator": "chars:10-80",
                    },
                    {
                        "spanRef": "span:2",
                        "sourceId": "2512.13564",
                        "sourceContentHash": "sha256:memory-survey",
                        "spanLocator": "chars:20-100",
                    },
                ],
            },
            "evidencePolicy": {"policyKey": "paper_compare_policy"},
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["AlphaFoo Retrieval 논문과 BetaBar Memory 논문을 비교해줘", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["comparePacket"]["coverage"]["answerable"] is False
    assert payload["comparePacket"]["dimensions"][0]["dimensionId"] == "retrieved-source-coverage"
    assert not any("paper knowledge slots" in warning for warning in payload["warnings"])
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_compare_json_rejects_low_signal_slot_text_for_answerability(monkeypatch):
    monkeypatch.setattr(
        substrate_cmd,
        "_run_answer_facade",
        lambda *_args, **_kwargs: {
            "status": "no_result",
            "answer": "근거가 부족합니다.",
            "sourceType": "paper",
            "paperFamily": "paper_compare",
            "queryFrame": {
                "source_type": "paper",
                "family": "paper_compare",
                "resolved_source_ids": ["2005.11401", "2007.01282"],
            },
            "v2": {
                "runtimeExecution": {"used": "ask_v2"},
                "claimCards": [],
                "claimAlignment": {"groups": []},
                "paperKnowledgeSlots": [
                    {
                        "paperId": "2005.11401",
                        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "RAG combines retrieval and generation for open-domain QA.",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-rag-method",
                                        "sourceId": "2005.11401",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:10-80",
                                        "sourceContentHash": "sha256:rag",
                                        "quote": "RAG method.",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "paperId": "2007.01282",
                        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                        "slots": [
                            {
                                "slotType": "method",
                                "text": "\\newcommand{\\parents}{Pa} \\DeclareMathOperator*{\\argmax}{arg\\,max}",
                                "evidenceRefs": [
                                    {
                                        "anchorId": "slot-fid-method",
                                        "sourceId": "2007.01282",
                                        "sourceType": "paper",
                                        "spanLocator": "chars:20-100",
                                        "sourceContentHash": "sha256:fid",
                                        "quote": "FiD preamble.",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
            "citations": [
                {"label": "S1", "target": "2005.11401", "kind": "paper"},
                {"label": "S2", "target": "2007.01282", "kind": "paper"},
            ],
            "sources": [
                {"source_id": "2005.11401", "source_type": "paper", "title": "RAG"},
                {"source_id": "2007.01282", "source_type": "paper", "title": "FiD"},
            ],
            "evidencePacketContract": {
                "packet_id": "epkt_1",
                "spans": [
                    {
                        "spanRef": "span:1",
                        "sourceId": "2005.11401",
                        "sourceContentHash": "sha256:rag",
                        "spanLocator": "chars:10-80",
                    },
                    {
                        "spanRef": "span:2",
                        "sourceId": "2007.01282",
                        "sourceContentHash": "sha256:fid",
                        "spanLocator": "chars:20-100",
                    },
                ],
            },
            "evidencePolicy": {"policyKey": "paper_compare_policy"},
        },
    )

    result = CliRunner().invoke(
        substrate_cmd.compare_cmd,
        ["RAG와 Fusion-in-Decoder를 논문 기준으로 비교해줘", "--json"],
        obj={"khub": object()},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["comparePacket"]["coverage"]["answerable"] is False
    assert payload["comparePacket"]["dimensions"][0]["dimensionId"] == "retrieved-source-coverage"
    assert not any("paper knowledge slots" in warning for warning in payload["warnings"])
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
