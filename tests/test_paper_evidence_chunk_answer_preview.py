from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.paper_labs_cmd import paper_labs_group
from knowledge_hub.papers.evidence_chunk_answer_preview import (
    PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID,
    build_evidence_chunk_query_plan,
    build_paper_evidence_chunk_answer_preview,
)


class _FakeSearcher:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def generate_answer(self, question, **kwargs):  # noqa: ANN001
        self.calls.append({"question": question, **kwargs})
        query_plan = dict(kwargs.get("query_plan") or {})
        paper_ids = list(query_plan.get("resolvedPaperIds") or [])
        return {
            "answer": "Evidence-backed preview answer.",
            "evidencePacket": {
                "answerable": True,
                "selectedEvidenceCount": 2,
                "citationCount": 2,
                "parsedArtifactEvidenceChunkAdapter": {
                    "status": "applied",
                    "resolvedPaperIds": paper_ids,
                    "candidateRowsConsidered": 2,
                    "rowsAdded": 2,
                },
            },
            "evidencePacketContract": {
                "answerable": True,
                "spans": [{"sourceId": paper_ids[0] if paper_ids else ""}],
            },
            "citations": [{"source_id": paper_ids[0] if paper_ids else "", "span_locator": "chars:0-12"}],
            "sources": [{"source_type": "paper", "source_id": paper_ids[0] if paper_ids else ""}],
            "warnings": [],
        }


class _FakeFactory:
    def __init__(self, searcher):
        self._searcher = searcher

    def searcher(self):
        return self._searcher


class _FakeKhub:
    def __init__(self, searcher):
        self.factory = _FakeFactory(searcher)
        self.config = SimpleNamespace()


def test_build_evidence_chunk_query_plan_sets_runtime_opt_in() -> None:
    plan = build_evidence_chunk_query_plan(["2501.00001", "2501.00001", "2501.00002"])

    assert plan["parsed_artifact_evidence_chunk_adapter"] == "runtime_v1"
    assert plan["parsedArtifactEvidenceChunkAdapter"] == "runtime_v1"
    assert plan["resolvedPaperIds"] == ["2501.00001", "2501.00002"]


def test_build_paper_evidence_chunk_answer_preview_forces_paper_local_scope() -> None:
    searcher = _FakeSearcher()

    payload = build_paper_evidence_chunk_answer_preview(
        searcher,
        question="What method evidence exists?",
        paper_ids=["2501.00001"],
        retrieval_mode="semantic",
        allow_external=False,
    )

    assert payload["status"] == "ok"
    assert payload["sourceType"] == "paper"
    assert payload["allowExternal"] is False
    assert payload["queryPlan"]["parsed_artifact_evidence_chunk_adapter"] == "runtime_v1"
    assert payload["evidencePacketSummary"]["adapterRowsAdded"] == 2
    assert searcher.calls[-1]["source_type"] == "paper"
    assert searcher.calls[-1]["allow_external"] is False
    assert searcher.calls[-1]["query_plan"]["resolvedPaperIds"] == ["2501.00001"]
    assert validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True).ok


def test_build_paper_evidence_chunk_answer_preview_rejects_external() -> None:
    searcher = _FakeSearcher()

    try:
        build_paper_evidence_chunk_answer_preview(
            searcher,
            question="What method evidence exists?",
            paper_ids=["2501.00001"],
            allow_external=True,
        )
    except ValueError as error:
        assert "external model calls" in str(error)
    else:
        raise AssertionError("expected external-call request to be rejected")


def test_labs_cli_evidence_chunk_ask_outputs_schema_valid_json() -> None:
    searcher = _FakeSearcher()

    result = CliRunner().invoke(
        paper_labs_group,
        [
            "evidence-chunk-ask",
            "What method evidence exists?",
            "--paper-id",
            "2501.00001",
            "--json",
        ],
        obj={"khub": _FakeKhub(searcher)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID
    assert payload["mode"] == "labs_opt_in_preview"
    assert payload["paperIds"] == ["2501.00001"]
    assert payload["queryPlan"]["resolvedPaperIds"] == ["2501.00001"]
    assert payload["safety"]["labsOnly"] is True
    assert validate_payload(payload, PAPER_EVIDENCE_CHUNK_ANSWER_PREVIEW_SCHEMA_ID, strict=True).ok


def test_labs_cli_evidence_chunk_ask_rejects_external() -> None:
    result = CliRunner().invoke(
        paper_labs_group,
        [
            "evidence-chunk-ask",
            "What method evidence exists?",
            "--paper-id",
            "2501.00001",
            "--allow-external",
            "--json",
        ],
        obj={"khub": _FakeKhub(_FakeSearcher())},
    )

    assert result.exit_code != 0
    assert "--allow-external is not enabled" in result.output
