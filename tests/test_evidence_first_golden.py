from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_hub.ai.answer_contracts import build_answer_contract
from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.config import Config
from tests.test_rag_search import DummyEmbedder, DummyVectorDB, StaticLLM, _build_records


FIXTURES_PATH = Path("eval/knowledgeos/fixtures/evidence_first_golden_cases.json")


def _local_searcher() -> RAGSearcher:
    config = Config()
    config.set_nested("routing", "llm", "tasks", "local", "provider", "ollama")
    config.set_nested("routing", "llm", "tasks", "local", "model", "qwen3:14b")
    config.set_nested("routing", "llm", "tasks", "local", "timeout_sec", 45)
    return RAGSearcher(DummyEmbedder(), DummyVectorDB(_build_records()), llm=StaticLLM("unused"), config=config)


@pytest.mark.parametrize(
    "case",
    json.loads(FIXTURES_PATH.read_text(encoding="utf-8")),
    ids=lambda case: str(case["case_id"]),
)
def test_evidence_first_golden_cases(case):
    if case["mode"] == "verification":
        searcher = _local_searcher()
        verification = searcher._verify_answer(
            query=case["query"],
            answer=case["answer"],
            evidence=list(case.get("evidence") or []),
            answer_signals={"contradictory_source_count": 0},
            contradicting_beliefs=list(case.get("contradicting_beliefs") or []),
            allow_external=False,
        )
        for key, expected_value in dict(case["expected"]).items():
            assert verification.get(key) == expected_value
        return

    if case["mode"] == "answer_contract":
        evidence_packet = SimpleNamespace(
            evidence=list(case.get("evidence") or []),
            citations=list(case.get("citations") or []),
            evidence_packet={},
        )
        contract = build_answer_contract(
            answer=case["answer"],
            evidence_packet=evidence_packet,
            verification=dict(case.get("verification") or {}),
            rewrite={},
            routing_meta={},
        )
        assert len(contract["citations"]) == int(case["expected"]["citationCount"])
        assert len(contract["retrievalSignals"]) == int(case["expected"]["retrievalSignalCount"])
        return

    raise AssertionError(f"unsupported golden mode: {case['mode']}")
