from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.ai.answer_contracts import build_answer_contract, build_evidence_packet_contract
from knowledge_hub.ai.rag import RAGSearcher
from tests.test_rag_search import DummyEmbedder, DummyVectorDB, FakeLLM


FIXTURES_PATH = Path("eval/knowledgeos/fixtures/retrieval_span_golden_cases.json")


def _record(*, record_id: str, source_type: str, title: str, document: str, distance: float = 0.2) -> dict:
    return {
        "id": record_id,
        "document": document,
        "distance": distance,
        "metadata": {
            "title": title,
            "source_type": source_type,
            "source_id": record_id,
            "source_content_hash": f"hash:{record_id}",
            "span_locator": f"chars:0-{min(240, len(document))}",
            "chunk_index": 0,
        },
    }


def _golden_records() -> list[dict]:
    return [
        _record(
            record_id="paper:1706.03762#0",
            source_type="paper",
            title="Attention Is All You Need",
            document="Transformer self attention enables sequence transduction with attention heads and encoder decoder layers.",
        ),
        _record(
            record_id="paper:2005.11401#0",
            source_type="paper",
            title="Retrieval-Augmented Generation",
            document="Retrieval augmented generation combines a generator with non parametric memory passages for grounded answers.",
        ),
        _record(
            record_id="paper:alexnet#0",
            source_type="paper",
            title="AlexNet",
            document="CNN convolutional neural networks improved ImageNet recognition with deep convolutional layers.",
        ),
        _record(
            record_id="paper:vit#0",
            source_type="paper",
            title="Vision Transformer",
            document="ViT uses image patches and a transformer encoder for vision classification.",
        ),
        _record(
            record_id="paper:dqn#0",
            source_type="paper",
            title="Playing Atari with Deep Reinforcement Learning",
            document="DQN combines Atari deep reinforcement learning with Q-learning and replay memory.",
        ),
        _record(
            record_id="paper:survey#0",
            source_type="paper",
            title="AI Benchmark Survey",
            document="Survey overview of general AI benchmark datasets and model comparisons.",
            distance=0.45,
        ),
        _record(
            record_id="web:version-grounding#0",
            source_type="web",
            title="Web Version Grounding",
            document="Web runtime version grounding requires explicit version, date, and latest update evidence.",
        ),
        _record(
            record_id="web:reference-source#0",
            source_type="web",
            title="Reference Source Policy",
            document="Reference explainer queries should prefer stable documentation, guide, and API reference sources.",
        ),
        _record(
            record_id="web:observed-at-guard#0",
            source_type="web",
            title="Observed At Guard",
            document="Observed_at alone is weak temporal evidence and cannot prove a latest claim without version markers.",
        ),
        _record(
            record_id="web:rerank-signal#0",
            source_type="web",
            title="Rerank Signal",
            document="Rerank cross encoder signal improves precision but is not canonical evidence.",
        ),
        _record(
            record_id="web:generic#0",
            source_type="web",
            title="Generic Web Note",
            document="Generic article about unrelated application examples.",
            distance=0.45,
        ),
        _record(
            record_id="vault:Project State.md#0",
            source_type="vault",
            title="Project State",
            document="The default loop is discover, index, search or ask, then evidence review.",
        ),
        _record(
            record_id="vault:Architecture.md#0",
            source_type="vault",
            title="Architecture",
            document="Ontology, learning graph, and memory relation rows are retrieval signals, not citations.",
        ),
        _record(
            record_id="vault:KL Divergence.md#0",
            source_type="vault",
            title="KL Divergence",
            document="KL divergence measures distribution difference and appears in variational inference evidence lower bound.",
        ),
        _record(
            record_id="vault:Learning Coach.md#0",
            source_type="vault",
            title="Learning Coach",
            document="The learning coach tracks progress, weak areas, quiz attempts, and checkpoint state.",
        ),
        _record(
            record_id="learning_edge:rag:prereq#0",
            source_type="learning_edge",
            title="RAG prerequisite edge",
            document="A prerequisite learning edge links retrieval concepts to generation concepts.",
        ),
    ]


def _searcher() -> RAGSearcher:
    return RAGSearcher(DummyEmbedder(), DummyVectorDB(_golden_records()), llm=FakeLLM())


def _result_source_id(result) -> str:
    return str(result.document_id or (result.metadata or {}).get("source_id") or "").strip()


def _evidence_from_result(result) -> dict:
    metadata = dict(result.metadata or {})
    return {
        "title": metadata.get("title", ""),
        "excerpt": result.document,
        "source_id": _result_source_id(result),
        "source_type": metadata.get("source_type", ""),
        "source_content_hash": metadata.get("source_content_hash", ""),
        "span_locator": metadata.get("span_locator", ""),
        "score": result.score,
        "lexical_score": result.lexical_score,
        "semantic_score": result.semantic_score,
    }


def _search_case(case: dict) -> list:
    return _searcher().search(
        case["query"],
        top_k=5,
        source_type=None,
        retrieval_mode="keyword",
        use_ontology_expansion=False,
        metadata_filter={"source_type": case["source_type"]},
    )


def test_retrieval_span_golden_cases_find_expected_source_and_terms():
    cases = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    assert 15 <= len(cases) <= 20

    for case in cases:
        results = _search_case(case)
        if case["must_abstain"] and not case["expected_source_id"]:
            joined = "\n".join(f"{result.document} {result.metadata.get('title', '')}" for result in results).casefold()
            direct_hits = [term for term in case["expected_text_terms"] if str(term).casefold() in joined]
            assert direct_hits == [], case["case_id"]
            continue
        source_ids = [_result_source_id(result) for result in results]
        assert case["expected_source_id"] in source_ids, case["case_id"]
        rank = source_ids.index(case["expected_source_id"]) + 1
        assert rank <= int(case["min_rank"]), case["case_id"]
        expected = next(result for result in results if _result_source_id(result) == case["expected_source_id"])
        text = f"{expected.document} {expected.metadata.get('title', '')}".casefold()
        missing_terms = [term for term in case["expected_text_terms"] if str(term).casefold() not in text]
        assert missing_terms == [], case["case_id"]


def test_retrieval_span_golden_non_evidence_result_stays_signal_only():
    case = next(
        item
        for item in json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
        if item["case_id"] == "non_evidence_learning_edge_signal"
    )
    result = _search_case(case)[0]
    evidence = [_evidence_from_result(result)]
    packet = SimpleNamespace(
        evidence=evidence,
        citations=[{"label": "S1", "target": _result_source_id(result), "kind": "source"}],
        evidence_packet={"answerable": True, "answerableDecisionReason": "retrieved signal"},
        evidence_policy={"policyKey": "test"},
    )
    pipeline_result = SimpleNamespace(plan=SimpleNamespace(to_dict=lambda: {"queryFrame": {"source_type": "learning_edge"}}))

    evidence_contract = build_evidence_packet_contract(
        query=case["query"],
        retrieval_mode="keyword",
        pipeline_result=pipeline_result,
        evidence_packet=packet,
    )
    answer_contract = build_answer_contract(
        answer="RAG uses a prerequisite learning edge.",
        evidence_packet=packet,
        verification={"status": "failed", "needsCaution": True, "unsupportedClaimCount": 0},
        rewrite={},
        routing_meta={"provider": "local", "model": "test"},
    )

    assert evidence_contract["spans"] == []
    assert evidence_contract["coverage"]["excluded_non_evidence"] == 1
    assert answer_contract["citations"] == []
    assert answer_contract["retrievalSignals"][0]["source_id"] == case["expected_source_id"]
