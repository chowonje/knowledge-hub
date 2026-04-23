from __future__ import annotations

import json

from click.testing import CliRunner

from knowledge_hub.ai import evidence_assembly as evidence_module
from knowledge_hub.core.config import Config
from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.interfaces.cli.commands.paper_labs_cmd import paper_labs_group
from knowledge_hub.papers.topic_synthesis import (
    PaperTopicSynthesisService,
    _dynamic_selected_limit_v2,
    _infer_topic_profile_v2,
    _prune_candidates_for_profile_v2,
    _score_candidate_for_profile_v2,
)


class _StubSQLite:
    def __init__(self):
        self._papers = {
            "2401.00001": {
                "arxiv_id": "2401.00001",
                "title": "Mamba",
                "year": 2024,
                "field": "AI",
                "primary_lane": "architecture",
            },
            "2307.00002": {
                "arxiv_id": "2307.00002",
                "title": "RetNet",
                "year": 2023,
                "field": "AI",
                "primary_lane": "architecture",
            },
            "2106.01345": {
                "arxiv_id": "2106.01345",
                "title": "Decision Transformer",
                "year": 2021,
                "field": "AI",
                "primary_lane": "architecture",
            },
        }

    def get_paper(self, paper_id):
        return self._papers.get(paper_id)

    def search_papers(self, query, limit=5):  # noqa: ARG002
        return list(self._papers.values())[:limit]


class _StubPaperMemoryRetriever:
    def __init__(self, sqlite_db):  # noqa: ARG002
        pass

    def search(self, query, limit=10, include_refs=True):  # noqa: ARG002
        return [
            {
                "paperId": "2401.00001",
                "title": "Mamba",
                "paperCore": "State space model family proposed as an alternative to Transformer attention.",
                "methodCore": "Selective state space sequence modeling.",
                "evidenceCore": "Improves long-context efficiency with linear-time behavior.",
                "limitations": "Needs careful hardware-aware kernels.",
                "conceptLinks": ["state space model"],
                "retrievalSignals": {"score": 3.4},
            },
            {
                "paperId": "2307.00002",
                "title": "RetNet",
                "paperCore": "Retention-based sequence model presented as a scaling alternative to Transformers.",
                "methodCore": "Retention mechanism replaces standard self-attention.",
                "evidenceCore": "Targets favorable scaling at inference.",
                "limitations": "Not every retention paper fully replaces Transformer stacks in practice.",
                "conceptLinks": ["retention"],
                "retrievalSignals": {"score": 3.1},
            },
            {
                "paperId": "2106.01345",
                "title": "Decision Transformer",
                "paperCore": "Transformer-based reinforcement learning sequence model.",
                "methodCore": "Autoregressive transformer conditioned on return-to-go.",
                "evidenceCore": "Shows strong offline RL results.",
                "limitations": "Still uses a transformer stack rather than replacing it.",
                "conceptLinks": ["transformer", "reinforcement learning"],
                "retrievalSignals": {"score": 4.2},
            },
        ][:limit]


class _StubDocumentMemoryRetriever:
    def __init__(self, sqlite_db):  # noqa: ARG002
        pass

    def search(self, query, limit=10):  # noqa: ARG002
        return [
            {
                "documentId": "paper:2401.00001",
                "documentTitle": "Mamba",
                "sourceType": "paper",
                "matchedUnit": {
                    "title": "Abstract",
                    "sourceRef": "2401.00001",
                    "sourceExcerpt": "Mamba replaces attention with selective state spaces for sequence modeling.",
                },
                "retrievalSignals": {"score": 2.2},
            },
            {
                "documentId": "paper:2307.00002",
                "documentTitle": "RetNet",
                "sourceType": "paper",
                "matchedUnit": {
                    "title": "Abstract",
                    "sourceRef": "2307.00002",
                    "sourceExcerpt": "RetNet proposes retention as a scalable sequence architecture beyond vanilla attention.",
                },
                "retrievalSignals": {"score": 2.0},
            },
            {
                "documentId": "paper:2106.01345",
                "documentTitle": "Decision Transformer",
                "sourceType": "paper",
                "matchedUnit": {
                    "title": "Abstract",
                    "sourceRef": "2106.01345",
                    "sourceExcerpt": "Decision Transformer models reinforcement learning trajectories with a transformer architecture.",
                },
                "retrievalSignals": {"score": 3.0},
            },
        ][:limit]


class _StubSearcher:
    def search_with_diagnostics(self, query, **kwargs):  # noqa: ARG002
        results = [
            SearchResult(
                document="Mamba is framed as a next-generation sequence architecture beyond Transformers.",
                metadata={"title": "Mamba", "source_type": "paper", "arxiv_id": "2401.00001"},
                distance=0.1,
                score=0.83,
                semantic_score=0.83,
                lexical_score=0.21,
                retrieval_mode="hybrid",
                document_id="paper:2401.00001",
            ),
            SearchResult(
                document="RetNet studies retention as a scalable replacement path for attention-heavy stacks.",
                metadata={"title": "RetNet", "source_type": "paper", "arxiv_id": "2307.00002"},
                distance=0.11,
                score=0.8,
                semantic_score=0.8,
                lexical_score=0.2,
                retrieval_mode="hybrid",
                document_id="paper:2307.00002",
            ),
            SearchResult(
                document="Decision Transformer applies transformer sequence modeling to reinforcement learning.",
                metadata={"title": "Decision Transformer", "source_type": "paper", "arxiv_id": "2106.01345"},
                distance=0.08,
                score=0.92,
                semantic_score=0.92,
                lexical_score=0.24,
                retrieval_mode="hybrid",
                document_id="paper:2106.01345",
            ),
        ]
        return {"results": results, "diagnostics": {"queryIntent": "paper_topic"}}

    def _verify_answer(self, **kwargs):  # noqa: ANN003
        return {
            "status": "verified",
            "summary": "stub verification",
            "warnings": [],
            "claims": [],
        }


class _StubLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def generate(self, prompt, context=""):  # noqa: ANN001, ARG002
        return self._responses.pop(0)


def _topic_candidate(
    paper_id: str,
    title: str,
    *,
    paper_core: str,
    method_core: str = "",
    evidence_core: str = "",
    limitations: str = "",
    score: float = 1.5,
) -> dict[str, object]:
    return {
        "paperId": paper_id,
        "title": title,
        "year": 2025,
        "field": "AI",
        "paperCore": paper_core,
        "methodCore": method_core,
        "evidenceCore": evidence_core,
        "limitations": limitations,
        "conceptLinks": [],
        "documentSnippets": [],
        "matchedVia": ["paper_memory"],
        "retrievalSignals": {
            "candidateScore": score,
            "paperMemoryScore": score,
            "documentMemoryScore": 0.0,
            "paperRetrievalScore": 0.0,
        },
    }


def test_paper_topic_synthesis_local_fallback_returns_multi_paper_payload():
    service = PaperTopicSynthesisService(
        searcher=_StubSearcher(),
        sqlite_db=_StubSQLite(),
        config=Config(),
        llm_resolver=lambda **kwargs: (None, {}, ["llm unavailable for test"]),
        paper_memory_retriever_cls=_StubPaperMemoryRetriever,
        document_memory_retriever_cls=_StubDocumentMemoryRetriever,
    )

    payload = service.synthesize(
        query="트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘",
        source_mode="hybrid",
        top_k=2,
        candidate_limit=6,
        selected_limit=2,
    )

    assert payload["status"] == "ok"
    assert payload["effectiveSourceMode"] == "local"
    assert payload["enrichment"]["used"] is False
    assert payload["enrichment"]["mode"] == "none"
    assert len(payload["candidatePapers"]) >= 2
    assert len(payload["selectedPapers"]) == 2
    assert {item["paperId"] for item in payload["selectedPapers"]} == {"2401.00001", "2307.00002"}
    assert validate_payload(payload, payload["schema"], strict=True).ok


def test_paper_topic_synthesis_llm_flow_preserves_selection_rationales():
    judge_payload = json.dumps(
        {
            "selectedPapers": [
                {"paperId": "2401.00001", "title": "Mamba", "decision": "keep", "rationale": "Direct architecture alternative.", "groupLabel": "state_space"},
                {"paperId": "2307.00002", "title": "RetNet", "decision": "keep", "rationale": "Retention is presented as a replacement path.", "groupLabel": "retention"},
            ],
            "excludedPapers": [],
        },
        ensure_ascii=False,
    )
    synthesis_payload = json.dumps(
        {
            "topicSummary": "Mamba and RetNet are the strongest local alternatives in this corpus.",
            "architectureGroups": [{"label": "state_space", "paperIds": ["2401.00001"], "summary": "State space path."}],
            "comparisonPoints": [{"paperId": "2401.00001", "title": "Mamba", "point": "Selective state spaces avoid standard attention."}],
            "limitations": ["Both are still narrower than the full transformer ecosystem."],
            "gaps": ["This local corpus may miss newer hybrid sequence models."],
        },
        ensure_ascii=False,
    )
    llm = _StubLLM([judge_payload, synthesis_payload])
    service = PaperTopicSynthesisService(
        searcher=_StubSearcher(),
        sqlite_db=_StubSQLite(),
        config=Config(),
        llm_resolver=lambda **kwargs: (llm, {"route": "local", "provider": "ollama", "model": "exaone3.5:7.8b"}, []),
        paper_memory_retriever_cls=_StubPaperMemoryRetriever,
        document_memory_retriever_cls=_StubDocumentMemoryRetriever,
    )

    payload = service.synthesize(
        query="트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘",
        top_k=2,
        candidate_limit=6,
    )

    assert payload["selectedPapers"][0]["rationale"]
    assert payload["topicSummary"].startswith("Mamba and RetNet")
    assert payload["verification"]["status"] == "verified"


def test_topic_profile_penalizes_transformer_optimizations_for_alternative_queries():
    service = PaperTopicSynthesisService(
        searcher=_StubSearcher(),
        sqlite_db=_StubSQLite(),
        config=Config(),
        llm_resolver=lambda **kwargs: (None, {}, ["llm unavailable for test"]),
        paper_memory_retriever_cls=_StubPaperMemoryRetriever,
        document_memory_retriever_cls=_StubDocumentMemoryRetriever,
    )

    payload = service.synthesize(
        query="트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘",
        top_k=3,
        candidate_limit=6,
        selected_limit=2,
    )

    selected_ids = {item["paperId"] for item in payload["selectedPapers"]}
    assert "2106.01345" not in selected_ids
    assert {"2401.00001", "2307.00002"} <= {item["paperId"] for item in payload["candidatePapers"]}


def test_bad5_profiles_map_to_stricter_intents():
    assert _infer_topic_profile_v2("state space model 계열 논문들 중에서 transformer 대안으로 읽을 만한 것들을 골라 비교해줘")["name"] == "state_space_models_as_transformer_alternatives"
    assert _infer_topic_profile_v2("audio-visual 또는 multimodal understanding에서 transformer 변형보다 다른 reasoning 구조를 강조하는 논문들을 찾아 정리해줘")["name"] == "multimodal_non_transformer_reasoning"
    assert _infer_topic_profile_v2("장기 기억이 핵심인 agent application 논문들과 장기 기억을 평가하는 benchmark 논문들을 비교 정리해줘")["name"] == "long_term_memory_application_vs_benchmark"
    assert _infer_topic_profile_v2("world model이나 alternative sequential modeling과 연결될 수 있는 논문들을 찾아서 transformer 대체 맥락에서 읽을 가치가 있는지 정리해줘")["name"] == "world_model_as_transformer_alternative"
    assert _infer_topic_profile_v2("논문을 AI agent로 재구성하거나 paper-to-agent를 다루는 논문들을 찾아서 system paper와 conceptual framing을 구분해줘")["name"] == "paper_to_agent_system_vs_framing"


def test_state_space_prune_removes_transformer_optimizations():
    profile = _infer_topic_profile_v2("state space model 계열 논문들 중에서 transformer 대안으로 읽을 만한 것들을 골라 비교해줘")
    candidates = [
        _topic_candidate("m1", "Mamba", paper_core="State space model architecture proposed as a transformer alternative.", method_core="Selective state space sequence modeling.", score=1.8),
        _topic_candidate("r1", "RetNet", paper_core="Retention-based architecture beyond attention.", method_core="Retentive sequence modeling.", score=1.7),
        _topic_candidate("s1", "Switch Transformers", paper_core="Sparse transformer architecture with routing.", method_core="Mixture of experts transformer.", score=2.8),
        _topic_candidate("c1", "Chain of Thought Prompting", paper_core="Reasoning prompting strategy for transformers.", method_core="Chain of thought prompting.", score=2.2),
        _topic_candidate("d1", "Direct Preference Optimization", paper_core="Preference optimization for language models.", method_core="DPO fine-tuning.", score=2.0),
    ]
    for candidate in candidates:
        candidate["topicSignals"] = _score_candidate_for_profile_v2(candidate, profile)

    kept, pruned, diag = _prune_candidates_for_profile_v2(candidates, profile)

    assert {item["paperId"] for item in kept} == {"m1", "r1"}
    assert {"s1", "c1", "d1"} <= {item["paperId"] for item in pruned}
    assert diag["prunedCount"] == 3
    assert _dynamic_selected_limit_v2(kept, 6, profile) == 2


def test_bad5_prune_rules_filter_security_and_generic_agent_noise():
    profile = _infer_topic_profile_v2("장기 기억이 핵심인 agent application 논문들과 장기 기억을 평가하는 benchmark 논문들을 비교 정리해줘")
    candidates = [
        _topic_candidate("a1", "AMA-Bench", paper_core="Long-horizon memory benchmark for agentic applications.", method_core="Benchmarking long-horizon agent memory.", score=1.9),
        _topic_candidate("a2", "Zep", paper_core="Temporal knowledge graph architecture for agent memory applications.", method_core="Graph memory system for long-term agent recall.", score=2.1),
        _topic_candidate("a3", "Prompt Injection Benchmark", paper_core="Security benchmark for prompt injection attacks on agents.", method_core="Prompt injection evaluation.", score=2.3),
        _topic_candidate("a4", "General Agent Survey", paper_core="Survey of LLM-powered agent systems and applications in industry.", method_core="Survey overview.", score=1.9),
    ]
    for candidate in candidates:
        candidate["topicSignals"] = _score_candidate_for_profile_v2(candidate, profile)

    kept, pruned, _ = _prune_candidates_for_profile_v2(candidates, profile)

    assert {item["paperId"] for item in kept} == {"a1", "a2"}
    assert {"a3", "a4"} <= {item["paperId"] for item in pruned}


def test_paper_to_agent_prune_keeps_direct_framing_and_drops_generic_agent_papers():
    profile = _infer_topic_profile_v2("논문을 AI agent로 재구성하거나 paper-to-agent를 다루는 논문들을 찾아서 system paper와 conceptual framing을 구분해줘")
    candidates = [
        _topic_candidate("p1", "Paper2Agent", paper_core="Reimagining research papers as interactive and reliable AI agents.", method_core="Systemizes papers into interactive agents.", score=1.8),
        _topic_candidate("p2", "Paper-to-Agent Framing Survey", paper_core="Conceptual framing for paper-to-agent systems and interactive papers.", method_core="Survey and framing.", score=1.7),
        _topic_candidate("p3", "MAGMA", paper_core="Multi-graph based agentic memory architecture for AI agents.", method_core="Agent memory architecture.", score=2.4),
        _topic_candidate("p4", "General Agent Survey", paper_core="Survey of general AI agents and their applications.", method_core="Survey overview.", score=2.1),
    ]
    for candidate in candidates:
        candidate["topicSignals"] = _score_candidate_for_profile_v2(candidate, profile)

    kept, pruned, _ = _prune_candidates_for_profile_v2(candidates, profile)

    assert {item["paperId"] for item in kept} == {"p1", "p2"}
    assert {"p3", "p4"} <= {item["paperId"] for item in pruned}


def test_strict_profile_returns_fewer_selected_papers_instead_of_padding():
    service = PaperTopicSynthesisService(
        searcher=_StubSearcher(),
        sqlite_db=_StubSQLite(),
        config=Config(),
        llm_resolver=lambda **kwargs: (None, {}, ["llm unavailable for test"]),
        paper_memory_retriever_cls=_StubPaperMemoryRetriever,
        document_memory_retriever_cls=_StubDocumentMemoryRetriever,
    )

    payload = service.synthesize(
        query="state space model 계열 논문들 중에서 transformer 대안으로 읽을 만한 것들을 골라 비교해줘",
        top_k=3,
        candidate_limit=6,
        selected_limit=6,
    )

    assert len(payload["selectedPapers"]) == 2
    assert {item["paperId"] for item in payload["selectedPapers"]} == {"2401.00001", "2307.00002"}


def test_paper_topic_queries_do_not_trigger_single_paper_scope_narrowing():
    result = SearchResult(
        document="Mamba is a candidate alternative.",
        metadata={"title": "Mamba", "source_type": "paper", "arxiv_id": "2401.00001"},
        distance=0.1,
        score=0.9,
        semantic_score=0.9,
        lexical_score=0.2,
        retrieval_mode="hybrid",
        document_id="paper:2401.00001",
    )

    narrowed, scope = evidence_module._derive_paper_answer_scope(
        query="트랜스포머를 대체할 차세대 아키텍처 논문들을 찾아서 정리해줘",
        source_type="paper",
        filtered=[result],
        paper_memory_prefilter={"matchedPaperIds": ["2401.00001"]},
        metadata_filter=None,
    )

    assert narrowed == [result]
    assert scope["applied"] is False
    assert scope["queryIntent"] == "paper_topic"


def test_paper_topic_synthesize_cli_emits_json(monkeypatch):
    payload = {
        "schema": "knowledge-hub.paper-topic-synthesis.result.v1",
        "status": "ok",
        "query": "test",
        "sourceMode": "local",
        "effectiveSourceMode": "local",
        "enrichment": {"eligible": False, "used": False, "mode": "none", "reason": "", "queryIntent": "paper_topic"},
        "candidatePapers": [{"paperId": "2401.00001", "title": "Mamba"}],
        "selectedPapers": [{"paperId": "2401.00001", "title": "Mamba", "decision": "keep", "rationale": "fit", "groupLabel": "architecture"}],
        "excludedPapers": [],
        "selectionDiagnostics": {},
        "topicSummary": "summary",
        "architectureGroups": [],
        "comparisonPoints": [],
        "limitations": [],
        "gaps": [],
        "citations": [],
        "verification": {"status": "verified", "summary": "ok", "warnings": [], "claims": []},
        "warnings": [],
    }

    class _StubKhub:
        def __init__(self):
            self.config = Config()

        def searcher(self):
            return object()

        def sqlite_db(self):
            return object()

    monkeypatch.setattr(
        "knowledge_hub.interfaces.cli.commands.paper_labs_cmd.PaperTopicSynthesisService",
        type(
            "_StubService",
            (),
            {
                "__init__": lambda self, **kwargs: None,  # noqa: ARG005
                "synthesize": lambda self, **kwargs: payload,  # noqa: ARG005
            },
        ),
    )

    result = CliRunner().invoke(
        paper_labs_group,
        ["topic-synthesize", "transformer alternatives papers", "--json"],
        obj={"khub": _StubKhub()},
    )

    assert result.exit_code == 0
    loaded = json.loads(result.output)
    assert loaded["selectedPapers"][0]["paperId"] == "2401.00001"
