from __future__ import annotations

from types import SimpleNamespace

from knowledge_hub.ai.answer_orchestrator import AnswerOrchestrator
from knowledge_hub.ai.answer_payload_builder import AnswerPayloadBuilder


class _SearcherStub:
    def _resolve_query_entities(self, query):  # noqa: ANN001
        _ = query
        return []


def _pipeline_result(
    *,
    family: str,
    source_type: str,
    runtime_used: str,
    v2: dict | None = None,
    memory_route: dict | None = None,
    memory_prefilter: dict | None = None,
    paper_memory_prefilter: dict | None = None,
):
    return SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": family,
                "queryPlan": {"family": family, "answerMode": "representative_paper_explainer"},
                "queryFrame": {"family": family, "source_type": source_type},
                "resolvedSourceScopeApplied": False,
                "canonicalEntitiesApplied": [],
                "metadataFilterApplied": {},
                "prefilterReason": "none",
            }
        ),
        related_clusters=[],
        active_profile=None,
        paper_memory_prefilter=dict(paper_memory_prefilter or {}),
        memory_route=dict(memory_route or {}),
        memory_prefilter=dict(memory_prefilter or {}),
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=False,
        mixed_fallback_used=False,
        v2_diagnostics={
            "runtimeExecution": {"used": runtime_used},
            **dict(v2 or {}),
        },
    )


def _evidence_packet(*, representative_title: str, representative_paper_id: str, evidence: list[dict]):
    return SimpleNamespace(
        answer_signals={
            "representative_paper": {
                "paperId": representative_paper_id,
                "title": representative_title,
            }
        },
        evidence_policy={"policyKey": "concept_explainer_policy"},
        paper_answer_scope={},
        evidence=evidence,
        citations=[],
        evidence_budget={},
        evidence_packet={},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )


def test_answer_payload_builder_reports_retrieval_objects_for_concept_explainer():
    builder = AnswerPayloadBuilder(_SearcherStub())
    payload = builder.base_payload(
        query="CNN을 쉽게 설명해줘",
        retrieval_mode="hybrid",
        pipeline_result=_pipeline_result(family="concept_explainer", source_type="paper", runtime_used="legacy"),
        evidence_packet=_evidence_packet(
            representative_title="ImageNet Classification with Deep Convolutional Neural Networks",
            representative_paper_id="alexnet-2012",
            evidence=[{"title": "ImageNet Classification with Deep Convolutional Neural Networks", "source_type": "paper"}],
        ),
        answer="answer",
    )

    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary", "SectionCard"]
    assert payload["retrievalObjectsUsed"] == ["RawEvidenceUnit"]
    assert payload["representativeRole"] == "anchor"


def test_answer_payload_builder_preserves_memory_contract_diagnostics():
    builder = AnswerPayloadBuilder(_SearcherStub())
    payload = builder.base_payload(
        query="RAG란?",
        retrieval_mode="hybrid",
        pipeline_result=_pipeline_result(
            family="general",
            source_type="paper",
            runtime_used="legacy",
            memory_route={
                "contractRole": "ask_retrieval_memory_prefilter",
                "requestedMode": "prefilter",
                "effectiveMode": "compat",
                "modeAliasApplied": True,
            },
            memory_prefilter={"contractRole": "retrieval_memory_prefilter", "applied": True},
            paper_memory_prefilter={"contractRole": "paper_source_memory_prefilter", "applied": True},
        ),
        evidence_packet=_evidence_packet(
            representative_title="",
            representative_paper_id="",
            evidence=[{"title": "RAG Note", "source_type": "paper"}],
        ),
        answer="answer",
    )

    assert payload["memoryRoute"]["contractRole"] == "ask_retrieval_memory_prefilter"
    assert payload["memoryRoute"]["effectiveMode"] == "compat"
    assert payload["memoryPrefilter"]["contractRole"] == "retrieval_memory_prefilter"
    assert payload["paperMemoryPrefilter"]["contractRole"] == "paper_source_memory_prefilter"


def test_apply_claim_consensus_to_verification_keeps_advisory_claims_out_of_answer_gate():
    verification = {
        "status": "verified",
        "supportedClaimCount": 1,
        "unsupportedClaimCount": 0,
        "uncertainClaimCount": 0,
        "needsCaution": False,
        "warnings": [],
    }
    consensus = {
        "supportCount": 0,
        "conflictCount": 0,
        "weakClaimCount": 1,
        "unsupportedClaimCount": 2,
        "claimVerificationSummary": "weak",
        "conflicts": [],
    }

    payload = AnswerPayloadBuilder.apply_claim_consensus_to_verification(
        verification,
        consensus,
        merge_mode="advisory",
    )

    assert payload["claimConsensusMode"] == "advisory"
    assert payload["claimWeakCount"] == 1
    assert payload["claimUnsupportedCount"] == 2
    assert payload["unsupportedClaimCount"] == 0
    assert payload["needsCaution"] is False
    assert payload["status"] == "verified"


def test_claim_consensus_merge_mode_is_strict_only_for_compare_family():
    compare_result = SimpleNamespace(
        v2_diagnostics={"answerProvenance": {"mode": "claim_cards_verified"}},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "paper_compare",
                "queryFrame": {"family": "paper_compare"},
            }
        )
    )
    lookup_result = SimpleNamespace(
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "paper_lookup",
                "queryFrame": {"family": "paper_lookup"},
            }
        )
    )

    assert AnswerOrchestrator._claim_consensus_merge_mode(
        claim_native_used=True,
        pipeline_result=compare_result,
    ) == "strict"
    assert AnswerOrchestrator._claim_consensus_merge_mode(
        claim_native_used=True,
        pipeline_result=lookup_result,
    ) == "advisory"


def test_claim_consensus_merge_mode_is_advisory_for_compare_weak_claim_fallback():
    compare_result = SimpleNamespace(
        v2_diagnostics={"answerProvenance": {"mode": "weak_claim_fallback"}},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "paper_compare",
                "queryFrame": {"family": "paper_compare"},
            }
        )
    )

    assert AnswerOrchestrator._claim_consensus_merge_mode(
        claim_native_used=True,
        pipeline_result=compare_result,
    ) == "advisory"


def test_answer_payload_builder_marks_survey_representative_role_for_paper_discover():
    builder = AnswerPayloadBuilder(_SearcherStub())
    payload = builder.base_payload(
        query="RAG 관련 논문 찾아줘",
        retrieval_mode="hybrid",
        pipeline_result=_pipeline_result(family="paper_discover", source_type="paper", runtime_used="legacy"),
        evidence_packet=_evidence_packet(
            representative_title="Retrieval-Augmented Generation Survey and Overview",
            representative_paper_id="2312.10997",
            evidence=[{"title": "Retrieval-Augmented Generation Survey and Overview", "source_type": "paper", "unit_type": "document_summary"}],
        ),
        answer="answer",
    )

    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary", "SectionCard"]
    assert payload["retrievalObjectsUsed"] == ["DocSummary", "RawEvidenceUnit"]
    assert payload["representativeRole"] == "survey"


def test_answer_payload_builder_falls_back_to_resolved_lookup_representative():
    builder = AnswerPayloadBuilder(_SearcherStub())
    pipeline_result = SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "paper_lookup",
                "queryPlan": {
                    "family": "paper_lookup",
                    "answerMode": "paper_scoped_answer",
                    "resolvedPaperIds": ["2010.11929"],
                },
                "queryFrame": {
                    "family": "paper_lookup",
                    "source_type": "paper",
                    "resolved_source_ids": ["2010.11929"],
                },
                "resolvedSourceScopeApplied": False,
                "canonicalEntitiesApplied": [],
                "metadataFilterApplied": {},
                "prefilterReason": "none",
            }
        ),
        related_clusters=[],
        active_profile=None,
        paper_memory_prefilter={},
        memory_route={},
        memory_prefilter={},
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=False,
        mixed_fallback_used=False,
        v2_diagnostics={"runtimeExecution": {"used": "ask_v2"}},
    )
    evidence_packet = SimpleNamespace(
        answer_signals={},
        evidence_policy={"policyKey": "paper_lookup_policy"},
        paper_answer_scope={"applied": False, "matchedPaperIds": ["2010.11929"]},
        evidence=[
            {
                "title": "An Image is Worth 16x16 Words",
                "arxiv_id": "2010.11929",
                "citation_label": "S1",
                "source_type": "paper",
            }
        ],
        citations=[],
        evidence_budget={},
        evidence_packet={"uniquePaperCount": 1},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )

    payload = builder.base_payload(
        query="An Image is Worth 16x16 Words 논문 요약해줘",
        retrieval_mode="hybrid",
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
        answer="answer",
    )

    assert payload["representativePaper"]["paperId"] == "2010.11929"
    assert payload["representativePaper"]["title"] == "An Image is Worth 16x16 Words"
    assert payload["representativeRole"] == "anchor"


def test_answer_payload_builder_uses_title_like_expanded_term_when_lookup_evidence_misses():
    builder = AnswerPayloadBuilder(_SearcherStub())
    pipeline_result = SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "paper_lookup",
                "queryPlan": {"family": "paper_lookup", "answerMode": "paper_scoped_answer"},
                "queryFrame": {
                    "family": "paper_lookup",
                    "source_type": "paper",
                    "resolved_source_ids": ["1312.5602"],
                    "expanded_terms": ["Playing Atari with Deep Reinforcement Learning", "Playing", "Atari"],
                },
                "resolvedSourceScopeApplied": False,
                "canonicalEntitiesApplied": [],
                "metadataFilterApplied": {},
                "prefilterReason": "none",
            }
        ),
        related_clusters=[],
        active_profile=None,
        paper_memory_prefilter={},
        memory_route={},
        memory_prefilter={},
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=False,
        mixed_fallback_used=False,
        v2_diagnostics={"runtimeExecution": {"used": "ask_v2"}},
    )
    evidence_packet = SimpleNamespace(
        answer_signals={},
        evidence_policy={"policyKey": "paper_lookup_policy"},
        paper_answer_scope={"applied": False, "matchedPaperIds": []},
        evidence=[
            {
                "title": "Asynchronous Methods for Deep Reinforcement Learning",
                "arxiv_id": "1602.01783",
                "citation_label": "S2",
                "source_type": "paper",
            }
        ],
        citations=[],
        evidence_budget={},
        evidence_packet={"uniquePaperCount": 1},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )

    payload = builder.base_payload(
        query="Playing Atari with Deep Reinforcement Learning 논문을 설명해줘",
        retrieval_mode="hybrid",
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
        answer="answer",
    )

    assert payload["representativePaper"]["paperId"] == "1312.5602"
    assert payload["representativePaper"]["title"] == "Playing Atari with Deep Reinforcement Learning"


def test_answer_payload_builder_reports_web_route_diagnostics_and_objects():
    builder = AnswerPayloadBuilder(_SearcherStub())
    pipeline_result = SimpleNamespace(
        context_expansion={},
        plan=SimpleNamespace(
            to_dict=lambda: {
                "paperFamily": "reference_explainer",
                "queryPlan": {"family": "reference_explainer", "answerMode": "concise_summary"},
                "queryFrame": {"family": "reference_explainer", "source_type": "web"},
                "resolvedSourceScopeApplied": True,
                "canonicalEntitiesApplied": ["concept_rerank"],
                "metadataFilterApplied": {"source_type": "web", "reference_only": True},
                "prefilterReason": "reference_source_bias",
                "referenceSourceApplied": True,
                "watchlistScopeApplied": False,
                "temporalRouteApplied": False,
                "temporalSignals": {"enabled": False},
            }
        ),
        related_clusters=[],
        active_profile=None,
        paper_memory_prefilter={},
        memory_route={},
        memory_prefilter={},
        candidate_sources=[],
        rerank_signals={},
        source_scope_enforced=True,
        mixed_fallback_used=False,
        v2_diagnostics={
            "runtimeExecution": {"used": "ask_v2"},
            "sectionCards": [{"section_card_id": "web-sec-1"}],
        },
    )
    evidence_packet = SimpleNamespace(
        answer_signals={},
        evidence_policy={"policyKey": "web_reference_explainer_policy", "family": "reference_explainer"},
        paper_answer_scope={},
        evidence=[{"title": "Vector Search Rerank Guide", "source_type": "web"}],
        citations=[],
        evidence_budget={},
        evidence_packet={},
        supporting_beliefs=[],
        contradicting_beliefs=[],
        belief_updates_suggested=[],
        claims=[],
    )

    payload = builder.base_payload(
        query="web card v2에서 version grounding이 필요한 이유는 무엇인가?",
        retrieval_mode="hybrid",
        pipeline_result=pipeline_result,
        evidence_packet=evidence_packet,
        answer="answer",
    )

    assert payload["paperFamily"] == "general"
    assert payload["queryFrame"]["family"] == "reference_explainer"
    assert payload["evidencePolicy"]["policyKey"] == "web_reference_explainer_policy"
    assert payload["retrievalObjectsAvailable"] == ["RawEvidenceUnit", "DocSummary", "SectionCard"]
    assert payload["retrievalObjectsUsed"] == ["RawEvidenceUnit", "SectionCard"]
    assert payload["familyRouteDiagnostics"]["referenceSourceApplied"] is True
    assert payload["familyRouteDiagnostics"]["watchlistScopeApplied"] is False
    assert payload["familyRouteDiagnostics"]["resolvedSourceScopeApplied"] is True
