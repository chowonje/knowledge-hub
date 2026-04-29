from __future__ import annotations

from knowledge_hub.ai.paper_query_plan import (
    PAPER_FAMILY_COMPARE,
    PAPER_FAMILY_CONCEPT_EXPLAINER,
    PAPER_FAMILY_DISCOVER,
    PAPER_FAMILY_LOOKUP,
    build_rule_based_query_frame,
    build_rule_query_plan,
    classify_paper_family,
)


class DummySQLite:
    def list_ontology_entities(self, limit=5000):
        _ = limit
        return [
            {
                "entity_id": "deep_convolutional_neural_networks",
                "entity_type": "concept",
                "canonical_name": "Deep Convolutional Neural Networks",
            }
        ]

    def get_entity_aliases(self, entity_id):
        if str(entity_id or "").strip() == "deep_convolutional_neural_networks":
            return ["CNN", "Convolutional Neural Networks"]
        return []

    def get_concept_papers(self, concept_id):
        if str(concept_id or "").strip() == "deep_convolutional_neural_networks":
            return [
                {
                    "arxiv_id": "alexnet-2012",
                    "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                }
            ]
        return []


class CompareSQLite(DummySQLite):
    def list_ontology_entities(self, limit=5000):
        _ = limit
        return [
            {
                "entity_id": "deep_convolutional_neural_networks",
                "entity_type": "concept",
                "canonical_name": "Deep Convolutional Neural Networks",
            },
            {
                "entity_id": "vision_transformers",
                "entity_type": "concept",
                "canonical_name": "Vision Transformers",
            },
        ]

    def get_entity_aliases(self, entity_id):
        if str(entity_id or "").strip() == "deep_convolutional_neural_networks":
            return ["CNN", "Convolutional Neural Networks"]
        if str(entity_id or "").strip() == "vision_transformers":
            return ["ViT", "Vision Transformer"]
        return []

    def get_concept_papers(self, concept_id):
        if str(concept_id or "").strip() == "deep_convolutional_neural_networks":
            return [
                {
                    "arxiv_id": "alexnet-2012",
                    "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                }
            ]
        if str(concept_id or "").strip() == "vision_transformers":
            return [
                {
                    "arxiv_id": "2010.11929",
                    "title": "An Image is Worth 16x16 Words",
                }
            ]
        return []


class LookupSQLite:
    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        if str(query or "").strip() == "AlexNet":
            return [
                {
                    "paper_id": "alexnet-2012",
                    "title": "ImageNet Classification with Deep Convolutional Neural Networks",
                    "search_text": "AlexNet ImageNet Classification with Deep Convolutional Neural Networks",
                }
            ]
        if str(query or "").strip() == "BERT":
            return [
                {
                    "paper_id": "1907.11692",
                    "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                    "search_text": "RoBERTa robustly optimized bert pretraining approach",
                },
                {
                    "paper_id": "1810.04805",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "search_text": "BERT pre-training of deep bidirectional transformers for language understanding",
                },
            ]
        return []


class LocalTitleLookupSQLite:
    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        token = str(query or "").strip()
        if token == "A Neural Image Caption Generator":
            return [
                {
                    "paper_id": "nmt-1409.0473",
                    "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
                    "search_text": "a neural image caption generator neural machine translation jointly learning align translate",
                }
            ]
        if token == "Attention":
            return [
                {
                    "paper_id": "attnres2026",
                    "title": "Attention Residuals",
                    "search_text": "attention residuals",
                }
            ]
        if token == "Retrieval-Augmented":
            return [
                {
                    "paper_id": "2510.15682",
                    "title": "SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation",
                    "search_text": "retrieval-augmented generation squai",
                }
            ]
        return []

    def search_papers(self, query, limit=20):
        _ = limit
        token = str(query or "").strip()
        if token == "Auto-Encoding Variational Bayes":
            return [
                {
                    "arxiv_id": "aevb-local",
                    "title": "Auto-Encoding Variational Bayes",
                }
            ]
        if token == "A Compound AI Architecture for Scientific Discovery":
            return [
                {
                    "arxiv_id": "2511.18298",
                    "title": "Cross-Disciplinary Knowledge Retrieval and Synthesis: A Compound AI Architecture for Scientific Discovery",
                },
                {
                    "arxiv_id": "compound-local",
                    "title": "A Compound AI Architecture for Scientific Discovery",
                },
            ]
        if token == "A Neural Image Caption Generator":
            return [
                {
                    "arxiv_id": "caption-local",
                    "title": "A Neural Image Caption Generator",
                }
            ]
        if token == "Deep Residual Learning":
            return [
                {
                    "arxiv_id": "deep-local",
                    "title": "Deep Residual Learning",
                }
            ]
        if token == "Attention Is All You Need":
            return [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                }
            ]
        if token == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks":
            return [
                {
                    "arxiv_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                }
            ]
        if token == "An Image is Worth 16x16 Words":
            return [
                {
                    "arxiv_id": "2010.11929",
                    "title": "An Image is Worth 16x16 Words",
                }
            ]
        if token == "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering":
            return [
                {
                    "arxiv_id": "2007.01282",
                    "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
                }
            ]
        if token == "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection":
            return [
                {
                    "arxiv_id": "2310.11511",
                    "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
                }
            ]
        if token == "Deep":
            return [
                {
                    "arxiv_id": "2501.12948",
                    "title": "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
                }
            ]
        return []


class CompareLookupRescueSQLite(DummySQLite):
    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        token = str(query or "").strip()
        if token == "An Image is Worth 16x16 Words":
            return [
                {
                    "paper_id": "2010.11929",
                    "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
                    "search_text": "vision transformer vit an image is worth 16x16 words",
                }
            ]
        return []


class RepresentativeFilterSQLite(DummySQLite):
    def list_ontology_entities(self, limit=5000):
        _ = limit
        return [
            {
                "entity_id": "retrieval_augmented_generation",
                "entity_type": "concept",
                "canonical_name": "Retrieval-Augmented Generation",
            },
            {
                "entity_id": "deep_q_networks",
                "entity_type": "concept",
                "canonical_name": "Deep Q-Networks",
            },
        ]

    def get_entity_aliases(self, entity_id):
        if str(entity_id or "").strip() == "retrieval_augmented_generation":
            return ["RAG"]
        if str(entity_id or "").strip() == "deep_q_networks":
            return ["DQN", "Deep Q-Network"]
        return []

    def get_concept_papers(self, concept_id):
        if str(concept_id or "").strip() == "retrieval_augmented_generation":
            return [
                {
                    "arxiv_id": "2312.10997",
                    "title": "Retrieval-Augmented Generation Survey and Overview",
                },
                {
                    "arxiv_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                },
            ]
        if str(concept_id or "").strip() == "deep_q_networks":
            return [
                {
                    "arxiv_id": "1312.5602",
                    "title": "Playing Atari with Deep Reinforcement Learning",
                }
            ]
        return []

    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        token = str(query or "").strip()
        if token == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks":
            return [
                {
                    "paper_id": "2005.11401",
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "search_text": "retrieval augmented generation knowledge intensive nlp tasks",
                }
            ]
        if token == "Playing Atari with Deep Reinforcement Learning":
            return [
                {
                    "paper_id": "1312.5602",
                    "title": "Playing Atari with Deep Reinforcement Learning",
                    "search_text": "deep q network dqn playing atari with deep reinforcement learning",
                }
            ]
        if token == "Attention Is All You Need":
            return [
                {
                    "paper_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "search_text": "transformer attention is all you need",
                }
            ]
        if token == "Language Models are Few-Shot Learners":
            return [
                {
                    "paper_id": "2005.14165",
                    "title": "Language Models are Few-Shot Learners",
                    "search_text": "gpt language models are few-shot learners",
                }
            ]
        if token == "Proximal Policy Optimization Algorithms":
            return [
                {
                    "paper_id": "1707.06347",
                    "title": "Proximal Policy Optimization Algorithms",
                    "search_text": "ppo proximal policy optimization algorithms",
                }
            ]
        if token == "Generative Adversarial Nets":
            return [
                {
                    "paper_id": "1406.2661",
                    "title": "Generative Adversarial Nets",
                    "search_text": "gan generative adversarial nets",
                }
            ]
        if token == "Denoising Diffusion Probabilistic Models":
            return [
                {
                    "paper_id": "2006.11239",
                    "title": "Denoising Diffusion Probabilistic Models",
                    "search_text": "diffusion denoising diffusion probabilistic models",
                }
            ]
        if token == "An Image is Worth 16x16 Words":
            return [
                {
                    "paper_id": "2010.11929",
                    "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
                    "search_text": "vit an image is worth 16x16 words transformers for image recognition at scale",
                }
            ]
        return []


class CompareResolverConflictSQLite(RepresentativeFilterSQLite):
    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        token = str(query or "").strip()
        if token == "GPT":
            return [
                {
                    "paper_id": "2303.08774",
                    "title": "GPT-4 Technical Report",
                    "search_text": "gpt gpt-4 technical report",
                }
            ]
        if token == "Language Models are Few-Shot Learners":
            return [
                {
                    "paper_id": "2005.14165",
                    "title": "Language Models are Few-Shot Learners",
                    "search_text": "gpt language models are few-shot learners",
                }
            ]
        return super().search_paper_cards_v2(query, limit=limit)

    def search_papers(self, query, limit=20):
        _ = limit
        token = str(query or "").strip()
        if token == "GPT":
            return [
                {
                    "arxiv_id": "2303.08774",
                    "title": "GPT-4 Technical Report",
                }
            ]
        if token == "Language Models are Few-Shot Learners":
            return [
                {
                    "arxiv_id": "2005.14165",
                    "title": "Language Models are Few-Shot Learners",
                }
            ]
        return []


class TransformerMambaResolverSQLite(DummySQLite):
    def list_ontology_entities(self, limit=5000):
        _ = limit
        return [
            {
                "entity_id": "transformers",
                "entity_type": "concept",
                "canonical_name": "Transformers",
            }
        ]

    def get_entity_aliases(self, entity_id):
        if str(entity_id or "").strip() == "transformers":
            return ["Transformer"]
        return []

    def get_concept_papers(self, concept_id):
        if str(concept_id or "").strip() == "transformers":
            return [
                {
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                },
                {
                    "arxiv_id": "attnres2026",
                    "title": "Attention Residuals",
                },
            ]
        return []

    def search_paper_cards_v2(self, query, limit=5):
        _ = limit
        token = str(query or "").strip()
        if token == "Attention Is All You Need":
            return [
                {
                    "paper_id": "1706.03762",
                    "title": "Attention Is All You Need",
                    "search_text": "transformer attention is all you need",
                }
            ]
        if token == "Mamba: Linear-Time Sequence Modeling with Selective State Spaces":
            return []
        return []


class PrefixTitleSQLite(DummySQLite):
    def search_papers(self, query, limit=20):
        _ = limit
        if str(query or "").strip().casefold() == "dino":
            return [
                {
                    "arxiv_id": "dinov3-local",
                    "title": "DINOv3",
                },
                {
                    "arxiv_id": "grounding-dino",
                    "title": "Grounding DINO",
                },
            ]
        return []


def test_classify_paper_family_routes_default_paper_queries_into_four_families():
    assert classify_paper_family("CNN을 쉽게 설명해줘", source_type="paper") == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert classify_paper_family("BERT가 뭐야?", source_type="paper") == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert classify_paper_family("ViT가 뭐야?", source_type="paper") == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert classify_paper_family("AlexNet 논문 요약해줘", source_type="paper") == PAPER_FAMILY_LOOKUP
    assert classify_paper_family("BERT 논문의 초록을 정리해줘", source_type="paper") == PAPER_FAMILY_LOOKUP
    assert classify_paper_family("Playing Atari with Deep Reinforcement Learning 논문을 설명해줘", source_type="paper") == PAPER_FAMILY_LOOKUP
    assert classify_paper_family("CNN vs ViT 비교해줘", source_type="paper") == PAPER_FAMILY_COMPARE
    assert classify_paper_family("RAG 관련 논문 찾아줘", source_type="paper") == PAPER_FAMILY_DISCOVER
    assert classify_paper_family("rag query", source_type="paper") == PAPER_FAMILY_DISCOVER


def test_build_rule_query_plan_expands_concept_aliases_and_representative_titles():
    plan = build_rule_query_plan(
        "CNN을 쉽게 설명해줘",
        source_type="paper",
        sqlite_db=DummySQLite(),
    ).to_dict()

    assert plan["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert plan["queryIntent"] == "definition"
    assert "CNN" in plan["expandedTerms"]
    assert "Deep Convolutional Neural Networks" in plan["expandedTerms"]
    assert "ImageNet Classification with Deep Convolutional Neural Networks" in plan["expandedTerms"]


def test_build_rule_based_query_frame_captures_domain_family_and_canonical_ids():
    frame = build_rule_based_query_frame(
        "CNN을 쉽게 설명해줘",
        source_type="paper",
        sqlite_db=DummySQLite(),
    ).to_dict()

    assert frame["domain_key"] == "ai_papers"
    assert frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert frame["query_intent"] == "definition"
    assert frame["answer_mode"] == "representative_paper_explainer_beginner"
    assert frame["canonical_entity_ids"] == ["deep_convolutional_neural_networks"]
    assert frame["evidence_policy_key"] == "concept_explainer_policy"
    assert frame["resolved_source_ids"] == ["alexnet-2012"]


def test_build_rule_based_query_frame_treats_short_korean_concept_questions_as_concept_explainer():
    frame = build_rule_based_query_frame(
        "BERT가 뭐야?",
        source_type="paper",
        sqlite_db=LookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert frame["query_intent"] == "definition"
    assert frame["answer_mode"] == "representative_paper_explainer"
    assert "BERT" in frame["entities"]


def test_build_rule_query_plan_resolves_lookup_alias_to_representative_paper():
    plan = build_rule_query_plan(
        "AlexNet 논문 요약해줘",
        source_type="paper",
        sqlite_db=LookupSQLite(),
    ).to_dict()

    assert plan["family"] == PAPER_FAMILY_LOOKUP
    assert plan["resolvedPaperIds"] == ["alexnet-2012"]
    assert plan["evidencePolicyKey"] == "paper_lookup_policy"
    assert "ImageNet Classification with Deep Convolutional Neural Networks" in plan["expandedTerms"]


def test_build_rule_based_query_frame_resolves_lookup_source_scope_for_near_exact_titles():
    frame = build_rule_based_query_frame(
        "AlexNet 논문 요약해줘",
        source_type="paper",
        sqlite_db=LookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["alexnet-2012"]
    assert frame["canonical_entity_ids"] == []
    assert frame["evidence_policy_key"] == "paper_lookup_policy"


def test_build_rule_query_plan_prefers_exact_title_prefix_for_lookup_disambiguation():
    plan = build_rule_query_plan(
        "BERT 논문 요약해줘",
        source_type="paper",
        sqlite_db=LookupSQLite(),
    ).to_dict()

    assert plan["family"] == PAPER_FAMILY_LOOKUP
    assert plan["resolvedPaperIds"] == ["1810.04805"]


def test_build_rule_based_query_frame_resolves_multiple_compare_candidates_from_concepts():
    frame = build_rule_based_query_frame(
        "CNN vs ViT 비교해줘",
        source_type="paper",
        sqlite_db=CompareSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["resolved_source_ids"] == ["alexnet-2012", "2010.11929"]
    assert "ImageNet Classification with Deep Convolutional Neural Networks" in frame["expanded_terms"]
    assert any("An Image is Worth 16x16 Words" in item for item in frame["expanded_terms"])
    assert frame["entities"] == ["CNN", "ViT"]


def test_build_rule_based_query_frame_handles_korean_compare_connector_and_keeps_vit_rescue():
    frame = build_rule_based_query_frame(
        "CNN이랑 ViT를 논문 관점에서 비교해서 핵심 차이와 각각 잘하는 상황을 설명해줘",
        source_type="paper",
        sqlite_db=CompareSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["entities"] == ["CNN", "ViT"]
    assert frame["resolved_source_ids"] == ["alexnet-2012", "2010.11929"]
    assert "Vision Transformer" in frame["expanded_terms"]


def test_build_rule_based_query_frame_seeds_compare_queries_with_multiple_representative_papers():
    frame = build_rule_based_query_frame(
        "CNN vs ViT 비교해줘",
        source_type="paper",
        sqlite_db=CompareSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["canonical_entity_ids"] == [
        "deep_convolutional_neural_networks",
        "vision_transformers",
    ]
    assert frame["resolved_source_ids"] == ["alexnet-2012", "2010.11929"]


def test_build_rule_based_query_frame_uses_bounded_alias_rescue_for_compare_entities():
    frame = build_rule_based_query_frame(
        "CNN vs ViT 비교해줘",
        source_type="paper",
        sqlite_db=CompareLookupRescueSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["entities"] == ["CNN", "ViT"]
    assert "An Image is Worth 16x16 Words" in frame["expanded_terms"]
    assert frame["resolved_source_ids"] == ["alexnet-2012", "2010.11929"]


def test_build_rule_based_query_frame_uses_bounded_rescue_forms_for_rag_and_dqn():
    rag_frame = build_rule_based_query_frame(
        "RAG를 쉽게 설명해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()
    dqn_frame = build_rule_based_query_frame(
        "DQN의 핵심 아이디어를 설명해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert rag_frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert "Retrieval-Augmented Generation" in rag_frame["expanded_terms"]
    assert "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" in rag_frame["expanded_terms"]
    assert rag_frame["resolved_source_ids"] == ["2005.11401"]

    assert dqn_frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert "Deep Q-Network" in dqn_frame["expanded_terms"]
    assert "Playing Atari with Deep Reinforcement Learning" in dqn_frame["expanded_terms"]
    assert dqn_frame["resolved_source_ids"] == ["1312.5602"]


def test_build_rule_based_query_frame_uses_transformer_rescue_title_for_concept_queries():
    frame = build_rule_based_query_frame(
        "Transformer의 핵심 아이디어를 설명해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert "Attention Is All You Need" in frame["expanded_terms"]
    assert frame["resolved_source_ids"] == ["1706.03762"]


def test_build_rule_based_query_frame_uses_local_prefix_title_rescue_for_versioned_papers():
    frame = build_rule_based_query_frame(
        "DINO에 대해서 설명해줘",
        source_type="paper",
        sqlite_db=PrefixTitleSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_CONCEPT_EXPLAINER
    assert "DINOv3" in frame["expanded_terms"]
    assert "Grounding DINO" not in frame["expanded_terms"]
    assert frame["resolved_source_ids"] == ["dinov3-local"]


def test_build_rule_based_query_frame_strips_lookup_task_words_for_title_queries():
    frame = build_rule_based_query_frame(
        "An Image is Worth 16x16 Words 논문 요약해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["2010.11929"]
    assert any("An Image is Worth 16x16 Words" in item for item in frame["expanded_terms"])


def test_build_rule_based_query_frame_applies_curated_title_rescue_when_lookup_search_is_missing():
    frame = build_rule_based_query_frame(
        "Playing Atari with Deep Reinforcement Learning 논문을 설명해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["1312.5602"]
    assert frame["expanded_terms"][0] == "Playing Atari with Deep Reinforcement Learning"


def test_build_rule_based_query_frame_uses_local_exact_title_fallback_for_lookup_without_cards():
    frame = build_rule_based_query_frame(
        "Auto-Encoding Variational Bayes 논문 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["aevb-local"]
    assert frame["expanded_terms"][0] == "Auto-Encoding Variational Bayes"


def test_build_rule_based_query_frame_prefers_exact_local_title_over_containing_title_fallback():
    frame = build_rule_based_query_frame(
        "A Compound AI Architecture for Scientific Discovery 논문 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["compound-local"]
    assert frame["expanded_terms"][0] == "A Compound AI Architecture for Scientific Discovery"


def test_build_rule_based_query_frame_prefers_local_exact_title_over_card_overlap_for_lookup():
    frame = build_rule_based_query_frame(
        "A Neural Image Caption Generator 논문 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["caption-local"]
    assert frame["expanded_terms"][0] == "A Neural Image Caption Generator"


def test_build_rule_based_query_frame_does_not_expand_local_title_lookup_with_short_overlap_tokens():
    frame = build_rule_based_query_frame(
        "Deep Residual Learning 논문 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["deep-local"]
    assert frame["expanded_terms"][0] == "Deep Residual Learning"


def test_build_rule_based_query_frame_strips_result_suffix_for_explicit_lookup_titles():
    frame = build_rule_based_query_frame(
        "Attention Is All You Need 논문의 실험 결과를 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["1706.03762"]
    assert frame["expanded_terms"][0] == "Attention Is All You Need"


def test_build_rule_based_query_frame_strips_method_suffix_for_explicit_lookup_titles():
    frame = build_rule_based_query_frame(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 논문의 방법을 설명해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_LOOKUP
    assert frame["resolved_source_ids"] == ["2005.11401"]
    assert frame["expanded_terms"][0] == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"


def test_build_rule_based_query_frame_adds_compare_rescue_forms_for_gpt_and_ppo():
    bert_gpt = build_rule_based_query_frame(
        "BERT와 GPT 계열의 차이를 논문 기준으로 비교해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()
    dqn_ppo = build_rule_based_query_frame(
        "DQN과 PPO를 비교해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert bert_gpt["family"] == PAPER_FAMILY_COMPARE
    assert "Language Models are Few-Shot Learners" in bert_gpt["expanded_terms"]
    assert "2005.14165" in bert_gpt["resolved_source_ids"]

    assert dqn_ppo["family"] == PAPER_FAMILY_COMPARE
    assert "Proximal Policy Optimization Algorithms" in dqn_ppo["expanded_terms"]
    assert "1707.06347" in dqn_ppo["resolved_source_ids"]


def test_build_rule_based_query_frame_prefers_gpt3_anchor_over_gpt4_lookup_overlap():
    frame = build_rule_based_query_frame(
        "BERT와 GPT 계열의 차이를 논문 기준으로 비교해줘",
        source_type="paper",
        sqlite_db=CompareResolverConflictSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["resolved_source_ids"][:2] == ["1810.04805", "2005.14165"]
    assert "2303.08774" not in frame["resolved_source_ids"][:2]
    assert "Language Models are Few-Shot Learners" in frame["expanded_terms"]


def test_build_rule_based_query_frame_uses_mamba_anchor_before_second_transformer_related_paper():
    frame = build_rule_based_query_frame(
        "Transformer와 Mamba를 비교해줘",
        source_type="paper",
        sqlite_db=TransformerMambaResolverSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert frame["resolved_source_ids"][:2] == ["1706.03762", "2312.00752"]
    assert "attnres2026" not in frame["resolved_source_ids"][:2]
    assert "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" in frame["expanded_terms"]


def test_build_rule_based_query_frame_adds_compare_rescue_forms_for_rag_and_fid():
    frame = build_rule_based_query_frame(
        "RAG와 FiD를 비교해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering" in frame["expanded_terms"]
    assert "2005.11401" in frame["resolved_source_ids"]
    assert "2007.01282" in frame["resolved_source_ids"]


def test_build_rule_based_query_frame_adds_compare_rescue_forms_for_gan_and_diffusion():
    frame = build_rule_based_query_frame(
        "GAN과 Diffusion 모델 논문을 비교해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert "Generative Adversarial Nets" in frame["expanded_terms"]
    assert "Denoising Diffusion Probabilistic Models" in frame["expanded_terms"]
    assert "1406.2661" in frame["resolved_source_ids"]
    assert "2006.11239" in frame["resolved_source_ids"]


def test_build_rule_based_query_frame_prefers_explicit_compare_titles_over_generic_token_overlap():
    frame = build_rule_based_query_frame(
        "Deep Residual Learning와 An Image is Worth 16x16 Words를 비교해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert "deep-local" in frame["resolved_source_ids"]
    assert "2010.11929" in frame["resolved_source_ids"]
    assert frame["expanded_terms"][0] == "Deep Residual Learning"
    assert frame["expanded_terms"][1] == "An Image is Worth 16x16 Words"


def test_build_rule_based_query_frame_resolves_long_title_compare_pairs():
    frame = build_rule_based_query_frame(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks와 Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection을 비교해줘",
        source_type="paper",
        sqlite_db=LocalTitleLookupSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_COMPARE
    assert "2005.11401" in frame["resolved_source_ids"]
    assert "2310.11511" in frame["resolved_source_ids"]


def test_build_rule_based_query_frame_adds_discover_rescue_terms_for_state_space_models():
    frame = build_rule_based_query_frame(
        "state space model 계열 논문들을 찾아 정리해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_DISCOVER
    assert "Mamba" in frame["expanded_terms"]
    assert "RetNet" in frame["expanded_terms"]


def test_build_rule_based_query_frame_adds_discover_rescue_terms_for_transformer_alternatives():
    frame = build_rule_based_query_frame(
        "Transformer 대안 아키텍처 논문들을 추천해줘",
        source_type="paper",
        sqlite_db=RepresentativeFilterSQLite(),
    ).to_dict()

    assert frame["family"] == PAPER_FAMILY_DISCOVER
    assert "Mamba" in frame["expanded_terms"]
    assert "RetNet" in frame["expanded_terms"]
