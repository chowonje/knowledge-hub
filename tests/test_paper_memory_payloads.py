from __future__ import annotations

from knowledge_hub.papers.memory_adapter import paper_memory_card_to_section_cards
from knowledge_hub.papers.memory_models import PaperMemoryCard
from knowledge_hub.papers.memory_payloads import shared_slot_payload


def test_shared_slot_payload_normalizes_paper_memory_card_into_card_v2_slot_shape():
    payload = shared_slot_payload(
        PaperMemoryCard(
            memory_id="paper-memory:flashattention",
            paper_id="2205.14135",
            title="FlashAttention",
            paper_core="IO-aware exact attention.",
            problem_context="Attention is bottlenecked by memory movement.",
            method_core="Tile attention into SRAM-sized blocks.",
            evidence_core="2-4x speedups on long sequence benchmarks.",
            limitations="Depends on accelerator memory hierarchy.",
            concept_links=["attention", "systems"],
            claim_refs=["claim:1"],
            published_at="2022-05-27T00:00:00+00:00",
            search_text="flashattention io-aware exact attention",
            quality_flag="ok",
            updated_at="2026-04-07T00:00:00+00:00",
        )
    )

    assert payload["memory_id"] == "paper-memory:flashattention"
    assert payload["paper_id"] == "2205.14135"
    assert payload["paper_core"] == "IO-aware exact attention."
    assert payload["problem_core"] == "Attention is bottlenecked by memory movement."
    assert payload["method_core"] == "Tile attention into SRAM-sized blocks."
    assert payload["evidence_core"] == "2-4x speedups on long sequence benchmarks."
    assert payload["limitations_core"] == "Depends on accelerator memory hierarchy."
    assert payload["quality_flag"] == "ok"


def test_paper_memory_card_to_section_cards_maps_slot_roles_into_section_shape():
    cards = paper_memory_card_to_section_cards(
        {
            "memory_id": "paper-memory:flashattention",
            "paper_id": "2205.14135",
            "title": "FlashAttention",
            "paper_core": "IO-aware exact attention.",
            "problem_context": "Attention is bottlenecked by memory movement.",
            "method_core": "Tile attention into SRAM-sized blocks.",
            "evidence_core": "2-4x speedups on long sequence benchmarks.",
            "limitations": "Depends on accelerator memory hierarchy.",
            "concept_links": ["attention", "systems"],
            "claim_refs": ["claim:1"],
        },
        source_card_id="paper-card-v2:2205.14135",
        paper_id="2205.14135",
        title="FlashAttention",
    )

    assert [item["role"] for item in cards] == ["problem", "method", "results", "limitations"]
    assert cards[0]["origin"] == "paper_memory_adapter_v1"
    assert cards[1]["section_path"] == "Paper Memory > Method"
    assert cards[2]["claims"] == ["claim:1"]
