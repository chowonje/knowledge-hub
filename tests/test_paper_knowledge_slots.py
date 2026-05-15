from __future__ import annotations

from knowledge_hub.core.schema_validator import validate_payload
from knowledge_hub.papers.knowledge_slots import (
    PAPER_KNOWLEDGE_SLOTS_SCHEMA,
    build_paper_knowledge_slots_payload,
    load_paper_knowledge_slots_payload,
)


def _base_card() -> dict:
    return {
        "card_id": "paper-card-v2:2205.14135",
        "paper_id": "2205.14135",
        "title": "FlashAttention",
        "source_content_hash": "sha256:paper",
        "paper_core": "IO-aware exact attention.",
        "problem_core": "Attention is bottlenecked by memory movement.",
        "method_core": "Tile attention into SRAM-sized blocks.",
        "result_core": "2-4x speedups on long sequence benchmarks.",
        "limitations_core": "Depends on accelerator memory hierarchy.",
        "dataset_core": "Long Range Arena",
        "metric_core": "wall-clock throughput",
        "when_not_to_use": "",
        "slot_coverage": {
            "paperCore": "complete",
            "problemCore": "complete",
            "methodCore": "complete",
            "resultCore": "complete",
            "limitationsCore": "complete",
            "datasetCore": "complete",
            "metricCore": "complete",
            "whenNotToUse": "missing",
        },
    }


def test_paper_knowledge_slots_builds_contract_from_card_refs_and_anchors():
    payload = build_paper_knowledge_slots_payload(
        card=_base_card(),
        claim_refs=[
            {
                "claim_id": "claim:method",
                "role": "method",
                "slot_key": "method_core",
                "confidence": 0.92,
                "rank": 1,
                "reason": "representative_claim",
                "normalization": {"task": "efficient attention"},
            }
        ],
        anchors=[
            {
                "anchor_id": "anchor:method",
                "claim_id": "claim:method",
                "paper_id": "2205.14135",
                "document_id": "paper:2205.14135",
                "chunk_id": "chunk:method",
                "span_locator": "chars:100-240",
                "snippet_hash": "snippet:method",
                "evidence_role": "method",
                "excerpt": "Tile attention into SRAM-sized blocks.",
                "score": 0.95,
            }
        ],
    )

    assert payload["schema"] == PAPER_KNOWLEDGE_SLOTS_SCHEMA
    assert payload["authority"] == "derived_read_model"
    assert payload["coverage"]["slotCount"] == 8
    assert payload["coverage"]["strictSlotCount"] == 1
    method_slot = next(item for item in payload["slots"] if item["slotType"] == "method")
    assert method_slot["text"] == "Tile attention into SRAM-sized blocks."
    assert method_slot["claimRefs"][0]["claimId"] == "claim:method"
    assert method_slot["strictEvidence"] is True
    assert method_slot["strictEvidenceRefCount"] == 1
    ref = method_slot["evidenceRefs"][0]
    assert ref["documentId"] == "paper:2205.14135"
    assert ref["chunkId"] == "chunk:method"
    assert ref["spanLocator"] == "chars:100-240"
    assert ref["sourceContentHash"] == "sha256:paper"
    assert ref["contentHash"] == "snippet:method"
    assert ref["strictSpanBacked"] is True
    assert ref["locatorOnly"] is False
    assert validate_payload(payload, PAPER_KNOWLEDGE_SLOTS_SCHEMA, strict=True).ok


def test_paper_knowledge_slots_keep_summary_only_and_locator_only_non_strict():
    payload = build_paper_knowledge_slots_payload(
        card=_base_card(),
        claim_refs=[],
        anchors=[
            {
                "anchor_id": "anchor:result",
                "paper_id": "2205.14135",
                "document_id": "paper:2205.14135",
                "span_locator": "unit:result",
                "snippet_hash": "snippet:result",
                "evidence_role": "result",
                "excerpt": "2-4x speedups on long sequence benchmarks.",
            }
        ],
    )

    result_slot = next(item for item in payload["slots"] if item["slotType"] == "result")
    limitation_slot = next(item for item in payload["slots"] if item["slotType"] == "limitation")
    assert result_slot["strictEvidence"] is False
    assert result_slot["evidenceRefs"][0]["strictSpanBacked"] is False
    assert result_slot["evidenceRefs"][0]["locatorOnly"] is True
    assert limitation_slot["coverage"] == "complete"
    assert limitation_slot["evidenceRefs"] == []
    assert payload["diagnostics"]["summaryOnlySlotCount"] >= 1
    assert validate_payload(payload, PAPER_KNOWLEDGE_SLOTS_SCHEMA, strict=True).ok


def test_load_paper_knowledge_slots_reads_existing_surfaces_without_writing():
    class _Db:
        def __init__(self) -> None:
            self.reads: list[str] = []

        def get_paper_card_v2(self, paper_id: str) -> dict:
            self.reads.append(f"card:{paper_id}")
            return _base_card()

        def list_paper_card_claim_refs_v2(self, *, card_id: str) -> list[dict]:
            self.reads.append(f"claims:{card_id}")
            return [{"claim_id": "claim:dataset", "role": "dataset", "slot_key": "dataset_core"}]

        def list_evidence_anchors_v2(self, *, card_id: str) -> list[dict]:
            self.reads.append(f"anchors:{card_id}")
            return [
                {
                    "anchor_id": "anchor:dataset",
                    "claim_id": "claim:dataset",
                    "paper_id": "2205.14135",
                    "span_locator": "chars:400-440",
                    "source_content_hash": "sha256:paper",
                    "snippet_hash": "snippet:dataset",
                    "evidence_role": "dataset",
                    "excerpt": "Long Range Arena",
                }
            ]

    db = _Db()
    payload = load_paper_knowledge_slots_payload(sqlite_db=db, paper_id="2205.14135")

    assert payload is not None
    assert db.reads == [
        "card:2205.14135",
        "claims:paper-card-v2:2205.14135",
        "anchors:paper-card-v2:2205.14135",
    ]
    dataset_slot = next(item for item in payload["slots"] if item["slotType"] == "dataset")
    assert dataset_slot["strictEvidence"] is True
    assert validate_payload(payload, PAPER_KNOWLEDGE_SLOTS_SCHEMA, strict=True).ok
