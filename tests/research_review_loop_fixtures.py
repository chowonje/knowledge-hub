from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final


SCHEMA_PATH: Final = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "research-review-loop-result.v1.json"
EVALUATION_PAPER_ID: Final = "2603.14473"
EVALUATION_PAPER_CARD_ID: Final = f"paper-card-v2:{EVALUATION_PAPER_ID}"
SUPPORTED_CLAIM: Final = (
    "The appendix presents representative outputs from the 30B model and its 4B variant for research paper "
    "comparison across fields and datasets."
)
REJECTED_CLAIM: Final = (
    "In this example, institutional prominence and broad downstream relevance outweigh a small publication-date "
    "difference."
)
UNSUPPORTED_CLAIM: Final = "The case study highlights temporal generalization to high-visibility frontier AI reports."
MISSING_LOCATOR_CLAIM: Final = "The paper's scientific taste benchmark is fully supported by an unlocatable span."
UNREVIEWED_CLAIM: Final = (
    "The examples are drawn from an OOD Year test set intended to test extrapolation to 2025 papers outside "
    "the training distribution."
)


def claim_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class ReviewLoopDb:
    def list_claim_cards(self, *, source_kind: str, limit: int):
        assert source_kind == "paper"
        assert limit >= 1
        return [
            {
                "claim_card_id": "claim-card-v1:paper:claim-supported",
                "claim_id": "claim-supported",
                "claim_text": SUPPORTED_CLAIM,
                "paper_id": EVALUATION_PAPER_ID,
                "source_id": EVALUATION_PAPER_ID,
                "task_canonical": "research paper comparison",
                "dataset_canonical": "OOD Year",
                "metric_canonical": "scientific taste",
            },
            {
                "claim_card_id": "claim-card-v1:paper:claim-rejected",
                "claim_id": "claim-rejected",
                "claim_text": REJECTED_CLAIM,
                "paper_id": EVALUATION_PAPER_ID,
                "source_id": EVALUATION_PAPER_ID,
                "task_canonical": "scientific taste evaluation",
            },
            {
                "claim_card_id": "claim-card-v1:paper:claim-unsupported",
                "claim_id": "claim-unsupported",
                "claim_text": UNSUPPORTED_CLAIM,
                "paper_id": EVALUATION_PAPER_ID,
                "source_id": EVALUATION_PAPER_ID,
                "task_canonical": "scientific taste evaluation",
            },
            {
                "claim_card_id": "claim-card-v1:paper:claim-unreviewed",
                "claim_id": "claim-unreviewed",
                "claim_text": UNREVIEWED_CLAIM,
                "paper_id": EVALUATION_PAPER_ID,
                "source_id": EVALUATION_PAPER_ID,
                "task_canonical": "scientific taste evaluation",
            },
            {
                "claim_card_id": "claim-card-v1:paper:claim-missing-locator",
                "claim_id": "claim-missing-locator",
                "claim_text": MISSING_LOCATOR_CLAIM,
                "paper_id": EVALUATION_PAPER_ID,
                "source_id": EVALUATION_PAPER_ID,
                "task_canonical": "scientific taste evaluation",
            },
        ]

    def list_claim_card_source_refs(self, *, claim_card_id: str = ""):
        match claim_card_id:
            case "claim-card-v1:paper:claim-supported":
                return [{"source_card_id": EVALUATION_PAPER_CARD_ID}]
            case "claim-card-v1:paper:claim-rejected":
                return [{"source_card_id": EVALUATION_PAPER_CARD_ID}]
            case "claim-card-v1:paper:claim-unsupported":
                return [{"source_card_id": EVALUATION_PAPER_CARD_ID}]
            case "claim-card-v1:paper:claim-unreviewed":
                return [{"source_card_id": EVALUATION_PAPER_CARD_ID}]
            case "claim-card-v1:paper:claim-missing-locator":
                return [{"source_card_id": EVALUATION_PAPER_CARD_ID}]
            case "":
                return []
            case unreachable:
                raise AssertionError(f"unexpected claim_card_id: {unreachable}")

    def list_evidence_anchors_v2(self, *, card_id: str, claim_ids: list[str]):
        assert card_id == EVALUATION_PAPER_CARD_ID
        match claim_ids:
            case ["claim-supported"]:
                return [
                    {
                        "anchor_id": "span-supported",
                        "source_id": EVALUATION_PAPER_ID,
                        "locator": "chars:100-188",
                        "sourceContentHash": "source-hash-supported",
                        "snippet_hash": "hash-supported",
                        "quote": "Representative outputs compare research papers across fields and datasets.",
                    }
                ]
            case ["claim-rejected"]:
                return [
                    {
                        "anchor_id": "span-rejected",
                        "source_id": EVALUATION_PAPER_ID,
                        "locator": "chars:240-312",
                        "sourceContentHash": "source-hash-rejected",
                        "snippet_hash": "hash-rejected",
                        "quote": "The case weighs institutional prominence against publication date.",
                    }
                ]
            case ["claim-unsupported"]:
                return []
            case ["claim-unreviewed"]:
                return []
            case ["claim-missing-locator"]:
                return [
                    {
                        "anchor_id": "span-missing-locator",
                        "source_id": EVALUATION_PAPER_ID,
                        "snippet_hash": "hash-missing-locator",
                        "quote": "This span has quote text but no stable locator.",
                    }
                ]
            case unreachable:
                raise AssertionError(f"unexpected claim ids: {unreachable}")


class ReviewLoopKhub:
    def __init__(self, db: ReviewLoopDb) -> None:
        self._db = db

    def sqlite_db(self) -> ReviewLoopDb:
        return self._db
