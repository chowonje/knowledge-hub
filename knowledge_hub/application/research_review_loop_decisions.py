from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.application.research_review_loop_types import JsonValue, ReviewDecision, ReviewDecisionValue


def claim_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_review_decisions(decision_file: str | None) -> list[ReviewDecision]:
    if not decision_file:
        return []
    path = Path(decision_file).expanduser()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, list):
        return [_coerce_decision(item) for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        raw_items = parsed.get("reviewDecisions", [])
        if isinstance(raw_items, list):
            return [_coerce_decision(item) for item in raw_items if isinstance(item, dict)]
    return []


def decision_applies_to_claim(decision: ReviewDecision | None, *, claim_id: str, claim_text: str) -> bool:
    if decision is None:
        return False
    required = (
        str(decision.get("decisionId") or ""),
        str(decision.get("targetId") or ""),
        str(decision.get("reviewer") or ""),
        str(decision.get("reviewedAt") or ""),
        str(decision.get("reason") or ""),
        str(decision.get("claimTextHash") or ""),
    )
    if not all(required):
        return False
    if str(decision.get("targetType") or "") != "claim":
        return False
    if str(decision.get("targetId") or "") != claim_id:
        return False
    return str(decision.get("claimTextHash") or "") == claim_text_hash(claim_text)


def _coerce_decision(item: dict[str, JsonValue]) -> ReviewDecision:
    return {
        "decisionId": str(item.get("decisionId") or ""),
        "targetType": str(item.get("targetType") or ""),
        "targetId": str(item.get("targetId") or ""),
        "decision": _decision_value(str(item.get("decision") or "unsure")),
        "confidence": str(item.get("confidence") or "medium"),
        "reviewer": str(item.get("reviewer") or ""),
        "reviewedAt": str(item.get("reviewedAt") or ""),
        "reason": str(item.get("reason") or ""),
        "claimTextHash": str(item.get("claimTextHash") or ""),
        "snippetHashes": _snippet_hashes(item.get("snippetHashes")),
        "decisionSource": str(item.get("decisionSource") or "imported_file"),
    }


def _decision_value(value: str) -> ReviewDecisionValue:
    match value:
        case "accept":
            return "accept"
        case "reject":
            return "reject"
        case "unsure":
            return "unsure"
        case "needs_more_evidence":
            return "needs_more_evidence"
        case "archive":
            return "archive"
        case _:
            return "unsure"


def _snippet_hashes(value: JsonValue) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
