from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knowledge_hub.application.research_review_loop_types import ClaimInput, JsonValue, Row


def load_claims_file_inputs(claims_file: str | None, *, paper_ids: list[str]) -> list[ClaimInput]:
    if not claims_file:
        return []
    path = Path(claims_file).expanduser()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    claim_rows = _claim_rows(parsed)
    selected = set(paper_ids)
    inputs: list[ClaimInput] = []
    for item in claim_rows:
        source_id = _source_id(item)
        claim_text = _text_value(item, ("claimText", "claim_text"))
        if not source_id or source_id not in selected or not claim_text:
            continue
        claim_id = _claim_id(item, source_id=source_id, claim_text=claim_text)
        inputs.append(
            ClaimInput(
                row={
                    "claim_card_id": f"claims-file:{source_id}:{claim_id}",
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "paper_id": source_id,
                    "source_id": source_id,
                },
                anchors=_anchors(item, source_id=source_id, claim_id=claim_id),
            )
        )
    return inputs


def _claim_rows(parsed: JsonValue) -> list[dict[str, JsonValue]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        claims = parsed.get("claims")
        if isinstance(claims, list):
            return [item for item in claims if isinstance(item, dict)]
    return []


def _anchors(item: dict[str, JsonValue], *, source_id: str, claim_id: str) -> list[Row]:
    spans = item.get("evidenceSpans") or item.get("evidence_spans")
    if not isinstance(spans, list):
        return []
    anchors: list[Row] = []
    for index, span in enumerate(spans, start=1):
        if not isinstance(span, dict):
            continue
        anchors.append(
            {
                "anchor_id": _text_value(span, ("evidenceSpanId", "anchor_id", "anchorId"))
                or f"claims-file-span:{claim_id}:{index}",
                "source_id": _text_value(span, ("sourceId", "source_id")) or source_id,
                "locator": _text_value(span, ("locator", "stableSpanLocator", "stable_span_locator")),
                "resolvedLocator": _text_value(span, ("resolvedLocator", "resolved_locator")),
                "sourceContentHash": _text_value(span, ("sourceContentHash", "source_content_hash")),
                "snippetHash": _text_value(span, ("snippetHash", "snippet_hash")),
                "quote": _text_value(span, ("quote", "excerpt", "text")),
            }
        )
    return anchors


def _claim_id(item: dict[str, JsonValue], *, source_id: str, claim_text: str) -> str:
    explicit = _text_value(item, ("claimId", "claim_id"))
    if explicit:
        return explicit
    digest = hashlib.sha1(f"{source_id}:{claim_text}".encode("utf-8")).hexdigest()[:12]
    return f"claims-file:{source_id}:{digest}"


def _source_id(item: dict[str, JsonValue]) -> str:
    return _text_value(item, ("sourceId", "source_id", "paperId", "paper_id"))


def _text_value(item: dict[str, JsonValue], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
