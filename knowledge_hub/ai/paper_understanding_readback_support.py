from __future__ import annotations

from pathlib import Path
from typing import Final
import hashlib
import json
import re

PRIVATE_PATH_MARKERS: Final = ("/Users/", "Mobile Documents", "iCloud")
FORBIDDEN_MARKERS: Final = ("paper-card-v2", "card_id", "source_card_id")
BRIEF_PROFILE_SLOTS: Final = ("claim", "method", "evidence", "limitation", "purpose_relevance")
CORE_PROFILE_SLOTS: Final = ("claim", "method", "evidence")
SAFE_PAPER_ID: Final = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_EXCERPT_CHARS: Final = 700
SLOT_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "claim": ("argue", "claim", "contribution", "demonstrate", "propose", "show"),
    "method": ("architecture", "approach", "framework", "method", "retrieval", "generation"),
    "evidence": ("benchmark", "evaluation", "experiment", "result", "study"),
    "limitation": ("challenge", "future work", "limitation", "limited", "risk"),
    "purpose_relevance": ("application", "goal", "purpose", "relevant", "task", "useful", "workflow"),
}

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]


def as_map(value: JsonValue | None) -> JsonMap:
    return value if isinstance(value, dict) else {}


def as_maps(value: JsonValue | None) -> list[JsonMap]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def as_strings(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def clean_text(value: JsonValue | None, *, limit: int = MAX_EXCERPT_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def unique(values: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value)]


def contains_marker(payload: JsonValue, markers: tuple[str, ...]) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return any(marker in text for marker in markers)


def has_warning(row: JsonMap, warning: str) -> bool:
    warnings = as_strings(row.get("warnings"))
    return any(item == warning or item.endswith(f":{warning}") for item in warnings)


def _source_id(value: JsonValue | None) -> str:
    text = clean_text(value, limit=120)
    for prefix in ("paper:", "arxiv:"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected_source_ids(row: JsonMap) -> list[str]:
    prompt_packet = as_map(row.get("promptPacket"))
    expected_ids = [_source_id(item) for item in as_strings(prompt_packet.get("expectedSourceIds"))]
    if expected_ids:
        return unique(expected_ids)
    return unique([_source_id(item) for item in as_strings(row.get("packetSourceIds"))])


def _citation_label(row: JsonMap, source_id: str) -> str:
    prompt_packet = as_map(row.get("promptPacket"))
    for span in as_maps(prompt_packet.get("spans")):
        span_source_id = _source_id(span.get("sourceId") or span.get("source_id"))
        if span_source_id == source_id:
            return clean_text(span.get("citationLabel") or span.get("citation_label"), limit=80)
    return ""


def _document_path(papers_dir: Path, paper_id: str) -> Path | None:
    if not SAFE_PAPER_ID.fullmatch(paper_id):
        return None
    return papers_dir / "parsed" / paper_id / "document.md"


def _document_segments(document: str) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for line in document.splitlines():
        start = document.find(line, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(line)
        text = line.strip()
        if text and not text.startswith("#"):
            segments.append((text, start, start + len(line)))
    if not segments and document.strip():
        stripped = document.strip()
        start = max(document.find(stripped), 0)
        segments.append((stripped, start, start + len(stripped)))
    return segments


def _slot_segment(document: str, slot: str) -> tuple[str, int, int] | None:
    for text, start, end in _document_segments(document):
        lower_text = text.lower()
        if any(keyword in lower_text for keyword in SLOT_KEYWORDS[slot]):
            return text, start, end
    return None


def _slot_readback(*, document: str, source_id: str, citation_label: str, source_hash: str, slot: str) -> JsonMap:
    source_ref = f"papers_dir/parsed/{source_id}/document.md"
    segment = _slot_segment(document, slot)
    if segment is None:
        return {
            "slot": slot,
            "status": "blocked",
            "sourceId": source_id,
            "citationLabel": citation_label,
            "locator": "",
            "sourceRef": source_ref,
            "contentHash": source_hash,
            "snippetHash": "",
            "excerpt": "",
            "warnings": [f"missing_readback_slot:{slot}"],
        }
    text, start, end = segment
    excerpt = clean_text(text)
    return {
        "slot": slot,
        "status": "ready",
        "sourceId": source_id,
        "citationLabel": citation_label,
        "locator": f"chars:{start}-{end}",
        "sourceRef": source_ref,
        "contentHash": source_hash,
        "snippetHash": _sha256_text(excerpt),
        "excerpt": excerpt,
        "warnings": [],
    }


def _blocked_paper_readback(source_id: str, warnings: list[str]) -> JsonMap:
    return {
        "sourceId": source_id,
        "slots": [],
        "availableSlots": [],
        "missingBriefSlots": list(BRIEF_PROFILE_SLOTS),
        "coreReady": False,
        "briefReady": False,
        "warnings": warnings,
    }


def _paper_readback(*, row: JsonMap, papers_dir: Path, source_id: str) -> JsonMap:
    if not SAFE_PAPER_ID.fullmatch(source_id):
        return _blocked_paper_readback(source_id, ["unsafe_paper_id"])
    citation_label = _citation_label(row, source_id)
    if not citation_label:
        return _blocked_paper_readback(source_id, ["missing_citation_label"])
    path = _document_path(papers_dir, source_id)
    if path is None or not path.exists():
        return _blocked_paper_readback(source_id, ["missing_parsed_document"])
    document = path.read_text(encoding="utf-8")
    if contains_marker(document, PRIVATE_PATH_MARKERS):
        return _blocked_paper_readback(source_id, ["private_path_marker"])

    source_hash = _sha256_text(document)
    slots = [
        _slot_readback(document=document, source_id=source_id, citation_label=citation_label, source_hash=source_hash, slot=slot)
        for slot in BRIEF_PROFILE_SLOTS
    ]
    available_slots = [str(slot["slot"]) for slot in slots if slot["status"] == "ready"]
    missing_brief_slots = [slot for slot in BRIEF_PROFILE_SLOTS if slot not in available_slots]
    warnings = [warning for slot in slots for warning in as_strings(slot.get("warnings"))]
    return {
        "sourceId": source_id,
        "slots": slots,
        "availableSlots": available_slots,
        "missingBriefSlots": missing_brief_slots,
        "coreReady": not any(slot for slot in CORE_PROFILE_SLOTS if slot not in available_slots),
        "briefReady": not missing_brief_slots,
        "warnings": warnings,
    }


def _prefixed_warnings(source_id: str, warnings: JsonValue | None) -> list[str]:
    return [f"{source_id}:{warning}" for warning in as_strings(warnings)]


def _row_shell(row: JsonMap, *, status: str, expected_ids: list[str], warnings: list[str]) -> JsonMap:
    return {
        "runId": clean_text(row.get("runId"), limit=160),
        "caseId": clean_text(row.get("caseId"), limit=120),
        "variantId": clean_text(row.get("variantId"), limit=120),
        "status": status,
        "expectedSourceIds": expected_ids,
        "paperReadbacks": [],
        "readyPaperCount": 0,
        "blockedPaperCount": len(expected_ids) if status == "blocked" else 0,
        "warnings": warnings,
    }


def row_readback(row: JsonMap, *, papers_dir: Path) -> JsonMap:
    expected_ids = _expected_source_ids(row)
    if not expected_ids:
        return _row_shell(row, status="not_applicable", expected_ids=[], warnings=["not_applicable_abstention"])

    row_private = contains_marker(row, PRIVATE_PATH_MARKERS)
    row_forbidden = contains_marker(row, FORBIDDEN_MARKERS)
    if row_private or row_forbidden:
        warnings = []
        if row_private:
            warnings.append("private_path_marker")
        if row_forbidden:
            warnings.append("forbidden_raw_marker")
        return _row_shell(row, status="blocked", expected_ids=expected_ids, warnings=warnings)

    paper_readbacks = [_paper_readback(row=row, papers_dir=papers_dir, source_id=source_id) for source_id in expected_ids]
    row_warnings = [
        warning
        for paper in paper_readbacks
        for warning in _prefixed_warnings(str(paper.get("sourceId") or ""), paper.get("warnings"))
    ]
    if clean_text(row.get("status"), limit=20) == "blocked":
        row_warnings.append("packet_input_row_blocked")

    ready_paper_count = sum(1 for paper in paper_readbacks if bool(paper.get("briefReady")))
    blocked_paper_count = len(paper_readbacks) - ready_paper_count
    return {
        "runId": clean_text(row.get("runId"), limit=160),
        "caseId": clean_text(row.get("caseId"), limit=120),
        "variantId": clean_text(row.get("variantId"), limit=120),
        "status": "ready" if blocked_paper_count == 0 and not row_warnings else "blocked",
        "expectedSourceIds": expected_ids,
        "paperReadbacks": paper_readbacks,
        "readyPaperCount": ready_paper_count,
        "blockedPaperCount": blocked_paper_count,
        "warnings": unique(row_warnings),
    }
