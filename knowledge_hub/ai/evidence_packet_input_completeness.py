from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping
import json

EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID: Final = "knowledge-hub.evidence-packet.input-completeness.v1"
FORBIDDEN_MARKERS: Final = ("paper-card-v2", "card_id", "source_card_id")
PRIVATE_PATH_MARKERS: Final = ("/Users/", "Mobile Documents", "iCloud")

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap = dict[str, JsonValue]


def _as_map(value: JsonValue | None) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_maps(value: JsonValue | None) -> list[JsonMap]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_strings(value: JsonValue | None) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _clean_text(value: JsonValue | None, *, limit: int = 700) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit]


def _source_id(value: JsonValue | None) -> str:
    text = _clean_text(value, limit=120)
    for prefix in ("paper:", "arxiv:"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def _unique(values: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(value for value in values if value)]


def _candidate_ids(run: JsonMap, raw: JsonMap) -> list[str]:
    ids = [_source_id(value) for value in _as_strings(run.get("expectedIds"))]
    for citation in _as_maps(raw.get("citations")):
        ids.extend(_source_id(citation.get(field)) for field in ("target", "source_id", "source_ref"))
    for row in [*_as_maps(raw.get("sources")), *_as_maps(raw.get("evidence"))]:
        ids.extend(_source_id(row.get(field)) for field in ("arxiv_id", "citation_target", "source_id", "source_ref"))
    return _unique(ids)


def _span_from_contract(span: JsonMap) -> JsonMap:
    return {
        "sourceId": _source_id(span.get("source_id") or span.get("sourceId")),
        "citationLabel": _clean_text(span.get("citation_label") or span.get("citationLabel"), limit=80),
        "locator": _clean_text(span.get("locator") or span.get("spanLocator"), limit=160),
        "text": _clean_text(span.get("text")),
        "contentHashAvailable": bool(span.get("content_hash") or span.get("contentHash")),
    }


def _span_from_raw(row: JsonMap, *, index: int) -> JsonMap:
    locator = _clean_text(row.get("span_locator") or row.get("spanLocator"), limit=160)
    return {
        "sourceId": _source_id(row.get("source_id") or row.get("citation_target") or row.get("arxiv_id")),
        "citationLabel": _clean_text(row.get("citation_label") or f"S{index}", limit=80),
        "locator": locator,
        "text": _clean_text(row.get("excerpt") or row.get("text")),
        "contentHashAvailable": bool(
            row.get("content_hash")
            or row.get("contentHash")
            or row.get("snippet_hash")
            or row.get("source_content_hash")
        ),
    }


def _prompt_spans(raw: JsonMap) -> list[JsonMap]:
    contract_spans = _as_maps(_as_map(raw.get("evidencePacketContract")).get("spans"))
    spans = [_span_from_contract(span) for span in contract_spans]
    if not spans:
        spans = [_span_from_raw(row, index=index) for index, row in enumerate(_as_maps(raw.get("evidence")), start=1)]
    if not spans:
        spans = [_span_from_raw(row, index=index) for index, row in enumerate(_as_maps(raw.get("sources")), start=1)]
    return [span for span in spans if span.get("sourceId") and span.get("text")]


def _row_report(run: JsonMap, raw: JsonMap) -> JsonMap:
    spans = _prompt_spans(raw)
    packet_source_ids = _unique([_source_id(span.get("sourceId")) for span in spans])
    source_candidate_ids = _candidate_ids(run, raw)
    expected_ids = [_source_id(value) for value in _as_strings(run.get("expectedIds"))]
    missing_ids = [expected_id for expected_id in expected_ids if expected_id not in packet_source_ids]
    citation_ids = _unique([*expected_ids, *[_source_id(citation.get("target")) for citation in _as_maps(raw.get("citations"))]])
    warnings = [f"missing_expected_source_id:{source_id}" for source_id in missing_ids]
    if expected_ids and not spans:
        warnings.append("packet_input_has_no_prompt_spans")
    status = "blocked" if warnings else "ready"
    prompt_packet = {
        "query": _clean_text(run.get("query"), limit=900),
        "expectedSourceIds": expected_ids,
        "spans": spans,
    }
    return {
        "runId": _clean_text(run.get("runId"), limit=160),
        "caseId": _clean_text(run.get("caseId"), limit=120),
        "variantId": _clean_text(run.get("variantId"), limit=120),
        "status": status,
        "spanCount": len(spans),
        "packetSourceIds": packet_source_ids,
        "sourceCandidateIds": source_candidate_ids,
        "citationCandidateIds": citation_ids,
        "missingExpectedSourceIds": missing_ids,
        "promptPacket": prompt_packet,
        "warnings": warnings,
    }


def _contains_marker(row: JsonMap, markers: tuple[str, ...]) -> bool:
    return any(marker in json.dumps(row, ensure_ascii=False, sort_keys=True) for marker in markers)


def build_evidence_packet_input_completeness(
    *,
    manifest: JsonMap,
    raw_payloads: Mapping[str, JsonMap],
    generated_at: str,
) -> JsonMap:
    rows = [_row_report(run, raw_payloads.get(str(run.get("runId") or ""), {})) for run in _as_maps(manifest.get("runs"))]
    private_rows = sum(1 for row in rows if _contains_marker(row, PRIVATE_PATH_MARKERS))
    forbidden_rows = sum(1 for row in rows if _contains_marker(row, FORBIDDEN_MARKERS))
    blocked_rows = sum(1 for row in rows if row["status"] == "blocked")
    return {
        "schema": EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID,
        "status": "blocked" if private_rows or forbidden_rows else "ready",
        "generatedAt": generated_at,
        "profile": "evidence-packet-v1-llm-only-harness",
        "policy": {
            "reportOnly": True,
            "localOnlyRawArtifactRefs": True,
            "publicDefaultPromotionAllowed": False,
        },
        "counts": {
            "rowCount": len(rows),
            "nonEmptyPacketRows": sum(1 for row in rows if int(row["spanCount"]) > 0),
            "blockedRows": blocked_rows,
            "privatePathLeakRows": private_rows,
            "forbiddenRawMarkerRows": forbidden_rows,
        },
        "rows": rows,
        "warnings": [] if not private_rows and not forbidden_rows else ["shareable packet input report contains blocked markers"],
    }


def write_evidence_packet_input_completeness(report: JsonMap, *, report_json: Path) -> dict[str, str]:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"json": str(report_json)}


__all__ = [
    "EVIDENCE_PACKET_INPUT_COMPLETENESS_SCHEMA_ID",
    "build_evidence_packet_input_completeness",
    "write_evidence_packet_input_completeness",
]
