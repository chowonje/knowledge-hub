"""Internal schema-backed extraction helpers for document-memory."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except Exception:
            candidates = [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _clean_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _preview_text(value: Any, *, limit: int = 240) -> str:
    body = str(value or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit].rstrip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


class DocumentMemoryExtractionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_preview: str = "",
        raw_payload_bytes: int = 0,
        parse_stage: str = "parse",
    ) -> None:
        super().__init__(message)
        self.raw_preview = raw_preview
        self.raw_payload_bytes = int(raw_payload_bytes or 0)
        self.parse_stage = _clean_text(parse_stage) or "parse"


@dataclass
class DocumentSectionExtractionV1:
    title: str = ""
    section_path: str = ""
    unit_type: str = "section"
    contextual_summary: str = ""
    source_excerpt: str = ""
    evidence_spans: list[dict[str, Any]] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    field_confidence: float = 0.0
    coverage_status: str = "partial"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "DocumentSectionExtractionV1 | None":
        if not isinstance(raw, dict):
            return None
        title = _clean_text(raw.get("title"))
        section_path = _clean_text(raw.get("section_path") or raw.get("sectionPath"))
        if not title and not section_path:
            return None
        evidence_spans = []
        for item in list(raw.get("evidence_spans") or raw.get("evidenceSpans") or []):
            if isinstance(item, dict):
                cleaned = {str(key): value for key, value in item.items()}
                if cleaned:
                    evidence_spans.append(cleaned)
        return cls(
            title=title,
            section_path=section_path,
            unit_type=_clean_text(raw.get("unit_type") or raw.get("unitType")) or "section",
            contextual_summary=_clean_text(raw.get("contextual_summary") or raw.get("contextualSummary")),
            source_excerpt=_clean_text(raw.get("source_excerpt") or raw.get("sourceExcerpt")),
            evidence_spans=evidence_spans[:8],
            claims=_clean_list(raw.get("claims"), limit=12),
            concepts=_clean_list(raw.get("concepts"), limit=12),
            field_confidence=_safe_float(raw.get("field_confidence") or raw.get("fieldConfidence"), 0.0),
            coverage_status=_clean_text(raw.get("coverage_status") or raw.get("coverageStatus")) or "partial",
        )


@dataclass
class DocumentMemoryExtractionV1:
    document_thesis: str = ""
    summary_unit: DocumentSectionExtractionV1 | None = None
    section_units: list[DocumentSectionExtractionV1] = field(default_factory=list)
    top_claims: list[str] = field(default_factory=list)
    core_concepts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage_status: str = "partial"
    extractor_model: str = "exaone3.5:7.8b"

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any] | None,
        *,
        default_model: str = "exaone3.5:7.8b",
    ) -> "DocumentMemoryExtractionV1 | None":
        if not isinstance(raw, dict):
            return None
        summary_unit = DocumentSectionExtractionV1.from_dict(raw.get("summary_unit") or raw.get("summaryUnit"))
        sections: list[DocumentSectionExtractionV1] = []
        for item in list(raw.get("section_units") or raw.get("sectionUnits") or []):
            parsed = DocumentSectionExtractionV1.from_dict(item)
            if parsed is not None:
                sections.append(parsed)
        if not _clean_text(raw.get("document_thesis") or raw.get("documentThesis")) and not summary_unit and not sections:
            return None
        return cls(
            document_thesis=_clean_text(raw.get("document_thesis") or raw.get("documentThesis")),
            summary_unit=summary_unit,
            section_units=sections[:24],
            top_claims=_clean_list(raw.get("top_claims") or raw.get("topClaims"), limit=16),
            core_concepts=_clean_list(raw.get("core_concepts") or raw.get("coreConcepts"), limit=16),
            warnings=_clean_list(raw.get("warnings"), limit=16),
            coverage_status=_clean_text(raw.get("coverage_status") or raw.get("coverageStatus")) or "partial",
            extractor_model=_clean_text(raw.get("extractor_model") or raw.get("extractorModel")) or default_model,
        )


class DocumentMemorySchemaExtractor:
    """LLM-backed internal extractor for document-memory payloads."""

    schema = "knowledge-hub.document-memory-extraction.v1"

    def __init__(self, llm: Any, *, model: str = "exaone3.5:7.8b") -> None:
        self.llm = llm
        self.model = _clean_text(model) or "exaone3.5:7.8b"

    def extract(self, *, document: dict[str, Any], compact: bool = False) -> dict[str, Any]:
        payload, _ = self.extract_with_metadata(document=document, compact=compact)
        return payload

    def extract_with_metadata(
        self,
        *,
        document: dict[str, Any],
        compact: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = self._prompt(document=document, compact=compact)
        raw = self.llm.generate(prompt, max_tokens=900)
        payload = self._coerce_payload(raw)
        metadata = {
            "rawPayloadBytes": len(str(raw or "").encode("utf-8")),
            "parsedFields": sorted(payload.keys()),
            "promptMode": "compact" if compact else "default",
        }
        return payload, metadata

    def _coerce_payload(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        body = str(raw or "").strip()
        if not body:
            raise DocumentMemoryExtractionError(
                "document-memory extraction payload must be a non-empty object",
                raw_preview="",
                raw_payload_bytes=0,
                parse_stage="empty",
            )
        body = re.sub(r"<think>[\s\S]*?</think>", " ", body, flags=re.IGNORECASE).strip()
        candidates = [body]
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", body, re.IGNORECASE)
        if fenced:
            candidates.append(fenced.group(1).strip())
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            candidates.append(body[start : end + 1])
        seen: set[str] = set()
        for candidate in candidates:
            token = str(candidate or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            try:
                parsed = json.loads(token)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise DocumentMemoryExtractionError(
            "invalid document-memory extraction payload: could not parse JSON object",
            raw_preview=_preview_text(body),
            raw_payload_bytes=len(body.encode("utf-8")),
            parse_stage="json_parse",
        )

    def _prompt(self, *, document: dict[str, Any], compact: bool = False) -> str:
        payload = json.dumps(document, ensure_ascii=False)
        section_limit = 3 if compact else 6
        return f"""You are extracting an internal schema-backed document memory payload.
Return one JSON object only. Do not include markdown, explanations, or commentary.

Schema:
{{
  "documentThesis": "string",
  "summaryUnit": {{
    "title": "string",
    "sectionPath": "string",
    "unitType": "document_summary|summary|section|method|result|limitation|background|list_block|table_block|image_block",
    "contextualSummary": "string",
    "sourceExcerpt": "string",
    "evidenceSpans": [{{"label": "string", "quote": "string"}}],
    "claims": ["string"],
    "concepts": ["string"],
    "fieldConfidence": 0.0,
    "coverageStatus": "missing|partial|complete"
  }},
  "sectionUnits": [<same section object without requiring every field>],
  "topClaims": ["string"],
  "coreConcepts": ["string"],
  "warnings": ["string"],
  "coverageStatus": "missing|partial|complete",
  "extractorModel": "{self.model}"
}}

Rules:
- Use only evidence present in the input.
- Prefer short retrieval-friendly phrases, not paragraphs.
- Keep `topClaims` and per-section `claims` to at most 3 items.
- Keep `coreConcepts` and per-section `concepts` to at most 5 items.
- Keep `sectionUnits` to at most {section_limit} sections.
- Leave weak fields empty instead of inventing content.
{"- Recovery mode: prioritize a non-empty `documentThesis` and `summaryUnit` over rich section coverage." if compact else ""}

Input:
{payload}
"""


__all__ = [
    "DocumentMemoryExtractionError",
    "DocumentMemoryExtractionV1",
    "DocumentMemorySchemaExtractor",
    "DocumentSectionExtractionV1",
]
