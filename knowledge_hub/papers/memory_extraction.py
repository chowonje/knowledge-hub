"""Internal schema-backed extraction helpers for paper-memory."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from knowledge_hub.papers.memory_models import normalize_final_cause, normalize_formal_cause


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


class PaperMemoryExtractionError(ValueError):
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
class PaperMemoryExtractionV1:
    thesis: str = ""
    problem_context: str = ""
    method_core: str = ""
    evidence_core: str = ""
    claims: list[str] = field(default_factory=list)
    limitations: str = ""
    concept_links: list[str] = field(default_factory=list)
    claim_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    formal_cause: dict[str, Any] = field(default_factory=dict)
    final_cause: dict[str, Any] = field(default_factory=dict)
    quality_flag: str = "unscored"
    coverage_status_by_field: dict[str, str] = field(default_factory=dict)
    field_confidence: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extractor_model: str = "exaone3.5:7.8b"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, default_model: str = "exaone3.5:7.8b") -> "PaperMemoryExtractionV1 | None":
        if not isinstance(raw, dict):
            return None
        thesis = _clean_text(raw.get("thesis") or raw.get("paper_core") or raw.get("paperCore"))
        problem_context = _clean_text(raw.get("problem_context") or raw.get("problemContext"))
        method_core = _clean_text(raw.get("method_core") or raw.get("methodCore"))
        evidence_core = _clean_text(raw.get("evidence_core") or raw.get("evidenceCore"))
        claims = _clean_list(raw.get("claims"), limit=4)
        limitations = _clean_text(raw.get("limitations"))
        concept_links = _clean_list(raw.get("concept_links") or raw.get("conceptLinks"), limit=6)
        claim_refs = _clean_list(raw.get("claim_refs") or raw.get("claimRefs"), limit=8)
        evidence_refs = _clean_list(raw.get("evidence_refs") or raw.get("evidenceRefs"), limit=8)
        formal_cause = normalize_formal_cause(raw.get("formal_cause") or raw.get("formalCause"))
        final_cause = normalize_final_cause(raw.get("final_cause") or raw.get("finalCause"))
        if not any(
            (
                thesis,
                problem_context,
                method_core,
                evidence_core,
                claims,
                limitations,
                concept_links,
                claim_refs,
                evidence_refs,
                formal_cause,
                final_cause,
            )
        ):
            return None
        coverage = {
            str(key): _clean_text(value) or "partial"
            for key, value in _clean_dict(raw.get("coverage_status_by_field") or raw.get("coverageStatusByField")).items()
            if _clean_text(key)
        }
        raw_confidence = _clean_dict(raw.get("field_confidence") or raw.get("fieldConfidence"))
        field_confidence: dict[str, float] = {}
        for key, value in raw_confidence.items():
            token = _clean_text(key)
            if not token:
                continue
            try:
                field_confidence[token] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue
        return cls(
            thesis=thesis,
            problem_context=problem_context,
            method_core=method_core,
            evidence_core=evidence_core,
            claims=claims,
            limitations=limitations,
            concept_links=concept_links,
            claim_refs=claim_refs,
            evidence_refs=evidence_refs,
            formal_cause=formal_cause,
            final_cause=final_cause,
            quality_flag=_clean_text(raw.get("quality_flag") or raw.get("qualityFlag")) or "unscored",
            coverage_status_by_field=coverage,
            field_confidence=field_confidence,
            warnings=_clean_list(raw.get("warnings"), limit=16),
            extractor_model=_clean_text(raw.get("extractor_model") or raw.get("extractorModel")) or default_model,
        )


class PaperMemorySchemaExtractor:
    """LLM-backed internal extractor for paper-memory payloads."""

    schema = "knowledge-hub.paper-memory-extraction.v2"

    def __init__(self, llm: Any, *, model: str = "exaone3.5:7.8b") -> None:
        self.llm = llm
        self.model = _clean_text(model) or "exaone3.5:7.8b"

    def extract(self, *, paper: dict[str, Any]) -> dict[str, Any]:
        payload, _ = self.extract_with_metadata(paper=paper)
        return payload

    def build_prompt(self, *, paper: dict[str, Any]) -> str:
        return self._prompt(paper=paper)

    def coerce_payload(self, raw: Any) -> dict[str, Any]:
        return self._coerce_payload(raw)

    def extract_with_metadata(self, *, paper: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = self._prompt(paper=paper)
        raw = self.llm.generate(prompt, max_tokens=700)
        try:
            payload = self._coerce_payload(raw)
        except PaperMemoryExtractionError:
            raise
        metadata = {
            "rawPayloadBytes": len(str(raw or "").encode("utf-8")),
            "parsedFields": sorted(payload.keys()),
        }
        return payload, metadata

    def _coerce_payload(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        body = str(raw or "").strip()
        if not body:
            raise PaperMemoryExtractionError(
                "paper-memory extraction payload must be a non-empty object",
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
        raise PaperMemoryExtractionError(
            "invalid paper-memory extraction payload: could not parse JSON object",
            raw_preview=_preview_text(body),
            raw_payload_bytes=len(body.encode("utf-8")),
            parse_stage="json_parse",
        )

    def _prompt(self, *, paper: dict[str, Any]) -> str:
        payload = json.dumps(paper, ensure_ascii=False)
        return f"""You are extracting an internal schema-backed paper memory payload.
Return one JSON object only. Do not include markdown, explanations, or commentary.

Schema:
{{
  "thesis": "string",
  "problemContext": "string",
  "methodCore": "string",
  "evidenceCore": "string",
  "claims": ["string"],
  "limitations": "string",
  "conceptLinks": ["string"],
  "formalCause": {{
    "summary": "string",
    "basis": "author_stated|inferred|mixed|missing",
    "confidence": 0.0,
    "coverage": "missing|partial|complete",
    "evidenceRefs": ["string"],
    "warnings": ["string"]
  }},
  "finalCause": {{
    "authorStatedSummary": "string",
    "inferredSummary": "string",
    "basis": "author_stated|inferred|mixed|missing",
    "confidence": 0.0,
    "coverage": "missing|partial|complete",
    "evidenceRefs": ["string"],
    "warnings": ["string"]
  }},
  "qualityFlag": "ok|needs_review|unscored",
  "coverageStatusByField": {{
    "thesis": "missing|partial|complete",
    "problemContext": "missing|partial|complete",
    "methodCore": "missing|partial|complete",
    "evidenceCore": "missing|partial|complete",
    "claims": "missing|partial|complete",
    "limitations": "missing|partial|complete",
    "conceptLinks": "missing|partial|complete",
    "formalCause": "missing|partial|complete",
    "finalCause": "missing|partial|complete"
  }},
  "fieldConfidence": {{
    "thesis": 0.0,
    "problemContext": 0.0,
    "methodCore": 0.0,
    "evidenceCore": 0.0,
    "claims": 0.0,
    "limitations": 0.0,
    "conceptLinks": 0.0,
    "formalCause": 0.0,
    "finalCause": 0.0
  }},
  "warnings": ["string"],
  "extractorModel": "{self.model}"
}}

Rules:
- Use only evidence present in the input.
- Keep `claims` to at most 3 short bullets.
- Keep `conceptLinks` to at most 5 items.
- Keep `problemContext` focused on the paper's problem setting, motivation, or task framing. Do not copy author footnotes, affiliation metadata, or raw LaTeX.
- Keep `methodCore` and `evidenceCore` to short retrieval-friendly phrases, not paragraphs.
- Prefer `evidenceCore` that preserves concrete benchmark names, metrics, gains, or comparison outcomes when they are visible.
- If no explicit quantitative or benchmark evidence is visible, keep `evidenceCore` conservative and short instead of generalizing.
- `formalCause.summary` should capture the paper's defining structure, formulation, architecture, or organizing approach only when that structure is visible in the input. Prefer author-stated structure; allow `basis=inferred` only for strong method/formulation evidence.
- `finalCause.authorStatedSummary` should contain only an explicit author-stated objective, purpose, task, or motivation. `finalCause.inferredSummary` may contain a conservative inferred purpose, but do not copy inference into `authorStatedSummary`.
- If `finalCause` is only inferred, keep `authorStatedSummary` empty.
- Leave weak or unsupported cause fields empty instead of speculating.
- For `limitations`, only report explicit limitations supported by the input excerpt. If limitations are not explicit, return exactly `limitations not explicit in visible excerpt`.
- Leave weak fields empty instead of inventing content.
- Prefer short retrieval-friendly phrases, not paragraphs.

Input:
{payload}
"""


__all__ = ["PaperMemoryExtractionError", "PaperMemoryExtractionV1", "PaperMemorySchemaExtractor"]
