"""Build compact paper memory cards from existing paper artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_DIR
from knowledge_hub.papers.memory_extraction import PaperMemoryExtractionError, PaperMemoryExtractionV1
from knowledge_hub.papers.memory_models import PaperMemoryCard
from knowledge_hub.papers.text_sanitizer import PaperTextNormalization, extract_keyword_window, normalize_paper_texts

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_EVIDENCE_MARKER_RE = re.compile(
    r"\b(?:outperform(?:s|ed)?|improv(?:e|es|ed)|achiev(?:e|es|ed)|gain(?:s|ed)?|accuracy|fid|auc|auprc|auroc|benchmark|results?|score|scores|evaluation|experiment(?:s)?|ablation|win(?:s|ning)?)\b",
    re.IGNORECASE,
)
_LIMITATION_MARKER_RE = re.compile(
    r"\b(?:limitation(?:s)?|future work|we do not|we don't|cannot|fails to|restricted to|limited to|only evaluated|only tested|not explicit|한계)\b",
    re.IGNORECASE,
)
_LATEX_MARKERS = (
    "\\documentclass",
    "\\usepackage",
    "\\begin{document}",
    "\\maketitle",
    "\\thanks{",
    "\\title{",
    "\\author{",
    "\\hypersetup",
    "\\includepdf",
    "\\section{",
    "\\subsection{",
    "\\paragraph{",
    "\\begin{abstract}",
)
_SUMMARY_REFUSAL_PATTERNS = (
    "원문",
    "전문 텍스트",
    "직접 확인할 수 없어",
    "현재 주신 정보",
    "제목·저자뿐",
    "구체적 수치",
    "가상의 요약",
    "가정에 기반",
    "실제 논문의 내용을 반영하지 않습니다",
    "insufficient information",
    "not enough information",
    "paper text is required",
    "need the paper text",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _iso_utc(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    if re.fullmatch(r"\d{4}", token):
        return datetime(int(token), 1, 1, tzinfo=timezone.utc).isoformat()
    return ""


def _clean_lines(values: list[str], *, limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
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


def _first_nonempty(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return ""


def _read_text(path_value: str) -> str:
    path = Path(str(path_value or "").strip())
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _contains_latex_markup(text: Any) -> bool:
    lowered = str(text or "").casefold()
    return any(marker in lowered for marker in _LATEX_MARKERS)


def _paper_storage_roots(paper: dict[str, Any] | None) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        token = str(path)
        if not token or token in seen:
            return
        seen.add(token)
        result.append(path)

    for key in ("translated_path", "text_path", "pdf_path"):
        raw = str((paper or {}).get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        _add(path.parent)
        for ancestor in path.parents:
            _add(ancestor)
            if ancestor.name.casefold() in {
                "translated",
                "translation",
                "translations",
                "text",
                "texts",
                "raw",
                "pdf",
                "pdfs",
                "downloads",
                "summaries",
                "parsed",
            }:
                _add(ancestor.parent)
    _add((DEFAULT_CONFIG_DIR / "papers").expanduser())
    return result


def _load_structured_summary_artifact(*, paper: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    token = str(paper_id or "").strip()
    if not token:
        return None
    for root in _paper_storage_roots(paper):
        path = root / "summaries" / token / "summary.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and _structured_summary_payload_is_trustworthy(payload):
            return payload
    return None


def _load_parsed_markdown_artifact(*, paper: dict[str, Any], paper_id: str) -> str:
    token = str(paper_id or "").strip()
    if not token:
        return ""
    for root in _paper_storage_roots(paper):
        path = root / "parsed" / token / "document.md"
        if not path.exists():
            continue
        text = _read_text(str(path))
        if _clean_text(text):
            return text
    return ""


def _summary_value_is_unusable(value: Any) -> bool:
    token = _clean_text(_strip_markdown(value))
    if not token:
        return True
    lowered = token.casefold()
    if _contains_latex_markup(lowered):
        return True
    return any(pattern in lowered for pattern in _SUMMARY_REFUSAL_PATTERNS)


def _structured_summary_payload_is_trustworthy(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    status = _clean_text(payload.get("status") or "ok").casefold()
    if status and status not in {"ok", "success"}:
        return False
    parser_used = _clean_text(payload.get("parserUsed") or payload.get("parser_used") or "").casefold()
    fallback_used = bool(payload.get("fallbackUsed") or payload.get("fallback_used"))
    diagnostics = dict(payload.get("documentMemoryDiagnostics") or {})
    structured_sections = int(diagnostics.get("structuredSectionsDetected") or 0)
    parse_artifact_path = _clean_text(diagnostics.get("parseArtifactPath") or "")
    if parser_used == "raw" and fallback_used and structured_sections <= 0 and not parse_artifact_path:
        return False
    return True


def _summary_values(summary_payload: dict[str, Any] | None, *field_names: str, limit: int = 4) -> list[str]:
    summary = dict((summary_payload or {}).get("summary") or {})
    values: list[str] = []
    for field_name in field_names:
        raw = summary.get(field_name)
        if isinstance(raw, list):
            values.extend(str(item or "") for item in raw)
        else:
            values.append(str(raw or ""))
    cleaned = []
    for value in values:
        excerpt = _context_excerpt(_strip_markdown(value), limit=420)
        if _summary_value_is_unusable(excerpt):
            continue
        cleaned.append(excerpt)
    return _clean_lines(cleaned, limit=limit)


def _structured_summary_slot_values(summary_payload: dict[str, Any] | None) -> dict[str, Any]:
    paper_core_values = _summary_values(summary_payload, "oneLine", "coreIdea", "whatIsNew", limit=3)
    problem_values = _summary_values(summary_payload, "problem", limit=2)
    method_values = _summary_values(summary_payload, "coreIdea", "methodSteps", limit=4)
    evidence_values = _summary_values(summary_payload, "keyResults", "whenItMatters", limit=4)
    limitation_values = _summary_values(summary_payload, "limitations", limit=3)
    return {
        "paper_core": _cap_join(paper_core_values, limit=520),
        "problem_context": _cap_join(problem_values or paper_core_values[:1], limit=520),
        "method_core": _cap_join(method_values or paper_core_values[:1], limit=520),
        "evidence_core": _cap_join(evidence_values, limit=520),
        "limitations": _cap_join(limitation_values, limit=420),
        "search_terms": _clean_lines(
            paper_core_values + problem_values + method_values[:2] + evidence_values[:2] + limitation_values[:1],
            limit=8,
        ),
    }


def _parsed_markdown_slot_values(parsed_markdown: str) -> dict[str, Any]:
    text = _clean_text(parsed_markdown)
    if not text:
        return {
            "paper_core": "",
            "problem_context": "",
            "method_core": "",
            "evidence_core": "",
            "limitations": "",
            "search_terms": [],
        }
    sections = _split_sections(parsed_markdown)
    stripped = _strip_markdown(parsed_markdown)
    abstract_text = _find_section(sections, "abstract", "요약")
    introduction_text = _find_section(sections, "introduction", "background", "motivation", "problem", "문제")
    method_text = _find_section(sections, "method", "approach", "algorithm", "training", "architecture", "방법", "접근")
    result_text = _find_section(sections, "result", "evaluation", "experiment", "finding", "evidence", "결과")
    limitation_text = _find_section(sections, "limitation", "future work", "한계")
    paper_core = _first_nonempty(
        _cap_join(_extract_sentences(_first_nonempty(abstract_text, stripped), limit=2), limit=520),
        _context_excerpt(stripped, limit=520),
    )
    problem_context = _first_nonempty(
        _context_excerpt(_first_nonempty(abstract_text, introduction_text), limit=520),
        extract_keyword_window(
            stripped,
            ("problem", "challenge", "task", "goal", "objective", "motivation", "background", "문제", "과제", "목표"),
            limit=520,
        ),
        paper_core,
    )
    method_core = _first_nonempty(
        _context_excerpt(method_text, limit=520),
        extract_keyword_window(
            stripped,
            ("method", "approach", "algorithm", "training", "architecture", "optimization", "방법", "접근"),
            limit=520,
        ),
        paper_core,
    )
    evidence_core = _first_nonempty(
        _context_excerpt(result_text, limit=520),
        " ".join(_extract_evidence_sentences(stripped, limit=3)),
        extract_keyword_window(
            stripped,
            ("result", "results", "evaluation", "experiment", "benchmark", "outperform", "BLEU", "accuracy", "결과"),
            limit=520,
        ),
    )
    limitations = _first_nonempty(
        _context_excerpt(limitation_text, limit=420),
        extract_keyword_window(
            stripped,
            ("limitation", "limitations", "future work", "restricted", "only", "한계"),
            limit=420,
        ),
    )
    return {
        "paper_core": _first_usable(paper_core),
        "problem_context": _first_usable(problem_context),
        "method_core": _first_usable(method_core),
        "evidence_core": _first_usable(evidence_core),
        "limitations": _first_usable(limitations),
        "search_terms": _clean_lines(
            _extract_sentences(paper_core, limit=2)
            + _extract_sentences(problem_context, limit=2)
            + _section_headings(sections, limit=6),
            limit=8,
        ),
    }


def _prefer_structured_slot(structured_value: Any, fallback_value: Any) -> str:
    preferred = _clean_text(structured_value)
    if preferred:
        return preferred
    return _clean_text(fallback_value)


def _first_usable(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if not token:
            continue
        if _summary_value_is_unusable(token):
            continue
        return token
    return ""


def _strip_markdown(text: str) -> str:
    body = str(text or "")
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    body = re.sub(r"^#{1,6}\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\s*[-*]\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"\n{2,}", "\n", body)
    return _clean_text(body)


def _split_sections(markdown: str) -> dict[str, str]:
    text = str(markdown or "")
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = _clean_text(match.group(1)).casefold()
        sections[title] = text[start:end].strip()
    return sections


def _extract_sentences(text: str, *, limit: int = 2) -> list[str]:
    body = _clean_text(_strip_markdown(text))
    if not body:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(body)
    return _clean_lines(sentences, limit=limit)


def _extract_evidence_sentences(text: str, *, limit: int = 3) -> list[str]:
    sentences = _extract_sentences(text, limit=max(limit * 6, 8))
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        token = _clean_text(sentence)
        if not token:
            continue
        score = 0
        if re.search(r"\d", token):
            score += 2
        if _EVIDENCE_MARKER_RE.search(token):
            score += 2
        if "%" in token:
            score += 1
        if score <= 0:
            continue
        ranked.append((score, token))
    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return _clean_lines([sentence for _, sentence in ranked], limit=limit)


def _find_section(sections: dict[str, str], *tokens: str) -> str:
    for key, value in sections.items():
        if any(token in key for token in tokens):
            return _strip_markdown(value)
    return ""


def _section_headings(sections: dict[str, str], *, limit: int = 8) -> list[str]:
    return _clean_lines([str(key or "") for key in sections.keys()], limit=limit)


def _claim_ids(rows: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    ordered = sorted(rows, key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
    for row in ordered:
        claim_id = _clean_text(row.get("claim_id"))
        if not claim_id or claim_id in seen:
            continue
        seen.add(claim_id)
        result.append(claim_id)
        if len(result) >= limit:
            break
    return result


def _claim_texts(rows: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    ordered = sorted(rows, key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
    return _clean_lines([str(row.get("claim_text") or "") for row in ordered], limit=limit)


def _concept_names(rows: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    values = [str(row.get("canonical_name") or row.get("id") or row.get("entity_id") or "") for row in rows]
    return _clean_lines(values, limit=limit)


def _quality_flag(note: dict[str, Any] | None, claims: list[dict[str, Any]]) -> str:
    metadata: dict[str, Any] = {}
    if note and isinstance(note.get("metadata"), str):
        try:
            metadata = json.loads(note.get("metadata") or "{}")
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
    elif note and isinstance(note.get("metadata"), dict):
        metadata = dict(note.get("metadata") or {})
    explicit = _clean_text(metadata.get("quality_flag") or (metadata.get("quality") or {}).get("flag"))
    if explicit:
        return explicit
    strong_claims = [row for row in claims if float(row.get("confidence") or 0.0) >= 0.8]
    if note and len(strong_claims) >= 2:
        return "ok"
    if claims:
        return "needs_review"
    return "unscored"


def _cap_join(parts: list[str], *, limit: int = 800) -> str:
    body = " ".join(_clean_lines(parts))
    if len(body) <= limit:
        return body
    return body[:limit].rstrip()


def _context_excerpt(text: str, *, limit: int = 2200) -> str:
    body = _clean_text(_strip_markdown(text))
    if len(body) <= limit:
        return body
    return body[:limit].rstrip()


def _trim_section_payload(sections: dict[str, str], *, section_limit: int = 8, text_limit: int = 400) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in list((sections or {}).items())[:section_limit]:
        title = _clean_text(key)
        if not title:
            continue
        excerpt = _context_excerpt(value, limit=text_limit)
        if excerpt:
            result[title] = excerpt
    return result


def _fixed_slot_excerpt(note_sections: dict[str, str], *tokens: str, limit: int = 500) -> str:
    return _context_excerpt(_find_section(note_sections, *tokens), limit=limit)


def _problem_excerpt(note_sections: dict[str, str], preferred_text: str, *, limit: int = 420) -> str:
    return _first_nonempty(
        _fixed_slot_excerpt(note_sections, "abstract", "introduction", "motivation", "background", "problem", "요약", "문제", limit=limit),
        extract_keyword_window(
            preferred_text,
            ("problem", "challenge", "task", "goal", "objective", "motivation", "background", "문제", "과제", "목표"),
            limit=limit,
        ),
        _context_excerpt(preferred_text, limit=limit),
    )


def _has_explicit_limitation_support(*parts: Any) -> bool:
    for part in parts:
        token = _clean_text(part)
        if not token:
            continue
        if len(token) >= 24:
            return True
        if _LIMITATION_MARKER_RE.search(token):
            return True
    return False


def _normalize_limitations_value(value: str, *, has_explicit_support: bool) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    if has_explicit_support:
        return token
    return "limitations not explicit in visible excerpt"


def _coverage_status_from_value(value: Any) -> str:
    token = _clean_text(value)
    return "complete" if token else "missing"


def _latency_bucket(latency_ms: float) -> str:
    if latency_ms < 1500:
        return "fast"
    if latency_ms < 5000:
        return "warm"
    if latency_ms < 15000:
        return "slow"
    return "very_slow"


def _stable_relation_id(src_form: str, src_id: str, dst_form: str, dst_id: str, relation_type: str) -> str:
    digest = hashlib.sha1(f"{src_form}|{src_id}|{dst_form}|{dst_id}|{relation_type}".encode("utf-8")).hexdigest()[:16]
    return f"memory-relation:{relation_type}:{digest}"


def _memory_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9A-Za-z가-힣]+", str(text or "").casefold())
        if len(token) >= 3 and token not in {"paper", "summary", "method", "result", "results"}
    }


def _memory_overlap(left: str, right: str) -> float:
    left_tokens = _memory_tokens(left)
    right_tokens = _memory_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _update_markers(*parts: Any) -> bool:
    haystack = " ".join(str(part or "").casefold() for part in parts)
    return any(token in haystack for token in ("updated", "revised", "benchmark update", " v2", " v3", "version 2", "version 3"))


class PaperMemoryBuilder:
    def __init__(self, sqlite_db, *, schema_extractor: Any | None = None, extraction_mode: str = "deterministic"):
        self.sqlite_db = sqlite_db
        self._schema_extractor = schema_extractor
        self._extraction_mode = str(extraction_mode or "deterministic").strip().lower()
        self._last_extraction_diagnostics: dict[str, dict[str, Any]] = {}

    def get_last_extraction_diagnostics(self, paper_id: str) -> dict[str, Any]:
        return dict(self._last_extraction_diagnostics.get(str(paper_id or "").strip(), {}))

    def _note_for_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self.sqlite_db.get_note(f"paper:{paper_id}")

    def _claims_for_paper(self, paper_id: str, note_id: str) -> list[dict[str, Any]]:
        claims = []
        if note_id:
            claims.extend(self.sqlite_db.list_claims_by_note(note_id, limit=50))
        entity_id = f"paper:{paper_id}"
        claims.extend(self.sqlite_db.list_claims_by_entity(entity_id, limit=50))
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for row in claims:
            claim_id = _clean_text(row.get("claim_id"))
            if not claim_id or claim_id in seen:
                continue
            seen.add(claim_id)
            result.append(row)
        return result

    def build_card(self, *, paper_id: str) -> PaperMemoryCard:
        paper = self.sqlite_db.get_paper(paper_id)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")
        note = self._note_for_paper(paper_id)
        note_id = str((note or {}).get("id") or f"paper:{paper_id}")
        note_content = str((note or {}).get("content") or "")
        note_sections = _split_sections(note_content)
        abstract_text = _first_usable(_find_section(note_sections, "abstract"))
        summary_text = _first_usable(_find_section(note_sections, "요약", "summary"))
        claims = self._claims_for_paper(paper_id, note_id if note else "")
        claim_texts = _claim_texts(claims, limit=4)
        concepts = self.sqlite_db.get_paper_concepts(paper_id)
        concept_links = _concept_names(concepts, limit=8)
        section_headings = _section_headings(note_sections, limit=8)
        structured_summary = _structured_summary_slot_values(_load_structured_summary_artifact(paper=paper, paper_id=paper_id))
        parsed_markdown_slots = _parsed_markdown_slot_values(_load_parsed_markdown_artifact(paper=paper, paper_id=paper_id))

        translated_text = _read_text(str(paper.get("translated_path") or ""))
        raw_text = _read_text(str(paper.get("text_path") or ""))
        text_normalization = normalize_paper_texts(translated_text=translated_text, raw_text=raw_text)
        translated_summary = " ".join(_extract_sentences(text_normalization.translated.sanitized_text, limit=3))
        raw_summary = " ".join(_extract_sentences(text_normalization.raw.sanitized_text, limit=3))

        paper_core = _first_nonempty(
            structured_summary.get("paper_core"),
            parsed_markdown_slots.get("paper_core"),
            summary_text,
            claim_texts[0] if claim_texts else "",
            translated_summary,
            raw_summary,
            _first_usable(paper.get("notes")),
            paper.get("title"),
        )
        problem_context = _first_nonempty(
            structured_summary.get("problem_context"),
            parsed_markdown_slots.get("problem_context"),
            abstract_text,
            structured_summary.get("paper_core"),
            parsed_markdown_slots.get("paper_core"),
            translated_summary,
            raw_summary,
            paper.get("field"),
        )
        method_core = _first_nonempty(
            structured_summary.get("method_core"),
            parsed_markdown_slots.get("method_core"),
            _first_usable(_find_section(note_sections, "method", "방법", "접근")),
            claim_texts[1] if len(claim_texts) > 1 else "",
            translated_summary,
        )
        evidence_core = _first_nonempty(
            structured_summary.get("evidence_core"),
            parsed_markdown_slots.get("evidence_core"),
            _first_usable(_find_section(note_sections, "result", "finding", "evidence", "결과")),
            " ".join(claim_texts[:2]),
            abstract_text,
            translated_summary,
        )
        limitations = _first_nonempty(
            structured_summary.get("limitations"),
            parsed_markdown_slots.get("limitations"),
            _first_usable(_find_section(note_sections, "limitation", "한계")),
            claim_texts[2] if len(claim_texts) > 2 else "",
        )
        search_text = _cap_join(
            [
                str(paper.get("title") or ""),
                str((note or {}).get("title") or ""),
                str((note or {}).get("id") or ""),
                str(paper.get("field") or ""),
                str(paper.get("year") or ""),
                " ".join(section_headings),
                paper_core,
                problem_context,
                method_core,
                evidence_core,
                limitations,
                " ".join(claim_texts[:4]),
                " ".join(concept_links[:5]),
                " ".join(structured_summary.get("search_terms") or []),
                " ".join(parsed_markdown_slots.get("search_terms") or []),
            ],
            limit=900,
        )

        memory_id = f"paper-memory:{paper_id}:{hashlib.sha1(paper_id.encode('utf-8')).hexdigest()[:10]}"
        published_at = _iso_utc(paper.get("published_at") or paper.get("year"))
        card = PaperMemoryCard(
            memory_id=memory_id,
            paper_id=paper_id,
            source_note_id=str((note or {}).get("id") or ""),
            title=str(paper.get("title") or paper_id),
            paper_core=paper_core,
            problem_context=problem_context,
            method_core=method_core,
            evidence_core=evidence_core,
            limitations=limitations,
            concept_links=concept_links,
            claim_refs=_claim_ids(claims, limit=8),
            published_at=published_at,
            evidence_window=published_at,
            search_text=search_text,
            quality_flag=_quality_flag(note, claims),
        )
        return self._apply_schema_extraction(
            paper_id=paper_id,
            card=card,
            paper=paper,
            note=note,
            note_sections=note_sections,
            claims=claims,
            concepts=concepts,
            translated_text=translated_text,
            raw_text=raw_text,
            text_normalization=text_normalization,
        )

    def build_compact_extraction_input(self, *, paper_id: str) -> dict[str, Any]:
        paper = self.sqlite_db.get_paper(paper_id)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")
        note = self._note_for_paper(paper_id)
        note_id = str((note or {}).get("id") or f"paper:{paper_id}")
        note_content = str((note or {}).get("content") or "")
        note_sections = _split_sections(note_content)
        claims = self._claims_for_paper(paper_id, note_id if note else "")
        concepts = self.sqlite_db.get_paper_concepts(paper_id)
        structured_summary = _structured_summary_slot_values(_load_structured_summary_artifact(paper=paper, paper_id=paper_id))
        parsed_markdown_slots = _parsed_markdown_slot_values(_load_parsed_markdown_artifact(paper=paper, paper_id=paper_id))
        translated_text = _read_text(str(paper.get("translated_path") or ""))
        raw_text = _read_text(str(paper.get("text_path") or ""))
        text_normalization = normalize_paper_texts(translated_text=translated_text, raw_text=raw_text)
        card = self.build_card(paper_id=paper_id)
        sanitized_translated_text = text_normalization.translated.sanitized_text
        sanitized_raw_text = text_normalization.raw.sanitized_text
        preferred_text = text_normalization.preferred_text
        explicit_limitations_excerpt = _first_nonempty(
            _fixed_slot_excerpt(note_sections, "limitation", "한계", limit=320),
            extract_keyword_window(preferred_text, ("limitation", "limitations", "future work", "limited", "한계"), limit=320),
        )
        has_explicit_limitations = bool(_clean_text(explicit_limitations_excerpt)) or _has_explicit_limitation_support(card.limitations)
        problem_excerpt = _problem_excerpt(note_sections, preferred_text, limit=420)
        return {
            "paperId": paper_id,
            "title": str(paper.get("title") or card.title or ""),
            "field": str(paper.get("field") or ""),
            "year": str(paper.get("year") or ""),
            "summaryExcerpt": _first_nonempty(
                structured_summary.get("paper_core"),
                parsed_markdown_slots.get("paper_core"),
                _fixed_slot_excerpt(note_sections, "요약", "summary", "abstract", limit=520),
                _context_excerpt(preferred_text, limit=520),
                _context_excerpt(sanitized_translated_text, limit=520),
                _context_excerpt(sanitized_raw_text, limit=520),
            ),
            "problemExcerpt": _first_nonempty(
                structured_summary.get("problem_context"),
                parsed_markdown_slots.get("problem_context"),
                problem_excerpt,
            ),
            "methodExcerpt": _first_nonempty(
                structured_summary.get("method_core"),
                parsed_markdown_slots.get("method_core"),
                _fixed_slot_excerpt(note_sections, "method", "방법", "접근", limit=420),
                extract_keyword_window(preferred_text, ("method", "approach", "architecture", "training", "방법", "접근"), limit=420),
            ),
            "findingsExcerpt": _first_nonempty(
                structured_summary.get("evidence_core"),
                parsed_markdown_slots.get("evidence_core"),
                _fixed_slot_excerpt(note_sections, "result", "finding", "결과", "evidence", limit=420),
                " ".join(_extract_evidence_sentences(preferred_text, limit=3)),
                extract_keyword_window(preferred_text, ("result", "results", "finding", "evaluation", "experiment", "결과"), limit=420),
                _context_excerpt(preferred_text, limit=420),
            ),
            "limitationsExcerpt": _first_nonempty(
                explicit_limitations_excerpt,
                structured_summary.get("limitations"),
                parsed_markdown_slots.get("limitations"),
            ),
            "topConceptCandidates": _concept_names(concepts, limit=5),
            "claimTexts": _claim_texts(claims, limit=3),
            "textSanitation": text_normalization.to_dict(),
            "deterministicBaseline": {
                "paperCore": card.paper_core,
                "problemContext": card.problem_context,
                "methodCore": card.method_core,
                "evidenceCore": card.evidence_core,
                "limitations": card.limitations,
                "conceptLinks": list(card.concept_links)[:5],
                "qualityFlag": card.quality_flag,
            },
            "limitationsPolicy": {
                "explicitSupportPresent": has_explicit_limitations,
                "fallbackText": "limitations not explicit in visible excerpt",
            },
        }

    def build_and_store(self, *, paper_id: str) -> dict[str, Any]:
        card = self.build_card(paper_id=paper_id)
        stored = self.sqlite_db.upsert_paper_memory_card(card=card.to_record())
        self._refresh_updates_relations(card=card)
        return stored

    def apply_extraction_payload_and_store(
        self,
        *,
        paper_id: str,
        raw_payload: dict[str, Any],
        extractor_model: str = "",
    ) -> dict[str, Any]:
        card = self.build_card(paper_id=paper_id)
        extraction = PaperMemoryExtractionV1.from_dict(raw_payload, default_model=_clean_text(extractor_model) or "openai-batch")
        if extraction is None:
            raise ValueError(f"invalid extraction payload for paper-memory apply: {paper_id}")
        compact_input = self.build_compact_extraction_input(paper_id=paper_id)
        has_explicit_limitations = bool((compact_input.get("limitationsPolicy") or {}).get("explicitSupportPresent"))
        merged = self._merge_extraction_into_card(
            card=card,
            extraction=extraction,
            has_explicit_limitations=has_explicit_limitations,
        )
        stored = self.sqlite_db.upsert_paper_memory_card(card=merged.to_record())
        self._refresh_updates_relations(card=merged)
        return stored

    def rebuild_all(self, *, limit: int = 5000) -> list[dict[str, Any]]:
        rows = self.sqlite_db.list_papers(limit=max(1, int(limit)))
        result: list[dict[str, Any]] = []
        for row in rows:
            paper_id = _clean_text(row.get("arxiv_id"))
            if not paper_id:
                continue
            result.append(self.build_and_store(paper_id=paper_id))
        return result

    def _refresh_updates_relations(self, *, card: PaperMemoryCard) -> None:
        if not getattr(self.sqlite_db, "list_memory_relations", None):
            return
        self.sqlite_db.delete_memory_relations_for_node(
            form="paper_memory",
            node_id=card.memory_id,
            relation_type="updates",
            direction="src",
        )
        current_date = _iso_utc(card.published_at or "")
        if not current_date:
            return
        current_dt = datetime.fromisoformat(current_date.replace("Z", "+00:00"))
        for row in self.sqlite_db.list_paper_memory_cards(limit=2000):
            other = PaperMemoryCard.from_row(row)
            if other is None or other.paper_id == card.paper_id:
                continue
            other_date = _iso_utc(other.published_at or "")
            if not other_date:
                continue
            other_dt = datetime.fromisoformat(other_date.replace("Z", "+00:00"))
            if current_dt <= other_dt:
                continue
            overlap = _memory_overlap(card.title + " " + card.search_text, other.title + " " + other.search_text)
            if overlap < 0.45:
                continue
            if not (_update_markers(card.title, other.title, card.search_text, other.search_text) or overlap >= 0.7):
                continue
            self.sqlite_db.upsert_memory_relation(
                relation_id=_stable_relation_id("paper_memory", other.memory_id, "paper_memory", card.memory_id, "updates"),
                src_form="paper_memory",
                src_id=other.memory_id,
                dst_form="paper_memory",
                dst_id=card.memory_id,
                relation_type="updates",
                confidence=round(min(0.99, 0.6 + (0.35 * overlap)), 4),
                provenance={
                    "rule": "paper_updates_v1",
                    "older_paper_id": other.paper_id,
                    "newer_paper_id": card.paper_id,
                    "older_date": other_date,
                    "newer_date": current_date,
                    "title_overlap": round(overlap, 4),
                },
            )

    def _apply_schema_extraction(
        self,
        *,
        paper_id: str,
        card: PaperMemoryCard,
        paper: dict[str, Any],
        note: dict[str, Any] | None,
        note_sections: dict[str, str],
        claims: list[dict[str, Any]],
        concepts: list[dict[str, Any]],
        translated_text: str,
        raw_text: str,
        text_normalization: PaperTextNormalization,
    ) -> PaperMemoryCard:
        diagnostics = {
            "mode": self._extraction_mode,
            "attempted": False,
            "applied": False,
            "fallbackUsed": False,
            "schema": "knowledge-hub.paper-memory-extraction.v1",
            "warnings": [],
        }
        diagnostics["textSanitation"] = text_normalization.to_dict()
        if self._extraction_mode not in {"shadow", "schema"} or self._schema_extractor is None:
            self._last_extraction_diagnostics[paper_id] = diagnostics
            return card
        diagnostics["attempted"] = True
        sanitized_translated_text = text_normalization.translated.sanitized_text
        sanitized_raw_text = text_normalization.raw.sanitized_text
        preferred_text = text_normalization.preferred_text
        structured_summary = _structured_summary_slot_values(_load_structured_summary_artifact(paper=paper, paper_id=paper_id))
        parsed_markdown_slots = _parsed_markdown_slot_values(_load_parsed_markdown_artifact(paper=paper, paper_id=paper_id))
        explicit_limitations_excerpt = _first_nonempty(
            _fixed_slot_excerpt(note_sections, "limitation", "한계", limit=320),
            structured_summary.get("limitations"),
            parsed_markdown_slots.get("limitations"),
            extract_keyword_window(preferred_text, ("limitation", "limitations", "future work", "limited", "한계"), limit=320),
        )
        has_explicit_limitations = bool(_clean_text(explicit_limitations_excerpt)) or _has_explicit_limitation_support(card.limitations)
        problem_excerpt = _problem_excerpt(note_sections, preferred_text, limit=420)
        extraction_input = {
            "paperId": paper_id,
            "title": str(paper.get("title") or card.title or ""),
            "field": str(paper.get("field") or ""),
            "year": str(paper.get("year") or ""),
            "summaryExcerpt": _first_nonempty(
                structured_summary.get("paper_core"),
                parsed_markdown_slots.get("paper_core"),
                _fixed_slot_excerpt(note_sections, "요약", "summary", "abstract", limit=520),
                _context_excerpt(preferred_text, limit=520),
                _context_excerpt(sanitized_translated_text, limit=520),
                _context_excerpt(sanitized_raw_text, limit=520),
            ),
            "problemExcerpt": _first_nonempty(
                structured_summary.get("problem_context"),
                parsed_markdown_slots.get("problem_context"),
                problem_excerpt,
            ),
            "methodExcerpt": _first_nonempty(
                structured_summary.get("method_core"),
                parsed_markdown_slots.get("method_core"),
                _fixed_slot_excerpt(note_sections, "method", "방법", "접근", limit=420),
                extract_keyword_window(preferred_text, ("method", "approach", "architecture", "training", "방법", "접근"), limit=420),
            ),
            "findingsExcerpt": _first_nonempty(
                structured_summary.get("evidence_core"),
                parsed_markdown_slots.get("evidence_core"),
                _fixed_slot_excerpt(note_sections, "result", "finding", "결과", "evidence", limit=420),
                " ".join(_extract_evidence_sentences(preferred_text, limit=3)),
                extract_keyword_window(preferred_text, ("result", "results", "finding", "evaluation", "experiment", "결과"), limit=420),
                _context_excerpt(preferred_text, limit=420),
            ),
            "limitationsExcerpt": _first_nonempty(explicit_limitations_excerpt, parsed_markdown_slots.get("limitations")),
            "topConceptCandidates": _concept_names(concepts, limit=5),
            "claimTexts": _claim_texts(claims, limit=3),
            "textSanitation": text_normalization.to_dict(),
            "deterministicBaseline": {
                "paperCore": card.paper_core,
                "problemContext": card.problem_context,
                "methodCore": card.method_core,
                "evidenceCore": card.evidence_core,
                "limitations": card.limitations,
                "conceptLinks": list(card.concept_links)[:5],
                "qualityFlag": card.quality_flag,
            },
            "limitationsPolicy": {
                "explicitSupportPresent": has_explicit_limitations,
                "fallbackText": "limitations not explicit in visible excerpt",
            },
        }
        start = time.perf_counter()
        try:
            metadata: dict[str, Any] = {}
            if hasattr(self._schema_extractor, "extract_with_metadata"):
                raw_payload, metadata = self._schema_extractor.extract_with_metadata(paper=extraction_input)
            else:
                raw_payload = self._schema_extractor.extract(paper=extraction_input)
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            extraction = PaperMemoryExtractionV1.from_dict(raw_payload)
            if extraction is None:
                diagnostics["warnings"].append("invalid_or_empty_payload")
                diagnostics["fallbackUsed"] = True
                self._last_extraction_diagnostics[paper_id] = diagnostics
                return card
            diagnostics["latencyMs"] = latency_ms
            diagnostics["latencyBucket"] = _latency_bucket(latency_ms)
            diagnostics["rawPayloadBytes"] = int(metadata.get("rawPayloadBytes") or len(json.dumps(raw_payload, ensure_ascii=False).encode("utf-8")))
            diagnostics["parsedFields"] = list(metadata.get("parsedFields") or sorted(raw_payload.keys()))
        except PaperMemoryExtractionError as exc:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            diagnostics["latencyMs"] = latency_ms
            diagnostics["latencyBucket"] = _latency_bucket(latency_ms)
            diagnostics["warnings"].append(f"extractor_error:{type(exc).__name__}")
            diagnostics["parseStage"] = exc.parse_stage
            diagnostics["rawOutputPreview"] = str(exc.raw_preview or "")
            diagnostics["rawPayloadBytes"] = int(exc.raw_payload_bytes or 0)
            diagnostics["parsedFields"] = []
            diagnostics["fieldConfidence"] = {}
            diagnostics["coverageByField"] = {}
            diagnostics["fallbackUsed"] = True
            self._last_extraction_diagnostics[paper_id] = diagnostics
            return card
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            diagnostics["latencyMs"] = latency_ms
            diagnostics["latencyBucket"] = _latency_bucket(latency_ms)
            diagnostics["warnings"].append(f"extractor_error:{type(exc).__name__}")
            diagnostics["fallbackUsed"] = True
            self._last_extraction_diagnostics[paper_id] = diagnostics
            return card

        diagnostics["extractorModel"] = extraction.extractor_model
        diagnostics["warnings"].extend(list(extraction.warnings or []))
        diagnostics["coverageByField"] = dict(extraction.coverage_status_by_field)
        diagnostics["fieldConfidence"] = dict(extraction.field_confidence)
        diagnostics["applied"] = self._extraction_mode == "schema"
        self._last_extraction_diagnostics[paper_id] = diagnostics

        if self._extraction_mode == "shadow":
            return card

        return self._merge_extraction_into_card(
            card=card,
            extraction=extraction,
            has_explicit_limitations=has_explicit_limitations,
        )

    def _merge_extraction_into_card(
        self,
        *,
        card: PaperMemoryCard,
        extraction: PaperMemoryExtractionV1,
        has_explicit_limitations: bool,
    ) -> PaperMemoryCard:
        merged_claim_refs = _clean_lines(list(card.claim_refs) + list(extraction.claim_refs), limit=12)
        merged_concepts = _clean_lines(list(card.concept_links) + list(extraction.concept_links), limit=12)
        thesis = _first_nonempty(extraction.thesis, card.paper_core)
        merged_problem_context = _first_nonempty(extraction.problem_context, card.problem_context)
        merged_limitations = _normalize_limitations_value(
            _first_nonempty(extraction.limitations, card.limitations),
            has_explicit_support=has_explicit_limitations,
        )
        return PaperMemoryCard(
            memory_id=card.memory_id,
            paper_id=card.paper_id,
            source_note_id=card.source_note_id,
            title=card.title,
            paper_core=thesis,
            problem_context=merged_problem_context,
            method_core=_first_nonempty(extraction.method_core, card.method_core),
            evidence_core=_first_nonempty(extraction.evidence_core, card.evidence_core),
            limitations=merged_limitations,
            concept_links=merged_concepts,
            claim_refs=merged_claim_refs,
            published_at=card.published_at,
            evidence_window=card.evidence_window,
            search_text=_cap_join(
                [
                    card.title,
                    thesis,
                    " ".join(_clean_lines(list(extraction.claims), limit=4)),
                    merged_problem_context,
                    _first_nonempty(extraction.method_core, card.method_core),
                    _first_nonempty(extraction.evidence_core, card.evidence_core),
                    merged_limitations,
                    " ".join(merged_concepts),
                    " ".join(merged_claim_refs),
                ],
                limit=900,
            ),
            quality_flag=_first_nonempty(extraction.quality_flag, card.quality_flag) or "unscored",
            version=card.version,
            created_at=card.created_at,
            updated_at=card.updated_at,
        )
