"""Build compact paper memory cards from existing paper artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

from knowledge_hub.core.keywords import STOP_ENGLISH
from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.infrastructure.config import DEFAULT_CONFIG_DIR
from knowledge_hub.papers.memory_extraction import PaperMemoryExtractionError, PaperMemoryExtractionV1
from knowledge_hub.papers.memory_models import (
    PaperMemoryCard,
    normalize_final_cause,
    normalize_formal_cause,
)
from knowledge_hub.papers.memory_quality import is_generic_limitation
from knowledge_hub.papers.memory_projection import (
    PROJECTED_ENRICHED_VERSION,
    PROJECTED_VERSION,
    PaperMemoryProjector,
)
from knowledge_hub.papers.source_text import extract_pdf_text_excerpt
from knowledge_hub.papers.text_quality import (
    contains_hangul as _contains_hangul,
    looks_author_stub as _looks_like_author_stub,
)
from knowledge_hub.papers.text_sanitizer import PaperTextNormalization, extract_keyword_window, normalize_paper_texts

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_EVIDENCE_MARKER_RE = re.compile(
    r"\b(?:outperform(?:s|ed)?|improv(?:e|es|ed)|achiev(?:e|es|ed)|gain(?:s|ed)?|accuracy|fid|auc|auprc|auroc|benchmark(?:s)?|results?|score|scores|evaluation|experiment(?:s)?|ablation|win(?:s|ning)?|성능|향상|개선|달성|결과|실험|평가|점수|정확도)\b",
    re.IGNORECASE,
)
_LIMITATION_MARKER_RE = re.compile(
    r"\b(?:limitation(?:s)?|future work|we do not|we don't|cannot|fails to|restricted to|limited to|only evaluated|only tested|not explicit|한계)\b",
    re.IGNORECASE,
)
_PROBLEM_MARKER_RE = re.compile(
    r"\b(?:problem|motivation|goal|objective|challenge|task|setting|context|background|문제|배경|목표|과제)\b",
    re.IGNORECASE,
)
_METHOD_MARKER_RE = re.compile(
    r"\b(?:method|approach|architecture|pipeline|training|framework|algorithm|design|instruction tuning|reinforcement learning|distill(?:ation|ed|ing)?|structure(?:d|ing)?|compress(?:ion|ed|es)?|방법|접근|구성|학습|구조화|증류|튜닝|생성|압축)\b",
    re.IGNORECASE,
)
_PAGE_EXCERPT_RE = re.compile(r"\[[^\]]+>\s*Page\s+\d+\]", re.IGNORECASE)
_TABLE_FIGURE_MARKER_RE = re.compile(r"\b(?:table|figure)\s+\d+\b", re.IGNORECASE)
_URL_MARKER_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_MARKER_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
_ARXIV_MARKER_RE = re.compile(r"\barxiv:\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_ET_AL_MARKER_RE = re.compile(r"\bet\s+al\.\b", re.IGNORECASE)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_BENCHMARK_MARKER_RE = re.compile(
    r"\b(?:mmlu(?:-pro|-redux)?|gsm8k|math|gpqa|humaneval\+?|livebench|auprc|auroc|f1|bleu|bar exam|lsat|drop|ifeval|simpleqa|mgsm)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_MARKER_RE = re.compile(
    r"\b(?:supported in part|supported by|funded by|grant(?:s)?|gift(?:s)? from|we thank|acknowledg(?:e|ement|ements)|legal advisory)\b",
    re.IGNORECASE,
)
_AUTHOR_INITIAL_RE = re.compile(r"\b[A-Z]\.")
_PROBLEM_SUBSTRING_MARKERS = ("문제", "배경", "목표", "과제", "맥락", "필요")
_METHOD_SUBSTRING_MARKERS = ("방법", "접근", "구성", "학습", "구조화", "증류", "튜닝", "생성", "압축")
_EVIDENCE_SUBSTRING_MARKERS = ("성능", "향상", "개선", "달성", "결과", "실험", "평가", "점수", "정확도")
_LIMITATION_SUBSTRING_MARKERS = ("한계", "제약", "민감", "의존")
_WEAK_SLOT_MARKERS = (
    "contents lists available at",
    "sciencedirect",
    "all rights reserved",
    "doi.org",
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
_TITLE_CONNECTOR_TOKENS = ("for", "via", "with", "using", "on", "by", "through", "under", "in", "from")
_TITLE_LEADING_FILLERS_RE = re.compile(
    r"^(?:a|an|the|across|evaluating|benchmarking|designing|scaling|towards?|understanding|improving|revisiting|beyond|analyzing)\s+",
    re.IGNORECASE,
)
_TITLE_FILELIKE_RE = re.compile(r"\.(?:md|markdown|txt|json|ya?ml|toml|csv|pdf|docx?|py|ipynb|js|ts|tsx)$", re.IGNORECASE)
_TITLE_SENTENCE_LIKE_TOKEN_RE = re.compile(
    r"\b(?:across|advanced|advancing|affordable|analyzing|analysing|beyond|can|designing|efficient|evaluating|improving|revisiting|scaling|towards?|understanding)\b",
    re.IGNORECASE,
)
_CITATION_STUB_RE = re.compile(r"^citations?\s*:\s*\d+\s*$", re.IGNORECASE)
_TITLE_GENERIC_BLACKLIST = {
    "agents",
    "benchmark",
    "benchmarks",
    "standard",
    "standards",
    "survey",
    "surveys",
}
_TITLE_CANONICAL_MAP = {
    "ai agent": "AI Agent",
    "ai agents": "AI Agent",
    "agent benchmarks": "Agent Benchmark",
    "agent memory": "Agent Memory",
    "ai scientist": "AI Scientist",
    "ai scientists": "AI Scientist",
    "ai coding agents": "AI Coding Agent",
    "coding agents": "Coding Agent",
    "coding agents?": "Coding Agent",
    "dense representations": "Dense representation",
    "evaluation metrics": "Evaluation metric",
    "foundation model": "Foundation Model",
    "foundation models": "Foundation Model",
    "language model": "Language Model",
    "language models": "Language Model",
    "large language model": "Large Language Model",
    "large language models": "Large Language Model",
    "large reasoning models": "Large Reasoning Model",
    "llm agents": "LLM Agent",
    "llm- agents": "LLM Agent",
    "llm-based agents": "LLM Agent",
    "memory-augmented generation": "Memory-Augmented Generation",
    "mistral 7b": "Mistral 7B",
    "neural networks": "Neural Network",
    "object-centric representations": "Object-centric representation",
    "open x-embodiment": "Open X-Embodiment",
    "prompt injection": "Prompt Injection",
    "prompt injection attack": "Prompt Injection",
    "prompt injection attacks": "Prompt Injection",
    "rag": "Retrieval-Augmented Generation",
    "reinforcement learning": "Reinforcement Learning",
    "retrieval-augmented generation": "Retrieval-Augmented Generation",
    "representation": "representation",
    "representations": "representation",
    "segment anything": "Segment Anything",
    "small language models": "Small Language Model",
    "table reasoning": "Table Reasoning",
    "vision-language-action model": "Vision-Language-Action Model",
    "vision-language-action models": "Vision-Language-Action Model",
}
_TRUSTED_TITLE_CONCEPTS = {
    "AI Agent",
    "AI Coding Agent",
    "AI Scientist",
    "Agent Benchmark",
    "Agent Memory",
    "Coding Agent",
    "Dense representation",
    "Evaluation metric",
    "Foundation Model",
    "Language Model",
    "Large Language Model",
    "Large Reasoning Model",
    "LLM Agent",
    "Memory-Augmented Generation",
    "Mistral 7B",
    "Neural Network",
    "Object-centric representation",
    "Open X-Embodiment",
    "Prompt Injection",
    "Reinforcement Learning",
    "Retrieval-Augmented Generation",
    "Segment Anything",
    "Small Language Model",
    "Table Reasoning",
    "Vision-Language-Action Model",
}


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


def _parse_timestamp(value: Any) -> datetime | None:
    token = _clean_text(value)
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    if re.fullmatch(r"\d{4}", token):
        return datetime(int(token), 1, 1, tzinfo=timezone.utc)
    return None


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


def _paper_text_inputs(paper: dict[str, Any]) -> tuple[str, str]:
    translated_text = _read_text(str(paper.get("translated_path") or ""))
    raw_text = _read_text(str(paper.get("text_path") or ""))
    if _clean_text(raw_text):
        return translated_text, raw_text
    pdf_excerpt = extract_pdf_text_excerpt(str(paper.get("pdf_path") or ""))
    if _clean_text(pdf_excerpt):
        return translated_text, pdf_excerpt
    return translated_text, raw_text


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
    if _CITATION_STUB_RE.fullmatch(token):
        return True
    if len(token.split()) <= 8 and _ET_AL_MARKER_RE.search(token):
        return True
    if len(token) <= 80 and ("## page" in lowered or _ARXIV_MARKER_RE.search(token)):
        return True
    return any(pattern in lowered for pattern in _SUMMARY_REFUSAL_PATTERNS)


def _summary_payload_has_usable_slots(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    summary = dict(payload.get("summary") or {})
    slot_values: list[Any] = [
        summary.get("oneLine"),
        summary.get("problem"),
        summary.get("coreIdea"),
        summary.get("whenItMatters"),
        summary.get("whatIsNew"),
    ]
    slot_values.extend(list(summary.get("methodSteps") or []))
    slot_values.extend(list(summary.get("keyResults") or []))
    slot_values.extend(list(summary.get("limitations") or []))
    usable = [value for value in slot_values if not _summary_value_is_unusable(value)]
    return len(usable) >= 3


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
    if (
        parser_used == "raw"
        and fallback_used
        and structured_sections <= 0
        and not parse_artifact_path
        and not _summary_payload_has_usable_slots(payload)
    ):
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
    evidence_values = _summary_values(summary_payload, "keyResults", limit=4)
    when_it_matters_values = _summary_values(summary_payload, "whenItMatters", limit=2)
    limitation_values = _summary_values(summary_payload, "limitations", limit=3)
    return {
        "paper_core": _cap_join(paper_core_values, limit=520),
        "problem_context": _cap_join(problem_values or paper_core_values[:1], limit=520),
        "method_core": _cap_join(method_values or paper_core_values[:1], limit=520),
        "evidence_core": _cap_join(evidence_values or when_it_matters_values, limit=520),
        "limitations": _cap_join(limitation_values, limit=420),
        "search_terms": _clean_lines(
            paper_core_values
            + problem_values
            + method_values[:2]
            + (evidence_values or when_it_matters_values)[:2]
            + limitation_values[:1],
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
    abstract_window = _extract_text_after_heading_keyword(stripped, "abstract", "요약", limit=1400)
    introduction_window = _extract_text_after_heading_keyword(
        stripped,
        "introduction",
        "background",
        "motivation",
        "problem",
        "문제",
        limit=1400,
    )
    method_window = _extract_text_after_heading_keyword(
        stripped,
        "method",
        "methods",
        "approach",
        "algorithm",
        "training",
        "architecture",
        "방법",
        "접근",
        limit=1400,
    )
    result_window = _extract_text_after_heading_keyword(
        stripped,
        "results",
        "result",
        "evaluation",
        "experiments",
        "experiment",
        "finding",
        "evidence",
        "결과",
        limit=1400,
    )
    limitation_window = _extract_text_after_heading_keyword(
        stripped,
        "limitations",
        "limitation",
        "future work",
        "한계",
        limit=1000,
    )
    abstract_source = _first_nonempty(abstract_text, abstract_window)
    introduction_source = _first_nonempty(introduction_text, introduction_window)
    method_source = _first_nonempty(method_text, method_window)
    result_source = _first_nonempty(result_text, result_window)
    limitation_source = _first_nonempty(limitation_text, limitation_window)
    paper_core = _first_nonempty(
        _cap_join(_extract_sentences(_first_nonempty(abstract_source, introduction_source, stripped), limit=2), limit=520),
        _context_excerpt(stripped, limit=520),
    )
    problem_context = _first_nonempty(
        _cap_join(_extract_sentences(_first_nonempty(introduction_source, abstract_source), limit=2), limit=520),
        extract_keyword_window(
            stripped,
            ("problem", "challenge", "task", "goal", "objective", "motivation", "background", "문제", "과제", "목표"),
            limit=520,
        ),
        paper_core,
    )
    method_core = _first_nonempty(
        _cap_join(_extract_sentences(_first_nonempty(method_source, abstract_source), limit=2), limit=520),
        extract_keyword_window(
            stripped,
            ("method", "approach", "algorithm", "training", "architecture", "optimization", "방법", "접근"),
            limit=520,
        ),
        paper_core,
    )
    evidence_core = _first_nonempty(
        _cap_join(_extract_evidence_sentences(_first_nonempty(result_source, abstract_source, stripped), limit=3), limit=520),
        _context_excerpt(result_source, limit=520),
        extract_keyword_window(
            stripped,
            ("result", "results", "evaluation", "experiment", "benchmark", "outperform", "BLEU", "accuracy", "결과"),
            limit=520,
        ),
    )
    limitations = _first_nonempty(
        _cap_join(_extract_sentences(limitation_source, limit=2), limit=420),
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


def _extract_text_after_heading_keyword(text: str, *keywords: str, limit: int = 1200) -> str:
    body = _clean_text(_strip_markdown(text))
    if not body:
        return ""
    for keyword in keywords:
        token = _clean_text(keyword)
        if not token:
            continue
        pattern = re.compile(
            rf"(?:^|\b)(?:\d+(?:\.\d+)*)?\s*{re.escape(token)}\b[.:]?\s*",
            re.IGNORECASE,
        )
        match = pattern.search(body)
        if not match:
            continue
        excerpt = _context_excerpt(body[match.end() :], limit=limit)
        if _clean_text(excerpt):
            return excerpt
    return ""


def _select_slot_candidate(*values: Any, field: str, title: str = "") -> str:
    candidates: list[str] = []
    for value in values:
        token = _clean_text(value)
        if not token:
            continue
        if _summary_value_is_unusable(token):
            continue
        candidates.append(token)
    if not candidates:
        return ""
    for token in candidates:
        if _slot_supports_quality(token, field=field, title=title):
            return token
    for token in candidates:
        if _slot_quality_score(token, field=field, title=title) >= 0:
            return token
    return candidates[0]
def _contains_marker(token: str, markers: tuple[str, ...]) -> bool:
    lowered = token.casefold()
    return any(marker.casefold() in lowered for marker in markers)


def _has_problem_signal(token: str) -> bool:
    return bool(_PROBLEM_MARKER_RE.search(token) or _contains_marker(token, _PROBLEM_SUBSTRING_MARKERS))


def _has_method_signal(token: str) -> bool:
    return bool(_METHOD_MARKER_RE.search(token) or _contains_marker(token, _METHOD_SUBSTRING_MARKERS))


def _has_evidence_signal(token: str) -> bool:
    numeric_tokens = len(_NUMERIC_TOKEN_RE.findall(token))
    return bool(
        _EVIDENCE_MARKER_RE.search(token)
        or _contains_marker(token, _EVIDENCE_SUBSTRING_MARKERS)
        or _BENCHMARK_MARKER_RE.search(token)
        or "%" in token
        or numeric_tokens >= 2
    )


def _has_limitation_signal(token: str) -> bool:
    return bool(_LIMITATION_MARKER_RE.search(token) or _contains_marker(token, _LIMITATION_SUBSTRING_MARKERS))


def _slot_quality_score(value: Any, *, field: str, title: str = "") -> int:
    token = _clean_text(_strip_markdown(value))
    if not token:
        return -100
    if _summary_value_is_unusable(token):
        return -80

    lowered = token.casefold()
    words = token.split()
    score = 0

    if 3 <= len(words) <= 60:
        score += 1
    if len(words) >= 10:
        score += 1
    if len(words) > 120:
        score -= 2
    if len(token) > 900:
        score -= 2

    if _PAGE_EXCERPT_RE.search(token):
        score -= 3
    if _TABLE_FIGURE_MARKER_RE.search(lowered):
        score -= 2 if field == "evidence_core" else 4
    if lowered.startswith(("table ", "figure ", "fig. ", "fig ")):
        score -= 3
    if _URL_MARKER_RE.search(lowered):
        score -= 3
    if _EMAIL_MARKER_RE.search(token):
        score -= 3
        if field in {"paper_core", "problem_context", "method_core"}:
            score -= 2
    if _ARXIV_MARKER_RE.search(token):
        score -= 2
    if any(marker in lowered for marker in _WEAK_SLOT_MARKERS):
        score -= 4
    if _looks_like_author_stub(token):
        score -= 4
    if len(words) <= 8 and _ET_AL_MARKER_RE.search(token):
        score -= 4
    if _ACKNOWLEDGEMENT_MARKER_RE.search(token):
        score -= 5 if field == "evidence_core" else 2

    title_token = _clean_text(title).casefold()
    if title_token and lowered.startswith(title_token):
        score -= 2
    if token.count(",") >= 8:
        score -= 2

    numeric_tokens = len(_NUMERIC_TOKEN_RE.findall(token))
    if numeric_tokens >= 14:
        score -= 2 if field == "evidence_core" else 3

    if field == "paper_core":
        if _has_problem_signal(token) or _has_method_signal(token):
            score += 1
    elif field == "problem_context":
        if _has_problem_signal(token):
            score += 2
        if _has_method_signal(token) and not _has_problem_signal(token):
            score -= 1
    elif field == "method_core":
        if _has_method_signal(token):
            score += 2
        if (_has_evidence_signal(token) or "latency" in lowered or "cost" in lowered) and not _has_method_signal(token):
            score -= 2
    elif field == "evidence_core":
        if _has_evidence_signal(token):
            score += 2
        if _BENCHMARK_MARKER_RE.search(token) or "%" in token:
            score += 1
    elif field == "limitations":
        if lowered == "limitations not explicit in visible excerpt":
            score -= 4
        if _has_limitation_signal(token):
            score += 2
        if is_generic_limitation(token):
            score -= 3
        if _has_method_signal(token) and not _has_limitation_signal(token):
            score -= 1

    return score


def _slot_supports_quality(value: Any, *, field: str, title: str = "") -> bool:
    token = _clean_text(_strip_markdown(value))
    if not token:
        return False

    score = _slot_quality_score(token, field=field, title=title)
    if field == "method_core":
        if score < 1:
            return False
        if _looks_like_author_stub(token):
            return False
        if _TABLE_FIGURE_MARKER_RE.search(token) or _EMAIL_MARKER_RE.search(token):
            return False
        if _has_method_signal(token):
            return True
        return len(token.split()) >= 8 and token.count(",") < 6
    if field == "problem_context":
        if score < 1:
            return False
        if _EMAIL_MARKER_RE.search(token) or _PAGE_EXCERPT_RE.search(token):
            return False
        return True
    if field == "evidence_core":
        if _ACKNOWLEDGEMENT_MARKER_RE.search(token):
            return False
        if token.casefold().startswith(("table ", "figure ", "fig. ", "fig ")):
            return False
        if _EMAIL_MARKER_RE.search(token):
            return False
        return score >= 0 and _has_evidence_signal(token)
    if field == "limitations":
        if score < 1:
            return False
        if _PAGE_EXCERPT_RE.search(token) or _EMAIL_MARKER_RE.search(token):
            return False
        return True

    if score < 1:
        return False

    return True


def _prefer_slot_value(primary: Any, secondary: Any, *, field: str, title: str = "") -> str:
    primary_token = _clean_text(primary)
    secondary_token = _clean_text(secondary)
    if not primary_token:
        return secondary_token
    if not secondary_token:
        return primary_token

    primary_score = _slot_quality_score(primary_token, field=field, title=title)
    secondary_score = _slot_quality_score(secondary_token, field=field, title=title)
    primary_supports = _slot_supports_quality(primary_token, field=field, title=title)
    secondary_supports = _slot_supports_quality(secondary_token, field=field, title=title)
    if primary_supports and secondary_supports:
        if secondary_score > primary_score:
            return secondary_token
        if secondary_score == primary_score and _contains_hangul(secondary_token) and not _contains_hangul(primary_token):
            return secondary_token
        return primary_token
    if primary_supports and not secondary_supports:
        return primary_token
    if secondary_supports and not primary_supports:
        return secondary_token
    if primary_score >= 1 and not secondary_supports:
        return primary_token
    if secondary_score >= primary_score:
        return secondary_token
    return primary_token


def _prefer_existing_slot_value(primary: Any, secondary: Any, *, field: str, title: str = "") -> str:
    primary_token = _clean_text(primary)
    secondary_token = _clean_text(secondary)
    if not primary_token:
        return secondary_token
    if not secondary_token:
        return primary_token

    primary_score = _slot_quality_score(primary_token, field=field, title=title)
    secondary_score = _slot_quality_score(secondary_token, field=field, title=title)
    if primary_score >= 0:
        return primary_token
    if secondary_score >= primary_score:
        return secondary_token
    return primary_token


def _merge_quality_flag(
    *,
    title: str,
    paper_core: str,
    problem_context: str,
    method_core: str,
    evidence_core: str,
    limitations: str,
    claim_ref_count: int,
    candidate_flags: list[str] | tuple[str, ...],
) -> str:
    flags = {
        _clean_text(flag).casefold()
        for flag in list(candidate_flags)
        if _clean_text(flag).casefold() in {"ok", "needs_review", "reject", "unscored"}
    }
    if "reject" in flags:
        return "reject"

    slot_strength = {
        "paper_core": _slot_supports_quality(paper_core, field="paper_core", title=title),
        "problem_context": _slot_supports_quality(problem_context, field="problem_context", title=title),
        "method_core": _slot_supports_quality(method_core, field="method_core", title=title),
        "evidence_core": _slot_supports_quality(evidence_core, field="evidence_core", title=title),
        "limitations": _slot_supports_quality(limitations, field="limitations", title=title),
    }
    strong_core_count = sum(1 for ok in slot_strength.values() if ok)
    has_context_anchor = slot_strength["paper_core"] or slot_strength["problem_context"]
    has_retrieval_core = slot_strength["method_core"] and slot_strength["evidence_core"]

    if "ok" in flags and has_context_anchor and has_retrieval_core and strong_core_count >= 4:
        return "ok"
    if has_context_anchor and has_retrieval_core and strong_core_count >= 5:
        return "ok"
    if claim_ref_count >= 2 and has_retrieval_core and strong_core_count >= 3:
        return "ok"
    if "needs_review" in flags or claim_ref_count > 0 or strong_core_count >= 2:
        return "needs_review"
    return "unscored"


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


def _legacy_evidence_payload(row: dict[str, Any]) -> dict[str, Any]:
    raw_reason = row.get("reason_json")
    if not raw_reason:
        return {}
    try:
        reason = json.loads(str(raw_reason))
    except Exception:
        return {}
    if isinstance(reason.get("legacy_evidence"), dict):
        return dict(reason.get("legacy_evidence") or {})
    legacy_text = reason.get("legacy_evidence_text")
    if not legacy_text:
        return {}
    try:
        parsed = json.loads(str(legacy_text))
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _is_bridge_math_concept(row: dict[str, Any]) -> bool:
    evidence = _legacy_evidence_payload(row)
    note_id = _clean_text(evidence.get("note_id"))
    if "Math Bridge - " in note_id:
        return True
    for ptr in list(evidence.get("evidence_ptrs") or []):
        if not isinstance(ptr, dict):
            continue
        path = _clean_text(ptr.get("path"))
        if "Math Bridge - " in path:
            return True
    return False


def _is_heuristic_title_fallback_concept(row: dict[str, Any]) -> bool:
    return _clean_text(row.get("source")) == "paper_memory_title_fallback"


def _concept_names(rows: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    if any(not _is_heuristic_title_fallback_concept(row) for row in rows):
        rows = [row for row in rows if not _is_heuristic_title_fallback_concept(row)]
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            0 if _is_bridge_math_concept(item[1]) else 1,
            -float(item[1].get("confidence") or 0.0),
            item[0],
        ),
    )
    values = [str(row.get("canonical_name") or row.get("id") or row.get("entity_id") or "") for _, row in ordered]
    return _clean_lines(values, limit=limit)


def _concept_entity_id(name: Any) -> str:
    return re.sub(r"\s+", "_", _clean_text(name)).strip("_").lower()


def _existing_concept_entity_id(sqlite_db: Any, canonical_name: Any) -> str:
    token = _clean_text(canonical_name)
    if not token:
        return ""
    conn = getattr(sqlite_db, "conn", None)
    if conn is None:
        return ""
    try:
        row = conn.execute(
            "SELECT entity_id FROM ontology_entities WHERE entity_type='concept' AND canonical_name=?",
            (token,),
        ).fetchone()
    except Exception:
        return ""
    if not row:
        return ""
    return _clean_text(row["entity_id"] if hasattr(row, "__getitem__") else row[0])


def _normalize_title_concept(text: Any) -> str:
    token = _clean_text(text).strip(" -:;,.")
    if not token:
        return ""
    if "/" in token or "\\" in token:
        return ""
    if _TITLE_FILELIKE_RE.search(token):
        return ""
    if any(marker in token for marker in ("$", "\\", "{", "}", "<", ">", "=")):
        return ""
    if "_" in token:
        return ""
    token = _TITLE_LEADING_FILLERS_RE.sub("", token)
    token = re.sub(r"\bbased\b", " ", token, flags=re.IGNORECASE)
    token = _clean_text(token).strip(" -:;,.")
    if not token:
        return ""
    words = token.split()
    if len(words) > 6:
        return ""
    if len(words) >= 3 and _TITLE_SENTENCE_LIKE_TOKEN_RE.search(token):
        return ""
    if not re.search(r"[A-Za-z0-9가-힣]", token):
        return ""
    lowered = token.casefold()
    if lowered in _TITLE_GENERIC_BLACKLIST:
        return ""
    if lowered in STOP_ENGLISH or lowered in {"paper", "study", "approach", "method", "framework"}:
        return ""
    if all(word.casefold() in STOP_ENGLISH for word in words):
        return ""
    return _TITLE_CANONICAL_MAP.get(lowered, token)


def _is_trusted_title_concept(name: Any) -> bool:
    return _clean_text(name) in _TRUSTED_TITLE_CONCEPTS


def _title_concept_candidates(title: Any, *, field: Any = "", limit: int = 8) -> list[str]:
    raw_title = _clean_text(title)
    if not raw_title:
        field_token = _normalize_title_concept(field)
        return [field_token] if field_token else []

    def _span_candidates(text: str) -> list[str]:
        words = _clean_text(text).split()
        if len(words) < 4:
            return []
        spans = [
            " ".join(words[:2]),
            " ".join(words[:3]),
            " ".join(words[-3:]),
        ]
        return _clean_lines([_normalize_title_concept(span) for span in spans], limit=3)

    candidates: list[str] = []
    for segment in re.split(r"\s*:\s*", raw_title):
        split_applied = False
        for connector in _TITLE_CONNECTOR_TOKENS:
            parts = re.split(rf"(?i)\b{re.escape(connector)}\b", segment, maxsplit=1)
            if len(parts) != 2:
                continue
            left = _normalize_title_concept(parts[0])
            right = _normalize_title_concept(parts[1])
            if left:
                candidates.append(left)
            if right:
                candidates.append(right)
            if not left and not right:
                candidates.extend(_span_candidates(segment))
            split_applied = bool(left or right)
            break
        if not split_applied:
            normalized_segment = _normalize_title_concept(segment)
            if normalized_segment:
                candidates.append(normalized_segment)
            else:
                candidates.extend(_span_candidates(segment))
    if not candidates:
        field_token = _normalize_title_concept(field)
        if field_token:
            candidates.append(field_token)
    return _clean_lines(candidates, limit=limit)


def _quality_flag(
    note: dict[str, Any] | None,
    claims: list[dict[str, Any]],
    *,
    title: str,
    paper_core: str,
    problem_context: str,
    method_core: str,
    evidence_core: str,
    limitations: str,
    summary_fallback_used: bool = False,
) -> str:
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
    has_local_note = bool(note and _clean_text((note or {}).get("content") or (note or {}).get("title") or ""))
    strong_claims = [row for row in claims if float(row.get("confidence") or 0.0) >= 0.8]
    if len(strong_claims) >= 2:
        return "ok"
    slot_strength = {
        "paper_core": _slot_supports_quality(paper_core, field="paper_core", title=title),
        "problem_context": _slot_supports_quality(problem_context, field="problem_context", title=title),
        "method_core": _slot_supports_quality(method_core, field="method_core", title=title),
        "evidence_core": _slot_supports_quality(evidence_core, field="evidence_core", title=title),
        "limitations": _slot_supports_quality(limitations, field="limitations", title=title),
    }
    strong_core_count = sum(1 for ok in slot_strength.values() if ok)
    has_context_anchor = slot_strength["paper_core"] or slot_strength["problem_context"]
    has_retrieval_core = slot_strength["method_core"] and slot_strength["evidence_core"]
    if not has_local_note and not summary_fallback_used and has_context_anchor and has_retrieval_core and strong_core_count >= 4:
        return "ok"
    if claims:
        return "needs_review"
    if strong_core_count >= 2:
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


def _coverage_rank(value: Any) -> int:
    token = _clean_text(value).casefold()
    return {"missing": 0, "partial": 1, "complete": 2}.get(token, 0)


def _prefer_coverage(*values: Any) -> str:
    best = "missing"
    best_rank = -1
    for value in values:
        token = _clean_text(value).casefold() or "missing"
        rank = _coverage_rank(token)
        if rank > best_rank:
            best = token
            best_rank = rank
    return best


def _formal_cause_excerpt(
    note_sections: dict[str, str],
    preferred_text: str,
    structured_summary: dict[str, Any],
    parsed_markdown_slots: dict[str, Any],
    *,
    limit: int = 420,
) -> str:
    return _first_nonempty(
        structured_summary.get("method_core"),
        parsed_markdown_slots.get("method_core"),
        _fixed_slot_excerpt(
            note_sections,
            "method",
            "approach",
            "architecture",
            "formulation",
            "model",
            "algorithm",
            "training",
            "방법",
            "접근",
            limit=limit,
        ),
        extract_keyword_window(
            preferred_text,
            ("method", "approach", "architecture", "formulation", "model", "algorithm", "training", "방법", "접근"),
            limit=limit,
        ),
    )


def _final_cause_excerpt(
    note_sections: dict[str, str],
    preferred_text: str,
    structured_summary: dict[str, Any],
    parsed_markdown_slots: dict[str, Any],
    *,
    limit: int = 420,
) -> str:
    return _first_nonempty(
        structured_summary.get("problem_context"),
        parsed_markdown_slots.get("problem_context"),
        _fixed_slot_excerpt(
            note_sections,
            "objective",
            "goal",
            "purpose",
            "task",
            "motivation",
            "why",
            "abstract",
            "introduction",
            "problem",
            "목적",
            "목표",
            "과제",
            "동기",
            "문제",
            limit=limit,
        ),
        extract_keyword_window(
            preferred_text,
            ("objective", "goal", "purpose", "task", "motivation", "why", "aim", "problem", "challenge", "목적", "목표", "과제", "동기", "문제"),
            limit=limit,
        ),
    )


def _cause_excerpts(
    note_sections: dict[str, str],
    preferred_text: str,
    structured_summary: dict[str, Any],
    parsed_markdown_slots: dict[str, Any],
) -> dict[str, str]:
    return {
        "formalCauseExcerpt": _formal_cause_excerpt(
            note_sections,
            preferred_text,
            structured_summary,
            parsed_markdown_slots,
        ),
        "finalCauseExcerpt": _final_cause_excerpt(
            note_sections,
            preferred_text,
            structured_summary,
            parsed_markdown_slots,
        ),
    }


def _merge_formal_cause(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    base_payload = normalize_formal_cause(base)
    incoming_payload = normalize_formal_cause(incoming)
    if not base_payload:
        return incoming_payload
    if not incoming_payload:
        return base_payload
    return normalize_formal_cause(
        {
            "summary": incoming_payload.get("summary") or base_payload.get("summary"),
            "basis": incoming_payload.get("basis") or base_payload.get("basis"),
            "confidence": max(
                float(base_payload.get("confidence") or 0.0),
                float(incoming_payload.get("confidence") or 0.0),
            ),
            "coverage": _prefer_coverage(base_payload.get("coverage"), incoming_payload.get("coverage")),
            "evidence_refs": _clean_lines(
                list(base_payload.get("evidence_refs") or []) + list(incoming_payload.get("evidence_refs") or []),
                limit=12,
            ),
            "warnings": _clean_lines(
                list(base_payload.get("warnings") or []) + list(incoming_payload.get("warnings") or []),
                limit=12,
            ),
        }
    )


def _merge_final_cause(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    base_payload = normalize_final_cause(base)
    incoming_payload = normalize_final_cause(incoming)
    if not base_payload:
        return incoming_payload
    if not incoming_payload:
        return base_payload
    return normalize_final_cause(
        {
            "author_stated_summary": incoming_payload.get("author_stated_summary")
            or base_payload.get("author_stated_summary"),
            "inferred_summary": incoming_payload.get("inferred_summary") or base_payload.get("inferred_summary"),
            "basis": incoming_payload.get("basis") or base_payload.get("basis"),
            "confidence": max(
                float(base_payload.get("confidence") or 0.0),
                float(incoming_payload.get("confidence") or 0.0),
            ),
            "coverage": _prefer_coverage(base_payload.get("coverage"), incoming_payload.get("coverage")),
            "evidence_refs": _clean_lines(
                list(base_payload.get("evidence_refs") or []) + list(incoming_payload.get("evidence_refs") or []),
                limit=12,
            ),
            "warnings": _clean_lines(
                list(base_payload.get("warnings") or []) + list(incoming_payload.get("warnings") or []),
                limit=12,
            ),
        }
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
        self._projector = PaperMemoryProjector(sqlite_db)

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

    def _ensure_document_memory(self, *, paper_id: str) -> list[dict[str, Any]]:
        document_id = f"paper:{paper_id}"
        units = list(self.sqlite_db.list_document_memory_units(document_id, limit=2000) or [])
        if units:
            return units
        return DocumentMemoryBuilder(self.sqlite_db).build_and_store_paper(paper_id=paper_id)

    def build_projected_card(
        self,
        *,
        paper_id: str,
        ensure_document_memory: bool = True,
    ) -> PaperMemoryCard:
        paper = self.sqlite_db.get_paper(paper_id)
        if not paper:
            raise ValueError(f"paper not found: {paper_id}")
        units = self._ensure_document_memory(paper_id=paper_id) if ensure_document_memory else list(
            self.sqlite_db.list_document_memory_units(f"paper:{paper_id}", limit=2000) or []
        )
        projected = self._projector.project(paper_id=paper_id, paper=paper, units=units)
        if projected is None:
            raise ValueError(f"document memory not found: paper:{paper_id}")
        return projected

    def needs_rebuild(self, paper_id: str) -> bool:
        token = _clean_text(paper_id)
        if not token:
            return False
        current = PaperMemoryCard.from_row(self.sqlite_db.get_paper_memory_card(token))
        rows = list(self.sqlite_db.list_document_memory_units(f"paper:{token}", limit=2000) or [])
        latest_document_updated = max(
            (_parse_timestamp(row.get("updated_at")) for row in rows if isinstance(row, dict)),
            default=None,
        )
        if latest_document_updated is None:
            return False
        if current is None:
            return True
        current_updated = _parse_timestamp(current.updated_at)
        if current_updated is None:
            return True
        return latest_document_updated > current_updated

    def _should_run_enrichment(self, *, card: PaperMemoryCard) -> bool:
        return self._schema_extractor is not None

    def _merge_additive_card_enrichment(
        self,
        *,
        base: PaperMemoryCard,
        enrichment: PaperMemoryCard,
    ) -> PaperMemoryCard:
        repeated_core_values = {
            value
            for value, count in Counter(
                [
                    _clean_text(base.paper_core),
                    _clean_text(base.problem_context),
                    _clean_text(base.method_core),
                    _clean_text(base.evidence_core),
                    _clean_text(base.limitations),
                ]
            ).items()
            if value and count >= 3
        }

        def _projected_slot(slot_value: Any) -> str:
            token = _clean_text(slot_value)
            if not token:
                return ""
            if token in repeated_core_values:
                return ""
            if len(token) <= 4 and " " not in token:
                return ""
            if token.casefold() in {
                _clean_text(base.title).casefold(),
                _clean_text(enrichment.title).casefold(),
                _clean_text(base.source_note_id).casefold(),
                _clean_text(enrichment.source_note_id).casefold(),
            }:
                return ""
            return token

        def _enrichment_slot(slot_value: Any) -> str:
            token = _clean_text(slot_value)
            if not token:
                return ""
            if len(token) <= 4 and " " not in token:
                return ""
            if token.casefold() in {
                _clean_text(base.title).casefold(),
                _clean_text(enrichment.title).casefold(),
                _clean_text(base.source_note_id).casefold(),
                _clean_text(enrichment.source_note_id).casefold(),
            }:
                return ""
            return token

        merged_claim_refs = _clean_lines(list(base.claim_refs) + list(enrichment.claim_refs), limit=12)
        # Prefer deterministic ontology-backed concept ordering for the visible slice.
        merged_concepts = _clean_lines(list(enrichment.concept_links) + list(base.concept_links), limit=12)
        merged_paper_core = _prefer_slot_value(
            _projected_slot(base.paper_core),
            _enrichment_slot(enrichment.paper_core),
            field="paper_core",
            title=_first_nonempty(base.title, enrichment.title),
        )
        merged_problem_context = _prefer_slot_value(
            _projected_slot(base.problem_context),
            _enrichment_slot(enrichment.problem_context),
            field="problem_context",
            title=_first_nonempty(base.title, enrichment.title),
        )
        merged_method_core = _prefer_slot_value(
            _projected_slot(base.method_core),
            _enrichment_slot(enrichment.method_core),
            field="method_core",
            title=_first_nonempty(base.title, enrichment.title),
        )
        merged_evidence_core = _prefer_slot_value(
            _projected_slot(base.evidence_core),
            _enrichment_slot(enrichment.evidence_core),
            field="evidence_core",
            title=_first_nonempty(base.title, enrichment.title),
        )
        merged_limitations = _prefer_slot_value(
            _projected_slot(base.limitations),
            _enrichment_slot(enrichment.limitations),
            field="limitations",
            title=_first_nonempty(base.title, enrichment.title),
        )
        merged_quality_flag = _merge_quality_flag(
            title=_first_nonempty(base.title, enrichment.title),
            paper_core=merged_paper_core,
            problem_context=merged_problem_context,
            method_core=merged_method_core,
            evidence_core=merged_evidence_core,
            limitations=merged_limitations,
            claim_ref_count=len(merged_claim_refs),
            candidate_flags=[base.quality_flag, enrichment.quality_flag],
        )
        merged = PaperMemoryCard(
            memory_id=base.memory_id,
            paper_id=base.paper_id,
            source_note_id=_first_nonempty(base.source_note_id, enrichment.source_note_id),
            title=_first_nonempty(base.title, enrichment.title),
            paper_core=merged_paper_core,
            problem_context=merged_problem_context,
            method_core=merged_method_core,
            evidence_core=merged_evidence_core,
            limitations=merged_limitations,
            concept_links=merged_concepts,
            claim_refs=merged_claim_refs,
            formal_cause=_merge_formal_cause(base.formal_cause, enrichment.formal_cause),
            final_cause=_merge_final_cause(base.final_cause, enrichment.final_cause),
            published_at=_first_nonempty(base.published_at, enrichment.published_at),
            evidence_window=_first_nonempty(base.evidence_window, enrichment.evidence_window),
            search_text=_cap_join(
                [
                    _first_nonempty(base.title, enrichment.title),
                    merged_paper_core,
                    merged_problem_context,
                    merged_method_core,
                    merged_evidence_core,
                    merged_limitations,
                    " ".join(merged_concepts),
                    " ".join(merged_claim_refs),
                ],
                limit=900,
            ),
            quality_flag=merged_quality_flag,
            version=base.version,
            created_at=base.created_at,
            updated_at=base.updated_at,
        )
        if not _clean_text(merged.paper_core):
            merged.paper_core = _first_nonempty(base.title, enrichment.title, base.paper_id)
        if not _clean_text(merged.search_text):
            merged.search_text = _cap_join(
                [
                    merged.title,
                    merged.paper_core,
                    merged.problem_context,
                    merged.method_core,
                    merged.evidence_core,
                    merged.limitations,
                    " ".join(merged.concept_links),
                    " ".join(merged.claim_refs),
                ],
                limit=900,
            )
        base_payload = base.to_payload()
        merged_payload = merged.to_payload()
        was_enriched = any(
            base_payload.get(key) != merged_payload.get(key)
            for key in (
                "sourceNoteId",
                "title",
                "paperCore",
                "problemContext",
                "methodCore",
                "evidenceCore",
                "limitations",
                "conceptLinks",
                "claimRefs",
                "publishedAt",
                "evidenceWindow",
                "searchText",
                "qualityFlag",
            )
        )
        merged.version = PROJECTED_ENRICHED_VERSION if was_enriched else (base.version or PROJECTED_VERSION)
        return merged

    def _build_cutover_base_card(self, *, paper_id: str) -> PaperMemoryCard:
        projected = self.build_projected_card(paper_id=paper_id)
        legacy = self.build_card(paper_id=paper_id, apply_schema_extraction=False)
        merged = self._merge_additive_card_enrichment(base=projected, enrichment=legacy)
        if (
            _clean_text(legacy.source_note_id)
            and _clean_text(legacy.quality_flag).casefold() != "ok"
            and _clean_text(merged.quality_flag).casefold() == "ok"
        ):
            merged.quality_flag = legacy.quality_flag
        if not _clean_text(legacy.source_note_id):
            imported_slot_strength = {
                "paper_core": _slot_supports_quality(merged.paper_core, field="paper_core", title=merged.title),
                "problem_context": _slot_supports_quality(merged.problem_context, field="problem_context", title=merged.title),
                "method_core": _slot_supports_quality(merged.method_core, field="method_core", title=merged.title),
                "evidence_core": _slot_supports_quality(merged.evidence_core, field="evidence_core", title=merged.title),
            }
            if all(imported_slot_strength.values()) and _clean_text(merged.quality_flag).casefold() != "reject":
                merged.quality_flag = "ok"
        return merged

    def build_card(self, *, paper_id: str, apply_schema_extraction: bool = True) -> PaperMemoryCard:
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
        title_fallback_concepts: list[str] = []
        if not concept_links:
            title_fallback_concepts = _title_concept_candidates(paper.get("title"), field=paper.get("field"), limit=8)
            concept_links = title_fallback_concepts
        section_headings = _section_headings(note_sections, limit=8)
        structured_summary_payload = _load_structured_summary_artifact(paper=paper, paper_id=paper_id)
        structured_summary = _structured_summary_slot_values(structured_summary_payload)
        parsed_markdown_slots = _parsed_markdown_slot_values(_load_parsed_markdown_artifact(paper=paper, paper_id=paper_id))
        paper_title = str(paper.get("title") or paper_id)

        translated_text, raw_text = _paper_text_inputs(paper)
        text_normalization = normalize_paper_texts(translated_text=translated_text, raw_text=raw_text)
        translated_summary = " ".join(_extract_sentences(text_normalization.translated.sanitized_text, limit=3))
        raw_summary = " ".join(_extract_sentences(text_normalization.raw.sanitized_text, limit=3))
        preferred_summary = _first_usable(translated_summary, raw_summary)

        paper_core = _select_slot_candidate(
            structured_summary.get("paper_core"),
            parsed_markdown_slots.get("paper_core"),
            summary_text,
            claim_texts[0] if claim_texts else "",
            translated_summary,
            raw_summary,
            paper.get("notes"),
            paper.get("title"),
            field="paper_core",
            title=paper_title,
        )
        problem_context = _select_slot_candidate(
            structured_summary.get("problem_context"),
            parsed_markdown_slots.get("problem_context"),
            abstract_text,
            structured_summary.get("paper_core"),
            parsed_markdown_slots.get("paper_core"),
            translated_summary,
            raw_summary,
            paper.get("field"),
            field="problem_context",
            title=paper_title,
        )
        method_core = _select_slot_candidate(
            structured_summary.get("method_core"),
            parsed_markdown_slots.get("method_core"),
            _first_usable(_find_section(note_sections, "method", "방법", "접근")),
            claim_texts[1] if len(claim_texts) > 1 else "",
            preferred_summary,
            field="method_core",
            title=paper_title,
        )
        evidence_core = _select_slot_candidate(
            structured_summary.get("evidence_core"),
            parsed_markdown_slots.get("evidence_core"),
            _first_usable(_find_section(note_sections, "result", "finding", "evidence", "결과")),
            " ".join(claim_texts[:2]),
            abstract_text,
            preferred_summary,
            field="evidence_core",
            title=paper_title,
        )
        limitations = _select_slot_candidate(
            structured_summary.get("limitations"),
            parsed_markdown_slots.get("limitations"),
            _first_usable(_find_section(note_sections, "limitation", "한계")),
            claim_texts[2] if len(claim_texts) > 2 else "",
            field="limitations",
            title=paper_title,
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
            quality_flag=_quality_flag(
                note,
                claims,
                title=paper_title,
                paper_core=paper_core,
                problem_context=problem_context,
                method_core=method_core,
                evidence_core=evidence_core,
                limitations=limitations,
                summary_fallback_used=bool(
                    (structured_summary_payload or {}).get("fallbackUsed")
                    or (structured_summary_payload or {}).get("fallback_used")
                ),
            ),
        )
        merged_card = card
        if apply_schema_extraction:
            merged_card = self._apply_schema_extraction(
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
        if title_fallback_concepts:
            self._sync_title_fallback_concepts(
                paper_id=paper_id,
                paper=paper,
                concept_links=list(merged_card.concept_links),
            )
        return merged_card

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
        translated_text, raw_text = _paper_text_inputs(paper)
        text_normalization = normalize_paper_texts(translated_text=translated_text, raw_text=raw_text)
        card = self._build_cutover_base_card(paper_id=paper_id)
        sanitized_translated_text = text_normalization.translated.sanitized_text
        sanitized_raw_text = text_normalization.raw.sanitized_text
        preferred_text = text_normalization.preferred_text
        explicit_limitations_excerpt = _first_nonempty(
            _fixed_slot_excerpt(note_sections, "limitation", "한계", limit=320),
            extract_keyword_window(preferred_text, ("limitation", "limitations", "future work", "limited", "한계"), limit=320),
        )
        has_explicit_limitations = bool(_clean_text(explicit_limitations_excerpt)) or _has_explicit_limitation_support(card.limitations)
        problem_excerpt = _problem_excerpt(note_sections, preferred_text, limit=420)
        cause_excerpts = _cause_excerpts(note_sections, preferred_text, structured_summary, parsed_markdown_slots)
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
            **cause_excerpts,
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

    def build_and_store(self, *, paper_id: str, materialize_card: bool = True) -> dict[str, Any]:
        card = self._build_cutover_base_card(paper_id=paper_id)
        if self._should_run_enrichment(card=card):
            paper = self.sqlite_db.get_paper(paper_id) or {}
            note = self._note_for_paper(paper_id)
            translated_text, raw_text = _paper_text_inputs(paper)
            card = self._apply_schema_extraction(
                paper_id=paper_id,
                card=card,
                paper=paper,
                note=note,
                note_sections=_split_sections(str((note or {}).get("content") or "")),
                claims=self._claims_for_paper(paper_id, str((note or {}).get("id") or f"paper:{paper_id}")),
                concepts=self.sqlite_db.get_paper_concepts(paper_id),
                translated_text=translated_text,
                raw_text=raw_text,
                text_normalization=normalize_paper_texts(translated_text=translated_text, raw_text=raw_text),
            )
        stored = self.sqlite_db.upsert_paper_memory_card(card=card.to_record())
        self._refresh_updates_relations(card=card)
        result: dict[str, Any] = dict(stored) if isinstance(stored, dict) else {}
        if materialize_card:
            try:
                from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder

                PaperCardV2Builder(self.sqlite_db).build_and_store(paper_id=paper_id)
            except Exception as exc:
                # Previously swallowed: paper_memory row exists but card v2 / anchors vanish with no signal.
                _logger.warning(
                    "paper card v2 materialize failed paper_id=%s",
                    paper_id,
                    exc_info=True,
                )
                result["card_v2_materialize_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def apply_extraction_payload_and_store(
        self,
        *,
        paper_id: str,
        raw_payload: dict[str, Any],
        extractor_model: str = "",
    ) -> dict[str, Any]:
        card = self._build_cutover_base_card(paper_id=paper_id)
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

    def _sync_title_fallback_concepts(self, *, paper_id: str, paper: dict[str, Any], concept_links: list[str]) -> None:
        upsert_entity = getattr(self.sqlite_db, "upsert_ontology_entity", None)
        add_relation = getattr(self.sqlite_db, "add_relation", None)
        if upsert_entity is None or add_relation is None:
            return
        if self.sqlite_db.get_paper_concepts(paper_id):
            return

        paper_title = _clean_text(paper.get("title") or paper_id)
        upsert_entity(
            entity_id=f"paper:{paper_id}",
            entity_type="paper",
            canonical_name=paper_title,
            properties={"arxiv_id": paper_id},
            source="paper_memory_title_fallback",
        )
        for concept_name in _clean_lines(concept_links, limit=8):
            concept_entity_id = _existing_concept_entity_id(self.sqlite_db, concept_name) or _concept_entity_id(concept_name)
            if not concept_entity_id:
                continue
            trusted_seed = _is_trusted_title_concept(concept_name)
            upsert_entity(
                entity_id=concept_entity_id,
                entity_type="concept",
                canonical_name=concept_name,
                properties={} if trusted_seed else {"heuristic_source": "paper_memory_title_fallback"},
                source="paper_title_seed" if trusted_seed else "paper_memory_title_fallback",
            )
            add_relation(
                source_type="paper",
                source_id=paper_id,
                relation="uses",
                target_type="concept",
                target_id=concept_entity_id,
                evidence_text=json.dumps(
                    {
                        "source": "paper_title_seed" if trusted_seed else "paper_memory_title_fallback",
                        "relation_norm": "uses",
                        "reason": "trusted title-derived concept links" if trusted_seed else "title-derived fallback concept links",
                    },
                    ensure_ascii=False,
                ),
                confidence=0.74 if trusted_seed else 0.42,
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
            "schema": str(getattr(self._schema_extractor, "schema", "knowledge-hub.paper-memory-extraction.v2")),
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
        cause_excerpts = _cause_excerpts(note_sections, preferred_text, structured_summary, parsed_markdown_slots)
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
            **cause_excerpts,
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
        thesis = _prefer_slot_value(card.paper_core, extraction.thesis, field="paper_core", title=card.title)
        merged_problem_context = _prefer_existing_slot_value(
            card.problem_context,
            extraction.problem_context,
            field="problem_context",
            title=card.title,
        )
        merged_limitations = _normalize_limitations_value(
            _prefer_existing_slot_value(card.limitations, extraction.limitations, field="limitations", title=card.title),
            has_explicit_support=has_explicit_limitations,
        )
        merged_method_core = _prefer_existing_slot_value(
            card.method_core,
            extraction.method_core,
            field="method_core",
            title=card.title,
        )
        merged_evidence_core = _prefer_existing_slot_value(
            card.evidence_core,
            extraction.evidence_core,
            field="evidence_core",
            title=card.title,
        )
        merged_formal_cause = _merge_formal_cause(card.formal_cause, extraction.formal_cause)
        merged_final_cause = _merge_final_cause(card.final_cause, extraction.final_cause)
        merged_quality_flag = _merge_quality_flag(
            title=card.title,
            paper_core=thesis,
            problem_context=merged_problem_context,
            method_core=merged_method_core,
            evidence_core=merged_evidence_core,
            limitations=merged_limitations,
            claim_ref_count=len(merged_claim_refs),
            candidate_flags=[card.quality_flag, extraction.quality_flag],
        )
        if _clean_text(card.quality_flag).casefold() == "ok" and merged_quality_flag != "reject":
            merged_quality_flag = "ok"
        was_enriched = any(
            [
                _clean_text(card.paper_core) != _clean_text(thesis),
                _clean_text(card.problem_context) != _clean_text(merged_problem_context),
                _clean_text(card.method_core) != _clean_text(merged_method_core),
                _clean_text(card.evidence_core) != _clean_text(merged_evidence_core),
                _clean_text(card.limitations) != _clean_text(merged_limitations),
                merged_claim_refs != list(card.claim_refs),
                merged_concepts != list(card.concept_links),
                _clean_text(card.quality_flag) != _clean_text(merged_quality_flag),
            ]
        )
        return PaperMemoryCard(
            memory_id=card.memory_id,
            paper_id=card.paper_id,
            source_note_id=card.source_note_id,
            title=card.title,
            paper_core=thesis,
            problem_context=merged_problem_context,
            method_core=merged_method_core,
            evidence_core=merged_evidence_core,
            limitations=merged_limitations,
            concept_links=merged_concepts,
            claim_refs=merged_claim_refs,
            formal_cause=merged_formal_cause,
            final_cause=merged_final_cause,
            published_at=card.published_at,
            evidence_window=card.evidence_window,
            search_text=_cap_join(
                [
                    card.title,
                    thesis,
                    " ".join(_clean_lines(list(extraction.claims), limit=4)),
                    merged_problem_context,
                    merged_method_core,
                    merged_evidence_core,
                    merged_limitations,
                    " ".join(merged_concepts),
                    " ".join(merged_claim_refs),
                ],
                limit=900,
            ),
            quality_flag=merged_quality_flag or "unscored",
            version=PROJECTED_ENRICHED_VERSION if was_enriched else (card.version or PROJECTED_VERSION),
            created_at=card.created_at,
            updated_at=card.updated_at,
        )
