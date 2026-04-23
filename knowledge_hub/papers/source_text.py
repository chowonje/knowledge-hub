"""Helpers for selecting usable paper source text."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

_DEGRADED_NOTE_HINTS = (
    "원문",
    "원문(또는",
    "원문을 보내주시면",
    "원문을 올려주시면",
    "논문 원문",
    "pdf 링크",
    "pdf)을 제공",
    "직접 검색하거나",
    "현재 주신 정보",
    "제목·저자",
    "제목과 저자",
    "arxiv / doi / 저널 링크",
    "arxiv id, doi",
    "provide the original paper text",
    "original paper text is required",
    "need the original paper text",
    "paper text is required",
    "unable to summarize",
    "cannot summarize",
)
_SPACE_RE = re.compile(r"\s+")
_ABSTRACT_MARKER_RE = re.compile(
    r"(?:\\begin\{abstract\}|(?:^|[\s:>])abstract(?:[\s:.\-]|$)|(?:^|[\s:>])초록(?:[\s:.\-]|$)|(?:^|[\s:>])요약(?:[\s:.\-]|$))",
    re.IGNORECASE,
)
_TRAILING_SECTION_RE = re.compile(r"(?:\\(?:bibliography|begin\{references\}|section\{references\})|\breferences\b|\backnowledg(?:e)?ments?\b)", re.IGNORECASE)
_BLOCK_PREFIX_RE = re.compile(r"^\s*\[Block\s+\d+\]\s*", re.IGNORECASE)
_LATEX_NOISE_RE = re.compile(
    r"\\(?:documentclass|usepackage|maketitle|title|author|thanks|affiliation|email|and|keywords|bibliographystyle)\b(?:\[[^\]]*\])?(?:\{[^{}]*\})?",
    re.IGNORECASE,
)
_LATEX_SECTION_RE = re.compile(r"\\(?:begin|end)\{abstract\}|\\(?:section|subsection|subsubsection|paragraph)\{([^{}]*)\}", re.IGNORECASE)
_LATEX_INLINE_RE = re.compile(r"\\(?:cite|ref|label|footnote|url|href)\b(?:\[[^\]]*\])?(?:\{[^{}]*\})?", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_CITATION_STUB_RE = re.compile(r"^citations?\s*:\s*\d+\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class PaperSourceSnapshot:
    source_key: str
    content: str
    source_content_hash: str
    path: str = ""
    warnings: tuple[str, ...] = ()


def clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())


def extract_salient_paper_text(value: Any, *, max_chars: int = 20_000) -> str:
    raw = str(value or "")
    if not raw.strip():
        return ""
    if _CITATION_STUB_RE.fullmatch(clean_text(raw)):
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLOCK_PREFIX_RE.sub("", text)
    marker = _ABSTRACT_MARKER_RE.search(text)
    if marker:
        text = text[marker.end() :]
    trailing = _TRAILING_SECTION_RE.search(text)
    if trailing:
        text = text[: trailing.start()]
    text = _LATEX_SECTION_RE.sub(lambda m: f" {m.group(1) or ' '} ", text)
    text = _LATEX_NOISE_RE.sub(" ", text)
    text = _LATEX_INLINE_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = clean_text(text)
    if not text or _CITATION_STUB_RE.fullmatch(text):
        return ""
    return text[:max_chars]


def looks_like_unusable_paper_notes(value: Any) -> bool:
    token = clean_text(value)
    if not token:
        return False
    lowered = token.casefold()
    return any(hint in lowered for hint in _DEGRADED_NOTE_HINTS)


def usable_paper_notes(value: Any) -> str:
    token = clean_text(value)
    if not token:
        return ""
    if looks_like_unusable_paper_notes(token):
        return ""
    return token


def source_hash_for_path(path_value: str) -> str:
    token = str(path_value or "").strip()
    if not token:
        return ""
    path = Path(token).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def source_hash_for_text(*parts: Any) -> str:
    text = "\n".join(clean_text(part) for part in parts if clean_text(part))
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_paper_source_snapshot(
    paper: dict[str, Any],
    *,
    pdf_text_extractor: Any = None,
) -> PaperSourceSnapshot:
    token = clean_text((paper or {}).get("arxiv_id") or (paper or {}).get("paper_id"))
    warnings: list[str] = []
    for key in ("translated_path", "text_path"):
        path_value = str((paper or {}).get(key) or "").strip()
        if not path_value:
            continue
        path = Path(path_value).expanduser()
        if not path.exists():
            warnings.append(f"missing_{key}")
            continue
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception:
            warnings.append(f"unreadable_{key}")
            continue
        text = extract_salient_paper_text(raw_text)
        if not text:
            warnings.append(f"unusable_{key}")
            continue
        path_hash = source_hash_for_path(str(path))
        return PaperSourceSnapshot(
            source_key=key,
            content=text,
            source_content_hash=path_hash or source_hash_for_text(raw_text, token, key),
            path=str(path),
            warnings=tuple(warnings),
        )

    pdf_path = str((paper or {}).get("pdf_path") or "").strip()
    if pdf_path:
        extractor = pdf_text_extractor or extract_pdf_text_excerpt
        try:
            text = extractor(pdf_path)
        except Exception:
            text = ""
        if text:
            return PaperSourceSnapshot(
                source_key="pdf_text",
                content=text,
                source_content_hash=source_hash_for_path(pdf_path) or source_hash_for_text(text, token, "pdf_text"),
                path=pdf_path,
                warnings=tuple(warnings),
            )
        warnings.append("unusable_pdf_text")

    notes = usable_paper_notes((paper or {}).get("notes"))
    if notes:
        return PaperSourceSnapshot(
            source_key="notes",
            content=notes,
            source_content_hash=source_hash_for_text(notes, token, "notes"),
            path="",
            warnings=tuple(warnings),
        )

    title = str((paper or {}).get("title") or token or "").strip()
    return PaperSourceSnapshot(
        source_key="title",
        content=title,
        source_content_hash=source_hash_for_text(title, token, "title"),
        path="",
        warnings=tuple(warnings),
    )


def extract_pdf_text_excerpt(
    pdf_path: str,
    *,
    max_pages: int = 5,
    max_chars: int = 20_000,
) -> str:
    token = str(pdf_path or "").strip()
    if not token:
        return ""
    path = Path(token).expanduser()
    if not path.exists():
        return ""
    text = _extract_pdf_text_excerpt_pymupdf(path, max_pages=max_pages, max_chars=max_chars)
    if text:
        return text
    return _extract_pdf_text_excerpt_pypdf(path, max_pages=max_pages, max_chars=max_chars)


def _extract_pdf_text_excerpt_pymupdf(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
) -> str:
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        document = fitz.open(str(path))
    except Exception:
        return ""

    parts: list[str] = []
    total = 0
    try:
        page_total = getattr(document, "page_count", None)
        if page_total is None:
            try:
                page_total = len(document)
            except Exception:
                page_total = 0
        page_count = min(max(1, int(max_pages)), int(page_total or 0))
        for page_index in range(page_count):
            try:
                page = document.load_page(page_index)
                text = extract_salient_paper_text(page.get_text("text") or "", max_chars=max_chars)
            except Exception:
                continue
            if not text:
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            parts.append(text[:remaining])
            total += len(parts[-1])
            if total >= max_chars:
                break
    finally:
        try:
            document.close()
        except Exception:
            pass
    return extract_salient_paper_text(" ".join(parts), max_chars=max_chars)


def _extract_pdf_text_excerpt_pypdf(
    path: Path,
    *,
    max_pages: int,
    max_chars: int,
) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""

    parts: list[str] = []
    total = 0
    for page in list(reader.pages)[: max(1, int(max_pages))]:
        try:
            text = extract_salient_paper_text(page.extract_text() or "", max_chars=max_chars)
        except Exception:
            continue
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        parts.append(text[:remaining])
        total += len(parts[-1])
        if total >= max_chars:
            break
    return extract_salient_paper_text(" ".join(parts), max_chars=max_chars)


__all__ = [
    "PaperSourceSnapshot",
    "clean_text",
    "extract_salient_paper_text",
    "extract_pdf_text_excerpt",
    "looks_like_unusable_paper_notes",
    "resolve_paper_source_snapshot",
    "source_hash_for_path",
    "source_hash_for_text",
    "usable_paper_notes",
]
