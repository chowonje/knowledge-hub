"""Additive sanitation helpers for extracted paper text."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

_LATEX_PREFIX_RE = re.compile(r"^\s*\\(?:documentclass|usepackage|input|title|author|date|begin|end|maketitle|tableofcontents|bibliographystyle|bibliography)\b")
_LATEX_INLINE_RE = re.compile(r"\\(?:documentclass|usepackage|title|author|begin\{document\}|maketitle)\b")
_SEMANTIC_SECTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:abstract|introduction|1[\.\s]+introduction|summary|overview)\b[:\s-]*$",
    re.IGNORECASE,
)
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:references|bibliography|works cited|appendix|supplementary material)\b[:\s-]*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
_MATHY_CHARS_RE = re.compile(r"[\\{}_^$]|\\[A-Za-z]+")


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _looks_like_prose(line: str) -> bool:
    body = str(line or "").strip()
    if len(body) < 80:
        return False
    if _LATEX_PREFIX_RE.match(body):
        return False
    words = _WORD_RE.findall(body)
    return len(words) >= 8


def _is_command_heavy_line(line: str) -> bool:
    body = str(line or "").strip()
    if not body:
        return False
    if _LATEX_PREFIX_RE.match(body):
        return True
    words = _WORD_RE.findall(body)
    command_hits = len(_MATHY_CHARS_RE.findall(body))
    if not words and command_hits >= 2:
        return True
    if len(words) <= 3 and command_hits >= 3:
        return True
    symbol_count = sum(1 for char in body if char in "\\{}_^$[]()")
    return len(body) < 120 and symbol_count >= 12 and len(words) <= 4


def _drop_leading_boilerplate(lines: list[str], *, start_index: int) -> tuple[list[str], int]:
    body = lines[start_index:]
    cleaned: list[str] = []
    skipping = True
    dropped_latex_line_count = 0
    for line in body:
        stripped = line.strip()
        if skipping and (not stripped or _is_command_heavy_line(stripped) or stripped.startswith("%")):
            if stripped:
                dropped_latex_line_count += 1
            continue
        skipping = False
        if _is_command_heavy_line(stripped):
            dropped_latex_line_count += 1
            continue
        cleaned.append(line)
    return cleaned, dropped_latex_line_count


def _drop_references_tail(lines: list[str], *, min_anchor_index: int) -> tuple[list[str], bool]:
    if not lines:
        return lines, False
    min_tail_index = max(min_anchor_index + 2, int(len(lines) * 0.35))
    for index, line in enumerate(lines):
        if index < min_tail_index:
            continue
        if _REFERENCES_HEADING_RE.match(str(line or "").strip()):
            return lines[:index], True
    return lines, False


def _find_semantic_start(lines: list[str]) -> tuple[int, str]:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _SEMANTIC_SECTION_RE.match(stripped):
            return index, "section_heading"
    for index, line in enumerate(lines):
        if _looks_like_prose(line):
            return index, "first_long_prose"
    return 0, "start_of_text"


def extract_keyword_window(text: str, keywords: tuple[str, ...], *, limit: int = 420) -> str:
    lowered = str(text or "").casefold()
    if not lowered:
        return ""
    positions = [lowered.find(token.casefold()) for token in keywords if token]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return ""
    start = min(positions)
    excerpt = _clean_text(str(text or "")[start : start + max(80, int(limit) * 2)])
    return excerpt[:limit].rstrip()


@dataclass
class SanitizedPaperText:
    original_length: int
    sanitized_text: str
    starts_with_latex: bool
    semantic_start_line: int
    semantic_start_reason: str
    trimmed_prefix_lines: int
    removed_references_tail: bool = False
    dropped_latex_line_count: int = 0
    selected_start_anchor: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def weak_content(self) -> bool:
        return len(_clean_text(self.sanitized_text)) < 200

    def to_dict(self) -> dict[str, object]:
        return {
            "originalLength": int(self.original_length),
            "sanitizedLength": len(_clean_text(self.sanitized_text)),
            "startsWithLatex": bool(self.starts_with_latex),
            "semanticStartLine": int(self.semantic_start_line),
            "semanticStartReason": str(self.semantic_start_reason),
            "trimmedPrefixLines": int(self.trimmed_prefix_lines),
            "removedReferencesTail": bool(self.removed_references_tail),
            "droppedLatexLineCount": int(self.dropped_latex_line_count),
            "selectedStartAnchor": str(self.selected_start_anchor),
            "weakContent": bool(self.weak_content),
            "warnings": list(self.warnings),
        }


@dataclass
class PaperTextNormalization:
    translated: SanitizedPaperText
    raw: SanitizedPaperText
    preferred_source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def preferred_text(self) -> str:
        if self.preferred_source == "translated":
            return self.translated.sanitized_text
        return self.raw.sanitized_text

    @property
    def weak_content(self) -> bool:
        if _clean_text(self.preferred_text):
            return len(_clean_text(self.preferred_text)) < 200
        return self.translated.weak_content and self.raw.weak_content

    def to_dict(self) -> dict[str, object]:
        return {
            "preferredSource": str(self.preferred_source),
            "preferredLength": len(_clean_text(self.preferred_text)),
            "weakContent": bool(self.weak_content),
            "warnings": list(self.warnings),
            "translated": self.translated.to_dict(),
            "raw": self.raw.to_dict(),
        }


def sanitize_paper_text(text: str) -> SanitizedPaperText:
    body = str(text or "")
    if not body.strip():
        return SanitizedPaperText(
            original_length=0,
            sanitized_text="",
            starts_with_latex=False,
            semantic_start_line=0,
            semantic_start_reason="empty",
            trimmed_prefix_lines=0,
            warnings=["empty_text"],
        )
    lines = body.splitlines()
    first_nonempty = next((line.strip() for line in lines if line.strip()), "")
    starts_with_latex = bool(_LATEX_PREFIX_RE.match(first_nonempty))
    start_index, reason = _find_semantic_start(lines)
    cleaned_lines, dropped_latex_line_count = _drop_leading_boilerplate(lines, start_index=start_index)
    cleaned_lines, removed_references_tail = _drop_references_tail(cleaned_lines, min_anchor_index=0)
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    warnings: list[str] = []
    if starts_with_latex:
        warnings.append("starts_with_latex")
    if _LATEX_INLINE_RE.search(cleaned[:200]):
        warnings.append("latex_marker_near_start")
    if removed_references_tail:
        warnings.append("references_tail_removed")
    result = SanitizedPaperText(
        original_length=len(body),
        sanitized_text=cleaned,
        starts_with_latex=starts_with_latex,
        semantic_start_line=start_index,
        semantic_start_reason=reason,
        trimmed_prefix_lines=max(0, start_index),
        removed_references_tail=removed_references_tail,
        dropped_latex_line_count=dropped_latex_line_count,
        selected_start_anchor=f"{reason}:{start_index}",
        warnings=warnings,
    )
    if result.weak_content:
        result.warnings.append("sanitized_too_short")
    return result


def normalize_paper_texts(*, translated_text: str, raw_text: str) -> PaperTextNormalization:
    translated = sanitize_paper_text(translated_text)
    raw = sanitize_paper_text(raw_text)
    preferred_source = "translated"
    if translated.weak_content and not raw.weak_content and _clean_text(raw.sanitized_text):
        preferred_source = "raw"
    elif not _clean_text(translated.sanitized_text) and _clean_text(raw.sanitized_text):
        preferred_source = "raw"
    warnings: list[str] = []
    if translated.starts_with_latex:
        warnings.append("translated_starts_with_latex")
    if raw.starts_with_latex:
        warnings.append("raw_starts_with_latex")
    if translated.weak_content and raw.weak_content:
        warnings.append("all_sources_weak")
    return PaperTextNormalization(
        translated=translated,
        raw=raw,
        preferred_source=preferred_source,
        warnings=warnings,
    )


__all__ = [
    "PaperTextNormalization",
    "SanitizedPaperText",
    "extract_keyword_window",
    "normalize_paper_texts",
    "sanitize_paper_text",
]
