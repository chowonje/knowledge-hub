from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

_CITATION_RE: Final = re.compile(r"\[S\d+\]")
_SOURCE_BACKED_KINDS: Final = {"raw_span", "raw_source", "parsed_artifact_evidence_chunk"}


def should_render_paper_lookup_text(source_type: str | None, result: Mapping[str, Any]) -> bool:
    if str(source_type or "").strip().lower() != "paper":
        return False
    query_frame = _mapping(result.get("queryFrame"))
    evidence_packet = _mapping(result.get("evidencePacket"))
    family = str(query_frame.get("family") or evidence_packet.get("paperFamily") or "").strip().lower()
    if family:
        return family == "paper_lookup"
    paper_ids = {_paper_id(source) for source in _paper_sources(result)}
    paper_ids.discard("")
    return len(paper_ids) == 1


def render_paper_lookup_text(question: str, result: Mapping[str, Any]) -> str:
    paper_sources = _paper_sources(result)
    source_backed = [source for source in paper_sources if _is_source_backed(source)]
    source_backed_excerpts = [source for source in source_backed if _excerpt(source)]
    card_sources = [source for source in paper_sources if _is_paper_card_v2(source)]
    citation_labels = _citation_labels(result, paper_sources)
    answer = _ensure_inline_citation(str(result.get("answer") or "").strip(), citation_labels)
    title, paper_id = _paper_identity(source_backed or paper_sources, result)
    state, note = _evidence_state(result, source_backed=source_backed, card_sources=card_sources)

    lines = [f"Q: {question}", "", answer or "No answer produced.", ""]
    lines.append(f"Paper: {title}{_arxiv_suffix(paper_id)}")
    lines.append(f"Evidence state: {state}")
    if note:
        lines.append(f"Evidence note: {note}")

    if source_backed_excerpts:
        lines.extend(["", "Source-backed excerpts:"])
        for source in source_backed_excerpts[:2]:
            label = _display_label(str(source.get("citation_label") or ""))
            prefix = f"{label} " if label else ""
            lines.append(f"{prefix}{_excerpt(source)}")
    elif source_backed:
        lines.append("Source-backed excerpts: unavailable in the selected source rows.")

    if citation_labels:
        labels = ", ".join(_display_label(label) for label in citation_labels)
        lines.extend(["", f"Citations: {labels}"])

    return "\n".join(lines).rstrip() + "\n"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _paper_sources(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for value in _sequence(result.get("sources")):
        source = _mapping(value)
        if _is_paper_source(source):
            sources.append(source)
    return sources


def _is_paper_source(source: Mapping[str, Any]) -> bool:
    source_type = str(source.get("source_type") or source.get("sourceType") or "").strip().lower()
    return source_type == "paper" or bool(_paper_id(source))


def _is_source_backed(source: Mapping[str, Any]) -> bool:
    evidence_kind = str(source.get("evidence_kind") or "").strip().lower()
    source_trace = _mapping(source.get("source_trace") or source.get("sourceTrace"))
    trace_kind = str(source_trace.get("evidenceKind") or "").strip().lower()
    retrieval_mode = str(source.get("retrieval_mode") or source.get("retrievalMode") or "").strip().lower()
    return evidence_kind in _SOURCE_BACKED_KINDS or trace_kind in _SOURCE_BACKED_KINDS or retrieval_mode == "active-vector-paper"


def _is_paper_card_v2(source: Mapping[str, Any]) -> bool:
    retrieval_mode = str(source.get("retrieval_mode") or source.get("retrievalMode") or "").strip().lower()
    memory_provenance = _mapping(source.get("memory_provenance") or source.get("memoryProvenance"))
    derivative_source = _mapping(source.get("derivative_source") or source.get("derivativeSource"))
    derivative_memory = _mapping(derivative_source.get("memoryProvenance"))
    return (
        retrieval_mode == "paper-card-v2"
        or str(memory_provenance.get("mode") or "").strip().lower() == "paper-card-v2"
        or str(derivative_memory.get("mode") or "").strip().lower() == "paper-card-v2"
    )


def _citation_labels(result: Mapping[str, Any], sources: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for value in _sequence(result.get("citations")):
        label = _normal_label(str(_mapping(value).get("label") or ""))
        if label and label not in labels:
            labels.append(label)
    for source in sources:
        label = _normal_label(str(source.get("citation_label") or source.get("citationLabel") or ""))
        if label and label not in labels:
            labels.append(label)
    return labels


def _normal_label(label: str) -> str:
    return label.strip().removeprefix("[").removesuffix("]")


def _display_label(label: str) -> str:
    normalized = _normal_label(label)
    return f"[{normalized}]" if normalized else ""


def _ensure_inline_citation(answer: str, labels: list[str]) -> str:
    if not answer or not labels or _CITATION_RE.search(answer):
        return answer
    return f"{answer} {_display_label(labels[0])}"


def _paper_identity(sources: list[dict[str, Any]], result: Mapping[str, Any]) -> tuple[str, str]:
    for source in sources:
        title = str(source.get("title") or source.get("parent_label") or "").strip()
        paper_id = _paper_id(source)
        if title or paper_id:
            return title or "Unknown paper", paper_id
    for value in _sequence(result.get("citations")):
        citation = _mapping(value)
        title = str(citation.get("title") or "").strip()
        paper_id = str(citation.get("target") or citation.get("source_id") or "").strip()
        if title or paper_id:
            return title or "Unknown paper", paper_id
    return "Unknown paper", ""


def _paper_id(source: Mapping[str, Any]) -> str:
    for key in ("arxiv_id", "arxivId", "citation_target", "target", "source_ref", "source_id"):
        value = str(source.get(key) or "").strip()
        if value:
            return value
    return ""


def _arxiv_suffix(paper_id: str) -> str:
    return f" (arXiv:{paper_id})" if paper_id else ""


def _evidence_state(
    result: Mapping[str, Any],
    *,
    source_backed: list[dict[str, Any]],
    card_sources: list[dict[str, Any]],
) -> tuple[str, str]:
    if not _answerable(result):
        return "insufficient", "source-backed evidence is insufficient for a checkable answer."
    if source_backed and _verification_caution(result):
        return "verifier weakness", "raw paper evidence is selected, but verification still marked caution."
    if source_backed:
        return "source-backed", ""
    if card_sources:
        return "insufficient", "paper-card-v2 only; no raw source-backed excerpt was selected."
    return "insufficient", "no raw source-backed excerpt was selected."


def _answerable(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status") or "").strip().lower()
    if status in {"blocked", "error", "failed", "no_result"}:
        return False
    evidence_packet = _mapping(result.get("evidencePacket"))
    if "answerable" in evidence_packet:
        return bool(evidence_packet.get("answerable"))
    if "answerable" in result:
        return bool(result.get("answerable"))
    return bool(str(result.get("answer") or "").strip())


def _verification_caution(result: Mapping[str, Any]) -> bool:
    verification = _mapping(result.get("answerVerification"))
    status = str(verification.get("status") or "").strip().lower()
    return bool(verification.get("needsCaution")) or status in {"caution", "failed", "fail", "warning"}


def _excerpt(source: Mapping[str, Any]) -> str:
    text = str(source.get("excerpt") or "").strip()
    if len(text) <= 700:
        return text
    return text[:697].rstrip() + "..."
