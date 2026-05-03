"""Shared raw paper-summary helpers.

This module mirrors the existing paper-summary generation path without mutating
`papers.notes`. It is intended for labs evaluation and baseline artifact
generation, not for replacing the default `khub paper summarize` command.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from knowledge_hub.learning.task_router import get_llm_for_task
from knowledge_hub.papers.source_text import extract_pdf_text_excerpt, extract_salient_paper_text, usable_paper_notes


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def collect_paper_text(paper: dict[str, Any]) -> str:
    """Collect paper text using the same fallback order as the legacy summary path."""
    for key in ("translated_path", "text_path"):
        path_value = str(paper.get(key) or "").strip()
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        try:
            text = extract_salient_paper_text(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if text.strip():
            return text
    pdf_text = extract_pdf_text_excerpt(str(paper.get("pdf_path") or ""))
    if pdf_text:
        return pdf_text
    notes = usable_paper_notes(paper.get("notes"))
    if notes:
        return notes
    return str(paper.get("title") or "").strip()


def _source_hash(*, paper: dict[str, Any], text: str, quick: bool) -> str:
    payload = {
        "paper_id": str(paper.get("arxiv_id") or ""),
        "title": str(paper.get("title") or ""),
        "text": str(text or ""),
        "quick": bool(quick),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _raw_summary_paths(*, papers_dir: str, paper_id: str) -> dict[str, Path]:
    root = Path(str(papers_dir or "")).expanduser() / "summaries" / str(paper_id).strip()
    return {
        "artifact_dir": root,
        "markdown": root / "raw-baseline.md",
        "json": root / "raw-baseline.json",
    }


def build_raw_summary_artifact(
    sqlite_db: Any,
    config: Any,
    *,
    paper_id: str,
    quick: bool = False,
    allow_external: bool = True,
) -> dict[str, Any]:
    token = str(paper_id).strip()
    paper = sqlite_db.get_paper(token)
    if not paper:
        raise ValueError(f"paper not found: {paper_id}")

    text = collect_paper_text(paper)
    task_type = "materialization_summary" if quick else "rag_answer"
    llm, decision, warnings = get_llm_for_task(
        config,
        task_type=task_type,  # type: ignore[arg-type]
        allow_external=bool(allow_external),
        query=str(paper.get("title") or ""),
        context=text[:8000],
        source_count=1,
        timeout_sec=60 if quick else 90,
    )

    summary_text = ""
    fallback_used = False
    if llm is not None:
        try:
            if quick:
                summary_text = llm.summarize(text, language="ko", max_sentences=5)
            else:
                summary_text = llm.summarize_paper(text, title=str(paper.get("title") or ""), language="ko")
        except Exception as error:
            warnings = [*warnings, f"raw summary generation failed: {error}"]

    if not summary_text.strip():
        fallback_used = True
        warnings = [*warnings, "raw summary fallback used"]
        excerpt = _clean_text(text)[:700]
        if quick:
            summary_text = excerpt or str(paper.get("title") or token)
        else:
            summary_text = (
                "### 한줄 요약\n"
                f"{_clean_text(paper.get('title') or token)}\n\n"
                "### 핵심 기여\n"
                f"- {_clean_text(excerpt) or '원문 텍스트 기반 fallback 요약'}\n\n"
                "### 방법론\n"
                "- 자동 raw baseline fallback\n\n"
                "### 주요 실험\n"
                "- 상세 요약 생성에 실패해 원문 일부만 보존했습니다.\n\n"
                "### 한계\n"
                "- LLM 경로를 사용할 수 없어 구조화 품질이 제한적입니다.\n"
            )

    paths = _raw_summary_paths(papers_dir=getattr(config, "papers_dir", ""), paper_id=token)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "paperId": token,
        "paperTitle": _clean_text(paper.get("title")),
        "provider": _clean_text(getattr(decision, "provider", "")),
        "model": _clean_text(getattr(decision, "model", "")),
        "route": _clean_text(getattr(decision, "route", "")),
        "fallbackUsed": bool(
            fallback_used
            or any(str(reason).startswith("fallback_from_") for reason in list(getattr(decision, "reasons", []) or []))
        ),
        "builtAt": built_at,
        "quick": bool(quick),
        "sourceHash": _source_hash(paper=paper, text=text, quick=bool(quick)),
        "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
        "summaryText": str(summary_text),
        "paths": {
            "artifactDir": str(paths["artifact_dir"]),
            "markdownPath": str(paths["markdown"]),
            "jsonPath": str(paths["json"]),
        },
    }
    paths["markdown"].write_text(str(summary_text), encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_raw_summary_artifact(*, papers_dir: str, paper_id: str) -> dict[str, Any] | None:
    paths = _raw_summary_paths(papers_dir=papers_dir, paper_id=paper_id)
    if not paths["json"].exists():
        return None
    try:
        return json.loads(paths["json"].read_text(encoding="utf-8"))
    except Exception:
        return None


__all__ = ["build_raw_summary_artifact", "collect_paper_text", "load_raw_summary_artifact"]
