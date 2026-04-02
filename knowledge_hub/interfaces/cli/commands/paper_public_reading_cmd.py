"""Public paper reading commands kept separate from maintenance operations."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown

from .paper_shared_runtime import (
    _compact_evidence_summary,
    _ensure_public_paper_summary,
    _public_paper_memory,
    _public_related_knowledge,
    _sqlite_db,
)

console = Console()


@click.command("summary")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_public_summary(ctx, paper_id, as_json):
    """사용자용 논문 요약 카드"""
    summary_payload, user_card = _ensure_public_paper_summary(ctx.obj["khub"], str(paper_id).strip())
    payload = {
        "schema": "knowledge-hub.paper.public.summary.v1",
        "status": user_card.get("status") or summary_payload.get("status"),
        "paperId": user_card.get("paperId"),
        "paperTitle": user_card.get("paperTitle"),
        "parserUsed": user_card.get("parserUsed"),
        "fallbackUsed": user_card.get("fallbackUsed"),
        "llmRoute": user_card.get("llmRoute"),
        "summary": user_card.get("summary"),
        "evidenceSummary": _compact_evidence_summary(summary_payload, include_refs=False),
        "claimCoverage": user_card.get("claimCoverage") or summary_payload.get("claimCoverage"),
        "warnings": user_card.get("warnings"),
    }
    if as_json:
        console.print_json(data=payload)
        return
    summary = dict(payload.get("summary") or {})
    lines = [
        f"# {payload.get('paperTitle') or payload.get('paperId')}",
        "",
        f"- parser: {payload.get('parserUsed')}",
        f"- fallback: {bool(payload.get('fallbackUsed'))}",
        "",
        "## 한줄 요약",
        "",
        str(summary.get("oneLine") or ""),
        "",
        "## 문제",
        "",
        str(summary.get("problem") or ""),
        "",
        "## 핵심 아이디어",
        "",
        str(summary.get("coreIdea") or ""),
        "",
        "## 방법",
        "",
    ]
    for item in list(summary.get("methodSteps") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 주요 결과", ""])
    for item in list(summary.get("keyResults") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 한계", ""])
    for item in list(summary.get("limitations") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 근거 요약", ""])
    for field in ("keyResults", "limitations", "whatIsNew"):
        entries = list((payload.get("evidenceSummary") or {}).get(field) or [])
        if not entries:
            continue
        lines.append(f"### {field}")
        for item in entries:
            lines.append(f"- {item}")
    console.print(Markdown("\n".join(lines)))


@click.command("evidence")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_public_evidence(ctx, paper_id, as_json):
    """사용자용 논문 근거 카드"""
    summary_payload, user_card = _ensure_public_paper_summary(ctx.obj["khub"], str(paper_id).strip())
    payload = {
        "schema": "knowledge-hub.paper.public.evidence.v1",
        "status": user_card.get("status") or summary_payload.get("status"),
        "paperId": user_card.get("paperId"),
        "paperTitle": user_card.get("paperTitle"),
        "evidenceSummary": _compact_evidence_summary(summary_payload, include_refs=True),
        "evidenceMap": list(summary_payload.get("evidenceMap") or []),
        "claimCoverage": user_card.get("claimCoverage") or summary_payload.get("claimCoverage"),
        "warnings": user_card.get("warnings"),
    }
    if as_json:
        console.print_json(data=payload)
        return
    lines = [
        f"# {payload.get('paperTitle') or payload.get('paperId')}",
        "",
        "## 결과 근거 요약",
        "",
    ]
    for item in list((payload.get("evidenceSummary") or {}).get("keyResults") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 한계 근거 요약", ""])
    for item in list((payload.get("evidenceSummary") or {}).get("limitations") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## Evidence Map", ""])
    for item in list(payload.get("evidenceMap") or []):
        lines.append(
            f"- `{item.get('field')}` :: {item.get('sectionPath') or item.get('unitId')} (page={item.get('page')}) :: {item.get('excerpt')}"
        )
    console.print(Markdown("\n".join(lines)))


@click.command("memory")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_public_memory(ctx, paper_id, as_json):
    """사용자용 논문 기억 카드"""
    khub = ctx.obj["khub"]
    payload, user_card = _ensure_public_paper_summary(khub, str(paper_id).strip())
    memory_card = _public_paper_memory(_sqlite_db(khub.config, khub=khub), config=khub.config, paper_id=str(paper_id).strip())
    claim_coverage = dict(user_card.get("claimCoverage") or {})
    evidence_map = list(user_card.get("evidenceMap") or [])
    page_grounded = sum(1 for item in evidence_map if item.get("page") is not None)
    result = {
        "schema": "knowledge-hub.paper.public.memory.v1",
        "status": user_card.get("status") or payload.get("status"),
        "paperId": user_card.get("paperId"),
        "paperTitle": user_card.get("paperTitle"),
        "claimCoverage": claim_coverage,
        "provenance": {
            "evidenceRefCount": len(evidence_map),
            "pageGroundedCount": page_grounded,
            "parserUsed": payload.get("parserUsed"),
            "paperSummaryAvailable": bool(payload),
            "paperMemoryAvailable": bool(memory_card),
        },
        "memory": {
            "paperCore": memory_card.get("paperCore", ""),
            "problemContext": memory_card.get("problemContext", ""),
            "methodCore": memory_card.get("methodCore", ""),
            "evidenceCore": memory_card.get("evidenceCore", ""),
        },
        "memoryCard": {
            "paperCore": memory_card.get("paperCore", ""),
            "problemContext": memory_card.get("problemContext", ""),
            "methodCore": memory_card.get("methodCore", ""),
            "evidenceCore": memory_card.get("evidenceCore", ""),
        },
        "warnings": user_card.get("warnings"),
    }
    if as_json:
        console.print_json(data=result)
        return
    memory = dict(result.get("memoryCard") or {})
    lines = [
        f"# {result.get('paperTitle') or result.get('paperId')}",
        "",
        f"- claims: {claim_coverage.get('normalizedClaims', 0)}/{claim_coverage.get('totalClaims', 0)} normalized",
        f"- evidence refs: {result['provenance']['evidenceRefCount']}",
        "",
        "## 기억",
        "",
        f"- paper core: {memory.get('paperCore')}",
        f"- problem context: {memory.get('problemContext')}",
        f"- method core: {memory.get('methodCore')}",
        f"- evidence core: {memory.get('evidenceCore')}",
    ]
    console.print(Markdown("\n".join(lines)))


@click.command("related")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--top-k", default=5, show_default=True, help="source별 최대 결과 수")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_public_related(ctx, paper_id, top_k, as_json):
    """사용자용 연결된 지식 카드"""
    khub = ctx.obj["khub"]
    summary_payload, user_card = _ensure_public_paper_summary(khub, str(paper_id).strip())
    related_groups = _public_related_knowledge(
        khub,
        paper_id=str(paper_id).strip(),
        paper_title=str(user_card.get("paperTitle") or ""),
        top_k=max(1, int(top_k)),
    )
    related = [
        {**dict(item or {}), "group": group}
        for group, items in related_groups.items()
        for item in list(items or [])
    ]
    payload = {
        "schema": "knowledge-hub.paper.public.related.v1",
        "status": user_card.get("status") or summary_payload.get("status"),
        "paperId": user_card.get("paperId"),
        "paperTitle": user_card.get("paperTitle"),
        "query": str(user_card.get("paperTitle") or paper_id),
        "relatedKnowledge": related,
        "claimCoverage": user_card.get("claimCoverage") or summary_payload.get("claimCoverage"),
        "warnings": user_card.get("warnings"),
    }
    if as_json:
        console.print_json(data=payload)
        return
    lines = [f"# {payload.get('paperTitle') or payload.get('paperId')}", "", "## 연결된 지식", ""]
    for item in list(related or []):
        lines.append(f"- {item.get('title')} ({item.get('sourceType')}, score={float(item.get('score') or 0.0):.3f})")
    console.print(Markdown("\n".join(lines)))


__all__ = [
    "paper_public_evidence",
    "paper_public_memory",
    "paper_public_related",
    "paper_public_summary",
]
