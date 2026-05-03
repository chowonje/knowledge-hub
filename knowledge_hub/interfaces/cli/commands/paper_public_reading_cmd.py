"""Public paper reading commands kept separate from maintenance operations."""

from __future__ import annotations

import click
from rich.console import Console
from rich.markdown import Markdown

from knowledge_hub.papers.public_surface import (
    build_public_evidence_card,
    build_public_memory_card,
    build_public_related_card,
    build_public_summary_card,
)

console = Console()


@click.command("summary")
@click.option("--paper-id", required=True, help="paper id / arXiv id")
@click.option("--json/--no-json", "as_json", default=False, show_default=True)
@click.pass_context
def paper_public_summary(ctx, paper_id, as_json):
    """사용자용 논문 요약 카드"""
    payload = build_public_summary_card(ctx.obj["khub"], paper_id=str(paper_id).strip())
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
        entries = list(((payload.get("evidenceSummary") or {}).get(field) or {}).get("summaryLines") or [])
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
    payload = build_public_evidence_card(ctx.obj["khub"], paper_id=str(paper_id).strip())
    if as_json:
        console.print_json(data=payload)
        return
    lines = [
        f"# {payload.get('paperTitle') or payload.get('paperId')}",
        "",
        "## 결과 근거 요약",
        "",
    ]
    for item in list(((payload.get("evidenceSummary") or {}).get("keyResults") or {}).get("summaryLines") or []):
        lines.append(f"- {item}")
    lines.extend(["", "## 한계 근거 요약", ""])
    for item in list(((payload.get("evidenceSummary") or {}).get("limitations") or {}).get("summaryLines") or []):
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
    payload = build_public_memory_card(ctx.obj["khub"], paper_id=str(paper_id).strip())
    if as_json:
        console.print_json(data=payload)
        return
    claim_coverage = dict(payload.get("claimCoverage") or {})
    memory = dict(payload.get("memoryCard") or {})
    lines = [
        f"# {payload.get('paperTitle') or payload.get('paperId')}",
        "",
        f"- claims: {claim_coverage.get('normalizedClaims', 0)}/{claim_coverage.get('totalClaims', 0)} normalized",
        f"- summary status: {((payload.get('artifactStatus') or {}).get('summary') or 'missing')}",
        f"- memory status: {((payload.get('artifactStatus') or {}).get('memory') or 'missing')}",
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
    payload = build_public_related_card(ctx.obj["khub"], paper_id=str(paper_id).strip(), top_k=max(1, int(top_k)))
    if as_json:
        console.print_json(data=payload)
        return
    lines = [f"# {payload.get('paperTitle') or payload.get('paperId')}", "", "## 연결된 지식", ""]
    for item in list(payload.get("relatedKnowledge") or []):
        lines.append(f"- {item.get('title')} ({item.get('sourceType')}, score={float(item.get('score') or 0.0):.3f})")
    console.print(Markdown("\n".join(lines)))


__all__ = [
    "paper_public_evidence",
    "paper_public_memory",
    "paper_public_related",
    "paper_public_summary",
]
