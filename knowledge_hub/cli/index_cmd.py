"""
khub index - 논문 + 개념 노트를 벡터DB에 통합 인덱싱 (배치 임베딩)

논문: 제목 + 요약 + 키워드를 임베딩
개념: 설명 + 관련 개념 + 관련 논문을 임베딩
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import click
from rich.console import Console

console = Console()
log = logging.getLogger("khub.index")

BATCH_SIZE = 20


def _get_embedder(config):
    """config 기반으로 적절한 Embedder 인스턴스 생성"""
    from knowledge_hub.providers.registry import get_embedder
    embed_cfg = config.get_provider_config(config.embedding_provider)
    return get_embedder(config.embedding_provider, model=config.embedding_model, **embed_cfg)


def _embed_batch_via_provider(embedder, texts: list[str]) -> list[list[float]]:
    """프로바이더 추상화를 통해 배치 임베딩"""
    results = embedder.embed_batch(texts, show_progress=False)
    failed = [i for i, r in enumerate(results) if r is None]
    if failed:
        raise RuntimeError(f"{len(failed)}개 텍스트 임베딩 실패 (인덱스: {failed[:5]})")
    return results


def _get_paper_keywords(vault_path: str) -> dict[str, list[str]]:
    """Obsidian 논문 노트에서 arxiv_id → 키워드 목록 매핑 추출"""
    papers_dir = Path(vault_path) / "Papers"
    if not papers_dir.exists():
        alt = Path(vault_path) / "Projects" / "AI" / "AI_Papers"
        if alt.exists():
            papers_dir = alt
        else:
            return {}

    mapping: dict[str, list[str]] = {}
    for md_path in papers_dir.glob("*.md"):
        if md_path.name == "00_Concept_Index.md":
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        arxiv_match = re.search(r'arxiv_id:\s*"?([0-9]+\.[0-9]+)"?', content)
        if not arxiv_match:
            continue
        aid = arxiv_match.group(1)
        concepts = re.findall(r'\[\[([^\]]+)\]\]', content)
        concepts = [c for c in concepts if c != "00_Concept_Index"]
        if concepts:
            mapping[aid] = concepts
    return mapping


def _load_concept_notes(vault_path: str) -> list[dict]:
    """Obsidian 개념 노트 로드 → 임베딩용 데이터 리스트 반환"""
    candidates = [
        Path(vault_path) / "Papers" / "Concepts",
        Path(vault_path) / "Projects" / "AI" / "AI_Papers" / "Concepts",
        Path(vault_path) / "Concepts",
    ]
    concepts_dir = None
    for c in candidates:
        if c.exists():
            concepts_dir = c
            break
    if concepts_dir is None:
        return []

    results = []
    for md_path in concepts_dir.glob("*.md"):
        try:
            content = md_path.read_text(encoding="utf-8")
        except Exception:
            continue
        name = md_path.stem

        desc_match = re.search(r'^# .+\n\n(.+?)(?:\n\n##|\Z)', content, re.MULTILINE | re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        related = re.findall(r'## 관련 개념\n((?:- \[\[.+?\]\]\n)*)', content)
        related_names = re.findall(r'\[\[([^\]]+)\]\]', related[0]) if related else []

        papers = re.findall(r'## 관련 논문\n((?:- \[\[.+?\]\]\n)*)', content)
        paper_names = re.findall(r'\[\[([^\]]+)\]\]', papers[0]) if papers else []

        text_parts = [f"Concept: {name}"]
        if description:
            text_parts.append(description)
        if related_names:
            text_parts.append(f"Related concepts: {', '.join(related_names)}")
        if paper_names:
            text_parts.append(f"Papers: {', '.join(paper_names)}")

        results.append({
            "name": name,
            "text": "\n\n".join(text_parts),
            "related": related_names,
            "papers": paper_names,
        })
    return results


@click.command("index")
@click.option("--all", "index_all", is_flag=True, help="이미 인덱싱된 논문도 재인덱싱")
@click.option("--concepts-only", is_flag=True, help="개념 노트만 인덱싱")
@click.pass_context
def index_cmd(ctx, index_all, concepts_only):
    """논문 + 개념 노트를 벡터DB에 통합 인덱싱"""
    from knowledge_hub.core.database import VectorDatabase, SQLiteDatabase

    khub = ctx.obj["khub"]
    config = khub.config

    try:
        embedder = _get_embedder(config)
    except Exception as e:
        console.print(f"[red]임베딩 프로바이더 초기화 실패: {e}[/red]")
        console.print("[dim]khub config providers 로 사용 가능한 프로바이더를 확인하세요.[/dim]")
        raise SystemExit(1)

    try:
        vector_db = VectorDatabase(config.vector_db_path, config.collection_name)
    except Exception as e:
        console.print(f"[red]벡터DB 초기화 실패: {e}[/red]")
        raise SystemExit(1)

    console.print(f"[dim]임베딩: {config.embedding_provider}/{config.embedding_model}[/dim]")

    t_start = time.time()
    failed_papers: list[dict] = []
    failed_concepts: list[dict] = []

    keyword_map = {}
    if config.vault_path:
        try:
            keyword_map = _get_paper_keywords(config.vault_path)
        except Exception as e:
            console.print(f"[yellow]키워드 로드 실패 (계속 진행): {e}[/yellow]")

    # ── Phase 1: 논문 인덱싱 ──
    paper_success = 0
    if not concepts_only:
        sqlite_db = SQLiteDatabase(config.sqlite_path)
        papers = sqlite_db.list_papers(limit=999)
        unindexed = papers if index_all else [p for p in papers if not p.get("indexed")]

        if unindexed:
            console.print(f"[bold]논문 인덱싱: {len(unindexed)}편[/bold]")

            for batch_start in range(0, len(unindexed), BATCH_SIZE):
                batch = unindexed[batch_start: batch_start + BATCH_SIZE]
                texts = []
                for p in batch:
                    t = f"Title: {p['title'] or p['arxiv_id']}"
                    keywords = keyword_map.get(p["arxiv_id"], [])
                    if keywords:
                        t += f"\nKeywords: {', '.join(keywords)}"
                    if p.get("notes"):
                        t += f"\n\n{p['notes']}"
                    texts.append(t)

                try:
                    embs = _embed_batch_via_provider(embedder, texts)

                    docs, embeddings, metas, ids = [], [], [], []
                    for p, text, emb in zip(batch, texts, embs):
                        docs.append(text)
                        embeddings.append(emb)
                        kw = keyword_map.get(p["arxiv_id"], [])
                        metas.append({
                            "title": p["title"] or "",
                            "arxiv_id": p["arxiv_id"],
                            "source_type": "paper",
                            "field": p.get("field", ""),
                            "keywords": ", ".join(kw[:10]),
                            "chunk_index": 0,
                        })
                        ids.append(f"paper_{p['arxiv_id']}_0")

                    vector_db.add_documents(
                        documents=docs, embeddings=embeddings, metadatas=metas, ids=ids,
                    )

                    for p in batch:
                        sqlite_db.conn.execute(
                            "UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (p["arxiv_id"],)
                        )
                    sqlite_db.conn.commit()

                    paper_success += len(batch)
                    console.print(
                        f"  [{paper_success}/{len(unindexed)}] "
                        f"[green]{len(batch)}편 OK[/green]"
                    )
                except Exception as e:
                    log.error("논문 배치 실패 (offset %d): %s", batch_start, e)
                    for p in batch:
                        failed_papers.append({"arxiv_id": p["arxiv_id"], "title": p["title"], "error": str(e)})
                    console.print(f"  배치 실패 ({len(batch)}편): [red]{e}[/red]")

            console.print(f"  [bold green]논문 {paper_success}/{len(unindexed)}편 완료[/bold green]")
        else:
            console.print("[green]모든 논문이 이미 인덱싱되어 있습니다.[/green]")

    # ── Phase 2: 개념 노트 인덱싱 ──
    concept_success = 0
    if config.vault_path:
        try:
            concept_notes = _load_concept_notes(config.vault_path)
        except Exception as e:
            console.print(f"[red]개념 노트 로드 실패: {e}[/red]")
            concept_notes = []

        if concept_notes:
            existing_ids = set()
            try:
                existing = vector_db.collection.get(
                    where={"source_type": "concept"},
                    include=[],
                )
                existing_ids = set(existing.get("ids", []))
            except Exception:
                pass

            if not index_all:
                concept_notes = [c for c in concept_notes
                                 if f"concept_{c['name']}" not in existing_ids]

            if concept_notes:
                console.print(f"\n[bold]개념 노트 인덱싱: {len(concept_notes)}개[/bold]")

                for batch_start in range(0, len(concept_notes), BATCH_SIZE):
                    batch = concept_notes[batch_start: batch_start + BATCH_SIZE]
                    texts = [c["text"] for c in batch]

                    try:
                        embs = _embed_batch_via_provider(embedder, texts)

                        docs, embeddings, metas, ids = [], [], [], []
                        for c, text, emb in zip(batch, texts, embs):
                            docs.append(text)
                            embeddings.append(emb)
                            metas.append({
                                "title": c["name"],
                                "source_type": "concept",
                                "related_concepts": ", ".join(c["related"][:5]),
                                "related_papers": ", ".join(c["papers"][:5]),
                                "chunk_index": 0,
                            })
                            ids.append(f"concept_{c['name']}")

                        vector_db.add_documents(
                            documents=docs, embeddings=embeddings, metadatas=metas, ids=ids,
                        )

                        concept_success += len(batch)
                        console.print(
                            f"  [{concept_success}/{len(concept_notes)}] "
                            f"[green]{len(batch)}개 OK[/green]"
                        )
                    except Exception as e:
                        log.error("개념 배치 실패 (offset %d): %s", batch_start, e)
                        for c in batch:
                            failed_concepts.append({"name": c["name"], "error": str(e)})
                        console.print(f"  배치 실패 ({len(batch)}개): [red]{e}[/red]")

                console.print(f"  [bold green]개념 {concept_success}/{len(concept_notes)}개 완료[/bold green]")
            else:
                console.print("[green]모든 개념 노트가 이미 인덱싱되어 있습니다.[/green]")
    else:
        console.print("[dim]Obsidian vault 미설정 - 개념 노트 인덱싱 건너뜀[/dim]")

    # ── 결과 리포트 ──
    elapsed = time.time() - t_start
    console.print(
        f"\n[bold]통합 인덱싱 완료 ({elapsed:.1f}초)[/bold]"
        f"\n  논문: {paper_success}편 | 개념: {concept_success}개"
        f"\n  벡터DB 총: {vector_db.count()}개 문서"
    )

    if failed_papers or failed_concepts:
        console.print("\n[bold red]실패 항목:[/bold red]")
        for fp in failed_papers:
            console.print(f"  논문 {fp['arxiv_id']}: {fp['error'][:80]}")
        for fc in failed_concepts:
            console.print(f"  개념 {fc['name']}: {fc['error'][:80]}")
