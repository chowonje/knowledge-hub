"""
논문 관리 모듈

CSV 기반 논문 목록 관리 + SQLite DB 연동 + 벡터 인덱싱
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional
from typing import Any

from rich.console import Console
from rich.table import Table

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import VectorDatabase, SQLiteDatabase
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    fallback_source_hash,
    mark_derivatives_stale_for_document,
)
from knowledge_hub.core.chunking import chunk_text_with_offsets as canonical_chunk_text_with_offsets
from knowledge_hub.core.keywords import extract_keywords_from_text
from knowledge_hub.providers.base import BaseEmbedder
from knowledge_hub.core.models import SourceType
from knowledge_hub.papers.downloader import PaperDownloader
from knowledge_hub.papers.translator import PaperTranslator
from knowledge_hub.papers.discoverer import DiscoveredPaper, discover_papers
from knowledge_hub.papers.judge import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_PASS_THRESHOLD,
    JUDGE_BACKEND,
    PaperJudgeService,
)
from knowledge_hub.papers.judge_feedback import PaperJudgeFeedbackLogger

console = Console()


class PaperManager:
    """논문 관리 통합 클래스"""

    def __init__(
        self,
        config: Config,
        vector_db: VectorDatabase,
        sqlite_db: SQLiteDatabase,
        embedder: BaseEmbedder,
    ):
        self.config = config
        self.vector_db = vector_db
        self.sqlite_db = sqlite_db
        self.embedder = embedder

    def import_csv(self, csv_path: str, overwrite: bool = False, check_vector_db: bool = True):
        """CSV 파일에서 논문 메타데이터를 SQLite로 가져오기"""
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        imported = 0
        skipped_sql = 0
        skipped_vector = 0
        skipped_duplicate_in_batch = 0
        skipped_invalid = 0
        seen_ids = set()

        for row in rows:
            arxiv_id = (row.get("arXivID") or row.get("arxiv_id") or "").strip()
            if not arxiv_id or arxiv_id == "-":
                skipped_invalid += 1
                continue
            if arxiv_id in seen_ids:
                skipped_duplicate_in_batch += 1
                continue

            # SQLite 중복: 기본적으로 skip, overwrite 옵션이면 최신값으로 덮어쓰기
            existing_paper = self.sqlite_db.get_paper(arxiv_id)
            if existing_paper and not overwrite:
                skipped_sql += 1
                continue

            # 벡터 DB 중복: paper source_type 기준으로 중복 여부 판별
            if check_vector_db and not existing_paper:
                if self.vector_db.has_metadata({
                    "source_type": SourceType.PAPER.value,
                    "arxiv_id": arxiv_id,
                }):
                    skipped_vector += 1
                    continue

            seen_ids.add(arxiv_id)

            paper = {
                "arxiv_id": arxiv_id,
                "title": (row.get("논문제목") or row.get("title") or "").strip(),
                "authors": (row.get("저자") or row.get("authors") or "").strip(),
                "year": int(row.get("연도") or row.get("year") or 0),
                "field": (row.get("분야") or row.get("field") or "").strip(),
                "importance": int(row.get("중요도") or row.get("importance") or 3),
                "notes": (row.get("비고") or row.get("notes") or "").strip(),
                "pdf_path": None,
                "text_path": None,
                "translated_path": None,
            }
            self.sqlite_db.upsert_paper(paper)

            # 태그로 분야 추가
            if paper["field"]:
                self.sqlite_db.ensure_tag(paper["field"])
            imported += 1

        summary = (
            f"{imported}개 논문 등록/갱신, "
            f"{skipped_sql}개(SQLite 중복) / "
            f"{skipped_vector}개(벡터 중복) / "
            f"{skipped_duplicate_in_batch}개(CSV 중복) / "
            f"{skipped_invalid}개(잘못된 ID) 건너뜀"
        )
        console.print(f"[green]{summary}[/green]")

    def download(self, csv_path: Optional[str] = None, arxiv_id: Optional[str] = None):
        """논문 다운로드"""
        downloader = PaperDownloader(self.config.papers_dir)

        if arxiv_id:
            paper = self.sqlite_db.get_paper(arxiv_id)
            title = paper["title"] if paper else arxiv_id
            result = downloader.download_single(arxiv_id, title)
            if result["success"]:
                self._update_paper_paths(result)
            return [result]

        path = csv_path or self.config.papers_csv
        if not path:
            console.print("[red]CSV 파일 경로가 필요합니다[/red]")
            return []

        results = downloader.download_from_csv(path)
        for r in results:
            if r["success"]:
                self._update_paper_paths(r)
        return results

    def translate(
        self,
        arxiv_id: str,
        model: Optional[str] = None,
        proofread: bool = True,
        generate_pdf: bool = False,
    ):
        """단일 논문 번역"""
        paper = self.sqlite_db.get_paper(arxiv_id)
        if not paper:
            console.print(f"[red]논문을 찾을 수 없습니다: {arxiv_id}[/red]")
            return

        text_path = paper.get("text_path")
        if not text_path:
            console.print(f"[red]텍스트 파일이 없습니다. 먼저 다운로드하세요: paper download {arxiv_id}[/red]")
            return

        translator = PaperTranslator(model=model or self.config.translate_model)
        output_dir = str(Path(self.config.papers_dir) / "translated")

        md_path = translator.translate_paper(
            text_path=text_path,
            output_dir=output_dir,
            arxiv_id=arxiv_id,
            title=paper["title"],
            proofread=proofread,
            generate_pdf=generate_pdf,
        )

        if md_path:
            self.sqlite_db.conn.execute(
                "UPDATE papers SET translated_path = ? WHERE arxiv_id = ?",
                (md_path, arxiv_id),
            )
            self.sqlite_db.conn.commit()

    def index_papers(self):
        """다운로드된 논문 텍스트를 벡터 DB에 인덱싱"""
        papers = self.sqlite_db.list_papers(limit=1000)
        indexed = 0

        for paper in papers:
            text_path = paper.get("text_path")
            if not text_path or paper.get("indexed"):
                continue

            try:
                text = Path(text_path).read_text(encoding="utf-8")
            except Exception:
                continue

            # 청크 분할
            chunks = self._chunk_text_with_offsets(text, chunk_size=1000, overlap=200)
            document_id = f"paper:{paper['arxiv_id']}"
            source_hash = fallback_source_hash(text, paper["arxiv_id"])
            if source_hash:
                mark_derivatives_stale_for_document(
                    self.sqlite_db.conn,
                    document_id=document_id,
                    source_content_hash=source_hash,
                    source_type=SourceType.PAPER.value,
                )
            delete_by_metadata = getattr(self.vector_db, "delete_by_metadata", None)
            if callable(delete_by_metadata):
                delete_by_metadata({"source_type": SourceType.PAPER.value, "arxiv_id": paper["arxiv_id"]})
            for chunk in chunks:
                chunk["metadata"] = {
                    "title": paper["title"],
                    "arxiv_id": paper["arxiv_id"],
                    "source_type": SourceType.PAPER.value,
                    "field": paper.get("field", ""),
                    "document_id": document_id,
                    "source_content_hash": source_hash,
                    "stale": 0,
                    "parent_id": f"{document_id}::document",
                    "parent_title": paper["title"],
                    "parent_type": "document",
                    "chunk_index": chunk["chunk_index"],
                    "chunk_size": chunk["chunk_end"] - chunk["chunk_start"],
                    "chunk_start": chunk["chunk_start"],
                    "chunk_end": chunk["chunk_end"],
                    "contextual_summary": self._build_contextual_summary(
                        title=paper["title"],
                        chunk_text=chunk["text"],
                    ),
                }

            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            metas = [c["metadata"] for c in chunks]
            ids = [f"paper_{paper['arxiv_id']}_{c['metadata']['chunk_index']}" for c in chunks]

            embeddings = self.embedder.embed_batch(texts, show_progress=False)
            valid = [(t, e, m, i) for t, e, m, i in zip(texts, embeddings, metas, ids) if e is not None]

            if valid:
                v_t, v_e, v_m, v_i = zip(*valid)
                self.vector_db.add_documents(
                    documents=list(v_t),
                    embeddings=list(v_e),
                    metadatas=list(v_m),
                    ids=list(v_i),
                )

            self.sqlite_db.conn.execute(
                "UPDATE papers SET indexed = 1 WHERE arxiv_id = ?", (paper["arxiv_id"],)
            )
            self.sqlite_db.conn.commit()
            indexed += 1
            console.print(f"  [green]인덱싱: {paper['title'][:50]}... ({len(chunks)} 청크)[/green]")

        console.print(f"\n[bold green]{indexed}개 논문 인덱싱 완료[/bold green]")

    @staticmethod
    def _build_contextual_summary(title: str, chunk_text: str) -> str:
        normalized = re.sub(r"\s+", " ", (chunk_text or "").strip())
        if not normalized:
            return title

        first_sentence = normalized.split(". ")[0].strip()
        if len(first_sentence) > 180:
            first_sentence = f"{first_sentence[:177]}..."

        return f"[{title}] {first_sentence}"

    @staticmethod
    def _chunk_text_with_offsets(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[dict[str, Any]]:
        return [
            {
                "text": str(item.get("text") or ""),
                "chunk_index": int(item.get("chunk_index", 0)),
                "chunk_start": int(item.get("start", 0)),
                "chunk_end": int(item.get("end", 0)),
            }
            for item in canonical_chunk_text_with_offsets(
                text,
                content_type="plain",
                chunk_size=chunk_size,
                overlap=overlap,
            )
        ]

    def list(self, field: Optional[str] = None):
        """논문 목록 표시"""
        papers = self.sqlite_db.list_papers(field=field, limit=100)

        table = Table(title="논문 목록")
        table.add_column("arXiv ID", style="cyan")
        table.add_column("제목", max_width=50)
        table.add_column("연도")
        table.add_column("분야", style="magenta")
        table.add_column("중요도")
        table.add_column("PDF", style="green")
        table.add_column("번역", style="yellow")

        for p in papers:
            table.add_row(
                p["arxiv_id"],
                p["title"][:50],
                str(p.get("year", "")),
                p.get("field", ""),
                str(p.get("importance", "")),
                "O" if p.get("pdf_path") else "-",
                "O" if p.get("translated_path") else "-",
            )

        console.print(table)
        console.print(f"총 {len(papers)}개")

    @staticmethod
    def _extract_keywords_from_translated_text(text: str, max_keywords: int = 12) -> list[str]:
        """번역본 텍스트에서 핵심 키워드 추출(휴리스틱)"""
        return extract_keywords_from_text(text, max_keywords=max_keywords)

    def sync_translated_keywords(
        self,
        arxiv_id: Optional[str] = None,
        top_k: int = 12,
        max_links_per_keyword: int = 5,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """번역본에서 핵심 키워드를 추출해 노트/링크로 저장"""
        papers: list[dict]
        if arxiv_id:
            paper = self.sqlite_db.get_paper(arxiv_id)
            papers = [paper] if paper else []
        else:
            papers = [p for p in self.sqlite_db.list_papers(limit=5000) if p.get("translated_path")]

        result_items: list[dict[str, Any]] = []
        updated = 0
        processed = 0
        skipped = 0

        for paper in papers:
            if not paper or not paper.get("translated_path"):
                skipped += 1
                continue

            processed += 1
            translated_path = paper.get("translated_path")
            try:
                raw_text = Path(translated_path).read_text(encoding="utf-8")
            except Exception:
                skipped += 1
                continue

            keywords = self._extract_keywords_from_translated_text(raw_text, max_keywords=top_k)
            if not keywords:
                skipped += 1
                continue

            item = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "keywords": keywords,
                "linked_notes": [],
                "keyword_nodes": [],
            }

            if dry_run:
                result_items.append(item)
                continue

            paper_note_id = f"paper:{paper['arxiv_id']}"
            paper_note_title = f"[논문 키워드] {paper['title']}"
            paper_metadata = {
                "kind": "paper_keyword_note",
                "arxiv_id": paper["arxiv_id"],
                "translated_path": translated_path,
                "top_k": top_k,
            }
            snippet = raw_text[:1200]
            paper_note_content = (
                f"아카이브: {paper['arxiv_id']}\n"
                f"제목: {paper['title']}\n"
                f"분야: {paper.get('field', '')}\n"
                f"핵심 키워드: {', '.join(keywords)}\n\n"
                f"요약 본문 일부:\n{snippet}"
            )

            self.sqlite_db.upsert_note(
                paper_note_id,
                paper_note_title,
                paper_note_content,
                file_path=translated_path,
                source_type="paper",
                para_category=None,
                metadata=paper_metadata,
            )

            for keyword in keywords:
                norm_kw = re.sub(r"\s+", " ", keyword.strip().lower()).replace(" ", "_")
                if not norm_kw:
                    continue
                keyword_id = f"keyword:{norm_kw[:120]}"
                keyword_title = f"[키워드] {keyword}"
                keyword_metadata = {
                    "kind": "keyword_node",
                    "source_arxiv": paper["arxiv_id"],
                }
                keyword_content = (
                    f"키워드: {keyword}\n"
                    f"관련 arXiv: {paper['arxiv_id']}\n"
                    f"관련 논문: {paper['title']}\n"
                )

                self.sqlite_db.upsert_note(
                    keyword_id,
                    keyword_title,
                    keyword_content,
                    file_path=f"keyword://{norm_kw}",
                    source_type="note",
                    para_category="resource",
                    metadata=keyword_metadata,
                )
                self.sqlite_db.add_link(paper_note_id, keyword_id, "contains_keyword")

                related_note_rows = self.sqlite_db.conn.execute(
                    "SELECT id, title FROM notes WHERE id != ? AND (LOWER(title) LIKE ? OR LOWER(content) LIKE ?) LIMIT ?",
                    (paper_note_id, f"%{keyword.lower()}%", f"%{keyword.lower()}%", max_links_per_keyword),
                ).fetchall()
                related_ids = []
                for row in related_note_rows:
                    target_id = row["id"]
                    if target_id == keyword_id:
                        continue
                    self.sqlite_db.add_link(keyword_id, target_id, "related_note")
                    related_ids.append(target_id)

                item["linked_notes"].append({
                    "keyword": keyword,
                    "note_id": keyword_id,
                    "related_notes": related_ids,
                })
                item["keyword_nodes"].append(keyword_id)

                # 벡터 검색에서 키워드 노트를 바로 찾을 수 있게 텍스트 임베딩 갱신
                keyword_text = f"{keyword_title}\n\n{keyword_content}"
                try:
                    keyword_embedding = self.embedder.embed_text(keyword_text)
                    if keyword_embedding:
                        self.vector_db.add_documents(
                            documents=[keyword_text],
                            embeddings=[keyword_embedding],
                            metadatas=[{
                                "title": keyword_title,
                                "source_type": "note",
                                "file_path": f"keyword://{norm_kw}",
                                "kind": "paper_keyword",
                                "arxiv_id": paper["arxiv_id"],
                            }],
                            ids=[f"note_{keyword_id}"],
                        )
                except Exception:
                    pass

            # paper note도 벡터에 반영
            paper_text = f"{paper_note_title}\n\n{paper_note_content}"
            try:
                paper_embedding = self.embedder.embed_text(paper_text)
                if paper_embedding:
                    self.vector_db.add_documents(
                        documents=[paper_text],
                        embeddings=[paper_embedding],
                        metadatas=[{
                            "title": paper_note_title,
                            "source_type": "paper",
                            "file_path": translated_path,
                            "kind": "paper_keyword_summary",
                            "arxiv_id": paper["arxiv_id"],
                        }],
                        ids=[f"note_{paper_note_id}"],
                    )
            except Exception:
                pass

            self.sqlite_db.conn.commit()
            updated += 1
            result_items.append(item)

        return {
            "mode": "dry_run" if dry_run else "applied",
            "processed": processed,
            "updated": updated,
            "skipped": skipped,
            "items": result_items,
        }

    def is_duplicate(self, arxiv_id: str) -> tuple[bool, str]:
        """SQLite + 벡터DB에서 중복 여부 확인. (중복여부, 사유) 반환"""
        existing = self.sqlite_db.get_paper(arxiv_id)
        if existing:
            return True, "sqlite"

        if self.vector_db.has_metadata({
            "source_type": SourceType.PAPER.value,
            "arxiv_id": arxiv_id,
        }):
            return True, "vector_db"

        return False, ""

    def discover_and_ingest(
        self,
        topic: str,
        max_papers: int = 5,
        year_start: Optional[int] = None,
        min_citations: int = 0,
        sort_by: str = "relevance",
        create_obsidian_note: bool = True,
        generate_summary: bool = True,
        llm: Optional[Any] = None,
        judge_enabled: bool = False,
        judge_threshold: float = DEFAULT_PASS_THRESHOLD,
        judge_candidates: Optional[int] = None,
        allow_external: bool = False,
        judge_llm: Optional[Any] = None,
        user_goal: str = "",
    ) -> dict[str, Any]:
        """
        논문 자동 발견 → 중복 체크 → 다운로드 → 요약 → 인덱싱 → 옵시디언 연결
        전체 파이프라인을 실행합니다.
        """
        discovered = discover_papers(
            topic=topic,
            max_papers=max(
                max_papers * 2,
                max(1, int(judge_candidates or (max_papers * DEFAULT_CANDIDATE_MULTIPLIER)))
                if judge_enabled
                else max_papers * 2,
            ),
            year_start=year_start,
            min_citations=min_citations,
            sort_by=sort_by,
        )

        report = {
            "schema": "knowledge-hub.paper.discover.result.v1",
            "status": "ok",
            "topic": topic,
            "discovered": len(discovered),
            "duplicates_skipped": 0,
            "ingested": [],
            "failed": [],
            "obsidian_notes_created": [],
            "warnings": [],
            "judge": {
                "enabled": bool(judge_enabled),
                "backend": JUDGE_BACKEND if judge_enabled else "",
                "threshold": round(float(judge_threshold or DEFAULT_PASS_THRESHOLD), 6),
                "candidateCount": 0,
                "selectedCount": 0,
                "degraded": False,
                "warnings": [],
                "items": [],
            },
        }

        if not discovered:
            report["message"] = "검색 결과가 없습니다."
            return report

        candidates: list[DiscoveredPaper] = []
        for paper in discovered:
            is_dup, _dup_reason = self.is_duplicate(paper.arxiv_id)
            if is_dup:
                report["duplicates_skipped"] += 1
                continue
            candidates.append(paper)

        if judge_enabled and candidates:
            judge_service = PaperJudgeService(
                self.config,
                llm=judge_llm or llm,
                allow_external=allow_external,
                pass_threshold=judge_threshold,
            )
            selected, judge_payload = judge_service.select_candidates(
                candidates,
                topic=topic,
                threshold=judge_threshold,
                top_k=max_papers,
                user_goal=user_goal,
            )
            report["judge"] = {
                "enabled": True,
                "backend": judge_payload.get("backend", JUDGE_BACKEND),
                "threshold": judge_payload.get("threshold", round(float(judge_threshold or DEFAULT_PASS_THRESHOLD), 6)),
                "candidateCount": int(judge_payload.get("candidateCount", len(candidates)) or 0),
                "selectedCount": int(judge_payload.get("selectedCount", len(selected)) or 0),
                "degraded": bool(judge_payload.get("degraded", False)),
                "warnings": list(judge_payload.get("warnings") or []),
                "items": list(judge_payload.get("items") or []),
            }
            PaperJudgeFeedbackLogger(self.config).log_judge_decisions(
                topic=topic,
                items=report["judge"]["items"],
                backend=str(report["judge"]["backend"] or JUDGE_BACKEND),
                threshold=float(report["judge"]["threshold"] or judge_threshold or DEFAULT_PASS_THRESHOLD),
                degraded=bool(report["judge"]["degraded"]),
                allow_external=bool(allow_external),
                source="discover_and_ingest",
            )
            candidates = selected
            report["warnings"].extend([item for item in report["judge"]["warnings"] if item not in report["warnings"]])
        else:
            candidates = candidates[:max_papers]

        if not candidates:
            report["message"] = "조건을 만족하는 신규 논문이 없습니다."
            return report

        downloader = PaperDownloader(self.config.papers_dir)
        ingested_count = 0

        for paper in candidates:
            if ingested_count >= max_papers:
                break

            try:
                result = downloader.download_single(paper.arxiv_id, paper.title)

                paper_data = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "field": ", ".join(paper.fields_of_study[:3]) if paper.fields_of_study else "",
                    "importance": self._estimate_importance(paper),
                    "notes": f"citations: {paper.citation_count}",
                    "pdf_path": result.get("pdf"),
                    "text_path": result.get("text"),
                    "translated_path": None,
                }
                self.sqlite_db.upsert_paper(paper_data)

                if paper_data["field"]:
                    for f in paper_data["field"].split(", "):
                        self.sqlite_db.ensure_tag(f.strip())

                summary = ""
                if generate_summary and llm and paper.abstract:
                    try:
                        summary = llm.generate(
                            f"다음 논문을 한국어로 간결하게 요약해주세요 (3~5문장):\n\n"
                            f"제목: {paper.title}\n초록: {paper.abstract}",
                            context="",
                        )
                    except Exception:
                        summary = paper.abstract[:500]
                elif paper.abstract:
                    summary = paper.abstract[:500]

                self._index_single_paper(paper, paper_data, summary)

                obsidian_path = ""
                if create_obsidian_note:
                    judge_assessment = next(
                        (
                            item
                            for item in list(report.get("judge", {}).get("items") or [])
                            if str(item.get("paper_id") or "") == str(paper.arxiv_id)
                        ),
                        None,
                    )
                    obsidian_path = self._create_obsidian_note(
                        paper,
                        summary,
                        topic,
                        judge_assessment=judge_assessment,
                    )

                ingested_count += 1
                entry = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "year": paper.year,
                    "citations": paper.citation_count,
                    "fields": paper.fields_of_study[:3],
                    "summary": summary[:300] if summary else "",
                    "pdf_downloaded": result.get("pdf") is not None,
                }
                report["ingested"].append(entry)
                if obsidian_path:
                    report["obsidian_notes_created"].append(obsidian_path)

            except Exception as e:
                report["failed"].append({
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "error": str(e),
                })

        report["message"] = (
            f"{report['discovered']}개 발견, "
            f"{len(report['ingested'])}개 수집, "
            f"{report['duplicates_skipped']}개 중복 건너뜀, "
            f"{len(report['failed'])}개 실패"
        )
        return report

    @staticmethod
    def _estimate_importance(paper: DiscoveredPaper) -> int:
        """인용수 기반 중요도 추정 (1~5)"""
        c = paper.citation_count
        if c >= 500:
            return 5
        if c >= 100:
            return 4
        if c >= 20:
            return 3
        if c >= 5:
            return 2
        return 1

    def _index_single_paper(
        self,
        paper: DiscoveredPaper,
        paper_data: dict,
        summary: str,
    ):
        """단일 논문의 abstract + summary를 벡터DB에 인덱싱"""
        text_parts = []
        if paper.title:
            text_parts.append(f"Title: {paper.title}")
        if paper.abstract:
            text_parts.append(f"Abstract: {paper.abstract}")
        if summary and summary != paper.abstract[:500]:
            text_parts.append(f"Summary: {summary}")

        full_text = "\n\n".join(text_parts)
        if not full_text.strip():
            return

        document_id = f"paper:{paper.arxiv_id}"
        source_hash = fallback_source_hash(full_text, paper.arxiv_id)
        if source_hash:
            mark_derivatives_stale_for_document(
                self.sqlite_db.conn,
                document_id=document_id,
                source_content_hash=source_hash,
                source_type=SourceType.PAPER.value,
            )
        delete_by_metadata = getattr(self.vector_db, "delete_by_metadata", None)
        if callable(delete_by_metadata):
            delete_by_metadata({"source_type": SourceType.PAPER.value, "arxiv_id": paper.arxiv_id})
        chunks = [
            {
                "text": str(item.get("text") or ""),
                "metadata": {
                    "title": paper.title,
                    "arxiv_id": paper.arxiv_id,
                    "source_type": SourceType.PAPER.value,
                    "field": paper_data.get("field", ""),
                    "document_id": document_id,
                    "source_content_hash": source_hash,
                    "stale": 0,
                    "parent_id": f"{document_id}::document",
                    "parent_title": paper.title,
                    "parent_type": "document",
                    "chunk_index": int(item.get("chunk_index", 0)),
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                },
            }
            for item in canonical_chunk_text_with_offsets(
                full_text,
                content_type="plain",
                chunk_size=1000,
                overlap=200,
            )
            if str(item.get("text") or "").strip()
        ]

        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        ids = [f"paper_{paper.arxiv_id}_{c['metadata']['chunk_index']}" for c in chunks]

        embeddings = self.embedder.embed_batch(texts, show_progress=False)
        valid = [
            (t, e, m, i)
            for t, e, m, i in zip(texts, embeddings, metas, ids)
            if e is not None
        ]

        if valid:
            v_t, v_e, v_m, v_i = zip(*valid)
            self.vector_db.add_documents(
                documents=list(v_t),
                embeddings=list(v_e),
                metadatas=list(v_m),
                ids=list(v_i),
            )

        self.sqlite_db.conn.execute(
            "UPDATE papers SET indexed = 1 WHERE arxiv_id = ?",
            (paper.arxiv_id,),
        )
        self.sqlite_db.conn.commit()

    def _create_obsidian_note(
        self,
        paper: DiscoveredPaper,
        summary: str,
        topic: str,
        judge_assessment: dict[str, Any] | None = None,
        translated_abstract: str = "",
    ) -> str:
        """논문 요약 노트를 Obsidian vault에 생성하고 관련 노트를 [[링크]]"""
        vault_path = self.config.vault_path
        if not vault_path:
            return ""

        vault = Path(vault_path)
        notes_folder = self.config.obsidian_notes_folder or "Papers"
        notes_root = Path(notes_folder)
        if notes_root.is_absolute():
            notes_dir = notes_root
        else:
            notes_dir = vault / notes_root

        papers_dir = notes_dir
        papers_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r'[\\/:*?"<>|]', '', paper.title)[:80].strip()
        note_path = papers_dir / f"{safe_title}.md"

        related_links = self._find_related_vault_notes(paper, topic)

        fields_tags = " ".join(
            f"#{f.replace(' ', '_').replace('.', '_')}"
            for f in paper.fields_of_study[:5]
        ) if paper.fields_of_study else ""

        content_lines = [
            "---",
            f"title: \"{paper.title}\"",
            f"arxiv_id: \"{paper.arxiv_id}\"",
            f"authors: \"{paper.authors}\"",
            f"year: {paper.year}",
            f"citations: {paper.citation_count}",
            f"topic: \"{topic}\"",
            f"tags: [paper, AI, {', '.join(paper.fields_of_study[:3])}]",
            "type: paper-summary",
        ]
        if judge_assessment:
            content_lines.extend(
                [
                    "judge_enabled: true",
                    f"judge_backend: \"{judge_assessment.get('backend', JUDGE_BACKEND)}\"",
                    f"judge_score: {float(judge_assessment.get('total_score', 0.0) or 0.0):.6f}",
                    f"judge_decision: \"{judge_assessment.get('decision', 'skip')}\"",
                    f"judge_topic: \"{topic}\"",
                ]
            )
        content_lines.extend(
            [
                "---",
                "",
                f"# {paper.title}",
                "",
                f"**arXiv:** [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})",
                f"**저자:** {paper.authors}",
                f"**연도:** {paper.year} | **인용수:** {paper.citation_count}",
                f"**분야:** {', '.join(paper.fields_of_study[:5]) if paper.fields_of_study else 'N/A'}",
                "",
            ]
        )

        if summary:
            content_lines.extend([
                "## 요약",
                "",
                summary,
                "",
            ])

        if judge_assessment:
            dimension_scores = dict(judge_assessment.get("dimension_scores") or {})
            content_lines.extend(
                [
                    "## Judge Assessment",
                    "",
                    f"- 총점: {float(judge_assessment.get('total_score', 0.0) or 0.0):.3f}",
                    f"- 결정: {judge_assessment.get('decision', 'skip')}",
                    f"- 백엔드: {judge_assessment.get('backend', JUDGE_BACKEND)}",
                    f"- relevance: {float(dimension_scores.get('relevance_score', 0.0) or 0.0):.3f}",
                    f"- novelty: {float(dimension_scores.get('novelty_score', 0.0) or 0.0):.3f}",
                    f"- read value: {float(dimension_scores.get('read_value_score', 0.0) or 0.0):.3f}",
                    f"- citation signal: {float(dimension_scores.get('citation_signal_score', 0.0) or 0.0):.3f}",
                    "",
                ]
            )
            reasons = [str(item).strip() for item in list(judge_assessment.get("top_reasons") or []) if str(item).strip()]
            if reasons:
                content_lines.extend(["### 이유", ""])
                for reason in reasons[:4]:
                    content_lines.append(f"- {reason}")
                content_lines.append("")

        if paper.abstract:
            content_lines.extend([
                "## Abstract",
                "",
                paper.abstract,
                "",
            ])

        if translated_abstract:
            content_lines.extend([
                "## 초록 (한국어)",
                "",
                translated_abstract,
                "",
            ])

        if related_links:
            content_lines.extend([
                "## 관련 노트",
                "",
            ])
            for link_info in related_links:
                content_lines.append(f"- [[{link_info['title']}]] (유사도: {link_info['score']:.2f})")
            content_lines.append("")

        if fields_tags:
            content_lines.extend([
                "## 태그",
                "",
                fields_tags,
                "",
            ])

        note_content = "\n".join(content_lines)
        note_path.write_text(note_content, encoding="utf-8")

        note_id = f"paper:{paper.arxiv_id}"
        self.sqlite_db.upsert_note(
            note_id=note_id,
            title=f"[논문] {paper.title}",
            content=note_content,
            file_path=str(note_path),
            source_type="paper",
            para_category="resource",
            metadata={
                "arxiv_id": paper.arxiv_id,
                "citations": paper.citation_count,
                "year": paper.year,
                "topic": topic,
            },
        )

        for link_info in related_links:
            self.sqlite_db.add_link(note_id, link_info.get("note_id", ""), "semantic_related")

        return str(note_path)

    def _find_related_vault_notes(
        self,
        paper: DiscoveredPaper,
        topic: str,
        top_k: int = 5,
    ) -> List[dict]:
        """벡터DB에서 논문과 관련된 기존 vault 노트를 찾아 반환"""
        query_text = f"{paper.title} {topic}"
        if paper.abstract:
            query_text += f" {paper.abstract[:200]}"

        try:
            query_embedding = self.embedder.embed_text(query_text)
        except Exception:
            return []

        results = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict={"source_type": "vault"},
        )

        related = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = max(0.0, min(1.0, 1.0 - distance))
                if score < 0.3:
                    continue

                title = meta.get("title", "Untitled")
                file_path = meta.get("file_path", "")
                stem = Path(file_path).stem if file_path else title

                related.append({
                    "title": stem,
                    "score": score,
                    "note_id": meta.get("id", f"vault:{stem}"),
                    "file_path": file_path,
                })

        return related

    def _update_paper_paths(self, result: dict):
        """다운로드 결과를 SQLite에 반영"""
        paper = self.sqlite_db.get_paper(result["arxiv_id"])
        if paper:
            self.sqlite_db.conn.execute(
                "UPDATE papers SET pdf_path = ?, text_path = ? WHERE arxiv_id = ?",
                (result.get("pdf"), result.get("text"), result["arxiv_id"]),
            )
        else:
            self.sqlite_db.upsert_paper(
                {
                    "arxiv_id": result["arxiv_id"],
                    "title": result.get("title", result["arxiv_id"]),
                    "authors": "",
                    "year": 0,
                    "field": "",
                    "importance": 3,
                    "notes": "",
                    "pdf_path": result.get("pdf"),
                    "text_path": result.get("text"),
                    "translated_path": None,
                }
            )
        self.sqlite_db.conn.commit()
