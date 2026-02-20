"""
RAG (Retrieval-Augmented Generation) 파이프라인

통합 지식 검색 + 답변 생성
- Obsidian 노트
- arXiv 논문
- 웹 문서
모두에서 통합 검색합니다.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from knowledge_hub.core.models import SearchResult
from knowledge_hub.core.database import VectorDatabase


class RAGSearcher:
    """통합 RAG 검색 및 답변 생성

    embedder: embed_text(str) -> List[float] 를 제공하는 객체
    llm: generate(prompt, context) 를 제공하는 객체
    """

    def __init__(
        self,
        embedder,
        database: VectorDatabase,
        llm=None,
    ):
        self.embedder = embedder
        self.database = database
        self.llm = llm

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        통합 의미론적 검색

        Args:
            query: 검색 질문
            top_k: 반환 결과 수
            source_type: 소스 필터 (vault, paper, web, note)
        """
        query_embedding = self.embedder.embed_text(query)

        filter_dict = None
        if source_type:
            filter_dict = {"source_type": source_type}

        results = self.database.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i]
                score = max(0.0, min(1.0, 1.0 - distance))

                search_results.append(
                    SearchResult(
                        document=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        distance=distance,
                        score=score,
                    )
                )

        return search_results

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 답변 생성: 검색 + LLM 응답"""
        if not self.llm:
            raise ValueError("LLM이 설정되지 않았습니다")

        search_results = self.search(query, top_k=top_k, source_type=source_type)
        filtered = [r for r in search_results if r.score >= min_score]

        if not filtered:
            return {
                "answer": "관련된 문서를 찾을 수 없습니다.",
                "sources": [],
                "query": query,
            }

        context_parts = []
        for i, result in enumerate(filtered, 1):
            title = result.metadata.get("title", "Untitled")
            file_path = result.metadata.get("file_path", "")
            src = result.metadata.get("source_type", "")

            context_parts.append(
                f"문서 {i}: {title} [{src}] ({file_path})\n"
                f"유사도: {result.score:.2f}\n\n"
                f"{result.document}\n---"
            )

        context = "\n\n".join(context_parts)
        answer = self.llm.generate(query, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "title": r.metadata.get("title", "Untitled"),
                    "file_path": r.metadata.get("file_path", ""),
                    "source_type": r.metadata.get("source_type", ""),
                    "score": r.score,
                    "excerpt": r.document[:200] + ("..." if len(r.document) > 200 else ""),
                }
                for r in filtered
            ],
            "query": query,
        }

    def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
    ):
        """스트리밍 RAG 답변"""
        if not self.llm:
            raise ValueError("LLM이 설정되지 않았습니다")

        search_results = self.search(query, top_k=top_k, source_type=source_type)
        filtered = [r for r in search_results if r.score >= min_score]

        if not filtered:
            yield "관련된 문서를 찾을 수 없습니다."
            return

        context_parts = []
        for i, result in enumerate(filtered, 1):
            title = result.metadata.get("title", "Untitled")
            context_parts.append(f"문서 {i}: {title}\n{result.document}\n---")

        context = "\n\n".join(context_parts)
        for chunk in self.llm.stream_generate(query, context):
            yield chunk
