"""
공유 데이터 모델

모든 모듈에서 사용하는 기본 데이터 클래스입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class SourceType(str, Enum):
    """지식 소스 유형"""
    VAULT = "vault"
    PAPER = "paper"
    WEB = "web"
    NOTE = "note"


class ParaCategory(str, Enum):
    """PARA 분류 체계"""
    PROJECT = "project"
    AREA = "area"
    RESOURCE = "resource"
    ARCHIVE = "archive"


@dataclass
class Document:
    """
    통합 문서 모델

    Obsidian 노트, 논문, 웹 문서 등을 모두 표현합니다.
    """
    content: str
    metadata: Dict[str, Any]
    file_path: str
    title: str
    tags: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    source_type: SourceType = SourceType.NOTE

    @property
    def id(self) -> str:
        return self.file_path


@dataclass
class SearchResult:
    """검색 결과"""
    document: str
    metadata: Dict[str, Any]
    distance: float
    score: float

    def __repr__(self):
        title = self.metadata.get("title", "Untitled")
        return f"SearchResult(title='{title}', score={self.score:.3f})"


@dataclass
class PaperInfo:
    """논문 메타데이터"""
    arxiv_id: str
    title: str
    authors: str = ""
    year: int = 0
    research_field: str = ""
    importance: int = 0
    notes: str = ""
    pdf_path: Optional[str] = None
    text_path: Optional[str] = None
    translated_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def has_pdf(self) -> bool:
        return self.pdf_path is not None

    @property
    def has_translation(self) -> bool:
        return self.translated_path is not None
