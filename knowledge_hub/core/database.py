"""
벡터 데이터베이스 + SQLite 통합 저장소

- ChromaDB: 벡터 임베딩 저장/검색 (의미론적 유사도)
- SQLite: 구조화된 메타데이터 (노트, 논문, PARA 분류, 태그)
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

log = logging.getLogger("khub.database")

SQLITE_BUSY_TIMEOUT_MS = 5000
SQLITE_MAX_RETRIES = 3
SQLITE_RETRY_BASE_SEC = 0.3


class _NoOpEmbeddingFunction:
    """ONNX 자동 로드를 방지하기 위한 더미 임베딩 함수. 사전 계산된 임베딩만 사용."""

    def __call__(self, input):
        return [[0.0]] * len(input)

    @staticmethod
    def name():
        return "noop"

    @staticmethod
    def build_from_config(config):
        return _NoOpEmbeddingFunction()

    def get_config(self):
        return {}

    def __init__(self):
        pass


class VectorDatabase:
    """ChromaDB 벡터 데이터베이스"""

    def __init__(self, db_path: str, collection_name: str = "knowledge_hub"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._ef = _NoOpEmbeddingFunction()
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        clean_metadatas = []
        for metadata in metadatas:
            clean = {}
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    clean[key] = json.dumps(value, ensure_ascii=False)
                elif value is None:
                    clean[key] = ""
                else:
                    clean[key] = value
            clean_metadatas.append(clean)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
        )

        if results["metadatas"]:
            for metadata_list in results["metadatas"]:
                for metadata in metadata_list:
                    for key, value in metadata.items():
                        if isinstance(value, str) and (
                            value.startswith("[") or value.startswith("{")
                        ):
                            try:
                                metadata[key] = json.loads(value)
                            except Exception:
                                pass
        return results

    def has_metadata(self, filter_dict: Dict[str, Any]) -> bool:
        """메타데이터 조건으로 문서 존재 여부를 빠르게 확인"""
        if not filter_dict:
            return False

        where_variants = [filter_dict]
        eq_filter = {
            key: {"$eq": value} if not isinstance(value, dict) else value
            for key, value in filter_dict.items()
        }
        if eq_filter != filter_dict:
            where_variants.append(eq_filter)

        for where in where_variants:
            try:
                result = self.collection.get(where=where, limit=1)
                ids = result.get("ids") if isinstance(result, dict) else None
                if ids:
                    return True
            except Exception:
                continue
        return False

    def delete_by_id(self, doc_ids: List[str]):
        self.collection.delete(ids=doc_ids)

    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()

    def count(self) -> int:
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        count = self.count()
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_path": str(self.db_path),
        }
        if count > 0:
            try:
                sample = self.collection.peek(limit=1)
                if sample and sample["metadatas"]:
                    stats["metadata_keys"] = list(sample["metadatas"][0].keys())
            except Exception:
                pass
        return stats


class SQLiteDatabase:
    """SQLite 구조화된 데이터 저장소 (노트, 논문, PARA, 태그)"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        self._init_tables()

    @contextmanager
    def transaction(self):
        """재시도 가능한 트랜잭션 컨텍스트 매니저.

        OperationalError(locked) → 지수 백오프 재시도. 그 외 에러 → 즉시 롤백.

        사용법:
            with db.transaction():
                db.conn.execute(...)
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise

    def _init_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                file_path TEXT,
                source_type TEXT DEFAULT 'note',
                para_category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                starred INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT DEFAULT '#6366f1'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS note_tags (
                note_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (note_id, tag_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                target_id TEXT REFERENCES notes(id) ON DELETE CASCADE,
                link_type TEXT DEFAULT 'related',
                strength REAL DEFAULT 0.5,
                PRIMARY KEY (source_id, target_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                field TEXT,
                importance INTEGER DEFAULT 3,
                notes TEXT,
                pdf_path TEXT,
                text_path TEXT,
                translated_path TEXT,
                indexed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS para_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                color TEXT DEFAULT '#6366f1',
                icon TEXT,
                sort_order INTEGER DEFAULT 0
            )
        """)

        # ── Knowledge Graph 테이블 ──

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                canonical_name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_aliases (
                alias TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                evidence_text TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_type, source_id, relation, target_type, target_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_source
            ON kg_relations(source_type, source_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_kg_target
            ON kg_relations(target_type, target_id)
        """)

        self.conn.commit()
        self._ensure_default_para()

    def _ensure_default_para(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM para_categories")
        if cursor.fetchone()[0] == 0:
            defaults = [
                ("project", "Projects", "활발히 진행 중인 프로젝트", "#3b82f6", "folder"),
                ("area", "Areas", "지속적으로 관리하는 영역", "#10b981", "layers"),
                ("resource", "Resources", "참고 자료와 리소스", "#f59e0b", "book-open"),
                ("archive", "Archives", "완료되거나 보관된 항목", "#6b7280", "archive"),
            ]
            cursor.executemany(
                "INSERT INTO para_categories (type, name, description, color, icon) VALUES (?, ?, ?, ?, ?)",
                defaults,
            )
            self.conn.commit()

    # --- Notes CRUD ---

    def upsert_note(self, note_id: str, title: str, content: str = "",
                    file_path: str = "", source_type: str = "note",
                    para_category: str = None, metadata: dict = None):
        self.conn.execute(
            """INSERT INTO notes (id, title, content, file_path, source_type, para_category, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, content=excluded.content,
                 file_path=excluded.file_path, source_type=excluded.source_type,
                 para_category=excluded.para_category, metadata=excluded.metadata,
                 updated_at=CURRENT_TIMESTAMP""",
            (note_id, title, content, file_path, source_type,
             para_category, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_note(self, note_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None

    def list_notes(self, source_type: str = None, para_category: str = None,
                   limit: int = 50, offset: int = 0) -> List[dict]:
        query = "SELECT * FROM notes WHERE 1=1"
        params = []
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if para_category:
            query += " AND para_category = ?"
            params.append(para_category)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def delete_note(self, note_id: str):
        self.conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self.conn.commit()

    def search_notes(self, query: str, limit: int = 20) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Tags ---

    def ensure_tag(self, name: str) -> int:
        row = self.conn.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
        if row:
            return row[0]
        cursor = self.conn.execute("INSERT INTO tags (name) VALUES (?)", (name,))
        self.conn.commit()
        return cursor.lastrowid

    def add_note_tag(self, note_id: str, tag_name: str):
        tag_id = self.ensure_tag(tag_name)
        self.conn.execute(
            "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
            (note_id, tag_id),
        )
        self.conn.commit()

    def get_note_tags(self, note_id: str) -> List[str]:
        rows = self.conn.execute(
            """SELECT t.name FROM tags t
               JOIN note_tags nt ON t.id = nt.tag_id
               WHERE nt.note_id = ?""",
            (note_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_tags(self) -> List[dict]:
        rows = self.conn.execute(
            """SELECT t.name, t.color, COUNT(nt.note_id) as count
               FROM tags t LEFT JOIN note_tags nt ON t.id = nt.tag_id
               GROUP BY t.id ORDER BY count DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Links ---

    def add_link(self, source_id: str, target_id: str, link_type: str = "related"):
        self.conn.execute(
            "INSERT OR IGNORE INTO links (source_id, target_id, link_type) VALUES (?, ?, ?)",
            (source_id, target_id, link_type),
        )
        self.conn.commit()

    def get_links(self, note_id: str) -> List[dict]:
        rows = self.conn.execute(
            """SELECT * FROM links WHERE source_id = ? OR target_id = ?""",
            (note_id, note_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Papers ---

    def upsert_paper(self, paper: dict):
        self.conn.execute(
            """INSERT INTO papers (arxiv_id, title, authors, year, field, importance, notes, pdf_path, text_path, translated_path)
               VALUES (:arxiv_id, :title, :authors, :year, :field, :importance, :notes, :pdf_path, :text_path, :translated_path)
               ON CONFLICT(arxiv_id) DO UPDATE SET
                 title=excluded.title, authors=excluded.authors, year=excluded.year,
                 field=excluded.field, importance=excluded.importance, notes=excluded.notes,
                 pdf_path=excluded.pdf_path, text_path=excluded.text_path,
                 translated_path=excluded.translated_path""",
            paper,
        )
        self.conn.commit()

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return dict(row) if row else None

    def list_papers(self, field: str = None, limit: int = 50) -> List[dict]:
        query = "SELECT * FROM papers WHERE 1=1"
        params = []
        if field:
            query += " AND field = ?"
            params.append(field)
        query += " ORDER BY year DESC, importance DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search_papers(self, query: str, limit: int = 20) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE title LIKE ? OR authors LIKE ? OR field LIKE ? ORDER BY importance DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- PARA ---

    def list_para_categories(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM para_categories ORDER BY sort_order").fetchall()
        return [dict(r) for r in rows]

    def get_para_stats(self) -> Dict[str, int]:
        rows = self.conn.execute(
            "SELECT para_category, COUNT(*) as cnt FROM notes WHERE para_category IS NOT NULL GROUP BY para_category"
        ).fetchall()
        return {r["para_category"]: r["cnt"] for r in rows}

    # --- Graph ---

    def get_graph_data(self) -> Dict[str, Any]:
        """지식 그래프 데이터 생성"""
        notes = self.conn.execute(
            "SELECT id, title, source_type, para_category FROM notes"
        ).fetchall()
        links = self.conn.execute("SELECT * FROM links").fetchall()

        nodes = [
            {
                "id": n["id"],
                "label": n["title"],
                "type": n["source_type"],
                "group": n["para_category"] or "none",
            }
            for n in notes
        ]
        edges = [
            {
                "source": l["source_id"],
                "target": l["target_id"],
                "type": l["link_type"],
            }
            for l in links
        ]
        return {"nodes": nodes, "edges": edges}

    # --- Stats ---

    def get_stats(self) -> Dict[str, Any]:
        note_count = self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        paper_count = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        tag_count = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        link_count = self.conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        return {
            "notes": note_count,
            "papers": paper_count,
            "tags": tag_count,
            "links": link_count,
        }

    # --- Concepts ---

    def upsert_concept(self, concept_id: str, canonical_name: str, description: str = ""):
        self.conn.execute(
            """INSERT INTO concepts (id, canonical_name, description)
               VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 canonical_name=excluded.canonical_name,
                 description=excluded.description""",
            (concept_id, canonical_name, description),
        )
        self.conn.commit()

    def get_concept(self, concept_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,)).fetchone()
        return dict(row) if row else None

    def get_concept_by_name(self, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE canonical_name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_concepts(self, limit: int = 500) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM concepts ORDER BY canonical_name LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def add_alias(self, alias: str, concept_id: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO concept_aliases (alias, concept_id) VALUES (?, ?)",
            (alias, concept_id),
        )
        self.conn.commit()

    def get_aliases(self, concept_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT alias FROM concept_aliases WHERE concept_id = ?", (concept_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def resolve_concept(self, name_or_alias: str) -> Optional[str]:
        """이름 또는 별칭에서 canonical name을 반환. 없으면 None."""
        row = self.conn.execute(
            "SELECT canonical_name FROM concepts WHERE canonical_name = ? OR id = ?",
            (name_or_alias, name_or_alias),
        ).fetchone()
        if row:
            return row[0]
        alias_row = self.conn.execute(
            "SELECT c.canonical_name FROM concept_aliases a "
            "JOIN concepts c ON a.concept_id = c.id "
            "WHERE a.alias = ?",
            (name_or_alias,),
        ).fetchone()
        return alias_row[0] if alias_row else None

    def delete_concept(self, concept_id: str):
        self.conn.execute("DELETE FROM concept_aliases WHERE concept_id = ?", (concept_id,))
        self.conn.execute("DELETE FROM kg_relations WHERE (source_type='concept' AND source_id=?) OR (target_type='concept' AND target_id=?)", (concept_id, concept_id))
        self.conn.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))
        self.conn.commit()

    # --- Knowledge Graph Relations ---

    def add_relation(
        self,
        source_type: str,
        source_id: str,
        relation: str,
        target_type: str,
        target_id: str,
        evidence_text: str = "",
        confidence: float = 0.5,
    ):
        self.conn.execute(
            """INSERT INTO kg_relations
                 (source_type, source_id, relation, target_type, target_id, evidence_text, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_type, source_id, relation, target_type, target_id) DO UPDATE SET
                 evidence_text=excluded.evidence_text,
                 confidence=excluded.confidence""",
            (source_type, source_id, relation, target_type, target_id, evidence_text, confidence),
        )
        self.conn.commit()

    def get_relations(self, entity_type: str, entity_id: str) -> List[dict]:
        rows = self.conn.execute(
            """SELECT * FROM kg_relations
               WHERE (source_type=? AND source_id=?) OR (target_type=? AND target_id=?)
               ORDER BY confidence DESC""",
            (entity_type, entity_id, entity_type, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_concept_papers(self, concept_id: str) -> List[dict]:
        """특정 개념을 사용하는 논문 목록 반환"""
        rows = self.conn.execute(
            """SELECT p.*, r.evidence_text, r.confidence
               FROM kg_relations r
               JOIN papers p ON r.source_id = p.arxiv_id
               WHERE r.source_type='paper'
                 AND r.relation='paper_uses_concept'
                 AND r.target_type='concept'
                 AND r.target_id=?
               ORDER BY r.confidence DESC""",
            (concept_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_paper_concepts(self, arxiv_id: str) -> List[dict]:
        """특정 논문이 사용하는 개념 목록 반환"""
        rows = self.conn.execute(
            """SELECT c.*, r.evidence_text, r.confidence
               FROM kg_relations r
               JOIN concepts c ON r.target_id = c.id
               WHERE r.source_type='paper'
                 AND r.relation='paper_uses_concept'
                 AND r.target_type='concept'
                 AND r.source_id=?
               ORDER BY r.confidence DESC""",
            (arxiv_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_related_concepts(self, concept_id: str) -> List[dict]:
        """특정 개념과 관련된 다른 개념 목록"""
        rows = self.conn.execute(
            """SELECT c.*, r.evidence_text, r.confidence
               FROM kg_relations r
               JOIN concepts c ON (
                 CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END
               ) = c.id
               WHERE r.relation='concept_related_to'
                 AND (
                   (r.source_type='concept' AND r.source_id=?)
                   OR (r.target_type='concept' AND r.target_id=?)
                 )
               ORDER BY r.confidence DESC""",
            (concept_id, concept_id, concept_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_relations(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM kg_relations").fetchone()[0]

    def count_concepts(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

    def get_kg_stats(self) -> Dict[str, Any]:
        concept_count = self.count_concepts()
        relation_count = self.count_relations()
        alias_count = self.conn.execute("SELECT COUNT(*) FROM concept_aliases").fetchone()[0]
        paper_count = self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

        isolated_concepts = self.conn.execute(
            """SELECT COUNT(*) FROM concepts c
               WHERE NOT EXISTS (
                 SELECT 1 FROM kg_relations r
                 WHERE (r.source_type='concept' AND r.source_id=c.id)
                    OR (r.target_type='concept' AND r.target_id=c.id)
               )"""
        ).fetchone()[0]

        rel_type_counts = self.conn.execute(
            "SELECT relation, COUNT(*) as cnt FROM kg_relations GROUP BY relation ORDER BY cnt DESC"
        ).fetchall()

        return {
            "concepts": concept_count,
            "aliases": alias_count,
            "papers": paper_count,
            "relations": relation_count,
            "isolated_concepts": isolated_concepts,
            "relation_types": {r["relation"]: r["cnt"] for r in rel_type_counts},
        }

    def close(self):
        self.conn.close()
