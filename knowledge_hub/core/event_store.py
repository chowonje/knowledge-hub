"""
온톨로지 이벤트 소싱 스토어

모든 온톨로지 변경을 append-only 이벤트로 기록하여
특정 시점의 지식 상태를 재구성할 수 있습니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.core.models import OntologyEvent


class EventStore:
    """
    온톨로지 이벤트 스토어
    
    - SQLite: 빠른 쿼리를 위한 인덱스
    - JSONL: append-only ground truth
    """
    
    def __init__(self, sqlite_db: SQLiteDatabase, jsonl_path: str | Path):
        self.db = sqlite_db
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_event_tables()
    
    def _ensure_event_tables(self):
        """ontology_events 테이블 생성"""
        cursor = self.db.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ontology_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_type TEXT,
                actor TEXT DEFAULT 'system',
                run_id TEXT,
                policy_class TEXT DEFAULT 'P2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_oe_entity
            ON ontology_events(entity_id, created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_oe_type
            ON ontology_events(event_type, created_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_oe_run
            ON ontology_events(run_id)
        """)
        
        self.db.conn.commit()
    
    def append(self, event: OntologyEvent) -> None:
        """
        이벤트 추가 (SQLite + JSONL 모두 기록)
        
        Args:
            event: 추가할 OntologyEvent
        """
        # 1) JSONL에 append (ground truth)
        event_json = json.dumps(event.to_dict(), ensure_ascii=False)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(event_json + "\n")
        
        # 2) SQLite에 인덱스 기록
        self.db.conn.execute(
            """INSERT INTO ontology_events 
               (event_id, event_type, entity_id, entity_type, actor, run_id, policy_class, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.event_type,
                event.entity_id,
                event.entity_type,
                event.actor,
                event.run_id,
                event.policy_class,
                event.timestamp,
            ),
        )
        self.db.conn.commit()
    
    def replay(
        self,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[OntologyEvent]:
        """
        이벤트 재생 (JSONL에서 읽기)
        
        Args:
            from_time: 시작 시간 (ISO 8601)
            to_time: 종료 시간 (ISO 8601)
            entity_id: 특정 엔티티만 필터
            event_type: 특정 이벤트 타입만 필터
        
        Returns:
            OntologyEvent 리스트
        """
        if not self.jsonl_path.exists():
            return []
        
        events: list[OntologyEvent] = []
        
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 필터링
                    if from_time and data.get("timestamp", "") < from_time:
                        continue
                    if to_time and data.get("timestamp", "") > to_time:
                        continue
                    if entity_id and data.get("entity_id") != entity_id:
                        continue
                    if event_type and data.get("event_type") != event_type:
                        continue
                    
                    # OntologyEvent로 변환
                    event = OntologyEvent(
                        event_id=data.get("event_id", ""),
                        timestamp=data.get("timestamp", ""),
                        event_type=data.get("event_type", ""),
                        entity_id=data.get("entity_id", ""),
                        entity_type=data.get("entity_type", ""),
                        actor=data.get("actor", "system"),
                        data=data.get("data", {}),
                        policy_class=data.get("policy_class", "P2"),
                        run_id=data.get("run_id"),
                    )
                    events.append(event)
                
                except Exception:
                    # 손상된 라인은 건너뜀
                    continue
        
        return events
    
    def snapshot_at(self, timestamp: str) -> dict[str, Any]:
        """
        특정 시점의 온톨로지 상태 재구성
        
        Args:
            timestamp: ISO 8601 시간
        
        Returns:
            {
                "entities": {entity_id: {...}},
                "claims": {claim_id: {...}},
                "relations": [...],
                "snapshot_time": timestamp
            }
        """
        events = self.replay(to_time=timestamp)
        
        entities: dict[str, dict] = {}
        claims: dict[str, dict] = {}
        relations: list[dict] = []
        
        for event in events:
            event_type = event.event_type
            entity_id = event.entity_id
            data = event.data
            
            if event_type == "entity_created":
                entities[entity_id] = {
                    "entity_id": entity_id,
                    "entity_type": data.get("entity_type", ""),
                    "canonical_name": data.get("canonical_name", ""),
                    "description": data.get("description", ""),
                    "properties": data.get("properties", {}),
                    "confidence": data.get("confidence", 1.0),
                    "source": data.get("source", "system"),
                }
            
            elif event_type == "entity_updated":
                if entity_id in entities:
                    entities[entity_id].update(data)
            
            elif event_type == "entity_deleted":
                if entity_id in entities:
                    del entities[entity_id]
            
            elif event_type == "claim_added":
                claim_id = data.get("claim_id", entity_id)
                claims[claim_id] = {
                    "claim_id": claim_id,
                    "claim_text": data.get("claim_text", ""),
                    "subject_entity_id": data.get("subject_entity_id", ""),
                    "predicate": data.get("predicate", ""),
                    "object_entity_id": data.get("object_entity_id"),
                    "object_literal": data.get("object_literal"),
                    "confidence": data.get("confidence", 0.5),
                    "evidence_ptrs": data.get("evidence_ptrs", []),
                }
            
            elif event_type == "claim_updated":
                claim_id = data.get("claim_id", entity_id)
                if claim_id in claims:
                    claims[claim_id].update(data)
            
            elif event_type == "claim_deleted":
                claim_id = data.get("claim_id", entity_id)
                if claim_id in claims:
                    del claims[claim_id]
            
            elif event_type == "relation_added":
                relations.append({
                    "source_id": data.get("source_id", ""),
                    "source_type": data.get("source_type", ""),
                    "relation": data.get("relation", ""),
                    "target_id": data.get("target_id", ""),
                    "target_type": data.get("target_type", ""),
                    "confidence": data.get("confidence", 0.5),
                })
        
        return {
            "entities": entities,
            "claims": claims,
            "relations": relations,
            "snapshot_time": timestamp,
            "entity_count": len(entities),
            "claim_count": len(claims),
            "relation_count": len(relations),
        }
    
    def get_entity_history(self, entity_id: str) -> list[OntologyEvent]:
        """특정 엔티티의 변경 이력 조회"""
        return self.replay(entity_id=entity_id)
    
    def list_recent_events(self, limit: int = 100) -> list[dict]:
        """최근 이벤트 조회 (SQLite 인덱스 사용)"""
        rows = self.db.conn.execute(
            """SELECT * FROM ontology_events 
               ORDER BY created_at DESC, id DESC 
               LIMIT ?""",
            (limit,)
        ).fetchall()
        
        return [dict(row) for row in rows]
