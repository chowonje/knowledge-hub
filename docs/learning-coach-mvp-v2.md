# Personal Learning Coach MVP v2

`knowledge-hub`의 개념 그래프/노트/논문 데이터를 이용해 학습 흐름을 자동화하는 CLI 중심 학습 코치입니다.

## 목표와 성공 기준

1. 관심 분야 입력 시 trunk/branch 구조가 재현 가능하게 생성된다.
2. 입력 표현이 달라도 canonical concept 기준으로 안정적으로 채점된다.
3. 엣지 수 부족/관계 불명확/근거 부족이 분리된 사유로 반환된다.
4. P0 원문은 외부 전송/결과 JSON/Obsidian writeback 본문에 저장되지 않는다.
5. 동일 `session_id` 재실행 시 writeback이 멱등하게 갱신된다.

## 아키텍처

1. Ingest: SQLite(`concepts`, `concept_aliases`, `kg_relations`, `papers`, `notes`) + `data/dynamic/*.jsonl` + opt-in 웹 신호
2. Identity: `canonical_id` + alias + fuzzy(0.88) + unknown 분류
3. Learning Core: map / assess-template / grade / next / run
4. Policy: P0 차단, pointerized evidence, external-call guard
5. Writeback: Obsidian `LearningHub/<topic-slug>`에 idempotent write
6. Event: `learning_events` append-only 감사 이벤트

## 명령 계약

```bash
khub labs learn map --topic "retrieval augmented generation" --source all --days 180 --top-k 12 --json --dry-run --no-writeback
khub labs learn assess-template --topic "retrieval augmented generation" --session-id "rag-001" --concept-count 6 --json --dry-run
khub labs learn grade --topic "retrieval augmented generation" --session-id "rag-001" --json --dry-run
khub labs learn next --topic "retrieval augmented generation" --session-id "rag-001" --json --dry-run
khub labs learn run --topic "retrieval augmented generation" --session-id "rag-001" --auto-next --json --dry-run
```

기본값은 `--no-writeback`이며, `--writeback`일 때만 Obsidian이 갱신됩니다.

## Concept Identity 정규화

해결 순서:
1. exact canonical/id match
2. alias table exact match
3. normalized string exact match
4. fuzzy match (threshold 0.88)
5. unknown (`normalization_failed`)

모든 trunk/branch/grade 계산은 원문 문자열이 아니라 resolved canonical ID를 기준으로 수행합니다.

## Trunk 점수 계산

`trunk_score = 0.45*topic_relevance + 0.25*graph_centrality + 0.20*evidence_coverage + 0.10*recency`

세부:
- `topic_relevance = 0.50 lexical + 0.35 semantic + 0.15 topic-proximity`
- `graph_centrality`는 topic-subgraph(관련 노드 우선)로 계산하고 generic penalty 적용
- `evidence_coverage`는 note/paper/web source 다양성 반영
- `recency`는 half-life 90일 보조 신호
- 기본 `top-k=12`, 결과에 `suggestedTopK` 반환

## 평가/게이트 규칙

- 관계 정규화 enum:
  - `causes`, `enables`, `part_of`, `contrasts`, `example_of`, `requires`, `improves`, `related_to`, `unknown_relation`
- `coverage = distinct_used_target_trunks / len(target_trunk_ids)`
- `edge_accuracy = (valid_edges + 1) / (total_edges + 2)`
- `explanation_quality = edges_with_valid_evidence_ptr / total_edges`
- `final = 0.50*edge_accuracy + 0.30*coverage + 0.20*explanation_quality`
- `min_edges = max(5, concept_count - 1)`

통과 조건:
- `final >= 0.75`
- `edge_accuracy >= 0.70`
- `coverage >= 0.60`
- `total_edges >= min_edges`

## P0 로컬-퍼스트 정책

- P0 원문은 external call / JSON / writeback 본문 저장 금지
- evidence는 `evidence_ptrs(path/heading/block_id/snippet_hash)`만 저장
- 외부 호출 기본 off, `--allow-external`일 때만 P1/P2 사실 전송 허용
- 정책 실패 시 `status=blocked`, `policyErrors[]` 반환 + 감사 이벤트 기록

## Obsidian Writeback 경로

- `/ABS_VAULT/LearningHub/<topic-slug>/00_Hub.md`
- `/ABS_VAULT/LearningHub/<topic-slug>/01_Trunk_Map.md`
- `/ABS_VAULT/LearningHub/<topic-slug>/sessions/<session-id>.md`
- `/ABS_VAULT/LearningHub/<topic-slug>/02_Next_Branches.md`
- `/ABS_VAULT/LearningHub/<topic-slug>/03_Web_Sources.md`
- `/ABS_VAULT/LearningHub/<topic-slug>/04_Web_Concepts.md`

`00_Hub.md` 고정 섹션:
- Active Session
- Recent Scores
- Unlocked Branches
- Next Action

세션 노트 frontmatter 필수:
- `topic`
- `session_id`
- `target_trunk_ids`
- `policy_mode`
- `created_at`

웹 수집 통합:
- `khub crawl ingest --extract-concepts`는 크롤링 직후 concepts/kg_relations를 반영합니다.
- 저신뢰 결과는 `web_ontology_pending` 큐로 적재되고 `khub labs crawl pending`으로 승인/거절합니다.
- `--writeback`일 때 `03_Web_Sources.md`, `04_Web_Concepts.md`를 marker 기반으로 멱등 갱신합니다.

## Sparse 그래프 fallback

조건:
- trunk 후보 `< 6` 또는 relation 품질 임계 미달

동작:
- papers + notes co-occurrence로 임시 edge 생성
- confidence 하향 적용
- `build-concepts / normalize-concepts` 권장 메시지 반환

## 저장/계약 파일

- Python service:
  - `knowledge_hub/learning/models.py`
  - `knowledge_hub/learning/mapper.py`
  - `knowledge_hub/learning/assessor.py`
  - `knowledge_hub/learning/recommender.py`
  - `knowledge_hub/learning/obsidian_writeback.py`
  - `knowledge_hub/learning/service.py`
- JSON schema:
  - `docs/schemas/learning-map-result.v1.json`
  - `docs/schemas/learning-grade-result.v1.json`
  - `docs/schemas/learning-next-result.v1.json`
- TypeScript contracts:
  - `foundry-core/src/contracts/learning.ts`

## MCP 연동

`khub-mcp` 서버에서 아래 도구로 동일 기능을 호출할 수 있습니다.

- `learn_map`
- `learn_assess_template`
- `learn_grade`
- `learn_next`
- `learn_run`

공통 입력 파라미터:
- `dry_run`: `true`면 writeback 비활성
- `writeback`: Obsidian 반영 여부
- `allow_external`: 외부 호출 opt-in (P1/P2만 허용)

## 가정

1. 단일 사용자 / 단일 vault
2. 기본 실행은 local-only(`--allow-external` off)
3. 한국어 기본, 영어 alias 동시 지원
4. 웹 소스는 명시적 opt-in일 때만 사용

## 그래프 백엔드 전략

현재 구현은 **A안(운영 백엔드)**으로 고정한다.
- 기본 저장: `SQLite + TTL` 경로(`knowledge-hub/local/web_ontology_graph.ttl` 등)
- 그래프 처리/검증: `rdflib`(`pySHACL`) 기반 내장
- Neo4j Desktop은 **미래 확장 대상**으로 보류
  - 이후 단계에서 Connector 형태로 읽기 전용 조회/시각화, 하이브리드 질의 강화에 사용

## 불확실 지점 대안

1. Concept 정규화
   - 대안 A(채택): Rule-based + alias + fuzzy
   - 대안 B(보류): LLM judge 기반 정규화
2. 결정 근거
   - A 장점: 재현성/비용/로컬 안정성
   - A 단점: alias 품질 의존
   - B 장점: 유연성
   - B 단점: 비용/비결정성/P0 리스크
