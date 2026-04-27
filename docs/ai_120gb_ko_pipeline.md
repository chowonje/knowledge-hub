# 120GB AI 지식 파이프라인 (영문 중심 → 한국어 활용)

## 목표
- 원본 120GB를 그대로 보존하면서, 한국어로 읽고 활용 가능한 지식 노트를 단계적으로 생성한다.
- 전체 번역이 아니라 "우선순위 기반 번역"으로 비용/시간/품질을 통제한다.

## 단계 (실행 순서)
1. 입력 수집 (Connector)
- `khub crawl run`으로 원본 수집
- 저장: `<pipeline-storage-root>/raw`

2. 정규화/중복 제거
- 저장: `<pipeline-storage-root>/normalized`
- `record_id`, `canonical_url_hash`, `content_sha256`로 추적/중복 차단

3. 온톨로지 추출 (Meaning Kernel)
- 엔티티/관계/클레임 후보 추출
- 미승인 관계는 pending 큐로 분리

4. 임베딩/인덱싱
- 저장: `<pipeline-storage-root>/indexed`
- 검색 가능한 벡터 인덱스 구축

5. 중요도 점수화 (노트 생성 후보 선정)
- 권장 기준:
  - `quality_score >= 0.70`
  - `entity >= 3`, `relation >= 1`
  - 중복 유사도 `< 0.85`
  - 도메인 신뢰도(allowlist) 가점
- 상위 문서만 다음 단계로 승격

6. 한국어 변환 (Translation/Summarization)
- 원칙: 영어 원문은 유지하고, 한국어는 "파생 산출물"로 생성
- 3단계 변환 정책:
  - T1 (기본): 제목/핵심요약(3~5문장) 한국어 생성
  - T2 (중요): abstract/핵심 단락 한국어 번역
  - T3 (최상위 5~10%): API 고급 모델로 심화 요약/구조화
- P0 원문은 외부 전송 금지, 외부 호출은 sanitized P1/P2만

7. Obsidian 노트 생성/반영
- 생성 위치(권장):
  - `AI/AI_Papers/Concepts/` (개념)
  - `LearningHub/ai/runs/` (실행 리포트)
- 노트 템플릿 필드:
  - `source_url`, `record_id`, `evidence`, `ko_summary`, `key_entities`, `key_relations`

8. 거버넌스/검증
- `khub health --check-events --check-pipeline`
- pending 승인/거절: `khub crawl pending ...`, `khub ontology pending ...`

## 영어가 대부분일 때의 실전 원칙
- 전체 본문 100% 번역하지 않는다.
- 먼저 "영문 원문 + 한국어 요약"으로 운영한다.
- 정말 중요한 문서만 부분/전체 번역한다.
- 용어 일관성을 위해 고정 용어집(Glossary)을 둔다.
  - 예: Transformer, Diffusion, Retrieval-Augmented Generation 등은 영문 유지 + 한국어 설명

## 지금 코드베이스에서 바로 가능한 것
- 수집/정규화/온톨로지/임베딩: `khub crawl run`
- 파이프라인 상태: `khub crawl status --job-id <JOB_ID> --json`
- 정합성 점검: `khub health --check-events --check-pipeline`
- 논문 한국어 번역/요약: `khub paper translate`, `khub paper summarize`

## 권장 운영 프로파일
- 대용량 안정 운영: `--profile safe`
- 기본 정책: `--source-policy hybrid`
- 외부 LLM은 기본 off, 필요 시 상위 후보만 on

## 최소 실행 예시
```bash
khub crawl run \
  --url-file /path/to/urls.txt \
  --topic "ai" \
  --source web \
  --profile safe \
  --source-policy hybrid \
  --index --extract-concepts --json

khub health --check-events --check-pipeline
```

## 권장 후속 운영
- `khub crawl collect`로 `run_pipeline -> ko-note-generate -> optional ko-note-apply`를 한 번에 묶을 수 있습니다.
- staging-first 운영을 유지하려면 `khub crawl collect --stage-only` 또는 `khub crawl ko-note-generate` 후 Obsidian에서 검토하고 `khub crawl ko-note-apply`를 실행합니다.
