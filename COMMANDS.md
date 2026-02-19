# Knowledge Hub CLI 명령어 가이드

## 시작하기

```bash
cd /Users/won/Desktop/allinone/knowledge-hub
source venv/bin/activate
```

## 초기 설정

```bash
khub init
```

대화형으로 OpenAI API 키, Obsidian vault 경로, LLM 프로바이더 등을 설정합니다.
설정은 `~/.khub/config.yaml`에 저장됩니다.

---

## 설정 관리

```bash
# 현재 설정 보기
khub config list

# 값 조회
khub config get embedding.provider

# 값 변경
khub config set embedding.provider openai
khub config set obsidian.vault_path "/path/to/vault"
```

---

## 시스템 상태

```bash
khub status
```

---

## 논문 자동 탐색 파이프라인

검색 → 다운로드 → 요약 → 번역 → 인덱싱 → Obsidian 연결을 한 번에 수행합니다.

```bash
# 기본 사용
khub discover "large language model agent" -n 5 --year 2024

# 확인 없이 바로 진행
khub discover "RAG" -n 10 --min-citations 50 -y

# 인용수 기준 정렬
khub discover "transformer" -n 5 --sort citationCount

# 병렬 다운로드 워커 수 지정
khub discover "AI agent" -w 8

# 번역/요약/인덱싱 개별 제어
khub discover "topic" --no-translate --no-summarize --no-index
```

### 옵션

| 플래그 | 설명 |
|--------|------|
| `-n N` | 최대 수집 논문 수 (기본: 5) |
| `--year YYYY` | 검색 시작 연도 |
| `--min-citations N` | 최소 인용수 |
| `--sort relevance/citationCount` | 정렬 기준 |
| `-y` | 확인 프롬프트 건너뛰기 |
| `-w N` | 병렬 다운로드 워커 수 (기본: 4) |
| `--no-translate` | 번역 건너뛰기 |
| `--no-summarize` | 요약 건너뛰기 |
| `--no-index` | 벡터 인덱싱 건너뛰기 |
| `--no-obsidian` | Obsidian 노트 생성 건너뛰기 |

---

## 개별 논문 관리

### 논문 추가 (URL)

```bash
# arXiv, OpenReview, PapersWithCode, HuggingFace, Semantic Scholar 등 지원
khub paper add "https://arxiv.org/abs/2501.06322"
```

### 단일 작업

```bash
khub paper download 2501.06322       # PDF/텍스트 다운로드
khub paper summarize 2501.06322      # 요약 생성
khub paper translate 2501.06322      # 전체 번역
khub paper embed 2501.06322          # 벡터 임베딩
khub paper info 2501.06322           # 상세 정보 조회
```

### 배치 작업

```bash
khub paper translate-all             # 미번역 논문 전체 번역
khub paper summarize-all             # 미요약 논문 전체 요약
khub paper embed-all                 # 미인덱싱 논문 전체 임베딩

# 옵션
khub paper translate-all -n 10       # 최대 10편만
khub paper summarize-all -f "CS"     # 분야 필터
```

### 키워드 & 개념 관리

```bash
# 논문에서 키워드 추출 → Obsidian 노트 + Concept Index 갱신
khub paper sync-keywords
khub paper sync-keywords --force     # 기존 키워드 재추출

# 개념 노트 자동 생성 (설명 + 관련 개념 + 관련 논문)
khub paper build-concepts
khub paper build-concepts --force    # 기존 노트 재생성
```

### 논문 목록

```bash
khub paper list
khub paper list -f "Computer Science" -n 100
```

---

## 벡터 인덱싱

논문 + 개념 노트를 벡터DB에 통합 인덱싱합니다.

```bash
khub index                           # 미인덱싱 항목만
khub index --all                     # 전체 재인덱싱
khub index --concepts-only           # 개념 노트만
```

---

## 통합 검색 & RAG

```bash
# 의미론적 검색 (논문 + 개념 노트)
khub search "transformer attention"
khub search "강화학습" --top-k 10

# RAG 질의 (관련 문서 기반 AI 답변)
khub ask "셀프 어텐션이 뭐야?"
```

---

## Google NotebookLM 연동

```bash
khub notebook create "AI Agent 연구"    # 노트북 생성
khub notebook list                      # 노트북 목록
khub notebook sync <id>                 # 논문 동기화
khub notebook study-pack <id>           # 학습 팩 생성
```

---

## 디버그 & 문제 해결

```bash
# 디버그 로그 출력
khub --verbose discover "topic" -n 1 -y

# 버전 확인
khub --version
```

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (임베딩/요약/번역/키워드 추출) |
| `ANTHROPIC_API_KEY` | Anthropic API 키 (선택) |
| `GOOGLE_API_KEY` | Google AI API 키 (선택) |
| `GOOGLE_CLOUD_PROJECT` | NotebookLM 프로젝트 번호 |

`.env` 파일에 설정하면 자동 로드됩니다.

---

## 테스트

```bash
python -m pytest tests/ -v
```
