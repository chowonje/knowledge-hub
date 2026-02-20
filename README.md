# Knowledge Hub

AI 논문을 자동으로 검색, 다운로드, 번역, 요약하고 Obsidian과 연결하는 CLI 파이프라인 도구입니다.

## Features

- **논문 자동 발견** - Semantic Scholar + arXiv에서 최신/중요 논문 검색
- **벡터DB 중복 체크** - ChromaDB + SQLite로 이미 수집된 논문 자동 건너뜀
- **플러그인 AI 프로바이더** - OpenAI, Anthropic, Google, Ollama, OpenAI-Compatible(DeepSeek/Groq 등) 자유롭게 교체
- **한국어 번역/요약** - 원하는 AI로 논문 번역 및 심층 요약 생성
- **Obsidian 연결** - vault에 논문 요약 노트 자동 생성, 관련 노트 `[[링크]]` 삽입
- **벡터 검색 + RAG** - 수집된 논문/노트에서 시맨틱 검색 및 질의응답
- **학술 탐색** - 저자 검색, 인용 네트워크 분석, 참고문헌 탐색
- **MCP 서버** - Cursor/Claude Code에서 모든 기능을 에이전트 도구로 사용 가능

## Quick Start

```bash
# 설치 (최소)
pip install -e .

# OpenAI 사용 시
pip install -e ".[openai]"

# Ollama(로컬) 사용 시
pip install -e ".[ollama]"

# 전체 프로바이더
pip install -e ".[all]"
```

### 최소 데모 흐름

```bash
# 1. 초기 설정 (프로바이더/API키/경로 설정)
khub init

# 2. 시스템 상태 확인
khub status

# 3. 논문 검색 + 수집 (Semantic Scholar → 다운로드 → 번역 → 요약 → Obsidian)
khub discover "large language model agent" -n 3

# 4. 수집 결과 확인
khub paper list

# 5. 벡터 인덱싱
khub index

# 6. 벡터 검색
khub search "attention mechanism"

# 7. RAG 질의
khub ask "Transformer의 핵심 아이디어는?"

# 8. 학술 탐색
khub explore author "Yoshua Bengio"
khub explore paper 1706.03762
khub explore citations 1706.03762
```

## Installation

### 기본 설치 (PyPI 미등록 시 로컬 설치)

```bash
git clone https://github.com/chowonje/knowledge-hub.git
cd knowledge-hub
pip install -e ".[openai]"
```

### 프로바이더별 설치

```bash
# OpenAI만 사용
pip install -e ".[openai]"

# Ollama(로컬) + OpenAI
pip install -e ".[ollama,openai]"

# 전체 프로바이더
pip install -e ".[all]"
```

### 사전 조건

| 구성 요소 | 필수 여부 | 설명 |
|---|---|---|
| Python >= 3.10 | 필수 | 런타임 |
| API 키 (하나 이상) | 필수 | OpenAI, Anthropic, Google 중 택 1 |
| [Ollama](https://ollama.ai/) | 선택 | 로컬 LLM/임베딩 사용 시 |
| [Obsidian](https://obsidian.md/) | 선택 | 지식 그래프 연결, 노트 생성 시 |

## Configuration

### 인터랙티브 설정

```bash
khub init
```

번역/요약/임베딩에 사용할 AI 프로바이더, API 키, 저장 경로 등을 대화형으로 설정합니다.
설정은 `~/.khub/config.yaml`에 저장됩니다.

### 설정 관리

```bash
# 전체 설정 보기
khub config list

# 개별 설정 변경
khub config set translation.provider openai
khub config set translation.model gpt-4o-mini
khub config set summarization.provider ollama
khub config set summarization.model qwen2.5:14b

# 사용 가능한 프로바이더 + 모델 확인
khub config providers --models
```

### 설정 파일 예시 (`~/.khub/config.yaml`)

```yaml
translation:
  provider: openai
  model: gpt-4o-mini

summarization:
  provider: ollama
  model: qwen2.5:14b

embedding:
  provider: ollama
  model: nomic-embed-text

storage:
  papers_dir: ~/.khub/papers
  vector_db: ~/.khub/chroma_db
  sqlite: ~/.khub/knowledge.db

obsidian:
  enabled: true
  vault_path: /path/to/your/obsidian/vault

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  ollama:
    base_url: http://localhost:11434
```

## AI Providers

| 프로바이더 | LLM | Embedding | 로컬 | 설치 |
|---|---|---|---|---|
| **OpenAI** | GPT-4o, 4o-mini, o1, o3, 4.1-nano | text-embedding-3-small/large | - | `[openai]` |
| **Anthropic** | Claude Opus 4, Sonnet 4, 3.5 Sonnet | - | - | `[anthropic]` |
| **Google** | Gemini 2.0 Flash/Pro/Lite | text-embedding-004 | - | `[google]` |
| **Ollama** | Qwen3, Llama4, Gemma3, DeepSeek-R1 등 | nomic-embed-text, bge-m3, snowflake | O | `[ollama]` |
| **OpenAI-Compatible** | DeepSeek, Groq, Together AI, Mistral 등 | 서비스별 지원 | - | 추가 의존성 없음 |

용도별로 다른 프로바이더를 조합할 수 있습니다:
- 번역: OpenAI GPT-4o-mini (저렴, 고품질)
- 요약: Ollama qwen2.5:14b (무료, 로컬)
- 임베딩: Ollama nomic-embed-text (무료, 로컬)

## Commands

### `khub discover` - 핵심 파이프라인

```bash
# 기본 사용
khub discover "topic" -n 5

# 연도 필터 + 인용수 필터
khub discover "RAG retrieval augmented generation" --year 2024 --min-citations 10

# 인용수 기준 정렬
khub discover "transformer" --sort citationCount -n 10

# Obsidian 노트 생성 포함
khub discover "AI agent" --obsidian
```

### `khub paper` - 개별 논문 관리

```bash
khub paper list                          # 목록
khub paper info 2401.12345               # 상세 정보
khub paper download 2401.12345           # 다운로드
khub paper translate 2401.12345          # 번역
khub paper summarize 2401.12345          # 요약
khub paper translate 2401.12345 -p anthropic -m claude-3-5-haiku-20241022
khub paper sync-keywords                 # 키워드 추출 + 개념 연결
khub paper build-concepts                # 개념 노트 자동 생성
khub paper normalize-concepts            # 동의어/약어 정규화
```

### `khub explore` - 학술 탐색

```bash
khub explore author "Yoshua Bengio"      # 저자 검색
khub explore author-papers <author_id>   # 저자 논문 목록
khub explore paper 1706.03762            # 논문 상세 정보
khub explore citations 1706.03762        # 인용 논문 목록
khub explore references 1706.03762       # 참고 논문 목록
khub explore network 1706.03762          # 인용 네트워크 분석
khub explore batch 1706.03762 2301.10226 # 배치 조회
```

### `khub search` / `khub ask` - 검색 & RAG

```bash
khub search "attention mechanism" -k 10  # 벡터 검색
khub ask "RAG의 장단점을 설명해줘"         # RAG 질의
```

### `khub index` - 벡터 인덱싱

```bash
khub index                               # 미인덱싱 논문 + 개념 인덱싱
khub index --all                         # 전체 재인덱싱
khub index --concepts-only               # 개념 노트만
```

### `khub notebook` - 지식 노트

```bash
khub notebook list                       # 노트 목록
khub notebook show <id>                  # 노트 상세
```

### `khub graph` - 지식 그래프

```bash
khub graph stats                         # 그래프 통계
khub graph show <id>                     # 노트 연결
khub graph isolated                      # 고립 노트
```

## 명령별 필수 의존성

| 명령 | LLM 필요 | 임베딩 필요 | Obsidian 필요 | API 키 |
|---|---|---|---|---|
| `khub discover` | O (번역/요약) | O (인덱싱) | 선택 | 프로바이더에 따라 |
| `khub paper list/info` | - | - | - | - |
| `khub paper translate` | O | - | - | 프로바이더에 따라 |
| `khub paper summarize` | O | - | - | 프로바이더에 따라 |
| `khub index` | - | O | 선택(개념 노트) | 프로바이더에 따라 |
| `khub search/ask` | O (ask만) | O | - | 프로바이더에 따라 |
| `khub explore *` | - | - | - | - (Semantic Scholar 무료) |
| `khub paper sync-keywords` | O | - | O | 프로바이더에 따라 |
| `khub paper build-concepts` | O | - | O | 프로바이더에 따라 |
| `khub status` | - | - | - | - |

## MCP 서버

Cursor / Claude Code에서 khub 기능을 에이전트 도구로 사용:

```bash
# MCP 서버 시작
khub-mcp
```

`~/.cursor/mcp.json` 또는 에이전트 설정에 추가하면 됩니다.

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: ollama` | ollama extra 미설치 | `pip install -e ".[ollama]"` 또는 다른 프로바이더로 변경 |
| `khub search` 실패 | 임베딩 프로바이더 미설정 | `khub init` → 임베딩 프로바이더 선택 |
| `OPENAI_API_KEY` 오류 | 환경변수 미설정 | `export OPENAI_API_KEY=sk-...` 또는 `.env` 파일 생성 |
| Obsidian 관련 명령 실패 | vault 경로 미설정 | `khub config set obsidian.vault_path /path/to/vault` |
| 인덱싱 0건 | 논문 미수집 | `khub discover "topic" -n 3` 먼저 실행 |

## License

MIT
