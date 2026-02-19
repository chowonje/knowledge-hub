# Knowledge Hub

AI 논문을 자동으로 검색, 다운로드, 번역, 요약하고 Obsidian과 연결하는 파이프라인 도구입니다.

## Features

- **논문 자동 발견** - Semantic Scholar + arXiv에서 최신/중요 논문 검색
- **벡터DB 중복 체크** - ChromaDB + SQLite로 이미 수집된 논문 자동 건너뜀
- **플러그인 AI 프로바이더** - OpenAI, Anthropic, Google, Ollama(로컬) 자유롭게 교체
- **한국어 번역/요약** - 원하는 AI로 논문 번역 및 요약 생성
- **Obsidian 연결** - vault에 논문 요약 노트 자동 생성, 관련 노트 `[[링크]]` 삽입
- **벡터 검색 + RAG** - 수집된 논문/노트에서 시맨틱 검색 및 질의응답

## Quick Start

```bash
# 설치
pip install knowledge-hub[openai,ollama]

# 초기 설정
khub init

# AI 논문 5편 자동 수집
khub discover "large language model agent" -n 5 --year 2024

# 수집된 논문 확인
khub paper list

# 벡터 검색
khub search "attention mechanism"

# RAG 질의
khub ask "Transformer의 핵심 아이디어는?"
```

## Installation

### 기본 설치

```bash
pip install knowledge-hub
```

### 프로바이더별 설치

```bash
# OpenAI만 사용
pip install knowledge-hub[openai]

# Ollama(로컬) + OpenAI
pip install knowledge-hub[ollama,openai]

# 전체 프로바이더
pip install knowledge-hub[all]
```

### 개발 모드 설치

```bash
git clone https://github.com/knowledge-hub/knowledge-hub.git
cd knowledge-hub
pip install -e ".[all]"
```

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

# 설정 확인
khub config get translation

# 사용 가능한 프로바이더 확인
khub config providers
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
| **OpenAI** | GPT-4o, GPT-4o-mini | text-embedding-3-small | - | `pip install knowledge-hub[openai]` |
| **Anthropic** | Claude 3.5 Sonnet/Haiku | - | - | `pip install knowledge-hub[anthropic]` |
| **Google** | Gemini 2.0 Flash/Pro | text-embedding-004 | - | `pip install knowledge-hub[google]` |
| **Ollama** | Qwen, Llama, Gemma 등 | nomic-embed-text 등 | O | `pip install knowledge-hub[ollama]` |

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

# 요약 없이 다운로드만
khub discover "AI safety" --no-summarize

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
khub paper translate 2401.12345 -p anthropic -m claude-3-5-haiku-20241022  # 프로바이더 지정
```

### `khub search` / `khub ask` - 검색 & RAG

```bash
khub search "attention mechanism" -k 10  # 벡터 검색
khub ask "RAG의 장단점을 설명해줘"         # RAG 질의
```

## Requirements

- Python >= 3.10
- 임베딩용 로컬 LLM 사용 시: [Ollama](https://ollama.ai/) 설치 필요

## License

MIT
