# Knowledge Hub Full Guide

This guide preserves the extended setup, command, provider, and operator
reference that previously lived in the root README. The root README is now a
short public Research Preview storefront focused on the supported
`add -> index -> search/ask -> evidence review` path.

## Original Extended README

Knowledge Hub는 Obsidian, papers, web, and optional project context를 근거로 LLM assistance를 제공하는 **local-first, policy-gated, retrieval-assistant-first** knowledge runtime입니다.

Status: **Research Preview**. The supported default path is `add -> index -> search/ask -> evidence review`. APIs, quality bars, and experimental surfaces may change without notice.

## Supported Default Path

Knowledge Hub의 public product promise는 의도적으로 좁습니다:

`add -> index -> search/ask -> evidence review`

현재 대표 시나리오는 다음입니다.

- 개인 연구자나 builder가 논문, 웹, vault 노트를 canonical local stores에 모은다.
- `khub`가 그 로컬 코퍼스를 인덱싱하고, `search`/`ask`로 근거 기반 결과를 준다.
- 사용자는 답을 그대로 소비하는 대신 evidence trail을 다시 확인하고 후속 읽기로 이어간다.

이 기본 경로의 현재 success 기준은 아래와 같습니다.

- `add`: 최소 1개의 source item이 canonical local storage에 등록된다.
- `index`: retrieval-facing vector/document surface가 비어 있지 않다.
- `search` or `ask`: grounded result 또는 explicit lack-of-evidence outcome을 반환한다.
- `evidence review`: source trace나 후속 reading surface로 grounding path를 다시 확인할 수 있다.

## Quick Start

Repository checkout 기준 최소 실행 경로:

```bash
pip install -e ".[ollama]"
khub doctor
khub search "attention mechanism"
khub ask "Transformer의 핵심 아이디어는?"
```

실제 코퍼스를 넣고 representative result를 확인하려면 최소 1개 source ingest 후 indexing을 먼저 수행하세요.

```bash
khub add "large language model agent" --type paper -n 3
khub index
```

## Known Limits

- This repository is a **Research Preview**, not a stable release.
- We do not claim full-repo green; the current green signal is limited to a narrow smoke gate / approval slice.
- Source quality is uneven. `paper` and `project` are currently stronger than at least one known `vault` compare path, which can still degrade to `0 source`.
- The Python / TypeScript boundary is structurally defined but still mid-migration in some areas, so less-traveled paths can still have rough edges.
- Several surfaces remain experimental or operator-facing: `khub labs ...`, `Agent Gateway`, learning workflows, and `foundry-core`.
- Not recommended for production use without internal validation.

## Default vs Experimental Surfaces

The default public/runtime promise is intentionally smaller than the full implementation surface.

### Core Runtime

The default surface is limited to:

- local ingestion and indexing
- grounded search and ask
- evidence review
- read-only task-context assembly
- policy / approval / provenance

### Experimental and Additive Surfaces

The repository also contains broader capabilities, but they are not the default product promise:

- `khub labs ...` operator and experimental workflows
- `Agent Gateway` context packing and approval-gated writeback lanes
- learning workflows
- `foundry-core` delegated runtime and bridge-facing execution

These surfaces stay visible for transparency, but they should be treated as `experimental`, `labs-first`, and `subject to change without notice`.

The default `khub --help` surface now favors the representative core loop. `khub add` is the preferred intake facade for web URLs, YouTube URLs, paper URLs, and paper discovery queries. Lower-level ingestion commands such as `khub discover`, `khub crawl`, `khub health`, `khub setup`, `khub vault`, and `khub mcp` still exist for compatibility or advanced use, but they are hidden from the default top-level help. Internal verification and operator-heavy personal commands remain outside public discovery.

## Features

- **Grounded retrieval** - vault, paper, web에서 evidence를 찾아 질의응답
- **Task context assembly** - repo context까지 읽기 전용으로 묶어 Codex-style assistance 지원
- **Paper ingestion** - Semantic Scholar + arXiv 기반 검색, 다운로드, 요약, 인덱싱
- **Obsidian 연결** - vault note를 로컬 source로 인덱싱하고 writeback은 명시 옵션으로만 수행
- **Provider setup** - 로컬/Ollama, API provider, Codex MCP, OpenAI-compatible custom 모델을 역할별로 설정
- **MCP 서버** - 기본 profile은 retrieval-assistant-first, 고급 기능은 labs profile로 노출

## Extended Setup and Product Flow

```bash
# 설치 (최소, PyPI)
pip install knowledge-hub-cli

# OpenAI 사용 시
pip install "knowledge-hub-cli[openai]"

# Ollama(로컬) 사용 시
pip install "knowledge-hub-cli[ollama]"

# crawl4ai 웹 수집 사용 시
pip install "knowledge-hub-cli[crawl4ai]"

# vault topology(UAMP/PCA) 사용 시
pip install "knowledge-hub-cli[topology]"

# 전체 프로바이더
pip install "knowledge-hub-cli[all]"

# 로컬 수정 후 개발/테스트
pip install -e .
```

### 기본 product 흐름

```bash
# 1. 초기 설정
khub init

# 2. 사용자용 환경 진단
khub doctor

# 3. 상세 상태 확인(엔지니어용)
khub status

# 4. 소스 추가(URL / YouTube / 논문 URL / 논문 검색)
khub add "large language model agent" --type paper -n 3
khub add "https://example.com/guide" --topic "rag"
khub add "https://youtu.be/<video-id>" --topic "agents"

# 5. 수집 결과 확인
khub paper list

# 6. 인덱싱
khub index

# 7. grounded search
khub search "attention mechanism"

# 8. grounded answer
khub ask "Transformer의 핵심 아이디어는?"

# 9. paper reading surface
khub paper summary --paper-id 2501.06322
khub paper evidence --paper-id 2501.06322
khub paper memory --paper-id 2501.06322
khub paper related --paper-id 2501.06322

# 10. Codex-style read-only task context
khub context "how should I refactor the RAG flow?" --repo-path .
```

주간 안정화에서 `index -> ask/search` 코어 루프를 반복 확인할 때는 아래 smoke gate를 사용합니다.

```bash
python scripts/check_release_smoke.py --mode weekly_core_loop --json
```

### Labs surface

비핵심/실험 기능은 기본 help에서 숨기고 `khub labs ...`로 노출합니다.

```bash
khub labs --help
khub labs crawl --help
khub labs learn --help
khub labs ops --help
```

MCP에서도 labs 도구는 기본적으로 숨겨지며, 필요하면 profile을 명시합니다.

```bash
export KHUB_MCP_PROFILE=default  # default | labs | all
```

ko-note는 기본적으로 `generate -> inspect staged notes in Obsidian -> apply` 흐름을 권장합니다. review/remediation/manual enrich는 `khub labs crawl ...` 아래의 고급 운영 명령으로 남아 있습니다.
YouTube URL ingest는 기본 intake에서는 `khub add <youtube-url>`로 진입하고, 고급 운영 옵션은 `khub labs crawl youtube-ingest ...`에 남아 있습니다. 구현은 `caption-first + description/chapters merge + optional local ASR fallback`이며, 로컬 fallback을 쓰려면 `yt-dlp`, `ffmpeg`, `openai-whisper`가 필요합니다.

### Stability Notes

- Canonical CLI/MCP entrypoints are `knowledge_hub.interfaces.cli.main` and `knowledge_hub.interfaces.mcp.server`.
- Legacy `knowledge_hub.cli.*` and `knowledge_hub.mcp_server` remain compatibility shims only.
- `khub health --check-events` remains a hard integrity check for ontology event drift; use `python scripts/repair_event_integrity.py --json` for explicit repair/backfill instead of manual SQL.
- `khub health --check-pipeline` now treats an unavailable external `storage_root` as a precondition warning/skip rather than as proof of pipeline corruption.
- Non-destructive checkpoint splitting now uses `ops/checkpoints/checkpoints.json` plus `python scripts/report_checkpoint_split.py --write-pathspec-dir <dir>` so the current dirty tree can be staged in exact bucket slices without rewriting history.

### Source-aware ask contract

`khub ask` now uses source-aware domain packs instead of a single generic ask route.

#### `--source paper`

`khub ask --source paper` treats paper questions as one of four default families:

- `concept_explainer`: concept resolver -> alias/acronym expansion -> representative paper -> grounded explanation
- `paper_lookup`: exact/near-exact paper resolution -> paper/document memory -> paper-scoped answer
- `paper_compare`: claim-aligned comparison across at least two resolved papers
- `paper_discover`: broad paper retrieval -> shortlist -> lightweight organized summary

Default `paper` ask is intentionally conservative:

- broad concept questions do not narrow to one weak paper match
- broad discovery questions stop at shortlist-style organization
- `concept_explainer` keeps the legacy hybrid route as the default, but now uses bounded fan-out on the paper path: at most `base + 2 planned terms + 1 rescue query`, at most 2 lexical forms, and at most one representative-paper scoped extra search
- `khub ask --json` keeps the existing paper diagnostics and now also exposes `retrievalObjectsAvailable`, `retrievalObjectsUsed`, and `representativeRole`

- deep multi-paper synthesis stays on `khub labs paper topic-synthesize`
- `khub ask ... --json` exposes `paperFamily`, `queryPlan`, `representativePaper`, `plannerFallback`, and `familyRouteDiagnostics`

#### `--source web`

`khub ask --source web` now resolves through `knowledge_hub.domain.web_knowledge` and classifies questions into four default families:

- `reference_explainer`: guide/reference/definition questions that should prefer stable reference material
- `temporal_update`: latest/change/update questions that require explicit version/date/observed grounding
- `relation_explainer`: connection/relationship questions that can use ontology or claim support but still verify against web evidence
- `source_disambiguation`: “reference article vs latest feed” style source-class selection questions

Default `web` ask is also conservative:

- explicit URL, host, or page-title matches create a hard scope before retrieval
- `latest`, `update`, `changed`, `최근`, `업데이트` push the route into temporal handling instead of the generic explainer path
- weak `observed_at`-only signals do not justify a strong “latest” answer
- `SectionCard` and `ClaimCard` are support objects only; the current default keeps raw/document-memory verification in charge

- `khub ask ... --json` uses `queryFrame.family`, `evidencePolicy`, and `familyRouteDiagnostics` as the canonical cross-source diagnostics, and web diagnostics add `temporalSignalsApplied`, `referenceSourceApplied`, and `watchlistScopeApplied`

The implementation boundary is now split on purpose:

- `knowledge_hub.ai` keeps the generic retrieval/answer engine
- `knowledge_hub.domain.ai_papers` owns paper-specific query families, lookup rules, representative-paper hints, and claim-card helpers
- `knowledge_hub.domain.web_knowledge` owns web-specific query families, temporal/reference/source-class routing, and evidence-policy selection
- `knowledge_hub.ai.paper_query_plan` and `knowledge_hub.ai.claim_cards` remain compatibility shims for one stabilization cycle

## Repository Layout

- `knowledge_hub/`: Python 제품 코드, CLI, MCP
- `foundry-core/`: TypeScript 런타임, 정책 경계, 브리지
- `data/`: 런타임 데이터와 curated watchlist
- `docs/`: 운영/설계 문서
- `scripts/`: 유지보수 및 일회성 자동화

### Architecture read order

새로 구조를 파악할 때는 아래 순서를 권장합니다.

1. [`docs/maps/README.md`](docs/maps/README.md)
2. [`docs/maps/canonical-ownership-map.md`](docs/maps/canonical-ownership-map.md)
3. [`docs/maps/agent-execution-map.md`](docs/maps/agent-execution-map.md)
4. [`docs/maps/data-policy-flow-map.md`](docs/maps/data-policy-flow-map.md)
5. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)

추가 구조 설명:
- [`docs/repo-layout.md`](docs/repo-layout.md)
- [`docs/guides/cli-commands.md`](docs/guides/cli-commands.md)

## Installation

### 기본 설치

```bash
# PyPI 설치(권장)
pip install "knowledge-hub-cli[openai]"
```

### 로컬 개발/테스트 설치

```bash
git clone https://github.com/chowonje/knowledge-hub.git
cd knowledge-hub
pip install -e .
```

### 프로바이더별 설치

```bash
# OpenAI만 사용
pip install "knowledge-hub-cli[openai]"

# Ollama(로컬) + OpenAI
pip install "knowledge-hub-cli[ollama,openai]"

# 전체 프로바이더
pip install "knowledge-hub-cli[all]"
```

### 사전 조건

| 구성 요소 | 필수 여부 | 설명 |
|---|---|---|
| Python >= 3.10 | 필수 | 런타임 |
| API 키 | 선택 | OpenAI, Anthropic, Google 같은 hosted provider를 사용할 때 필요 |
| [Ollama](https://ollama.ai/) | 선택 | local-first / API-key-free 시작 경로에서 권장 |
| [Obsidian](https://obsidian.md/) | 선택 | vault 연동, note writeback, workbench 흐름에서 사용 |

`local` profile은 Ollama-only 구성으로 시작할 수 있으므로 API 키를 전제하지 않습니다.

## Configuration

### 기본 설정

권장 시작 경로는 `setup`입니다.

```bash
khub setup --profile local
khub doctor
```

`setup`은 설정 파일만 저장합니다. `local` profile 또는 Ollama 기반 embedding을 쓰는 `hybrid` profile에서 local runtime이 꺼져 있으면 `khub doctor`는 계속 `blocked/degraded`를 보여주며, 이때 권장 복구 순서는 아래와 같습니다.

```bash
ollama serve
ollama pull qwen3:14b
ollama pull nomic-embed-text
python -m knowledge_hub.interfaces.cli.main doctor
```

지원 프로필:
- `local`
  - translation/summarization: `ollama/qwen3:14b`
  - embedding: `ollama/nomic-embed-text`
  - parser: `auto`
- `hybrid`
  - translation/summarization: `openai/gpt-5-nano`
  - embedding: `ollama/nomic-embed-text` 우선, 불가하면 `openai/text-embedding-3-small`
  - parser: `auto`
- `custom`
  - `khub init`으로 세부 provider/model/API key를 조정

### 고급/커스텀 설정

```bash
khub init
```

번역/요약/임베딩에 사용할 AI 프로바이더, API 키, 저장 경로 등을 대화형으로 세부 설정합니다.
설정은 `~/.khub/config.yaml`에 저장됩니다.

### 설정 관리

```bash
# 전체 설정 보기
khub config list

# 개별 설정 변경
khub config set translation.provider openai
khub config set translation.model gpt-5-nano
khub config set summarization.provider ollama
khub config set summarization.model qwen3:14b
khub config set paper.summary.parser auto

# 사용 가능한 프로바이더 + 모델 확인
khub config providers --models

# 역할별 추천/설정 surface
khub provider recommend
khub provider setup --profile local
khub provider setup --profile balanced
khub provider setup --profile quality
khub provider setup --profile codex-mcp
```

### Custom / 기타 AI 모델 연결

OpenAI-compatible API를 제공하는 모델은 provider alias로 등록할 수 있습니다. DeepSeek, OpenRouter, Together, Fireworks, Mistral, vLLM, LM Studio 같은 서비스나 사내 게이트웨이를 같은 방식으로 연결합니다.

```bash
# preset 사용
khub provider add deepseek \
  --from-service deepseek \
  --use-for answer

# 완전 수동 등록
khub provider add qwen-api \
  --adapter openai-compatible \
  --base-url https://api.example.com/v1 \
  --api-key-env QWEN_API_KEY \
  --llm-model qwen-plus \
  --region cn

# 로컬 OpenAI-compatible 서버
khub provider add lmstudio \
  --adapter openai-compatible \
  --base-url http://localhost:1234/v1 \
  --no-api-key \
  --llm-model local-model \
  --local

# 역할에 적용
khub provider use answer deepseek/deepseek-chat
khub provider use translation qwen-api/qwen-plus
khub provider use embedding pplx-st/perplexity-ai/pplx-embed-v1-0.6b
```

API 키는 raw 값보다 환경변수 참조를 권장합니다.

```bash
export DEEPSEEK_API_KEY="<provider-api-key>"
khub provider key deepseek --env DEEPSEEK_API_KEY
```

운영 원칙:
- 임베딩은 대량 텍스트가 외부로 나가기 쉬우므로 기본 추천은 로컬(`ollama`, `pplx-st`)입니다.
- 답변, 요약, 정규화처럼 품질이 중요한 생성/판단 구간은 API 모델을 선택할 수 있습니다.
- 알 수 없는 external provider는 명시적으로 설정하고, 답변 생성에서는 `--allow-external` 정책을 확인하세요.
- Codex MCP 답변 backend는 `khub provider setup --profile codex-mcp` 뒤 `khub ask "질문" --answer-route codex --allow-external`로 확인합니다.

### 설정 파일 예시 (`~/.khub/config.yaml`)

```yaml
translation:
  provider: openai
  model: gpt-5-nano

summarization:
  provider: ollama
  model: qwen3:14b

embedding:
  provider: ollama
  model: nomic-embed-text

paper:
  summary:
    parser: auto

storage:
  papers_dir: ~/.khub/papers
  vector_db: ~/.khub/chroma_db
  sqlite: ~/.khub/knowledge.db

obsidian:
  enabled: true
  vault_path: /path/to/your/obsidian/vault
  write_backend: filesystem   # filesystem | cli-preferred
  cli_binary: obsidian

providers:
  openai:
    api_key_env: OPENAI_API_KEY
  ollama:
    base_url: http://localhost:11434
  pplx-local:
    base_url: http://localhost:8080
    timeout: 60
  deepseek:
    adapter: openai-compatible
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    supports:
      llm: true
      embedding: false
    models:
      llm:
        - deepseek-chat
        - deepseek-reasoner
      embedding: []
    default_llm_model: deepseek-chat
```

## AI Providers

| 프로바이더 | LLM | Embedding | 로컬 | 설치 |
|---|---|---|---|---|
| **OpenAI** | GPT-4o, 4o-mini, o1, o3, 4.1-nano | text-embedding-3-small/large | - | `[openai]` |
| **Anthropic** | Claude Opus 4, Sonnet 4, 3.5 Sonnet | - | - | `[anthropic]` |
| **Google** | Gemini 2.0 Flash/Pro/Lite | text-embedding-004 | - | `[google]` |
| **Ollama** | Qwen3, Llama4, Gemma3, DeepSeek-R1 등 | nomic-embed-text, bge-m3, snowflake | O | `[ollama]` |
| **OpenAI-Compatible** | DeepSeek, Groq, Together AI, Mistral 등 | 서비스별 지원 | - | 추가 의존성 없음 |
| **PPLX-Local (TEI)** | - | pplx-embed-v1/context-v1 (0.6b/4b) | O | 추가 의존성 없음 |
| **PPLX-ST (SentenceTransformers)** | - | pplx-embed-v1/context-v1 (0.6b/4b) | O | `[st]` |

용도별로 다른 프로바이더를 조합할 수 있습니다:
- 번역: OpenAI GPT-5-nano (읽기 쉬운 기본값)
- 요약: Ollama qwen3:14b (무료, 로컬)
- 임베딩: Ollama nomic-embed-text (무료, 로컬)

### Local Perplexity Embeddings (TEI)

```bash
# 1) TEI 서버 실행 (예: 0.6b)
docker run --rm -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id perplexity-ai/pplx-embed-v1-0.6b

# 2) 임베딩 프로바이더 전환
khub config set embedding.provider pplx-local
khub config set embedding.model perplexity-ai/pplx-embed-v1-0.6b
khub config set providers.pplx-local.base_url http://localhost:8080

# 3) 모델 차원 변경 시 전체 재인덱싱 권장
khub index --all
```

참고:
- Apple Silicon(M1/M2/M3/M4)에서 TEI Docker 이미지 태그에 따라 `linux/arm64` 미지원일 수 있습니다.
- 이 경우 아래 `PPLX-ST` 방식을 사용하세요.

### Local Perplexity Embeddings (SentenceTransformers)

```bash
# 1) 의존성 설치
pip install -e ".[st]"

# 2) 임베딩 프로바이더 전환
khub config set embedding.provider pplx-st
khub config set embedding.model perplexity-ai/pplx-embed-v1-0.6b
khub config set providers.pplx-st.batch_size 8
khub config set providers.pplx-st.device mps
khub config set providers.pplx-st.torch_num_threads 1
khub config set providers.pplx-st.disable_tokenizers_parallelism true
khub config set providers.pplx-st.max_chars_per_chunk 1000
khub config set providers.pplx-st.chunk_overlap_chars 200
khub config set providers.pplx-st.normalize_embeddings true
khub config set providers.pplx-st.trust_remote_code true

# 3) 모델 차원 변경 시 전체 재인덱싱 권장
khub index --all
```

속도 튜닝(Apple Silicon):
- `providers.pplx-st.device`를 `mps`로 설정
- `providers.pplx-st.batch_size`를 8 → 16/24로 점진적으로 증가
- 락/교착이 보이면 `providers.pplx-st.torch_num_threads=1` 유지
- 락/교착이 보이면 `providers.pplx-st.disable_tokenizers_parallelism=true` 유지
- 긴 문서는 `providers.pplx-st.max_chars_per_chunk`/`chunk_overlap_chars`로 자동 분할
- 메모리 압박 시 batch_size를 다시 낮추기

## Commands

### `khub discover` - direct paper discovery compatibility

기본 intake는 `khub add "topic" --type paper -n 3`입니다. `khub discover`는 judge, 연도/인용수 필터, 정렬 등 paper discovery 세부 옵션이 필요할 때 직접 호출하는 compatibility surface입니다.

```bash
# 기본 사용
khub discover "topic" -n 5

# 연도 필터 + 인용수 필터
khub discover "RAG retrieval augmented generation" --year 2024 --min-citations 10

# 인용수 기준 정렬
khub discover "transformer" --sort citationCount -n 10

# Obsidian 노트 생성 포함
khub discover "AI agent" --obsidian

# JSON 계약으로 inspectable 결과 확인
khub discover "AI agent" --judge --json
```

`paper judge`는 기본 retrieval 코어가 아니라, 논문 discovery 입력단에서만 쓰는 선택형 필터입니다. 공식 opt-in은 호출별 `--judge`와 MCP `discover_and_ingest(judge_enabled=true)`뿐이며, 전역 config로 기본 on/off를 바꾸는 제품 계약은 현재 두지 않습니다. `allow_external=false`가 기본이고, 외부 judge가 허용되지 않거나 LLM judge를 쓸 수 없으면 rule-only fallback으로만 동작합니다.

judge를 켜서 discovery를 실행하면 keep/skip 판단이 로컬 `~/.khub/paper_judge_events.jsonl`에 자동 기록됩니다. 사람이 나중에 판단을 뒤집고 싶으면 `khub paper feedback <paper_id> --label keep|skip`으로 수동 피드백을 남겨 future calibration 데이터로 사용할 수 있습니다.

### `khub paper` - paper reading and maintenance

```bash
# user-facing reading surface
khub paper summary --paper-id 2401.12345
khub paper evidence --paper-id 2401.12345
khub paper memory --paper-id 2401.12345
khub paper related --paper-id 2401.12345

# maintenance / ingestion helpers
khub paper list
khub paper info 2401.12345
khub paper download 2401.12345
khub paper translate 2401.12345
khub paper summarize 2401.12345
khub paper summarize-all --bad-only
khub paper sync-keywords
khub paper build-concepts
khub paper normalize-concepts
```

`summary|evidence|memory|related`는 현재 promoted reading surface입니다. `summarize`와 `summarize-all`은 artifact 생성/갱신 쪽 maintenance surface입니다.

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

### `khub search` / `khub ask` - grounded retrieval & answer

```bash
khub search "attention mechanism" -k 10  # grounded retrieval search
khub ask "RAG의 장단점을 설명해줘"         # retrieval-backed answer
```

`search`/`ask`는 shared retrieval pipeline 위에서 동작합니다. 현재 검색은 vector-only가 아니라 semantic / keyword / hybrid retrieval을 additive하게 사용할 수 있습니다.

`khub ask --source paper`의 기본 contract는 4개 family로 고정됩니다.

- `concept_explainer`: 개념 해석 -> 개념 설명 우선 -> 대표 논문 예시 연결 -> grounded explanation
- `paper_lookup`: 특정 논문 resolve -> paper/document memory -> single-paper answer
- `paper_compare`: 최소 2편 정렬 -> claim/evidence comparison
- `paper_discover`: broad paper retrieval -> shortlist + lightweight summary

기본 `ask`는 `paper_discover`에서 shortlist 중심으로 멈추고, 깊은 multi-paper synthesis는 `khub labs paper topic-synthesize ...`에 남깁니다. `--json` payload에는 `paperFamily`, `queryPlan`, `representativePaper`, `plannerFallback`, `familyRouteDiagnostics`가 포함돼 route와 retrieval 품질을 답변 생성기 상태와 분리해서 진단할 수 있습니다.

### `khub context` - read-only task context

```bash
khub context "how should I refactor the RAG flow?" --repo-path .
```

task-context는 persistent vault/paper/web evidence와 optional repo snippets를 함께 묶습니다. repo context는 inspectable하고 ephemeral하며, canonical knowledge store로 영속화되지 않습니다.
Legacy `khub agent context` remains as a hidden compatibility alias; new docs and quickstarts should use `khub context`.

### `khub index` - 벡터 인덱싱

```bash
khub index                               # 미인덱싱 논문 + 개념 인덱싱
khub index --all                         # 전체 재인덱싱
khub index --concepts-only               # 개념 노트만
khub index --vault-all                   # Obsidian 전체(.md) 청크 인덱싱
khub index --vault-all --vault-clear     # 기존 vault 벡터만 지우고 재인덱싱
khub index --vault-all --vault-clear --json  # 실행 리포트(JSON)
```

운영 권장 순서:
1. `khub --verbose index --all`
2. `khub --verbose index --vault-all --vault-clear`
3. 리포트 파일(`~/.khub/runs/index-<run_id>.json`)에서 실패 항목 확인 후 재실행

멈춤처럼 보일 때 점검:
1. 5분 관찰 후 CPU/메모리/캐시 증가 여부 확인
2. 동시 `khub search/ask/index` 프로세스 정리
3. `providers.pplx-st.batch_size`를 `8 -> 4 -> 2`로 낮춰 재시도

### `khub graph` - 지식 그래프

```bash
khub graph stats                         # 그래프 통계
khub graph show <id>                     # 노트 연결
khub graph isolated                      # 고립 노트
```

### `khub labs learn` - Personal Learning Coach (Labs)

```bash
khub labs learn map --topic "retrieval augmented generation" --source all --days 180 --top-k 12 --json --dry-run
khub labs learn assess-template --topic "retrieval augmented generation" --session-id "rag-001" --concept-count 6 --writeback
khub labs learn grade --topic "retrieval augmented generation" --session-id "rag-001" --json --writeback
khub labs learn next --topic "retrieval augmented generation" --session-id "rag-001" --json --writeback
khub labs learn run --topic "retrieval augmented generation" --session-id "rag-001" --auto-next --json --dry-run
```

핵심 규칙:
- 기본은 `--no-writeback`입니다. Obsidian 반영은 `--writeback` 명시 시에만 수행합니다.
- 세션 채점은 `target_trunk_ids`(frontmatter 고정값) 기준으로 계산됩니다.
- P0 원문은 결과 JSON/노트 본문에 저장하지 않고 `evidence_ptrs(path/heading/block_id/snippet_hash)`만 기록합니다.
- learning coach는 기본 제품 표면이 아니라 `khub labs ...` 아래의 실험/고급 워크플로우입니다.

상세 설계/스키마: `docs/learning-coach-mvp-v2.md`, `docs/schemas/learning-*.v1.json`

### `khub crawl` - Web Ingest (crawl4ai)

```bash
# URL 목록 -> crawl -> ko-note stage/apply 한 번에
khub crawl collect --url-file ./urls.txt --topic "ai-trends" --apply

# URL 직접 입력
khub crawl ingest --url "https://example.com/post/1" --url "https://example.com/post/2" --topic "rag" --index

# 파일 입력 + crawl4ai 강제
khub crawl ingest --url-file ./urls.txt --engine crawl4ai --topic "agent" --index

# 온톨로지 자동 추출 + LearningHub writeback
khub crawl ingest --url-file ./urls.txt --topic "rag" --extract-concepts --writeback

# 수집 후 trunk map 갱신(labs)
khub crawl ingest --url-file ./urls.txt --topic "rag" --learn-map
khub labs learn map --topic "rag" --source all --days 30 --top-k 12 --json --dry-run

# pending 큐 관리(labs)
khub labs crawl pending list --topic "rag" --limit 50
khub labs crawl pending apply --id 12
khub labs crawl pending reject --id 13
```

핵심 규칙:
- `khub crawl collect`는 `run_pipeline -> ko-note-generate -> optional ko-note-apply`를 묶은 편의 명령입니다.
- 기본은 `--stage-only`이며, 실제 Vault 반영은 `--apply`를 명시해야 합니다.

핵심 규칙:
- `--engine auto`이면 crawl4ai 설치 시 우선 사용, 미설치면 기본 크롤러로 자동 fallback.
- 수집 원문은 로컬(`storage.sqlite` 인접 `web_docs/`)에 저장되고, `notes(source_type=web)`로 적재됩니다.
- 기본 실행에서 웹 온톨로지 추출이 동작하며 고신뢰만 즉시 반영, 저신뢰는 `pending` 큐로 저장됩니다.
- `--writeback` 활성 시 Obsidian `LearningHub/<topic-slug>/03_Web_Sources.md`, `04_Web_Concepts.md`를 멱등 갱신합니다.
- `--index` 활성 시 vector DB까지 즉시 인덱싱됩니다.
- `pending`, `domain-policy`, `reference-sync` 같은 고급 운영 명령은 `khub labs crawl ...` 아래에 있습니다.
- 결과 JSON 스키마: `docs/schemas/crawl-ingest-result.v1.json` (`knowledge-hub.crawl.ingest.result.v1`)

백엔드 전략(A안):
- 현재 온톨로지는 `SQLite` + 선택적 `RDF/Turtle` 출력(`ontologyGraph`) 중심으로 운영한다.
- Neo4j Desktop은 현재는 연결하지 않고, 추후 커넥터 형태로 확장한다.

## MCP Overview (Cursor/Codex)

`khub-mcp`로 MCP 서버를 실행하면 Cursor/Codex가 Knowledge Hub 도구를 직접 호출할 수 있습니다.

주의:
- 기본 MCP surface는 retrieval-assistant-first product surface에 맞춰 노출됩니다.
- learning / advanced crawl / operator 도구는 기본 discovery에 항상 나타나는 surface가 아닙니다.
- 비핵심 도구는 `KHUB_MCP_PROFILE=labs` 또는 `KHUB_MCP_PROFILE=all`일 때만 listed/callable 상태가 됩니다.

Cursor 예시 설정:

```json
{
  "knowledge-hub": {
    "command": "khub-mcp",
    "env": {}
  }
}
```

대표 MCP 도구 묶음:
- default retrieval / answer: `search_knowledge`, `ask_knowledge`, `build_task_context`
- default paper / ingest: `discover_and_ingest`, `get_paper_detail`, `paper_lookup_and_summarize`
- default status / jobs: `get_hub_stats`, `mcp_job_status`, `mcp_job_list`, `mcp_job_cancel`
- labs/operator profile only: `run_agentic_query`, `run_paper_ingest_flow`, `crawl_web_ingest`, learning, ko-note, transform, ontology, and ops tools

비동기 호출 규칙:
- 장시간 작업은 보통 `queued|running|ok|blocked|failed` 상태를 반환합니다.
- `jobId`가 있으면 `mcp_job_status`, `mcp_job_list`, `mcp_job_cancel`로 추적합니다.
- 정책 차단 시 `verify.policyAllowed=false`, `artifact.classification=P0`, `artifact.jsonContent="[REDACTED_BY_POLICY]"` 패턴을 우선 확인합니다.

운영 참고:
- MCP payload 스키마: `docs/schemas/`
- CLI/MCP 표면 개요: `docs/guides/cli-commands.md`

## 명령별 필수 의존성

| 명령 | LLM 필요 | 임베딩 필요 | Obsidian 필요 | API 키 |
|---|---|---|---|---|
| `khub add` | 소스/옵션에 따라 | O (기본 `--index`) | 선택 | 프로바이더에 따라 |
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

## Troubleshooting

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: ollama` | ollama extra 미설치 | `pip install -e ".[ollama]"` 또는 다른 프로바이더로 변경 |
| `khub search` 실패 | 임베딩 프로바이더 미설정 | `khub init` → 임베딩 프로바이더 선택 |
| `OPENAI_API_KEY` 오류 | 환경변수 미설정 | `export OPENAI_API_KEY=<openai-api-key>` 또는 `.env` 파일 생성 |
| Obsidian 관련 명령 실패 | vault 경로 미설정 | `khub config set obsidian.vault_path /path/to/vault` |
| 인덱싱 0건 | 소스 미수집 | `khub add "topic" --type paper -n 3` 또는 `khub add "https://example.com/guide" --topic "topic"` 먼저 실행 |

## Runtime Mode: CLI + MCP Only

This repository is now fixed to terminal/agent operation only.
Web UI and UI API components were removed.

Supported user entry points:

- CLI: `khub`
- MCP server: `khub-mcp`

Internal compatibility entry points remain available during the layering migration:

- CLI: `python -m knowledge_hub.cli.main`
- MCP server: `python -m knowledge_hub.cli.main mcp`

Node helper scripts:

- `npm run cli` -> CLI launcher
- `npm run mcp` -> MCP server launcher
- `npm run foundry:project` -> Foundry project CLI (`foundry-core`)

## License

MIT
