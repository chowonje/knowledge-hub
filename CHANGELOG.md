# Changelog

## [0.1.0] - 2026-02-19

### Added

- **논문 자동 발견 파이프라인** (`khub discover`) — Semantic Scholar + arXiv 검색, 병렬 다운로드, 자동 요약/번역/인덱싱
- **구조화된 심층 요약** — PDF 전문 기반 6개 섹션 분석 (기여/방법론/결과/한계/시사점)
- **플러그인 AI 프로바이더** — OpenAI, Anthropic, Google Gemini, Ollama(로컬) 자유 교체
- **벡터 검색 + RAG** (`khub search`, `khub ask`) — ChromaDB 기반 시맨틱 검색 및 질의응답
- **개별 논문 관리** (`khub paper`) — add, download, translate, summarize, embed 개별 실행
- **배치 작업** — `translate-all`, `summarize-all`, `embed-all`
- **지식 그래프** (`khub graph`) — 개념 추출, 동의어 정규화, 관계 근거 추적
- **Obsidian 연동** — vault에 논문 요약 노트 자동 생성, `[[개념]]` 위키링크 삽입
- **Concept Index** — 핵심 개념 자동 추출 및 개별 개념 노트 생성
- **Google NotebookLM 연동** (`khub notebook`)
- **MCP 서버** (`khub-mcp`) — Cursor IDE 통합 검색/RAG
- **인터랙티브 설정** (`khub init`) — 대화형 프로바이더/경로 설정
- **설정 관리** (`khub config`) — get/set/list/providers
- **입력 검증** — arXiv ID 형식 검증, CLI 에러 핸들링
- **API 재시도** — 지수 백오프, 부분 실패 격리
- **SQLite WAL 모드** — 동시성 안정성 강화
- **foundry-core** — connector-sdk dedupe 수정, 파일 기반 영속 store, 스냅샷 생성, relation evidence 추적
