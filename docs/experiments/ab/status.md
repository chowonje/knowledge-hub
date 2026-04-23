# A/B Status

외부 비교군별 현재 실행 상태, blocker, 다음 액션을 한 곳에 모읍니다.

| 비교군 | 상태 | 현재까지 한 일 | 주요 blocker | 다음 액션 | 현재 흡수 후보 |
|---|---|---|---|---|---|
| `Khoj` | 실행 완료(1차) | `knowledge-hub` baseline `q01~q05` 저장, repo/docs 검토, short-path venv에서 `khoj[local]` 설치, embedded DB + anonymous self-host 기동, `GET / -> 200` 확인, `scripts/ab/run_khoj_ab.py`로 22-file minimal corpus 업로드 후 `q01~q05` 결과 저장 | full-vault parity가 아니라 minimal shared corpus 기준이라 결과 해석 범위가 제한됨 | eval sheet에 반영하고 `query interpretation`/`ranking`만 흡수 후보로 좁히기 | bi-encoder + rerank 조합, related-note UX, 개인 문서 검색 흐름 |
| `PaperQA` | 실행 완료(1차) | 로컬 runner 추가, `PaperQA + Ollama(OpenAI-compatible)`로 `q06~q10` 생성 | `knowledge-hub ask` baseline이 local Ollama timeout으로 자주 실패 | `khub ask` timeout contract 확인 후 같은 질문셋 baseline 재수집 | paper-scoped answer formatting, citation assembly, evidence budget discipline |
| `Open Notebook` | 개념 비교 완료 / 추가 실행 보류 | repo clone, docs/quickstart/core concepts 검토, notebook mental model과 source-scope UX 후보 정리 | 이 머신엔 `docker compose`와 `docker-compose`가 모두 없음 | 지금 단계에선 추가 runtime A/B 없이 후보 아이디어만 유지 | notebook/sources/notes 모델, context level controls, ask vs chat 분리 |
| `GraphRAG` | 최소 판단 완료 / full runtime 보류 | repo clone, quickstart/CLI/requirements 검토, local/global search 설계와 현재 ontology/cluster rerank 구조 대조 | Python `3.11+` 요구, OpenAI/Azure 기반 indexing cost, graph indexing 운영비가 큼 | full runtime 대신 bounded `graph candidate reduction`/`explainable graph signal`만 후속 후보로 유지 | graph candidate reduction, graph/path explainability, bounded rerank boost |
| `Chroma Context-1 Data Gen` | 개념 비교 완료 / 적용 후보 분류 | repo README 구조, 요구 API key, synthetic data-gen 목적 검토 | 외부 API 의존(`Anthropic`, `Serper`, `Jina`, `OpenAI`, `Chroma`)이 강하고 도메인이 web/SEC/patents/email 중심이라 local-first 기본 runtime에는 바로 안 맞음 | runtime 통합 대신 `labs eval`용 synthetic multi-hop task generator 후보로 유지, 우리 eval schema로 변환 가능한지 후속 검토 | explore→verify→extend task synthesis, distractor generation, hard multi-hop eval set 확장 |

## 현재 하드콜

- 가장 먼저 비교를 정상화해야 하는 건 `knowledge-hub ask`의 local timeout 문제다.
- `PaperQA`는 이미 결과를 냈고, `Khoj`도 최소 공통 corpus 기준 1차 결과까지 나왔다. 다음 우선순위는 `PaperQA baseline 안정화`와 `Khoj 결과 평가표 반영`이다.
- `Open Notebook`은 이미 개념 추출 단계로는 충분하다. 지금은 runtime parity보다 `notebook/sources/notes` 모델과 context control 아이디어만 후보로 남기면 된다.
- `GraphRAG`는 full runtime parity를 목표로 잡기보다, 현재 retrieval에 부족한 `entity-neighborhood 기반 후보 축소`와 `설명 가능한 graph/community 근거 표면`만 후속 흡수 후보로 유지하는 편이 맞다.
- `Chroma Context-1 Data Gen`은 현재 프로젝트에 “데이터 생성형 eval 파이프라인”으로는 맞지만, 코어 retrieval/runtime dependency로 넣는 건 맞지 않다. 지금 단계에선 synthetic benchmark/task generation 아이디어만 흡수 후보로 유지하는 편이 맞다.

## 구현 반영 상태

- `PaperQA` 비교에서 나온 `paper-scoped narrowing`, `evidence budget`, `citation assembly`의 1차 버전은 기본 `ask` 경로에 additive diagnostics와 함께 반영됐다.
- `ask`는 이제 초기 로컬 answer generation이 timeout/예외로 실패해도 conservative evidence-grounded fallback payload를 반환하고, `answerGeneration` 진단 필드로 fallback 원인을 남긴다.
- `Khoj` 비교에서 나온 `query interpretation`/`vault ranking` 보정과 `related note` 제안은 기본 retrieval/search surface에 반영됐다.
- `Open Notebook`은 별도 runtime을 가져오지 않고 `full|summary|excluded` source context mode만 workbench/topic export에 흡수했다.
- `GraphRAG`는 full adoption 없이 `graph_query_signal` diagnostics와 bounded `graph candidate boost`만 search ranking에 흡수했고, hard candidate filtering이나 별도 graph indexing은 여전히 보류다.
