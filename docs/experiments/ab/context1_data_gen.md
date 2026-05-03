# Chroma Context-1 Data Gen Comparison

## 링크

- GitHub: [chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen)

## 목적

`knowledge-hub`의 수동/반수동 eval 질문셋과 비교해서, synthetic multi-hop retrieval task generation 파이프라인을 흡수할 가치가 있는지 판단한다.

## 무엇을 하는 프로젝트인가

- Chroma의 `Context-1` 기술 보고서 기반 synthetic data generation pipeline이다.
- README 기준으로 multi-hop search task를 `explore -> verify -> extend` 패턴으로 생성한다.
- 현재 공개 도메인은 `web`, `SEC`, `patents`, `email` 이다.

## 현재 프로젝트에 적용 가능한 부분

- `labs eval` 쪽에서 hard query / multi-hop query / distractor-heavy query를 자동 생성하는 보조 파이프라인
- retrieval regression set 확장
- memory-router / paper-memory / document-memory 평가용 synthetic case generation
- `explore -> verify -> extend` 식의 단계형 question synthesis 구조

## 바로 적용하기 어려운 부분

- README 기준 외부 의존이 강하다:
  - `ANTHROPIC_API_KEY`
  - `SERPER_API_KEY`
  - `JINA_API_KEY`
  - `OPENAI_API_KEY`
  - `CHROMA_API_KEY`
  - `CHROMA_DATABASE`
  - 일부 도메인은 `BASETEN_API_KEY`
- 현재 도메인 중심이 `web/SEC/patents/email` 이라서 `knowledge-hub`의 핵심인 `vault/paper/local-first memory`와는 바로 맞물리지 않는다.
- 따라서 코어 runtime dependency로 가져오면 local-first / policy-first 방향과 충돌할 가능성이 높다.

## 판단

- 적용 가능: `예`
- 단, 적용 위치는 `core runtime` 이 아니라 `labs-only eval/data generation` 이 맞다.
- 현재 프로젝트에 가장 맞는 흡수 방식은:
  - synthetic retrieval benchmark generator
  - hard negative / distractor generation
  - multi-hop question set expansion
  - 우리 eval CSV/JSON schema로 변환하는 import bridge

## 비적용 판단

다음 항목은 현재 단계에서 그대로 들여오는 것이 맞지 않다.

- 외부 API를 전제로 한 전체 pipeline runtime
- Chroma indexing 전제를 코어 저장소 구조로 직접 편입하는 것
- 도메인 구조 자체를 `knowledge-hub` 기본 질의응답 runtime으로 복사하는 것

## 다음 액션

1. `docs/experiments/ab/query_set_v1.md` 와 별도로 synthetic hard-set 생성 후보로 유지
2. `khub labs eval` 입력 포맷으로 변환 가능한 최소 adapter schema 설계
3. `paper` 또는 `vault` 전용으로 줄인 local synthetic generation이 가능한지 별도 검토

## 결론

`context-1-data-gen`은 지금 프로젝트에 "바로 붙이는 검색 엔진"이나 "메모리 시스템"이 아니라, 평가를 더 어렵고 현실적으로 만드는 `synthetic task generator` 후보로 보는 게 맞다. 흡수 가치 자체는 있지만, 위치는 runtime이 아니라 eval/labs 쪽이다.
