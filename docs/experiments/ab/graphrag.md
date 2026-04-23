# GraphRAG Comparison

## 목적

graph/ontology 기반 retrieval boost에서 `knowledge-hub`와 `GraphRAG`를 비교합니다.

## 비교 대상

- concept-heavy query 성능
- multi-hop relation query 성능
- graph signal explainability
- 운영 복잡도 대비 품질 개선폭

## 사용 질문셋

- `q16` ~ `q20`

## 우리 쪽 기록 예시

```bash
khub search "ontology-backed search" --json > runs/ab/graphrag/knowledge_hub_q19.json
```

## 볼 것

- `hit@3`
- concept query 유효 적중률
- latency 증가폭
- graph signal이 concrete evidence를 덮는지 여부

## 흡수 후보

- graph candidate reduction
- graph-based rerank boost
- explainable concept path surfacing
- bounded helper implementation under `knowledge_hub/knowledge/graph_signals.py`

## 반영 위치

- `retrieval_fit`
- `knowledge/graph`
- `ontology signal`

## 결론

- repo clone과 quickstart/CLI review는 완료했다.
- 현재 repo snapshot은 `runs/ab/graphrag/repo`에 보관했다.
- 공식 quickstart는 Python `3.11-3.12`와 OpenAI/Azure API key를 전제로 하며, indexing 비용과 운영 복잡도가 높다고 명시하고 있다.
- GraphRAG local search의 핵심 가치는 `entity neighborhood`에서 후보를 줄인 뒤 text units, entities, relationships, community reports를 한 컨텍스트로 조립하는 점이다.
- GraphRAG global search의 핵심 가치는 dataset-wide theme question에 대해 community report를 map-reduce로 집계하는 점이다.
- 현재 `knowledge-hub`는 이미 bounded ontology/entity overlap boost와 cluster proximity boost를 가지고 있으므로, 지금 당장 부족한 건 `graph candidate reduction`과 `graph path/community explanation`이지 전체 GraphRAG 런타임이 아니다.
- 따라서 지금 머신/프로젝트 단계에서 full runtime parity A/B를 돌리기보다는, 작은 샘플로 `graph candidate reduction`과 `explainable rerank signal`만 비교하는 쪽이 더 현실적이다.
- 현재로서는 `GraphRAG 전체 도입`보다 `bounded graph signal 흡수`가 더 유망하고, global-search style community-report pipeline은 코어 제품 정의와 운영비를 고려하면 보류가 맞다.
- bounded helper는 `knowledge_hub/knowledge/graph_signals.py`에 독립 모듈로 추가했으며, runtime wiring은 아직 하지 않고 query analysis / candidate-hint diagnostics만 책임지도록 남겨뒀다.
