# 프로젝트 직접 관련 핵심 논문 메모

날짜: 2026-03-19

목적:
- 최근 AI 동향 메모 중에서 `knowledge-hub`와 직접 관련이 큰 논문 3편만 따로 정리한다.
- 기준은 다음 3축이다.
  - 메모리
  - 에이전트 시스템 구조
  - 평가/검증

선정 기준:
- 이 프로젝트의 코어인 `retrieval`, `document memory`, `agent reliability`, `evaluation`에 직접 영향을 주는가
- 단순히 “흥미로운 최신 논문”이 아니라, 현재 구조 리뉴얼에 참고할 수 있는가

## 1. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

- ID: `2504.19413`
- 링크: [arXiv](https://arxiv.org/abs/2504.19413)

핵심:
- 장기 대화/세션에서 전체 히스토리를 그대로 넣는 방식은 비용과 일관성 면에서 한계가 있다.
- 중요한 정보를 추출, 통합, 검색하는 memory-centric architecture가 더 실용적이다.
- graph memory를 추가하면 base memory보다 소폭 더 좋아질 수 있지만, 핵심은 먼저 `persistent structured memory` 자체다.

왜 중요한가:
- 이 논문은 memory를 “RAG 옵션”이 아니라 “에이전트 시스템의 인프라”로 본다.
- 현재 `knowledge-hub`의 `Document -> MemoryUnit -> RetrievalChunk` 방향과 가장 직접 맞닿아 있다.

이 프로젝트에 주는 시사점:
- `document_memory`를 labs에만 두지 말고, 장기적으로는 코어 retrieval 앞단 후보 축소 계층으로 올릴 가치가 있다.
- 단순 chunk store보다:
  - 중요한 정보 추출
  - memory consolidation
  - query-aware retrieval
  - human-readable persistence
  가 더 중요하다.

실제 반영 포인트:
- `MemoryUnit`에 더 강한 consolidation 규칙 추가
- 같은 문서/세션의 중복 정보 압축
- query-class-aware memory retrieval

## 2. AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges

- ID: `2505.10468`
- 링크: [arXiv](https://arxiv.org/abs/2505.10468)

핵심:
- “AI agent”와 “agentic AI”를 같은 것으로 보면 설계가 흐려진다.
- AI agent는 도구 통합, prompt engineering, reasoning 강화 기반의 task automation 쪽에 가깝다.
- Agentic AI는 multi-agent collaboration, dynamic task decomposition, persistent memory, coordinated autonomy까지 포함하는 더 큰 패러다임이다.

왜 중요한가:
- 지금 프로젝트가 어디까지 가려는지 경계를 세우는 데 유용하다.
- 이 프로젝트는 아직 `범용 agentic platform`이 아니라, `grounded retrieval + memory + task context` 쪽이 중심이다.

이 프로젝트에 주는 시사점:
- `knowledge-hub`는 당장 “범용 AI 에이전트 플랫폼”으로 확장하기보다,
  - grounded retrieval
  - persistent memory
  - inspectable task context
  를 더 세게 만드는 편이 맞다.
- 즉 지금은 `agentic AI` 전체를 노리기보다, `agent-supporting knowledge runtime`으로 정의하는 게 더 정확하다.

실제 반영 포인트:
- 코어와 labs 경계를 더 분명히 하기
- `agent context`와 `document memory`를 중심축으로 재정의
- multi-agent orchestration은 supporting/labs에 두기

참고:
- 이 논문은 이전 탐색에서 `khub explore paper 2505.10468`로 abstract를 확인했지만, 같은 ID를 다시 조회할 때 Semantic Scholar 쪽에서 일시적으로 찾지 못하는 상태가 있었다. 본 메모는 앞선 확인 결과를 바탕으로 쓴다.

## 3. Safety by Measurement: A Systematic Literature Review of AI Safety Evaluation Methods

- ID: `2505.05541`
- 링크: [arXiv](https://arxiv.org/abs/2505.05541)

핵심:
- 안전 평가는 단순 benchmark score가 아니라,
  - 무엇을 측정하는가
  - 어떻게 측정하는가
  - 그 결과를 어떤 프레임워크에 연결하는가
  의 문제다.
- capability, propensity, control 같은 구분이 중요하다.

왜 중요한가:
- 이 프로젝트는 이미 `judge`, `answerVerification`, `retrieval diagnostics`, `quality-aware retrieval` 쪽으로 가고 있다.
- 이 논문은 그 방향이 맞다는 걸 정리된 프레임으로 뒷받침한다.

이 프로젝트에 주는 시사점:
- 평가 결과는 숨겨진 내부 점수보다 operator-facing artifact로 남겨야 한다.
- `khub ask`, `document_memory`, `paper judge` 모두 평가 루프와 연결되어야 한다.
- 특히 `document_memory_eval_template.csv` 같은 수동 평가 루프는 좋은 시작점이다.

실제 반영 포인트:
- `good / partial / bad` 같은 수동 평가 라벨을 축적
- query-type별 failure taxonomy 유지
- 나중에 memory retrieval 승격 여부를 평가 지표로 결정

## 보조 참고 논문

### Inference-Time Computations for LLM Reasoning and Planning

- ID: `2502.12521`
- 링크: [arXiv](https://arxiv.org/abs/2502.12521)

짧은 해석:
- reasoning을 잘하려면 무조건 더 많은 test-time compute를 주는 게 아니라, task 유형별로 적절한 inference-time 기법을 써야 한다.
- 이건 `ask`와 `document_memory search`를 query-type-aware로 만드는 근거가 된다.

왜 보조냐:
- 중요하긴 하지만, 지금 프로젝트 리뉴얼의 직접 중심은 `memory`와 `evaluation` 쪽이 더 강하다.

## 현재 프로젝트 기준 최종 판단

가장 직접적인 중심축:
1. `Mem0`
2. `Safety by Measurement`
3. `AI Agents vs. Agentic AI`

즉 현재 리뉴얼 방향을 한 줄로 압축하면:

`knowledge-hub는 범용 agent platform보다, persistent document memory와 grounded evaluation을 갖춘 retrieval-centered agent support system으로 가는 것이 더 자연스럽다.`

## 다음 액션

1. `document_memory`를 계속 labs에서 평가
2. `judge / answer verification / memory eval`을 하나의 평가 관점으로 정리
3. 코어 정의를 `retrieval + memory + evaluation` 중심으로 다시 쓰기
