# KnowledgeOS Multi-Model Eval Rubric v0

이 문서는 Claude, Gemini, Grok, GPT Pro 같은 외부 LLM에게 `machine_eval.csv`를 1차 판정시키기 위한 공통 rubric입니다.

목표:

- retrieval 품질 초벌 판정
- temporal 오류 후보 식별
- abstain 필요 케이스 식별
- 사람 최종 검토 전에 `pred_*` 컬럼을 보수적으로 채우기

입력:

- KnowledgeOS `machine eval` CSV
- 각 행에는 최소한 아래 정보가 들어 있다고 가정한다.
  - `query`
  - `source`
  - `query_type`
  - `expected_primary_source`
  - `expected_answer_style`
  - `top1_title`
  - `top1_source_type`
  - `top1_excerpt`
  - `answer_status`
  - `temporal_query`
  - `temporal_signals`
  - `insufficient_reasons`

출력:

- 기존 컬럼은 유지
- 아래 `pred_*` 5개 컬럼만 채운 CSV 전체
  - `pred_label`
  - `pred_wrong_era`
  - `pred_should_abstain`
  - `pred_confidence`
  - `pred_reason`

핵심 규칙:

- `label`, `wrong_era`, `answerable`, `no_result`는 참고만 하고 그대로 복사하지 않는다.
- `pred_*` 외 다른 컬럼은 절대 수정하지 않는다.
- 불확실하면 보수적으로 판단하되, 기계적으로 전부 `bad/abstain`으로 몰지 않는다.
- temporal 질문은 시점 불일치를 별도로 본다.
- 최종 정답을 확정하지 않는다. 이 출력은 사람 검토용 초벌이다.

## 판단 기준

### `pred_label`

- `good`
  - top1이 질문 핵심을 직접 다룬다.
  - source/type 기대가 크게 어긋나지 않는다.
- `partial`
  - 관련은 있지만 질문 핵심 일부만 맞는다.
  - 너무 일반적이거나, 질문의 특정 조건을 놓친다.
- `bad`
  - 사실상 다른 주제다.
  - source가 크게 어긋난다.
  - 근거가 없거나 지나치게 약하다.

### `pred_wrong_era`

- `1`
  - `temporal_query=1`이고, top1이 명백히 다른 시점의 정보다.
  - 예: 최신/업데이트 질문에 예전 버전 설명이 top1로 잡힘
- `0`
  - temporal miss가 아니거나 판정 근거가 약함

### `pred_should_abstain`

- `1`
  - `top1_title`, `top1_excerpt`가 사실상 비어 있음
  - `answer_status=no_result`
  - temporal 질문인데 temporal grounding이 없음
  - 질문 핵심에 답할 근거가 너무 약함
- `0`
  - top1이 완전하지 않더라도 답변 시도는 가능함

### `pred_confidence`

- `0.9 ~ 1.0`
  - 매우 명확
- `0.7 ~ 0.89`
  - 대체로 명확
- `0.4 ~ 0.69`
  - 애매함
- `0.0 ~ 0.39`
  - 거의 추정 수준

### `pred_reason`

- 한 줄로 쓴다.
- 기존 label을 반복하지 말고 실제 판단 근거를 적는다.
- 예:
  - `top1 excerpt directly explains retrieval failure causes`
  - `related paper topic but does not answer the asked mechanism`
  - `no retrieved evidence`
  - `temporal query with no grounded time signal`

## 검토 우선순위

사람이 먼저 봐야 하는 행:

- `pred_label=partial`
- `pred_wrong_era=1`
- `pred_confidence < 0.7`
- `pred_should_abstain=1`인데 top1 excerpt가 존재하는 행

## 운영 원칙

- 외부 LLM 판정은 추천안이다.
- 최종 truth는 `human review`의 `final_*`다.
- gate와 회귀 평가는 사람 확정본만 사용한다.
