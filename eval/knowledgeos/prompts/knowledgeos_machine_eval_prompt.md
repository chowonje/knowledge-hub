# KnowledgeOS Machine Eval Prompt v1

이 문서는 `knowledgeos_machine_eval.csv`의 `pred_*` 컬럼을 GPT/Codex가 1차로 채울 때 사용하는 운영 프롬프트입니다.

목표:
- retrieval/answer 품질의 초벌 판정을 자동화
- `pred_*`만 채우고 사람이 `final_*`를 확정하게 하기
- `wrong era`, `should abstain` 같은 고비용 판정을 먼저 좁히기

중요 규칙:
- `pred_*` 컬럼만 채운다.
- 기존 결과 컬럼(`query`, `top1_title`, `answer_status`, `temporal_signals` 등)은 수정하지 않는다.
- 확신이 낮으면 보수적으로 판단한다.
- 최종 정답을 확정하려 들지 말고, `추천 판정`만 제공한다.

## 평가 기준

### `pred_label`
- `good`
  - top1이 질문 핵심을 직접 다룸
  - source와 answer style이 기대와 크게 어긋나지 않음
- `partial`
  - 관련은 있지만 질문 핵심이 비껴감
  - 너무 일반적이거나 일부만 맞음
- `bad`
  - 사실상 다른 주제
  - source가 엉뚱하거나 stale해서 질문 요구를 충족하지 못함

### `pred_wrong_era`
- `1`
  - `temporal_query=1`이고, top1이 명백히 오래된 버전/예전 설명/업데이트 전 정보에 묶여 있음
- `0`
  - temporal miss가 아니거나 판정 근거가 약함

### `pred_should_abstain`
- `1`
  - `no_result=1`
  - `answerable=false`
  - `insufficient_reasons`에 `missing_temporal_grounding` 또는 강한 부족 신호가 있음
  - top1이 있어도 질문 핵심에 답할 근거가 부족함
- `0`
  - top1이 다소 약해도 답변 시도 자체는 가능해 보임

### `pred_confidence`
- `0.9~1.0`: 매우 명확
- `0.7~0.89`: 대체로 명확
- `0.4~0.69`: 애매함, 사람이 꼭 확인해야 함
- `0.0~0.39`: 거의 추정 수준

### `pred_reason`
- 한 줄로 쓴다.
- 왜 `good/partial/bad`인지, 왜 `wrong era` 또는 `abstain`이라고 봤는지 핵심만 적는다.

## 권장 판정 순서

각 행마다 아래 순서로 본다.

1. `query`, `source`, `query_type`, `expected_primary_source`, `expected_answer_style`
2. `top1_title`, `top1_source_type`, `top1_excerpt`
3. `answer_status`, `answerable`, `insufficient_reasons`
4. `memory_route_applied`, `memory_prefilter_reason`, `temporal_route_applied`, `temporal_signals`
5. `pred_label`, `pred_wrong_era`, `pred_should_abstain`, `pred_confidence`, `pred_reason` 결정

## 추천 판단 휴리스틱

- `source=paper`인데 top1이 `vault` 일반 노트면 기본적으로 `partial` 또는 `bad` 쪽으로 본다.
- `query_type=temporal`인데 `temporal_route_applied=false`이고 `temporal_signals`도 약하면 `pred_should_abstain=1`을 강하게 고려한다.
- `memory_prefilter_reason=no_memory_hits` 자체만으로 `bad`를 주지는 않는다.
  - top1이 실제로 질문에 맞으면 `good/partial` 가능
- `no_result=1`이면 `pred_should_abstain=1`을 기본값으로 둔다.
- `answerable=false`이고 `insufficient_reasons`에 temporal/freshness 관련 이유가 있으면 `pred_should_abstain=1` 쪽으로 둔다.
- temporal 질의가 아닌 경우 `pred_wrong_era`는 대부분 `0`으로 둔다.

## 출력 형식

입력 CSV의 각 행에 대해 아래 5개 컬럼만 채운다.

- `pred_label`
- `pred_wrong_era`
- `pred_should_abstain`
- `pred_confidence`
- `pred_reason`

다른 컬럼은 그대로 둔다.

## Prompt Template

아래 템플릿을 그대로 복붙해 사용한다.

```text
You are labeling KnowledgeOS machine-eval rows.

Task:
- Read each CSV row.
- Fill only these columns:
  - pred_label
  - pred_wrong_era
  - pred_should_abstain
  - pred_confidence
  - pred_reason
- Do not modify any other columns.

Label rules:
- pred_label: good | partial | bad
- pred_wrong_era: 0 | 1
- pred_should_abstain: 0 | 1
- pred_confidence: 0.0 to 1.0
- pred_reason: one short sentence

Use these criteria:
- good: top1 directly answers the query
- partial: related but misses part of the query
- bad: wrong topic, wrong source, or clearly stale/misaligned
- wrong_era=1 only when temporal_query=1 and the top1 is clearly from the wrong time/update state
- should_abstain=1 when no_result=1, answerable=false, missing_temporal_grounding is present, or evidence is too weak

Be conservative.
If uncertain, lower pred_confidence and prefer partial over overconfident good.

Return the updated CSV rows with the original columns preserved.
```

## Human handoff

- 사람이 보는 파일은 `knowledgeos_human_review.csv`
- 사람은 `pred_*`를 참고만 하고 `final_*`를 확정한다
- `pred_confidence < 0.7`인 행은 우선 검토 대상으로 본다
