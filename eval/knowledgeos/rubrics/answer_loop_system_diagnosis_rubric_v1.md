# Answer-Loop System Diagnosis Rubric v1

이 문서는 `answer-loop` 결과를 답변 문장력 기준이 아니라 시스템 레이어 기준으로 판단하기 위한 rubric이다.

목표:
- 같은 질문에 대한 여러 답변을 비교할 때
- 어떤 문제가 모델 탓인지, retrieval 탓인지, assembly 탓인지 분리한다
- 다음 코드 수정 우선순위를 정한다

## 1. Query Understanding

좋은 상태:
- 질문이 설명형인지 비교형인지 바로 맞춘다
- 비교 질문이면 공통 축을 세운다

나쁜 상태:
- 비교 질문인데 답변이 `A 소개 + B 소개`로 흐른다
- 사용자가 묻지 않은 축으로 답변이 새어 나간다

주된 귀속:
- `ASSEMBLY`
- 경우에 따라 `PROMPT`

## 2. Retrieval Quality

좋은 상태:
- 대표 논문만 모으는 게 아니라 질문 축에 맞는 근거를 모은다
- 비교 질문이면 direct compare 또는 axis-specific evidence가 있다

나쁜 상태:
- 대표 사례는 있지만 비교 근거는 없다
- task-specific example이 core difference evidence를 덮는다
- source balance가 한쪽으로 기운다

주된 귀속:
- `RETRIEVAL`

## 3. Evidence Grounding

좋은 상태:
- 답변이 packet 안의 근거 범위를 넘지 않는다
- 근거가 약한 축은 약하다고 표시한다

나쁜 상태:
- indirect example을 direct conclusion처럼 말한다
- packet에 없는 일반 지식을 섞는다

주된 귀속:
- `ASSEMBLY`
- 경우에 따라 `MODEL`

## 4. Answer Assembly

좋은 상태:
- 답변 구조가 질문 구조를 따른다
- concept / comparison / example / limitation이 분리된다

나쁜 상태:
- 논문 나열 후 마지막에 대충 요약한다
- 핵심 차이는 약하고 사례가 본문을 덮는다

주된 귀속:
- `ASSEMBLY`
- 경우에 따라 `PROMPT`

## 5. Reasoning Quality

좋은 상태:
- 왜 그런 차이가 나는지 조건부로 설명한다
- 답변이 사례 요약을 넘어서 비교 논리를 가진다

나쁜 상태:
- 사례를 말하지만 인과 설명이 약하다
- “잘 맞는 상황”이 구조적 이유 없이 붙는다

주된 귀속:
- `MODEL`
- `ASSEMBLY`

## 6. Overclaim Risk

좋은 상태:
- 없는 benchmark나 일반 우위를 만들지 않는다
- 약한 packet이면 abstain/caution을 유지한다

나쁜 상태:
- detection 사례만으로 CNN 일반 우위를 말한다
- 일부 self-supervised evidence만으로 ViT 일반 우위를 말한다

주된 귀속:
- `MODEL`
- `ASSEMBLY`

## 우선순위 규칙

1. 먼저 `RETRIEVAL`을 본다
2. 그 다음 `ASSEMBLY`를 본다
3. 마지막으로 `MODEL`과 `PROMPT`를 본다

이유:
- retrieval이 비교축에 맞지 않으면 assembly가 잘해도 한계가 크다
- retrieval이 충분한데도 답이 나쁘면 그때 assembly/prompt 문제가 더 크다
