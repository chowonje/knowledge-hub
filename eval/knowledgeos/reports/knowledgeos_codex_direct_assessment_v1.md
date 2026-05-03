# KnowledgeOS Codex Direct Assessment v1

기준 시점: 2026-03-24

평가 대상:

- current candidate run:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_v1.csv`
- external judged runs:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_claude_judged_v0.csv`
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_gpt_judged_v1.csv`
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline_gpt_judged_v1.csv`

## 내 판단

### 1. 프로젝트 자체는 파기 대상이 아니다

지금 드러난 문제는 주로 아래다.

- `paper` 질의 precision 저하
- `vault/web/all`의 높은 `no_result`
- evaluator 품질 편차
- temporal grounding 부족

이건 구조 가설이 틀렸다는 신호가 아니라, **retrieval coverage / routing / evaluation quality**가 아직 덜 정제됐다는 신호다.

### 2. 현재 evaluator 신뢰도 순위

1. `Claude`
   - 가장 균형적이다.
   - `partial`, `wrong_era`, `abstain`이 최소한 구분된다.
2. `GPT judged v1`
   - 보조 신호로는 쓸 수 있다.
   - Claude보다 조금 더 보수적이다.
3. `Grok`
   - 지나치게 보수적이지만 보조 신호로는 가능하다.
4. `Gemini`
   - `wrong_era`를 과하게 잡는 경향이 있어 참고 신호로만 써야 한다.
5. `baseline GPT judged v1`
   - 사실상 evaluator로서 구분력이 부족하다.
   - `bad 100 / abstain 100`은 디버깅 신호가 거의 없다.

### 3. 현재 평가 방식의 기술적 한계

외부 웹 LLM 평가는 대부분 `machine_eval.csv`만 보고 판단한다.
이 방식은 다음 한계가 있다.

- excerpt가 짧으면 직접 근거 판단이 어렵다
- top1 source mismatch가 왜 발생했는지 코드를 못 본다
- temporal grounding 부족과 wrong-era를 쉽게 혼동한다
- refusal/meta-response를 retrieval failure와 혼동하기 쉽다

즉 외부 웹 LLM 평가는 유용하지만, **repo/file-aware evaluator가 아니면 한계가 뚜렷하다**.

### 4. Cursor evaluator를 쓰는 게 맞는 이유

Cursor는 repo 파일을 직접 열 수 있으므로:

- CSV excerpt가 애매할 때 관련 코드/문서를 바로 확인할 수 있다
- 질문이 repo 구조 설명인지 실제 source retrieval 질문인지 구분하기 쉽다
- wrong-era와 no-grounding을 더 정확히 분리할 수 있다

따라서 현재 단계에서는:

- `Claude`를 primary external evaluator
- `Cursor`를 stronger file-aware evaluator
- `human review`를 final truth

구조가 가장 맞다.

### 5. 지금 기준의 실제 상태 평가

냉정하게 보면:

- `paper precision`은 개선 여지가 분명하고, 이미 일부 코드 수정 포인트가 드러났다
- `vault/web/all`은 ranking보다 coverage와 source support 문제가 더 크다
- evaluator는 아직 불안정해서 human review 없이 자동 승격하면 안 된다

즉 상태는:

- `망한 프로젝트`: 아님
- `바로 product-ready`: 아님
- `실패 유형이 보여서 고칠 수 있는 시스템`: 맞음

## 추천 운영

1. candidate와 baseline을 Cursor로 다시 `pred_*` 평가
2. Claude와 Cursor 불일치 행을 우선 human review
3. `final_*` 기준으로 failure taxonomy 생성
4. 그 taxonomy 기준으로 다음 패치를 결정

## 한 줄 결론

현재 KnowledgeOS는 파기 대상이 아니라, **평가자 노이즈를 줄이고 retrieval coverage/routing을 계속 정제해야 하는 단계**다.
