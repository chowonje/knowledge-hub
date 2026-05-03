# KnowledgeOS External Evaluator Runbook v0

이 문서는 Claude, Gemini, Grok, GPT Pro에게 같은 machine-eval CSV를 보내 1차 판정을 받는 운영 순서를 정리합니다.

## 입력 파일

- candidate:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval.csv`
- baseline:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline.csv`

## 모델별 프롬프트

- GPT Pro:
  - `prompts/knowledgeos_external_eval_gpt_pro_v0.md`
- Claude:
  - `prompts/knowledgeos_external_eval_claude_v0.md`
- Gemini:
  - `prompts/knowledgeos_external_eval_gemini_v0.md`
- Grok:
  - `prompts/knowledgeos_external_eval_grok_v0.md`

## 받으면 좋은 출력 파일 이름

- candidate
  - `knowledgeos_machine_eval_gpt_pro_judged_v0.csv`
  - `knowledgeos_machine_eval_claude_judged_v0.csv`
  - `knowledgeos_machine_eval_gemini_judged_v0.csv`
  - `knowledgeos_machine_eval_grok_judged_v0.csv`
- baseline
  - `knowledgeos_machine_eval_baseline_gpt_pro_judged_v0.csv`
  - `knowledgeos_machine_eval_baseline_claude_judged_v0.csv`
  - `knowledgeos_machine_eval_baseline_gemini_judged_v0.csv`
  - `knowledgeos_machine_eval_baseline_grok_judged_v0.csv`

## 운영 원칙

- 각 외부 모델은 `pred_*`만 채운다.
- 모델 간 판정이 다르면 그 행은 사람 우선 검토 대상이다.
- `pred_label=partial`, `pred_wrong_era=1`, `pred_confidence < 0.7`도 우선 검토 대상이다.
- 최종 truth는 사람의 `final_*`다.

## 권장 순서

1. candidate를 4개 외부 모델에 보낸다.
2. baseline을 4개 외부 모델에 보낸다.
3. 결과 CSV를 `eval/knowledgeos/runs/` 아래에 저장한다.
4. 사람 검토용 우선순위 파일을 만든다.
5. `human review`에서 `final_*`를 확정한다.
