# KnowledgeOS Multi-Model Priority Review v0

이 문서는 여러 외부 평가자 결과를 합쳐 사람이 먼저 봐야 할 행만 정리하는 review queue 스키마를 설명합니다.

입력:

- base machine eval
- 1개 이상 judged CSV

출력:

- 사람 검토 우선순위가 높은 순으로 정렬된 CSV
- 각 evaluator의 `pred_*`를 같은 행에 나란히 둠
- 사람이 `final_*`를 채울 수 있게 함

## 우선순위 규칙

점수가 올라가는 조건:

- evaluator 간 `pred_label` 불일치
- evaluator 중 하나라도 `pred_wrong_era=1`
- evaluator 간 `pred_should_abstain` 불일치
- evaluator 중 하나라도 `pred_label=partial`
- evaluator 중 하나라도 `pred_confidence < 0.7`
- top1 evidence가 있는데 evaluator 중 하나라도 `pred_should_abstain=1`
- `temporal_query=1`
- `no_result=1`

## 생성 스크립트

```bash
python scripts/build_multi_model_priority_review.py \
  --base eval/knowledgeos/runs/knowledgeos_machine_eval.csv \
  --evaluator claude=eval/knowledgeos/runs/knowledgeos_machine_eval_claude_judged_v0.csv \
  --evaluator grok=eval/knowledgeos/runs/knowledgeos_machine_eval_grok_judged_v0.csv \
  --evaluator gemini=eval/knowledgeos/runs/knowledgeos_machine_eval_gemini_judged_v0.csv \
  --out eval/knowledgeos/review/knowledgeos_multi_model_priority_review_v0.csv
```

현재 일부 evaluator만 있어도 생성 가능하다.

예:

```bash
python scripts/build_multi_model_priority_review.py \
  --base eval/knowledgeos/runs/knowledgeos_machine_eval.csv \
  --evaluator claude=eval/knowledgeos/runs/knowledgeos_machine_eval_claude_judged_v0.csv \
  --out eval/knowledgeos/review/knowledgeos_multi_model_priority_review_v0.csv
```
