# KnowledgeOS External Eval Prompt for GPT Pro v0

업로드할 파일:

- candidate: `eval/knowledgeos/runs/knowledgeos_machine_eval.csv`
- baseline: `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline.csv`

참고 rubric:

- `eval/knowledgeos/rubrics/knowledgeos_multi_model_eval_rubric_v0.md`

프롬프트:

```text
You are a KnowledgeOS evaluation assistant.

I will upload a CSV containing machine-eval rows from a retrieval-and-answer pipeline.
Your task is to read every row and fill only these five columns:

- pred_label
- pred_wrong_era
- pred_should_abstain
- pred_confidence
- pred_reason

Do not modify any other columns.
Do not reorder rows.
Return the full CSV.

Use this rubric:

- pred_label = good | partial | bad
- pred_wrong_era = 0 | 1
- pred_should_abstain = 0 | 1
- pred_confidence = 0.0 to 1.0
- pred_reason = one short sentence

Judging rules:
- Use top1_title, top1_excerpt, answer_status, temporal_query, temporal_signals, insufficient_reasons as the main evidence.
- Treat label, wrong_era, answerable, and no_result as hints only. Do not mechanically copy them.
- If the top1 directly answers the query, use good.
- If it is related but incomplete or indirect, use partial.
- If it is irrelevant, unsupported, or clearly misaligned, use bad.
- Use pred_wrong_era=1 only when a temporal query is clearly answered with the wrong time/update state.
- Use pred_should_abstain=1 when evidence is missing, too weak, or temporally ungrounded.
- Be conservative, but do not collapse everything into bad/abstain.

Return the updated CSV with all original columns preserved.
```

권장 요청 방식:

- candidate와 baseline을 각각 별도 세션으로 돌린다.
- CSV가 크면 25행씩 나눠도 되지만, 가능하면 파일 업로드로 전체를 유지한다.
- 결과 파일 이름 예:
  - `knowledgeos_machine_eval_gpt_pro_judged_v0.csv`
  - `knowledgeos_machine_eval_baseline_gpt_pro_judged_v0.csv`
