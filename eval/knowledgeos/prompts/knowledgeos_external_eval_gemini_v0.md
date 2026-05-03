# KnowledgeOS External Eval Prompt for Gemini v0

업로드할 파일:

- candidate: `eval/knowledgeos/runs/knowledgeos_machine_eval.csv`
- baseline: `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline.csv`

참고 rubric:

- `eval/knowledgeos/rubrics/knowledgeos_multi_model_eval_rubric_v0.md`

프롬프트:

```text
Please evaluate the uploaded KnowledgeOS machine-eval CSV.

For each row, fill only:
- pred_label
- pred_wrong_era
- pred_should_abstain
- pred_confidence
- pred_reason

Preserve all original columns and row order.
Return the complete CSV.

Use these rules:
- good = top1 directly answers the query
- partial = related but incomplete / indirect / only partly aligned
- bad = wrong topic, weak support, or no usable evidence
- pred_wrong_era = 1 only if a temporal query is clearly answered from the wrong time or update state
- pred_should_abstain = 1 if evidence is missing, too weak, or temporally ungrounded

Use top1_title, top1_excerpt, answer_status, temporal_query, temporal_signals, and insufficient_reasons as the main evidence.
Do not mechanically copy label, wrong_era, answerable, or no_result.
Be conservative, but do not force every row into bad or abstain.
Keep pred_reason to one short sentence.
```

권장 결과 파일 이름:

- `knowledgeos_machine_eval_gemini_judged_v0.csv`
- `knowledgeos_machine_eval_baseline_gemini_judged_v0.csv`
