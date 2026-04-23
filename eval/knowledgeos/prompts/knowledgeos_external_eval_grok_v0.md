# KnowledgeOS External Eval Prompt for Grok v0

업로드할 파일:

- candidate: `eval/knowledgeos/runs/knowledgeos_machine_eval.csv`
- baseline: `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline.csv`

참고 rubric:

- `eval/knowledgeos/rubrics/knowledgeos_multi_model_eval_rubric_v0.md`

프롬프트:

```text
You are labeling a KnowledgeOS machine-evaluation CSV.

Update only these five fields for every row:
- pred_label
- pred_wrong_era
- pred_should_abstain
- pred_confidence
- pred_reason

Do not change any other field.
Return the full CSV with the same row order.

Labeling guidance:
- good = directly answers the query
- partial = related but not directly sufficient
- bad = irrelevant, unsupported, or clearly mismatched
- pred_wrong_era = 1 only for clear temporal mismatch on temporal queries
- pred_should_abstain = 1 when evidence is missing, too weak, or not temporally grounded

Use top1_title, top1_excerpt, answer_status, temporal_query, temporal_signals, and insufficient_reasons as primary evidence.
Treat label, wrong_era, answerable, and no_result only as hints.
Be conservative, but avoid flattening all rows into bad/abstain.
pred_reason should be one short sentence.
```

권장 결과 파일 이름:

- `knowledgeos_machine_eval_grok_judged_v0.csv`
- `knowledgeos_machine_eval_baseline_grok_judged_v0.csv`
