# KnowledgeOS External Eval Prompt for Claude v0

업로드할 파일:

- candidate: `eval/knowledgeos/runs/knowledgeos_machine_eval.csv`
- baseline: `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline.csv`

참고 rubric:

- `eval/knowledgeos/rubrics/knowledgeos_multi_model_eval_rubric_v0.md`

프롬프트:

```text
You are helping evaluate KnowledgeOS retrieval results.

I will provide a CSV file. For each row, fill only these columns:
- pred_label
- pred_wrong_era
- pred_should_abstain
- pred_confidence
- pred_reason

Rules:
- Keep all existing columns exactly as they are.
- Do not add commentary outside the CSV.
- Return the full CSV with only pred_* fields updated.

Evaluation criteria:
- good: the retrieved top1 directly answers the query
- partial: related, but indirect, incomplete, or only partially aligned
- bad: wrong topic, weak evidence, or no meaningful support
- pred_wrong_era=1 only if a temporal query clearly points to the wrong time/update state
- pred_should_abstain=1 if evidence is missing, too weak, or lacks temporal grounding

Important:
- Use top1_title, top1_excerpt, answer_status, temporal_query, temporal_signals, and insufficient_reasons as your primary signals.
- Treat label, wrong_era, answerable, and no_result only as hints. Do not copy them automatically.
- Be conservative, but avoid making every row bad or abstain.
```

권장 결과 파일 이름:

- `knowledgeos_machine_eval_claude_judged_v0.csv`
- `knowledgeos_machine_eval_baseline_claude_judged_v0.csv`
