# KnowledgeOS Cursor Eval Prompt v1

이 프롬프트는 **Cursor 모델처럼 repo 파일에 직접 접근할 수 있는 evaluator**를 위한 것이다.

입력 파일:

- candidate:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_v1.csv`
- baseline:
  - `eval/knowledgeos/runs/knowledgeos_machine_eval_baseline_v1.csv`

관련 참고 파일:

- rubric:
  - `eval/knowledgeos/rubrics/knowledgeos_multi_model_eval_rubric_v0.md`
- project state:
  - `docs/PROJECT_STATE.md`
- retrieval code:
  - `knowledge_hub/ai/retrieval_pipeline.py`
  - `knowledge_hub/ai/retrieval_fit.py`
  - `knowledge_hub/ai/evidence_assembly.py`
  - `knowledge_hub/ai/memory_prefilter.py`

## Cursor에게 줄 프롬프트

```text
You are evaluating KnowledgeOS retrieval results with local repository access.

You will receive a machine-eval CSV. For each row, fill only these columns:
- pred_label
- pred_wrong_era
- pred_should_abstain
- pred_confidence
- pred_reason

Rules:
- Keep every existing column exactly as it is.
- Do not add any extra prose outside the CSV.
- Return the full CSV with only pred_* fields updated.

Important differences from web-only evaluation:
- You may inspect the local repository when the CSV excerpt is ambiguous.
- Use file access selectively, not exhaustively.
- Prefer direct evidence from the CSV first, then repo files if needed to resolve ambiguity.

Primary decision inputs per row:
- query
- source
- query_type
- expected_primary_source
- top1_title
- top1_source_type
- top1_excerpt
- answer_status
- memory_prefilter_reason
- temporal_signals
- insufficient_reasons

Secondary inputs when needed:
- docs/PROJECT_STATE.md
- retrieval/evidence code under knowledge_hub/ai/
- any repo files clearly implicated by the query or top1 title

Do not blindly copy:
- label
- wrong_era
- answerable
- no_result

You must judge them independently.

Evaluation criteria:

1. pred_label
- good:
  - top1 directly answers the query
  - expected source and actual source are aligned
  - evidence is substantive, not just a refusal/meta response
- partial:
  - related, but indirect, incomplete, weakly grounded, or source-misaligned
- bad:
  - wrong topic, no meaningful evidence, or severe mismatch

2. pred_wrong_era
- Set to 1 only when a temporal query clearly retrieves the wrong time/version/update state.
- Do not use 1 just because temporal grounding is missing.
- Missing temporal grounding should usually affect pred_should_abstain instead.

3. pred_should_abstain
- Set to 1 when evidence is absent, non-substantive, too weak, or temporal grounding is required but missing.
- Set to 0 when a cautious answer would still be justified by the retrieved evidence.

4. pred_confidence
- 0.9 to 1.0: very clear
- 0.7 to 0.89: mostly clear
- 0.4 to 0.69: ambiguous
- 0.0 to 0.39: weak basis

5. pred_reason
- One short sentence only.
- State the concrete reason, not a generic label.
- Good examples:
  - direct paper evidence from matching source
  - related topic but not the asked mechanism
  - source mismatch: expected paper but got vault
  - temporal query with no grounded time signal
  - refusal/meta-response excerpt, not substantive evidence
  - no retrieved evidence

Special guidance:
- For paper queries, prefer paper-source evidence when available.
- For web latest queries, do not over-trust observed_at alone as proof of freshness.
- For source=all or mixed queries, evaluate whether the fallback result is still substantively useful.
- If the CSV excerpt is too thin, you may inspect relevant repo files before deciding between partial and bad.
- Be conservative, but avoid collapsing everything into bad/abstain.

Output:
- Return the entire CSV.
- Update only pred_* columns.
```

## 권장 출력 파일 이름

- candidate:
  - `knowledgeos_machine_eval_cursor_judged_v1.csv`
- baseline:
  - `knowledgeos_machine_eval_baseline_cursor_judged_v1.csv`

## 운영 메모

- Cursor 평가는 **외부 웹 LLM보다 신뢰도가 높을 가능성**이 있다. 이유는 CSV만 보지 않고 repo 구조와 관련 파일을 직접 확인할 수 있기 때문이다.
- 그래도 Cursor 결과는 `pred_*` 초벌 평가일 뿐이고, 최종 truth는 여전히 사람의 `final_*`다.
