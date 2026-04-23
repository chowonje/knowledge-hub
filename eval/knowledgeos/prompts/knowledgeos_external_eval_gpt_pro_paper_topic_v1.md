# KnowledgeOS External Eval Prompt for GPT Pro Paper Topic v1

업로드할 파일:

- query set:
  - `eval/knowledgeos/queries/knowledgeos_paper_topic_eval_queries_20_v1.csv`
- candidate outputs:
  - `labs paper topic-synthesize` 결과를 query별로 정리한 CSV, JSONL, 또는 markdown bundle
- optional baseline outputs:
  - 기존 `khub ask` 또는 비교 대상 시스템의 같은 query 결과 묶음

권장 입력 형태:

- query CSV에는 `query_id`, `query`, `review_focus`, `must_avoid`가 포함되어 있다.
- candidate outputs는 query별로 아래 정보가 있으면 가장 좋다.
  - `query_id`
  - 최종 answer text
  - selected papers
  - excluded papers
  - citations 또는 supporting titles
  - verification/diagnostics

GPT Pro에게 줄 프롬프트:

```text
You are a strict evaluator for KnowledgeOS topic-style paper synthesis outputs.

I will upload:
1. a query CSV named `knowledgeos_paper_topic_eval_queries_20_v1.csv`
2. a candidate output file containing one answer per `query_id`
3. optionally a baseline output file for side-by-side comparison

Your task is to evaluate the candidate output for each query.

Use the query CSV as the primary intent specification.
Pay special attention to:
- `review_focus`
- `must_avoid`

For each query_id, produce exactly one row in a CSV with these columns:

- query_id
- overall_label
- intent_match
- selection_precision
- selection_coverage
- synthesis_quality
- grounding_quality
- abstain_quality
- major_failure_mode
- short_reason

Allowed values:
- overall_label = good | partial | bad
- intent_match = high | medium | low
- selection_precision = high | medium | low
- selection_coverage = high | medium | low
- synthesis_quality = high | medium | low
- grounding_quality = high | medium | low
- abstain_quality = good | partial | bad | n/a
- major_failure_mode = none | single_paper_collapse | irrelevant_inclusions | missed_core_papers | weak_comparison | unsupported_claims | bad_abstention | other
- short_reason = one short sentence

Judging rules:
- This is a topic-style multi-paper task, not a single-paper lookup task.
- A single-paper answer is usually bad unless the query explicitly narrows to one canonical paper.
- Prefer precision over noisy breadth, but do not reward over-pruning when the query clearly asks for a set or taxonomy.
- If the answer includes adjacent-but-wrong papers called out by `must_avoid`, penalize selection_precision.
- If the answer fails to separate categories requested in the query (for example architecture vs benchmark, survey vs system, benchmark vs application), penalize synthesis_quality.
- If the answer names papers without explaining why they belong, penalize grounding_quality.
- If the local corpus clearly seems insufficient and the system says so honestly with caveats, abstain_quality may be good or partial rather than bad.
- Do not reward confident hallucinated taxonomy or invented citations.
- When comparing candidate vs baseline, judge the candidate independently first. Then mention comparative advantage only in `short_reason` if it is clearly relevant.

Interpretation guide:
- `intent_match` asks whether the answer understood what kind of paper set the user wanted.
- `selection_precision` asks whether the kept papers are actually on-topic.
- `selection_coverage` asks whether the answer captured the core representative papers or subfamilies that the corpus appears to contain.
- `synthesis_quality` asks whether the answer compares, groups, and differentiates papers rather than listing them.
- `grounding_quality` asks whether claims are tied to cited or explicitly named papers and whether the explanation is supported by the provided evidence.
- `abstain_quality` matters only when the system admits uncertainty or missing corpus coverage.

Label guide:
- `good`: the answer is clearly multi-paper, mostly on-topic, distinguishes the requested categories, and is grounded.
- `partial`: the answer is somewhat useful but has notable noise, missed distinctions, weak grouping, or thin grounding.
- `bad`: the answer collapses to one paper, is mostly off-topic, ignores required distinctions, or makes unsupported claims.

Return only the CSV.
Do not add commentary before or after the CSV.
Do not reorder query_id values if the candidate file already has an order.
```

권장 파일 이름:

- candidate judgement:
  - `knowledgeos_paper_topic_eval_gpt_pro_judged_v1.csv`
- baseline judgement:
  - `knowledgeos_paper_topic_eval_baseline_gpt_pro_judged_v1.csv`

운영 팁:

- candidate와 baseline은 가능하면 separate run으로 평가한다.
- candidate output에 `selectedPapers`, `excludedPapers`, `citations`, `verification`가 함께 있으면 판정 품질이 훨씬 안정적이다.
- query당 결과가 길다면 markdown bundle보다는 CSV 또는 JSONL이 더 안정적이다.
