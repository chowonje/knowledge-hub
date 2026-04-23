# KnowledgeOS External Eval Prompt: Paper-Memory Triplet v1

You are evaluating three versions of paper-memory cards for the same papers.

Files to use:
- `knowledgeos_paper_memory_triplet_compare_12_compact_v3.csv`
- optionally `knowledgeos_paper_memory_vs_gptpro_sample_12_v1.csv` as historical context for the old baseline review

Variants:
- `baseline`: the existing stored card
- `exaone`: local `exaone3.5:7.8b` compact rebuild
- `openai`: `gpt-5.4` compact rebuild

Goal:
- Judge which variant is best grounded in the visible excerpt for each paper.
- Pay special attention to the earlier failure modes:
  - unsupported `evidence_core`
  - speculative `limitations`
  - overclaiming benchmark superiority not visible in the excerpt

Instructions:
1. Treat the visible excerpt as the source of truth.
2. Reward specificity only when it is actually supported by the excerpt.
3. Penalize invented or over-generalized evidence.
4. Penalize limitations that are not explicitly supported.
5. If a more conservative summary is better supported, prefer the more conservative variant even if it is less detailed.

Return exactly one CSV with this header:

```csv
paper_id,title,winner,runner_up,baseline_label,exaone_label,openai_label,winner_reason,main_risk
```

Field definitions:
- `winner`: one of `baseline`, `exaone`, `openai`
- `runner_up`: one of `baseline`, `exaone`, `openai`
- `*_label`: one of `good`, `partial`, `bad`
- `winner_reason`: short explanation focusing on grounding and fidelity
- `main_risk`: the biggest remaining weakness in the winning card

Evaluation rubric:
- `good`: core/method/evidence/limitations are all well aligned to the visible excerpt
- `partial`: mostly aligned but one or two fields are too vague, too aggressive, or incomplete
- `bad`: core claim or evidence is materially unsupported, or limitations are clearly invented

Important:
- Do not prefer a variant just because it is longer.
- Do not reward unsupported specificity.
- If `openai` says `limitations not explicit in visible excerpt` and the excerpt truly lacks explicit limitations, that is often a strength rather than a weakness.
