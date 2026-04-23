# Paper-Memory Triplet Internal Review v1

Source comparison:
- [knowledgeos_paper_memory_triplet_compare_12_compact_v3.csv](/Users/won/Desktop/allinone/knowledge-hub/eval/knowledgeos/runs/knowledgeos_paper_memory_triplet_compare_12_compact_v3.csv)

Scope:
- `baseline`: existing card
- `exaone`: local `exaone3.5:7.8b` compact rebuild
- `openai`: `gpt-5.4` compact rebuild

Evaluation lens:
- Prefer grounded `evidence_core`
- Penalize unsupported or speculative `limitations`
- Prefer concise but specific `method_core`
- Favor cards that match the earlier GPT Pro review style: conservative when evidence is not explicit

## Quick Summary

- Likely winner overall: `openai`
- Main reason: it is consistently more conservative on `evidence_core` and `limitations`, which matches the main failure mode found in the earlier GPT Pro review.
- Main tradeoff: `openai` can become too cautious when the visible excerpt actually supports a concrete limitation and the compact excerpt does not carry it clearly enough.
- Best local fallback: `exaone`
- Main weakness of `exaone`: still tends to keep unsupported `limitations` or over-confident evidence phrasing when the excerpt is thin.

## Per-Paper Judgment

| paper_id | title | likely_winner | confidence | short_reason |
| --- | --- | --- | --- | --- |
| 2510.27598 | InnovatorBench | openai | medium | OpenAI adds the 20-task structure and removes unsupported limitations; baseline/exaone stay too generic and limitation-heavy. |
| 2212.09748 | Scalable Diffusion Models with Transformers | openai | high | OpenAI preserves the key numeric evidence (`FID 2.27`) and avoids unsupported limitation claims. |
| 2307.15818 | RT-2 | exaone | medium | Exaone keeps the visible physical-skill limitation; OpenAI is safer but likely too conservative for this sample. |
| 2602.02262 | OmniCode | openai | high | OpenAI keeps the benchmark scope and avoids unsupported quantitative or limitation claims. |
| 2603.17216 | AI Scientist via Synthetic Task Scaling | openai | high | OpenAI removes unsupported MLGym limitation claims and stays aligned to the visible excerpt. |
| 2603.15798 | CUBE | openai | high | The source is weak; OpenAI's abstaining limitation style is safer than baseline/exaone's specific but unsupported limitation. |
| 2506.06326 | Memory OS of AI Agent | openai | high | OpenAI avoids carrying over unsupported numeric gains from the older card. |
| 2501.13956 | Zep | openai | high | OpenAI keeps the architecture description and benchmark framing without claiming superiority that is not visible. |
| 2502.15224 | Auto-Bench | openai | high | OpenAI stays conservative on results where the earlier card was judged directionally wrong on evidence. |
| 2503.23077 | Efficient Inference for Large Reasoning Models: A Survey | openai | medium | OpenAI drops a specific unsupported evidence claim and keeps the survey framing tighter. |
| 2507.02076 | Reasoning on a Budget | openai | medium | OpenAI's evidence is still summary-level but more grounded to the visible excerpt than baseline/exaone. |
| 2405.14093 | A Survey on Vision-Language-Action Models for Embodied AI | openai | high | OpenAI avoids unsupported experimental and limitation claims that baseline/exaone keep. |

## Aggregate Takeaways

- `openai` likely wins `11 / 12`
- `exaone` likely wins `1 / 12`
- `baseline` does not clearly win any of the `12` samples

## Recommended Next Step

1. Send the triplet CSV to GPT Pro with the external evaluation prompt.
2. If GPT Pro broadly agrees that `openai` is better on evidence/limitations, use:
   - `gpt-5.4` for high-value subset rebuilds
   - `exaone` as the local fallback path
3. If GPT Pro finds `openai` too conservative on specific papers like `RT-2`, widen the compact excerpt or add a dedicated limitations evidence slot before a larger rebuild.
