# Paper-Memory Improvement Priorities v1

Source:

- External review result provided by GPT Pro on `knowledgeos_paper_memory_vs_gptpro_sample_12_v1.csv`
- Internal card fields from `paper_memory_cards`

## Snapshot

- `good`: `1 / 12`
- `partial`: `10 / 12`
- `bad`: `1 / 12`

Interpretation:

- `paper_core` and `method_core` are often usable.
- The main weakness is `evidence_core`.
- The second weakness is `limitations`, especially when the source excerpt does not explicitly support a limitation claim.
- The worst failures still correlate with poor source text quality.

## Priority Buckets

### P0. Unusable Source / Card Trust Failure

These papers should not be trusted until source quality or extraction quality is fixed.

| paper_id | title | Why |
| --- | --- | --- |
| `2603.15798` | `CUBE: A Standard for Unifying Agent Benchmarks` | Rated `bad`. The excerpt was effectively front matter, so the card cannot be validated against meaningful content. |

### P1. Evidence Extraction Too Weak

These cards mostly capture the core idea, but `evidence_core` is too generic, omits the strongest result, or drifts away from the source passage.

| paper_id | title | Main issue |
| --- | --- | --- |
| `2510.27598` | `InnovatorBench: Evaluating Agents' Ability to Innovate in Machine Learning Research` | Evidence is broad and low-density relative to the benchmark description. |
| `2602.02262` | `OmniCode: A Benchmark for Evaluating Software Engineering Agents` | Evidence misses the sharper baseline result patterns in the excerpt. |
| `2603.17216` | `AI Scientist via Synthetic Task Scaling` | Evidence leaves out the strongest quantitative gains. |
| `2506.06326` | `Memory OS of AI Agent` | Evidence overstates support from the visible excerpt. |
| `2502.15224` | `Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs` | Evidence drifts from the excerpt and partially reverses the intended finding. |
| `2503.23077` | `Efficient Inference for Large Reasoning Models: A Survey` | Evidence inserts a specific critique not visible in the excerpt. |
| `2405.14093` | `A Survey on Vision-Language-Action Models for Embodied AI` | Evidence becomes broader than the quoted architectural examples. |

### P2. Limitation Claims Too Aggressive

These cards should be made more conservative. If the excerpt does not directly show limitations, the card should say that limits are not explicit in the visible evidence.

| paper_id | title | Main issue |
| --- | --- | --- |
| `2212.09748` | `Scalable Diffusion Models with Transformers` | Limitation not clearly grounded in the excerpt. |
| `2602.02262` | `OmniCode: A Benchmark for Evaluating Software Engineering Agents` | Limitation claim extends beyond the visible passage. |
| `2603.17216` | `AI Scientist via Synthetic Task Scaling` | Limitation not explicit in the source excerpt. |
| `2506.06326` | `Memory OS of AI Agent` | Limitation is plausible but not explicitly shown. |
| `2501.13956` | `Zep: A Temporal Knowledge Graph Architecture for Agent Memory` | Limitation/support is too weakly grounded in the visible text. |
| `2405.14093` | `A Survey on Vision-Language-Action Models for Embodied AI` | Limitation overreaches relative to the excerpt. |

### P3. Broadly Usable / Keep As Reference

These cards are acceptable reference points for “what good looks like” in the current pipeline.

| paper_id | title | Note |
| --- | --- | --- |
| `2307.15818` | `RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control` | Rated `good`; useful exemplar for grounded `core/method/evidence/limitations`. |
| `2212.09748` | `Scalable Diffusion Models with Transformers` | Core and method are strong even though evidence/limitations need tightening. |
| `2603.17216` | `AI Scientist via Synthetic Task Scaling` | Core/method are good; evidence should become more quantitative. |

## Required Pipeline Fixes

### 1. Evidence-First Extraction

Current behavior is too summary-oriented. `evidence_core` should explicitly prefer:

- sentences containing numbers, gains, benchmark names, or comparison markers
- lines with words like `improves`, `achieves`, `outperforms`, `FID`, `AUP`, `accuracy`, `gain`, `benchmark`
- excerpt-local evidence over generic paraphrase

Expected change:

- `evidence_core` becomes shorter and more concrete
- quantitative statements survive extraction more often

### 2. Conservative Limitation Policy

If the visible excerpt does not directly show a limitation, do not invent one.

Target behavior:

- either extract an explicit limitation from the source
- or emit a conservative statement like:
  - `limitations not explicit in visible excerpt`

Expected change:

- lower overclaim risk
- higher limitations fidelity

### 3. Source-Quality Gate Before Card Trust

If the source excerpt is still mostly LaTeX, front matter, or metadata, the card should be marked `needs_review` instead of pretending to be complete.

Target behavior:

- fail closed on unusable text
- do not treat front matter as usable evidence

Expected change:

- fewer `bad` cards
- clearer operator queue for re-extraction

## Recommended Next Pass

### Pass 1. Trust Repair

- Fix `2603.15798` first.
- Rebuild cards in `P0`.

### Pass 2. Evidence Repair

- Rebuild all `P1` cards with stricter evidence extraction.
- Compare old vs new `evidence_core` only.

### Pass 3. Limitation Guardrail

- Rebuild `P2` cards with “explicit limitation or abstain” behavior.

## Gate for the Next Review

The next sample review should pass these thresholds:

- `bad = 0`
- `good >= 4 / 12`
- `evidence_fidelity low <= 2 / 12`
- `limitations_fidelity low <= 2 / 12`
- no card should make a limitation claim unsupported by the provided excerpt

## Decision

Do not treat current `paper-memory` cards as citation-grade summaries.
They are usable as retrieval and synthesis scaffolding, but not yet as fully trusted standalone evidence cards.
