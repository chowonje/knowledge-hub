# Priority Corpus Source Join Report

Generated: `2026-05-21T05:25:56.626367+00:00`

## Scope

Report-only join of priority corpus candidate ledger rows against local source artifact inventory file rows. Does not mutate corpus_manifest.json, download sources, or register manifest entries.

## Inventory Input (file rows, not paper count)

- Inventory file rows: **841**
- Inventory unique source ids (unjoined): **531**

## Join Result

- Total candidate rows: **446**
- Unique source_id count: **446**
- Matched to artifact (`available`): **351**
- Source missing: **86**
- Metadata only: **0**
- Hash missing: **0**
- Hash mismatch: **0**
- Ambiguous: **9**
- Parsed missing (available only): **0**
- Already in manifest (excluded from allowlist): **100**

## Expansion Allowlist

- Allowlist rows: **251**
- First tranche recommended: **50**

### Allowlist by tier

- `recent_ai`: **127**
- `local_corpus_candidate`: **124**

## First Tranche Recommendation (up to 50)

- `2605.03838` (recent_ai, 2026) — TRACE: A Metrologically-Grounded Engineering Framework for Trustworthy
- `2605.03862` (recent_ai, 2026) — Correct Is Not Enough: Training Reasoning Planners with Executor-Groun
- `2605.03871` (recent_ai, 2026) — EvoLM: Self-Evolving Language Models through Co-Evolved Discriminative
- `2605.03903` (recent_ai, 2026) — CC-OCR V2: Benchmarking Large Multimodal Models for Literacy in Real-w
- `2605.03986` (recent_ai, 2026) — From Intent to Execution: Composing Agentic Workflows with Agent Recom
- `2605.03989` (recent_ai, 2026) — An Agent-Oriented Pluggable Experience-RAG Skill for Experience-Driven
- `2605.04036` (recent_ai, 2026) — OpenSeeker-v2: Pushing the Limits of Search Agents with Informative an
- `2605.10616` (recent_ai, 2026) — MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Ima
- `attnres2026` (recent_ai, 2026) — Attention Residuals
- `2501.05031` (recent_ai, 2025) — ECBench: Can Multi-modal Foundation Models Understand Embodied Cogniti
- `2501.13956` (recent_ai, 2025) — Zep: A Temporal Knowledge Graph Architecture for Agent Memory
- `2502.04463` (recent_ai, 2025) — Training Language Models to Reason Efficiently
- `2502.07191` (recent_ai, 2025) — Bag of Tricks for Inference-time Computation of LLM Reasoning
- `2502.12521` (recent_ai, 2025) — Inference-Time Computations for LLM Reasoning and Planning: A Benchmar
- `2502.13130` (recent_ai, 2025) — Magma: A Foundation Model for Multimodal AI Agents
- `2502.15224` (recent_ai, 2025) — Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs
- `2502.15840` (recent_ai, 2025) — Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agent
- `2502.19918` (recent_ai, 2025) — Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning
- `2503.03734` (recent_ai, 2025) — OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature E
- `2503.04412` (recent_ai, 2025) — Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Bran
- `2503.07885` (recent_ai, 2025) — Safety Guardrails for LLM-Enabled Robots
- `2503.10965` (recent_ai, 2025) — Auditing language models for hidden objectives
- `2503.11926` (recent_ai, 2025) — Monitoring Reasoning Models for Misbehavior and the Risks of Promoting
- `2503.16248` (recent_ai, 2025) — Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks
- `2503.16416` (recent_ai, 2025) — Survey on Evaluation of LLM-based Agents
- `2503.19786` (recent_ai, 2025) — Gemma 3 Technical Report
- `2503.23077` (recent_ai, 2025) — Efficient Inference for Large Reasoning Models: A Survey
- `2504.00983` (recent_ai, 2025) — WorldScore: A Unified Evaluation Benchmark for World Generation
- `2504.10449` (recent_ai, 2025) — M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models
- `2504.11168` (recent_ai, 2025) — Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks aga
- `2504.14191` (recent_ai, 2025) — AI Idea Bench 2025: AI Research Idea Generation Benchmark
- `2504.14891` (recent_ai, 2025) — Retrieval Augmented Generation Evaluation in the Era of Large Language
- `2504.16736` (recent_ai, 2025) — A Survey of AI Agent Protocols
- `2504.18575` (recent_ai, 2025) — WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks
- `2505.03574` (recent_ai, 2025) — LlamaFirewall: An open source guardrail system for building secure AI
- `2505.05541` (recent_ai, 2025) — Safety by Measurement: A Systematic Literature Review of AI Safety Eva
- `2505.05849` (recent_ai, 2025) — AgentXploit: End-to-End Redteaming of Black-Box AI Agents
- `2505.07062` (recent_ai, 2025) — Seed1.5-VL Technical Report
- `2505.08341` (recent_ai, 2025) — Benchmarking AI scientists for omics data driven biological discovery
- `2505.09388` (recent_ai, 2025) — Qwen3 Technical Report
- `2505.16100` (recent_ai, 2025) — BioDSA-1K: Benchmarking Data Science Agents for Biomedical Hypothesis
- `2505.20873` (recent_ai, 2025) — Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visua
- `2505.23450` (recent_ai, 2025) — Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action M
- `2505.23621` (recent_ai, 2025) — Table-R1: Inference-Time Scaling for Table Reasoning
- `2506.01844` (recent_ai, 2025) — SmolVLA: A Vision-Language-Action Model for Affordable and Efficient R
- `2506.02153` (recent_ai, 2025) — Small Language Models are the Future of Agentic AI
- `2506.05176` (recent_ai, 2025) — Qwen3 Embedding: Advancing Text Embedding and Reranking Through Founda
- `2506.05813` (recent_ai, 2025) — MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table R
- `2506.06326` (recent_ai, 2025) — Memory OS of AI Agent
- `2506.06941` (recent_ai, 2025) — The Illusion of Thinking: Understanding the Strengths and Limitations

## Status Breakdown

- `ambiguous`: **9**
- `available`: **351**
- `source_missing`: **86**
