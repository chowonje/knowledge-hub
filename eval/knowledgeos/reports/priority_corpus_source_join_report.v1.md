# Priority Corpus Source Join Report

Generated: `2026-05-21T06:04:17.825159+00:00`

## Scope

Report-only join of priority corpus candidate ledger rows against local source artifact inventory file rows. Does not mutate corpus_manifest.json, download sources, or register manifest entries.

## Inventory Input (file rows, not paper count)

- Inventory file rows: **888**
- Inventory unique source ids (unjoined): **627**

## Join Result

- Total candidate rows: **446**
- Unique source_id count: **446**
- Matched to artifact (`available`): **396**
- Source missing: **41**
- Metadata only: **0**
- Hash missing: **0**
- Hash mismatch: **0**
- Ambiguous: **9**
- Parsed missing (available only): **0**
- Already in manifest (excluded from allowlist): **200**

## Expansion Allowlist

- Allowlist rows: **196**
- First tranche recommended: **50**

### Allowlist by tier

- `recent_ai`: **68**
- `local_corpus_candidate`: **128**

## First Tranche Recommendation (up to 50)

- `2507.13575` (recent_ai, 2025) — Apple Intelligence Foundation Language Models
- `2507.20526` (recent_ai, 2025) — Security Challenges in AI Agent Deployment: Insights from a Large Scal
- `2508.02324` (recent_ai, 2025) — Qwen-Image Technical Report
- `2508.04039` (recent_ai, 2025) — Large Reasoning Models Are Autonomous Jailbreak Agents
- `2508.15763` (recent_ai, 2025) — Intern-S1: A Scientific Multimodal Foundation Model
- `2509.06917` (recent_ai, 2025) — Paper2Agent: Reimagining Research Papers As Interactive and Reliable A
- `2509.14260` (recent_ai, 2025) — Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs
- `2509.16861` (recent_ai, 2025) — AdaptiveGuard: Towards Adaptive Runtime Safety for LLM-Powered Softwar
- `2509.16870` (recent_ai, 2025) — DecipherGuard: Understanding and Deciphering Jailbreak Prompts for a S
- `2509.16941` (recent_ai, 2025) — SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering T
- `2509.17765` (recent_ai, 2025) — Qwen3-Omni Technical Report
- `2509.19480` (recent_ai, 2025) — OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigati
- `2509.21766` (recent_ai, 2025) — UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Sc
- `2509.24065` (recent_ai, 2025) — Open Opportunities in AI Safety, Alignment, and Ethics (AI SAE)
- `2510.01375` (recent_ai, 2025) — Fine-tuning with RAG for Improving LLM Learning of New Skills
- `2510.04852` (recent_ai, 2025) — FreshBrew: A Benchmark for Evaluating AI Agents on Java Code Migration
- `2510.07172` (recent_ai, 2025) — NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LL
- `2510.08002` (recent_ai, 2025) — Learning on the Job: An Experience-Driven Self-Evolving Agent for Long
- `2510.08996` (recent_ai, 2025) — Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Ev
- `2510.15682` (recent_ai, 2025) — SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmen
- `2510.21571` (recent_ai, 2025) — Scalable Vision-Language-Action Model Pretraining for Robotic Manipula
- `2510.21652` (recent_ai, 2025) — AstaBench: Rigorous Benchmarking of AI Agents with a Holistic Scientif
- `2510.22075` (recent_ai, 2025) — Agentic Reinforcement Learning for Real-World Code Repair
- `2510.24358` (recent_ai, 2025) — Automatically Benchmarking LLM Code Agents through Agent-Driven Annota
- `2510.24699` (recent_ai, 2025) — AgentFold: Long-Horizon Web Agents with Proactive Context Management
- `2510.26583` (recent_ai, 2025) — Emu3.5: Native Multimodal Models are World Learners
- `2510.27598` (recent_ai, 2025) — InnovatorBench: Evaluating Agents' Ability to Innovate in Machine Lear
- `2511.13646` (recent_ai, 2025) — Can Software Engineering Agents Self-Evolve on the Fly?
- `2511.18298` (recent_ai, 2025) — Cross-Disciplinary Knowledge Retrieval and Synthesis: A Compound AI Ar
- `2511.21631` (recent_ai, 2025) — Qwen3-VL Technical Report
- `2511.22138` (recent_ai, 2025) — TinyLLM: Evaluation and Optimization of Small Language Models for Agen
- `2511.23404` (recent_ai, 2025) — LFM2 Technical Report
- `2512.01822` (recent_ai, 2025) — InnoGym: Benchmarking the Innovation Potential of AI Agents
- `2512.02008` (recent_ai, 2025) — The Art of Scaling Test-Time Compute for Large Language Models
- `2512.02425` (recent_ai, 2025) — WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
- `2512.03262` (recent_ai, 2025) — Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Cod
- `2512.07582` (recent_ai, 2025) — See Once, Then Act: Vision-Language-Action Model with Task Learning fr
- `2512.13564` (recent_ai, 2025) — Memory in the Age of AI Agents
- `2512.15840` (recent_ai, 2025) — Large Video Planner Enables Generalizable Robot Control
- `2512.15943` (recent_ai, 2025) — Small Language Models for Efficient Agentic Tool Calling: Outperformin
- `2512.18470` (recent_ai, 2025) — SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution
- `2512.18552` (recent_ai, 2025) — Toward Training Superintelligent Software Agents through Self-Play
- `2512.21373` (recent_ai, 2025) — AInsteinBench: Benchmarking Coding Agents on Scientific Computing Deve
- `2512.21919` (recent_ai, 2025) — SWE-RM: Execution-free Feedback For Training and Test-Time Scaling of
- `2512.22414` (recent_ai, 2025) — Emergence of Human to Robot Transfer in Vision-Language-Action Models
- `2401.04088` (recent_ai, 2024) — Mixtral of Experts
- `2401.15391` (recent_ai, 2024) — MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Ho
- `2401.15884` (recent_ai, 2024) — Corrective Retrieval Augmented Generation
- `2401.17043` (recent_ai, 2024) — arXiv 2401.17043
- `2402.01881` (recent_ai, 2024) — Large Language Model Agent for Hyper-Parameter Optimization

## Status Breakdown

- `ambiguous`: **9**
- `available`: **396**
- `source_missing`: **41**
