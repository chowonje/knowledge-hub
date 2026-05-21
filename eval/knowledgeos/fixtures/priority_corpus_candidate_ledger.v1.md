# Priority Corpus Candidate Ledger

Date: 2026-05-21

## Scope

Metadata candidate ledger only. This tranche does not confirm source artifact availability, compute source hashes, or register rows as corpus_manifest available entries.

This ledger is the input for a later **source artifact join** tranche. Rows here are **not**
corpus availability declarations.

## Counts

- Candidate rows: **446**
- `eval_critical`: **55**
- `foundational`: **16**
- `recent_ai`: **210**
- `local_corpus_candidate`: **165**
- In manifest already: **100**
- Not in manifest yet: **346**
- Source verification still pending (unknown or join_pending): **446**

## Source Pools Surveyed

| Pool | Location | Rows |
| --- | --- | ---: |
| Local papers registry | `local_operator_registry::papers` | 446 |
| Public corpus manifest | `eval/knowledgeos/fixtures/corpus_manifest.json` | 100 artifacts / 100 source ids |
| Eval-critical references | queries + fixtures + complex QA seed ids | 28 unique source ids |
| Local PDF inventory hint | `local_operator_papers_dir/*.pdf` | 338 files (not availability proof) |

## Status Vocabulary

### current_manifest_status

- `in_manifest`: registered in `corpus_manifest.json`
- `not_in_manifest`: candidate only; not yet in public manifest

### source_artifact_status

- `unknown`: no artifact confirmation in this tranche
- `join_pending`: queued for source-byte/hash join; **not** `available` until join completes

## Candidate Preview (first 25 rows)

| source_id | title | year | tier | manifest | artifact status |
| --- | --- | ---: | --- | --- | --- |
| `alexnet-2012` | ImageNet Classification with Deep Convolutional Neural Netwo | 2012 | `eval_critical` | `in_manifest` | `join_pending` |
| `2501.06322` | Multi-Agent Collaboration Mechanisms: A Survey of LLMs | 2025 | `eval_critical` | `not_in_manifest` | `join_pending` |
| `2503.21460` | Large Language Model Agent: A Survey on Methodology, Applica | 2025 | `eval_critical` | `in_manifest` | `join_pending` |
| `2504.19413` | Mem0: Building Production-Ready AI Agents with Scalable Long | 2025 | `eval_critical` | `in_manifest` | `join_pending` |
| `2505.10468` | AI Agents vs. Agentic AI: A Conceptual Taxonomy, Application | 2025 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.01828` | Emergent Introspective Awareness in Large Language Models | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.03236` | MAGMA: A Multi-Graph based Agentic Memory Architecture for A | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.04720` | Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framewor | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.09668` | STEP3-VL-10B Technical Report | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.11077` | ABC-Bench: Benchmarking Agentic Backend Coding in Real-World | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.12542` | Rethinking the AI Scientist: Interactive Multi-Agent Workflo | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2601.23086` | Chain-of-thought obfuscation learned from output supervision | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.00103` | Autonomous Multi-Agent AI for High-Throughput Polymer Discov | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.00185` | QUASAR: A Universal Autonomous System for Atomistic Simulati | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.01237` | Predictive Scheduling for Efficient Inference-Time Reasoning | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.01655` | ProjDevBench: Benchmarking AI Coding Agents on End-to-End Pr | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.01853` | Designing Time Series Experiments in A/B Testing with Transf | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.02007` | Beyond RAG for Agent Memory: Retrieval by Decoupling and Agg | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.02262` | OmniCode: A Benchmark for Evaluating Software Engineering Ag | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.02905` | FIRE-Bench: Evaluating Agents on the Rediscovery of Scientif | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.03117` | AgentDyn: A Dynamic Open-Ended Benchmark for Evaluating Prom | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.03442` | A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hi | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.10715` | Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Fram | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.10975` | FeatureBench: Benchmarking Agentic Coding for Complex Featur | 2026 | `eval_critical` | `in_manifest` | `join_pending` |
| `2602.11964` | Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous E | 2026 | `eval_critical` | `in_manifest` | `join_pending` |

Full machine-readable ledger: `eval/knowledgeos/fixtures/priority_corpus_candidate_ledger.v1.json`
