# RAG-Based Answer System Evaluation Prompt v1

Use this prompt when you want Codex, GPT, Claude, or another evaluator to compare multiple answers to the same question from a system-design perspective.

Purpose:
- decide whether answer quality came from the base model or from system structure
- locate the dominant failure layer
- produce system-fix guidance rather than rewriting the answer only

Inputs:
- `Question`
- `Answer A` (project output)
- `Answer B` (Codex/GPT output)
- optional `Answer C` (reference answer)

Required evaluation dimensions:
- query understanding
- retrieval quality
- evidence grounding
- answer assembly
- reasoning quality
- overclaim / hallucination risk
- system attribution

Required output shape:
- overall verdict per answer
- top 3 issues per answer
- per-answer layer analysis
- system attribution buckets:
  - `MODEL`
  - `RETRIEVAL`
  - `ASSEMBLY`
  - `PROMPT`
- top 3 system improvement actions

Important rule:
- the goal is not “fix this one answer”
- the goal is “identify what to change in the system”
