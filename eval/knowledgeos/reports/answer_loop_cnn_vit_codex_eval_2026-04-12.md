# Answer-Loop CNN vs ViT Codex Eval 2026-04-12

Question:
- `CNN이랑 ViT를 논문 관점에서 비교해서 핵심 차이와 각각 잘하는 상황을 설명해줘`

Compared answers:
- Answer A: pre-axis-first compare answer from `answer_loop`
- Answer B: axis-first compare answer from `answer_loop`

Codex verdict:
- Answer A: `중`
- Answer B: `상`

Key findings:
- Answer A failed mainly at `ASSEMBLY` on top of a weak comparison-oriented retrieval set.
- Answer B improved because the packet and prompt forced axis-first comparison, but it remained somewhat defensive because the packet still lacked direct evidence for several axes.
- The dominant system issue is not raw model fluency. It is the combination of:
  - comparison retrieval that still leans toward representative papers
  - answer assembly that needs a stricter evidence audit before synthesis

Concrete follow-up:
1. retrieval should prefer axis-specific evidence, not only representative papers
2. comparison packets should label evidence by usage class
3. compare answers should treat task-specific examples as example-only support
4. judge prompts should describe failures in system-layer terms
