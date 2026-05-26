# ADR: Text Evidence v0.1 Scope

Date: 2026-05-26

## Status

Accepted for v0.1 planning.

## Context

The research-paper roadmap was drifting toward broad report-only gates and hard
PDF layout problems before a small user-visible vertical slice was proven. The
current FigureCaptionArtifact slice can extract caption-text candidates from a
small local AI-paper set and run QA readback with `sourceContentHash`, page,
bbox, and caption identity, but it does not interpret figure images or promote
strict evidence.

The v0.1 product should prove a local-first, evidence-first research workflow
with text-derived provenance before attempting visual or rich-layout semantics.

## Decision

v0.1 mainline is a text-evidence operating slice:

- source PDF/text registration and hash traceability
- parsed text artifacts and text spans
- section and paragraph spans
- figure caption text evidence
- table caption and table-like text candidates
- equation locator and surrounding text candidates
- answerability gates that return no-answer when provenance is insufficient
- QA readback that can cite `sourceContentHash` plus chars/page/bbox provenance

The following are deferred out of v0.1 mainline:

- figure image understanding
- chart, plot, bar, or diagram visual reasoning
- subfigure visual-object binding
- full PDF formatting reconstruction
- complete table-grid/cell guarantees for every PDF
- complete equation LaTeX reconstruction
- VLM or external-parser output as direct strict evidence authority

Deferred visual/layout work may happen on a separate research branch, but it must
not block or redefine v0.1 text-evidence completion.

## Roadmap

1. `text_figure_caption_qa_path`
   - Connect FigureCaptionArtifact candidates to a product/internal paper-QA readback path.
   - Answer "what does Figure N show?" from caption text only.
   - Return no-answer for visual-inspection questions or missing caption provenance.

2. `text_section_paragraph_span_artifacts`
   - Create stable SectionSpan and ParagraphSpan candidates from parsed text.
   - Preserve `sourceContentHash` plus chars/page provenance.
   - Keep layout repair candidate-only unless explicitly applied.

3. `text_table_caption_candidate_artifacts`
   - Extract table captions and table-like text blocks.
   - Treat row/column/cell structure as candidate-grade until provenance is complete.
   - Allow numeric QA only for rows with explicit cell or text-span provenance.

4. `text_equation_locator_context_artifacts`
   - Extract equation labels/numbers when available.
   - Attach nearby paragraph/context text and source provenance.
   - Do not require full LaTeX reconstruction for v0.1.

5. `text_complex_qa_eval_alignment`
   - Re-scope complex-paper QA around text-evidence capabilities.
   - Separate text-answerable, candidate-only, visual-unsupported, and no-answer cases.

6. `source_alias_normalization`
   - Prevent short aliases such as RAG, GPT, CNN, and VLM from overmatching.
   - Resolve aliases only inside explicit paper/query context.

7. `public_operator_surface_cleanup`
   - Keep default public CLI/MCP surfaces compact.
   - Gate labs/operator tools behind explicit advanced profiles.

8. `rc_hygiene_and_convergence`
   - Reconcile side branches, worktrees, schema records, reports, and docs.
   - Run release smoke, public hygiene, focused eval gates, and diff checks.

## Promotion Rules

An artifact can participate in answerability only when it has:

- `sourceContentHash`
- a source locator: chars, page, bbox, or a clearly bounded equivalent
- deterministic artifact identity
- extraction method
- confidence or blocker reason
- a schema-backed payload

An artifact remains blocked from strict evidence when:

- it is locator-only
- it is memory-unit-only
- it is fallback text without source hash
- it depends on visual inspection not represented in text evidence
- it depends on VLM/external output without local source provenance
- its row/column/cell or equation identity is candidate-only

## Stop Rules

Stop before:

- vault scans or vault writes
- DB/index mutation
- reindex/reembed
- external downloads
- canonical parsed artifact overwrite
- strict evidence promotion
- runtime answer-visible exposure
- broad parser-framework rewrites

Each stop can be lifted only by an explicit scoped instruction.

## Verification

Each tranche must leave:

- schema-backed JSON payloads or fixtures
- focused pytest coverage
- generated report or readback artifact when applicable
- private-path scan result
- `git diff --check`
- updated `CHANGELOG.md`
- updated `docs/PROJECT_STATE.md` when product scope or behavior changes

## Consequences

This narrows v0.1 from "understand any PDF layout" to "answer from auditable
text evidence." The tradeoff is intentional: figure visuals, exact table grids,
and equation LaTeX remain later work, while users get a safer and earlier paper
QA path that can explain exactly what source text supports an answer.
