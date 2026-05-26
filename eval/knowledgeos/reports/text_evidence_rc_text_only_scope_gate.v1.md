# Text Evidence RC Text-Only Scope Gate

- status: `ready`
- decision: `text_only_scope_ready_for_public_rc_review`
- textOnlyRcReady: `True`
- publicRcReady: `True`
- phaseRows: `6`
- textReadyRows: `6`
- deferredRows: `6`
- blockerRows: `0`
- nextAction: `ready_for_public_rc_review`

## Deferred
- `visual_layout_image_format_branch` -> `codex/visual-layout-image-format-evidence-20260526`: outside_text_evidence_v01_rc_scope
- `bbox_identity_context_probe` -> `codex/visual-layout-image-format-evidence-20260526`: requires layout-aware provenance recovery, not text-only answerability
- `table_cell_grid_numeric_extraction` -> `codex/table-cell-grid-evidence-20260526`: table captions are candidates only until cell identity is recovered
- `equation_visual_latex_reconstruction` -> `codex/equation-visual-latex-evidence-20260526`: equation locator/context is text-only; LaTeX reconstruction needs a later parser path
- `figure_visual_binding` -> `codex/figure-visual-binding-evidence-20260526`: figure captions can be text evidence; image/visual binding is a separate evidence type
- `vlm_visual_interpretation` -> `codex/visual-layout-image-format-evidence-20260526`: external or VLM-derived interpretation cannot become citation-grade text evidence by default

## Blockers
