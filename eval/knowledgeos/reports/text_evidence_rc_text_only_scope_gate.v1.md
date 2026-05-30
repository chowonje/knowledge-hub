# Text Evidence RC Text-Only Scope Gate

- status: `blocked`
- decision: `text_only_scope_blocked`
- textOnlyRcReady: `False`
- publicRcReady: `False`
- phaseRows: `6`
- textReadyRows: `4`
- deferredRows: `6`
- blockerRows: `3`
- nextAction: `replay_missing_text_evidence_phase_reports`

## Deferred
- `visual_layout_image_format_branch` -> `codex/visual-layout-image-format-evidence-20260526`: outside_text_evidence_v01_rc_scope
- `bbox_identity_context_probe` -> `codex/visual-layout-image-format-evidence-20260526`: requires layout-aware provenance recovery, not text-only answerability
- `table_cell_grid_numeric_extraction` -> `codex/table-cell-grid-evidence-20260526`: table captions are candidates only until cell identity is recovered
- `equation_visual_latex_reconstruction` -> `codex/equation-visual-latex-evidence-20260526`: equation locator/context is text-only; LaTeX reconstruction needs a later parser path
- `figure_visual_binding` -> `codex/figure-visual-binding-evidence-20260526`: figure captions can be text evidence; image/visual binding is a separate evidence type
- `vlm_visual_interpretation` -> `codex/visual-layout-image-format-evidence-20260526`: external or VLM-derived interpretation cannot become citation-grade text evidence by default

## Blockers
- `missing_phase_report:text_equation_locator_context_artifacts` (hold): required phase report text_equation_locator_context_artifacts.v1.json is not present in current reports_root
- `missing_phase_report:text_complex_qa_eval_alignment` (hold): required phase report text_complex_qa_eval_alignment.v1.json is not present in current reports_root
- `missing_convergence_report` (hold): required convergence report text_evidence_rc_convergence.v1.json is not present in current reports_root
