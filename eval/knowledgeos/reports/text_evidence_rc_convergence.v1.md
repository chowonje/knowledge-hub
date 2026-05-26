# Text Evidence RC Convergence

- status: `ready_for_integration_review`
- publicRcReady: `False`
- phaseRows: `9`
- readyPhaseRows: `9`
- blockedPhaseRows: `0`
- publicRcBlockerRows: `2`
- privatePathLeakRows: `0`
- reportHash: `sha256:8de8a79e0cb414f821ac68d483e4aa26c4b7f35de79579d236abc4a76a6a41b5`
- textOnlyScopeReady: `True`
- externalActionPreflightReady: `True`

## Blockers

| blockerId | severity | reason |
|---|---|---|
| `canonical_checkout_dirty` | `hold` | canonical dirty snapshot dry-run fingerprint is ready; physical cleanup still awaits PR #149 closure and approval |
| `pr_149_conflicting_or_draft` | `hold` | PR #149 preflight is ready for explicit close approval |

## Phase Reports

| phase | commit | reportStatus | disposition |
|---|---:|---|---|
| `figure_caption_artifact_vertical_slice` | `b5ffdf5` | `ready` | `included_in_stacked_branch` |
| `text_evidence_v01_roadmap` | `a4e7f68` | `accepted` | `included_in_stacked_branch` |
| `text_figure_caption_qa_path` | `9c1cf9a` | `ready` | `included_in_stacked_branch` |
| `text_section_paragraph_span_artifacts` | `41679e6` | `ready` | `included_in_stacked_branch` |
| `text_table_caption_candidate_artifacts` | `f2ebb44` | `ready` | `included_in_stacked_branch` |
| `text_equation_locator_context_artifacts` | `153eb79` | `ready` | `included_in_stacked_branch` |
| `text_complex_qa_eval_alignment` | `90585c8` | `ready` | `included_in_stacked_branch` |
| `source_alias_normalization` | `ac88f19` | `ready` | `included_in_stacked_branch` |
| `public_operator_surface_cleanup` | `ca4d794` | `accepted` | `included_in_stacked_branch` |

## Merge Queue

| order | branch | decision |
|---:|---|---|
| `1` | `codex/rc-hygiene-and-convergence-20260526` | `review_as_single_stacked_text_evidence_candidate` |
| `2` | `codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520` | `abandon_current_pr_before_public_rc` |
