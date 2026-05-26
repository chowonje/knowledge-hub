# Text Evidence RC Convergence

- status: `ready_for_integration_review`
- publicRcReady: `False`
- phaseRows: `9`
- readyPhaseRows: `9`
- blockedPhaseRows: `0`
- publicRcBlockerRows: `2`
- privatePathLeakRows: `0`
- reportHash: `sha256:e6344148460d5e681def688bd44ed15f70beb64c43b464e904a5e7c345788f0b`

## Blockers

| blockerId | severity | reason |
|---|---|---|
| `canonical_checkout_dirty` | `hold` | canonical checkout is dirty and remains evidence-only |
| `pr_149_conflicting_or_draft` | `hold` | open strict-evidence audit PR is not clean/mergeable |

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
| `2` | `codex/complex-qa-real-strict-evidence-availability-bridge-audit-20260520` | `recut_or_abandon_before_rc` |
