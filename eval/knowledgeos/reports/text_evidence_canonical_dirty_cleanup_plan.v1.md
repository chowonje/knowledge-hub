# Text Evidence Canonical Dirty Cleanup Plan

- status: `ready`
- dirtyRows: `168`
- planRows: `11`
- manualPlanRows: `0`
- requiresExplicitApproval: `True`
- requiresPr149ResolvedFirst: `True`
- requiresSnapshotBeforeCleanup: `True`
- allowsDirectTextRcMerge: `False`
- nextAction: `request_approval_to_close_pr149_then_snapshot_and_clean_canonical_checkout`

## Sequence Counts

| sequence | rows |
|---|---:|
| `clean_replay_candidate` | `3` |
| `held_side_track` | `3` |
| `owning_feature_review` | `1` |
| `post_rc_candidate_review` | `1` |
| `public_rc_exclusion` | `1` |
| `public_surface_review` | `1` |
| `record_reconciliation` | `1` |

## Cleanup Rows

| bucket | dirty rows | sequence | cleanup action |
|---|---:|---|---|
| `answer_runtime_or_query_stack` | `23` | `post_rc_candidate_review` | `archive_or_discard_after_text_rc_branch_review` |
| `cli_mcp_public_surface_stack` | `21` | `public_surface_review` | `compare_then_discard_or_clean_replay` |
| `docs_governance_stack` | `11` | `record_reconciliation` | `selective_record_reconciliation_then_discard_remainder` |
| `eval_or_test_support_stack` | `36` | `owning_feature_review` | `map_to_owning_feature_then_archive_or_discard` |
| `evidence_spine_or_source_contract_stack` | `29` | `clean_replay_candidate` | `preserve_for_later_clean_replay_then_clear_canonical_dirty` |
| `infrastructure_or_core_stack` | `11` | `clean_replay_candidate` | `preserve_for_later_clean_replay_then_clear_canonical_dirty` |
| `parser_artifact_side_stack` | `6` | `held_side_track` | `hold_for_parser_branch_or_archive` |
| `provider_hint_side_stack` | `8` | `held_side_track` | `hold_or_discard_shadow_experiment` |
| `research_objects_side_stack` | `12` | `held_side_track` | `archive_research_objects_side_stack` |
| `source_ingest_or_library_stack` | `7` | `clean_replay_candidate` | `preserve_for_later_clean_replay_then_clear_canonical_dirty` |
| `workspace_process_record` | `4` | `public_rc_exclusion` | `move_to_workspace_records_or_delete_from_product_checkout` |
