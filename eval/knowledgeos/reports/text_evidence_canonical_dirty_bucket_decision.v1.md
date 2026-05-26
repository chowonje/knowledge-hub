# Text Evidence Canonical Dirty Bucket Decision

- status: `ready`
- dirtyRows: `168`
- bucketRows: `11`
- directIncludeRows: `0`
- blockRows: `0`
- unknownBucketRows: `0`
- publicRcDecision: `do_not_merge_canonical_dirty_checkout_into_text_rc`
- nextAction: `clean_or_archive_canonical_dirty_checkout_after_external_pr149_resolution`

## Buckets

| bucket | rows | action | decision | follow-up |
|---|---:|---|---|---|
| `answer_runtime_or_query_stack` | `23` | `do_not_replay_now` | `exclude_from_text_rc_review_after_rc_candidate` | `answer_runtime_replay` |
| `cli_mcp_public_surface_stack` | `21` | `do_not_replay_now` | `exclude_from_text_rc_compare_with_public_operator_cleanup` | `public_surface_replay_review` |
| `docs_governance_stack` | `11` | `review_records_only` | `exclude_from_text_rc_record_sync_only` | `docs_record_reconciliation` |
| `eval_or_test_support_stack` | `36` | `do_not_replay_now` | `exclude_from_text_rc_map_to_owning_feature` | `owning_feature_eval_replay` |
| `evidence_spine_or_source_contract_stack` | `29` | `defer_to_clean_replay` | `exclude_from_text_rc_clean_replay_candidate` | `evidence_spine_clean_replay` |
| `infrastructure_or_core_stack` | `11` | `defer_to_clean_replay` | `exclude_from_text_rc_clean_replay_candidate` | `core_infrastructure_clean_replay` |
| `parser_artifact_side_stack` | `6` | `hold` | `hold_outside_text_rc_parser_track` | `parser_artifact_repair` |
| `provider_hint_side_stack` | `8` | `hold` | `hold_outside_text_rc_provider_hint_track` | `provider_hint_shadow` |
| `research_objects_side_stack` | `12` | `hold` | `hold_outside_text_rc_research_objects_track` | `research_objects` |
| `source_ingest_or_library_stack` | `7` | `defer_to_clean_replay` | `exclude_from_text_rc_clean_replay_candidate` | `source_ingest_library_clean_replay` |
| `workspace_process_record` | `4` | `exclude_from_public_rc` | `drop_or_move_out_of_product_checkout` | `workspace_record_cleanup` |
