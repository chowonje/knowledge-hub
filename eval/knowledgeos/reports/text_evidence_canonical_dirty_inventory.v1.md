# Text Evidence Canonical Dirty Inventory

- status: `ready`
- dirtyRows: `168`
- trackedDirtyRows: `79`
- untrackedRows: `89`
- unknownRows: `0`
- privatePathLeakRows: `0`
- nextAction: `decide_keep_drop_or_clean_replay_per_bucket_before_public_rc`

## Bucket Counts

| bucket | rows |
|---|---:|
| `answer_runtime_or_query_stack` | `23` |
| `cli_mcp_public_surface_stack` | `21` |
| `docs_governance_stack` | `11` |
| `eval_or_test_support_stack` | `36` |
| `evidence_spine_or_source_contract_stack` | `29` |
| `infrastructure_or_core_stack` | `11` |
| `parser_artifact_side_stack` | `6` |
| `provider_hint_side_stack` | `8` |
| `research_objects_side_stack` | `12` |
| `source_ingest_or_library_stack` | `7` |
| `workspace_process_record` | `4` |

## Disposition Counts

| disposition | rows |
|---|---:|
| `compare_against_public_operator_cleanup_before_replay` | `21` |
| `exclude_from_public_rc_or_move_to_workspace_records` | `4` |
| `hold_for_parser_track` | `6` |
| `hold_outside_text_evidence_rc` | `20` |
| `map_to_owning_feature_before_replay` | `36` |
| `review_after_text_evidence_rc_candidate` | `23` |
| `review_as_separate_clean_replay_candidate` | `47` |
| `review_for_record_sync_only` | `11` |

## Dirty Rows

| status | bucket | disposition | path |
|---|---|---|---|
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `CHANGELOG.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `README.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/ARCHITECTURE.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/PROJECT_STATE.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/ai_120gb_ko_pipeline.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/curated_ai_source_ingestion.md` |
| ` M` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/guides/cli-commands.md` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/evidence-packet.v1.json` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/fixtures/evidence-packet.v1.fixture.json` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/README.md` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/fixtures/answer_quality_golden_cases.json` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/scripts/check_answer_quality_gate.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/scripts/report_legacy_runtime_readiness.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/answer_contracts.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/answer_execution_setup.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/answer_payload_builder.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/answer_rewrite.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/answer_verification.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/ask_v2.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/evidence_assembly.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/rag.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/rag_answer_evidence.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/rag_answer_runtime.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/rag_support.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/retrieval_fit.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/retrieval_pipeline.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/retrieval_pipeline_search_core.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/index_freshness.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/mcp/responses.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/search.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/task_context.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/core/schema_validator.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/domain/ai_papers/query_plan.py` |
| ` M` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/domain/ai_papers/representative.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/config.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/persistence/sqlite.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/persistence/store_registry.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/persistence/stores/__init__.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/persistence/vector.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/crawl_cmd.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/crawl_support.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/doctor_cmd.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/index_cmd.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/paper_cmd.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/paper_import_support.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/paper_materialization_runtime.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/paper_shared_runtime.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/search_cmd.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/main.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/mcp/handlers/search.py` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/mcp/tool_specs.py` |
| ` M` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/papers/manager.py` |
| ` M` | `parser_artifact_side_stack` | `hold_for_parser_track` | `knowledge_hub/papers/pymupdf_adapter.py` |
| ` M` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/papers/source_text.py` |
| ` M` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/vault/indexer.py` |
| ` M` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/web/ingest.py` |
| ` M` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `pyproject.toml` |
| ` M` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `scripts/check_release_smoke.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_answer_contracts_runtime.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_answer_orchestrator_services.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_answer_quality_gate.py` |
| ` M` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_answer_verification_guards.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_cli_smoke_contract.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_config.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_doctor_cmd.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_index_freshness.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_interfaces_cli_main.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_mcp_search_handler.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_mcp_server.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_ask_v2.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_import_csv.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_materialization_cli.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_query_plan.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_source_freshness.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_rag_runtime_services.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_rag_search.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_retrieval_pipeline_services.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_search_cmd.py` |
| ` M` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_task_context.py` |
| `??` | `workspace_process_record` | `exclude_from_public_rc_or_move_to_workspace_records` | `artifacts/` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/adr/2026-05-09-evidence-spine-and-auxiliary-signal-plane.md` |
| `??` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/contracts/` |
| `??` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/record-cleanup-2026-05-10.md` |
| `??` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/research/external-rag-reference-analysis-2026-05-06.md` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `docs/research_objects/` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/evidence-answer-verification.v1.json` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/fixtures/evidence-answer-verification.v1.fixture.json` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/fixtures/prepared-source-record.v1.fixture.json` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/fixtures/source-ledger-record.v1.fixture.json` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/prepared-source-record.v1.json` |
| `??` | `docs_governance_stack` | `review_for_record_sync_only` | `docs/schemas/research-context-result.v1.json` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `docs/schemas/source-ledger-record.v1.json` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `eval/knowledgeos/fixtures/research_objects/` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/scripts/build_paper_answer_failure_bank.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `eval/knowledgeos/scripts/evaluate_research_objects_commit_preview.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `eval/knowledgeos/scripts/evaluate_research_objects_operator.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `eval/knowledgeos/scripts/run_paper_slot_diagnostics.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/ask_v2_retrieval_bridge.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/graph_planning_context.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/paper_answer_materializer.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/paper_evidence_artifacts.py` |
| `??` | `parser_artifact_side_stack` | `hold_for_parser_track` | `knowledge_hub/ai/paper_table_extraction.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/ai/prepared_evidence_policy.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/result_table_semantics.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/ai/result_table_slot.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/application/evidence_answer_verifier.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/application/evidence_packet.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/application/evidence_packet_renderers.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/application/evidence_sufficiency.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/application/evidence_workbench.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/paper_answer_failure_bank.py` |
| `??` | `parser_artifact_side_stack` | `hold_for_parser_track` | `knowledge_hub/application/parsed_artifact_auditor.py` |
| `??` | `answer_runtime_or_query_stack` | `review_after_text_evidence_rc_candidate` | `knowledge_hub/application/research_context.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `knowledge_hub/application/research_objects.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/core/prepared_source_record.py` |
| `??` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/core/retrieval_units.py` |
| `??` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/core/source_ledger_record.py` |
| `??` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/graph/` |
| `??` | `infrastructure_or_core_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/infrastructure/index_profiles.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `knowledge_hub/infrastructure/persistence/stores/research_object_candidate_store.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/context_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/index_profile_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/library_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/packet_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/project_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/public_surface_cmd.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `knowledge_hub/interfaces/cli/commands/research_objects_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/commands/workbench_cmd.py` |
| `??` | `cli_mcp_public_surface_stack` | `compare_against_public_operator_cleanup_before_replay` | `knowledge_hub/interfaces/cli/library_scope.py` |
| `??` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/library/` |
| `??` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/papers/vault_links.py` |
| `??` | `source_ingest_or_library_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/project/` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `knowledge_hub/research_objects/` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `knowledge_hub/web/prepared_source.py` |
| `??` | `workspace_process_record` | `exclude_from_public_rc_or_move_to_workspace_records` | `reviews/` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `scripts/check_research_object_graph_planning_mvp.sh` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `scripts/check_research_objects_stack.sh` |
| `??` | `workspace_process_record` | `exclude_from_public_rc_or_move_to_workspace_records` | `tasks/` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `tests/fixtures/research_objects/` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_context_cmd.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_evidence_packet.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_evidence_sufficiency.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_evidence_workbench.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_index_profile_cmd.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_legacy_runtime_readiness_report.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_library_mode.py` |
| `??` | `parser_artifact_side_stack` | `hold_for_parser_track` | `tests/test_paper_table_extraction.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_paper_vault_links_cli.py` |
| `??` | `parser_artifact_side_stack` | `hold_for_parser_track` | `tests/test_parsed_artifact_auditor.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_prepared_source_contract.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_project_mode.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_provider_candidate_policy.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_adapter.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_dry_run.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_dry_run_replay.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_promotion_policy.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_shadow_assembly.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_shadow_hook.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_hint_shadow_hook_eval.py` |
| `??` | `provider_hint_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_provider_policy_shadow_eval.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_public_surface_cmd.py` |
| `??` | `parser_artifact_side_stack` | `hold_for_parser_track` | `tests/test_pymupdf_adapter.py` |
| `??` | `evidence_spine_or_source_contract_stack` | `review_as_separate_clean_replay_candidate` | `tests/test_rag_answer_evidence.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_research_context.py` |
| `??` | `research_objects_side_stack` | `hold_outside_text_evidence_rc` | `tests/test_research_object_candidates.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_result_table_semantics.py` |
| `??` | `eval_or_test_support_stack` | `map_to_owning_feature_before_replay` | `tests/test_result_table_slot_v1.py` |
| `??` | `workspace_process_record` | `exclude_from_public_rc_or_move_to_workspace_records` | `worklog/` |
