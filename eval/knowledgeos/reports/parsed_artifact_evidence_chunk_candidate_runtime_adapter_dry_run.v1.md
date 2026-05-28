# Parsed Artifact Evidence Chunk Candidate Runtime Adapter Dry-run

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-candidate-runtime-adapter-dry-run.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_candidate_runtime_adapter_dry_run_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_candidate_runtime_adapter_implementation_opt_in`
- adapterId: `parsed_artifact_evidence_chunk_runtime_adapter_v1`
- resolvedPaperIds: `['1207.0580', '1301.3781']`
- inputRows: `1200`
- adapterDryRunReadyRows: `1200`
- candidateStoreRecordMatchedRows: `1200`
- excerptReadbackRows: `1200`
- positiveScenarioSelectedRows: `4`
- plannedEvidenceItemPreviewRows: `4`
- answerableRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- sourceSpanCreatedRows: `0`
- strictEvidenceRows: `0`
- citationGradeRows: `0`
- runtimeEvidenceRows: `0`
- answerVisibleRows: `0`
- answerGenerationRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- parserExecutionRows: `0`
- canonicalParsedArtifactWriteRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`

## Scenario Results

- `opt_in_off`: status=`skipped`, rowsAdded=`0`, skippedReason=`query_plan_opt_in_not_enabled`
- `opt_in_missing_resolved_paper_ids`: status=`skipped`, rowsAdded=`0`, skippedReason=`resolved_paper_ids_required`
- `source_type_not_paper`: status=`skipped`, rowsAdded=`0`, skippedReason=`source_type_not_paper`
- `opt_in_paper_resolved`: status=`applied_dry_run`, rowsAdded=`4`, skippedReason=``

## Adapter Dry-run Status

- `runtime_adapter_dry_run_ready_candidate_only`: `1200`
