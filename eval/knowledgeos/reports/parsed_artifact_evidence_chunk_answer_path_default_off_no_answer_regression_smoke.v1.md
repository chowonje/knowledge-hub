# Parsed Artifact Evidence Chunk Answer Path Default-Off No-Answer Regression Smoke

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-default-off-no-answer-regression-smoke.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_default_off_no_answer_regression_smoke_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_surface_design`
- inputScenarioRows: `3`
- passRows: `3`
- failRows: `0`
- noAnswerRows: `3`
- answerableRows: `0`
- adapterAppliedRows: `0`
- adapterRowsAdded: `0`
- localFakeLlmCallRows: `0`
- externalLlmCallRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- sourceSpanCreatedRows: `0`
- strictEvidenceRows: `0`
- parserExecutionRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- canonicalParsedArtifactWriteRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- publicCliFlagRows: `0`
- defaultOnRows: `0`
- externalLlmCallRows: `0`
- modelApiCallRows: `0`

## Scenarios

- `opt_in_absent` source=`paper` payload=`no_result` adapter=`disabled` reason=`query_plan_opt_in_not_enabled` pass=`True`
- `opt_in_missing_resolved_paper` source=`paper` payload=`no_result` adapter=`skipped` reason=`resolved_paper_ids_required` pass=`True`
- `opt_in_non_paper_source` source=`web` payload=`no_result` adapter=`skipped` reason=`source_type_not_paper` pass=`True`
