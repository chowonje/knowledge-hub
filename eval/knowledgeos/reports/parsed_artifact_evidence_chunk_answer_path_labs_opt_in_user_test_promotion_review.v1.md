# Parsed Artifact Evidence Chunk Answer Path Labs Opt-in User Test Promotion Review

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-labs-opt-in-user-test-promotion-review.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_labs_opt_in_user_test_promotion_review_ready`
- nextRecommendedTranche: `parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate`
- v01ScopeDecision: `labs_limited_rc_candidate_ready`
- publicDefaultDecision: `hold_public_default_promotion`
- capturedCommandRows: `5`
- outputCapturePassRows: `5`
- jsonAssertionPassRows: `47`
- labsLimitedPromotionReadyRows: `1`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Decisions

- `khub_labs_paper_evidence_chunk_ask`: `ready_for_v01_labs_limited_release_gate`; publicDefaultPromotionAllowed=`False`
- `khub_ask_and_default_mcp`: `hold_public_default_promotion`; publicDefaultPromotionAllowed=`False`

## Required Next Checks

- `release_smoke`: `pending`
- `public_hygiene`: `pending`
- `public_surface_design`: `pending`
- `no_answer_regression`: `pending`

## Mutation Guarantees

- candidateStoreWriteRows: `0`
- sourceSpanCreatedRows: `0`
- strictEvidenceRows: `0`
- citationGradeEvidenceRows: `0`
- runtimeEvidenceRows: `0`
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
- judgeModelCallRows: `0`
