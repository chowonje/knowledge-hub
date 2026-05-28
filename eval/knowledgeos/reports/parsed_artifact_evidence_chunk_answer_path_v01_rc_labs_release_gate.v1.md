# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Labs Release Gate

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-labs-release-gate.v1`
- status: `ready`
- decision: `parsed_artifact_evidence_chunk_answer_path_v01_rc_labs_release_gate_ready`
- nextRecommendedTranche: `knowledge_hub_v01_rc_branch_pr_readiness_review`
- v01LabsReleaseGateDecision: `ready`
- publicDefaultDecision: `hold_public_default_promotion`
- releaseSmokeCheckedRows: `10`
- releaseSmokePassedRows: `10`
- publicHygieneIssueRows: `0`
- noAnswerScenarioRows: `3`
- noAnswerPassRows: `3`
- labsSurfaceSmokePassRows: `1`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `promotion_review`: `pass`; blockers=`none`
- `release_smoke`: `pass`; blockers=`none`
- `public_hygiene`: `pass`; blockers=`none`
- `no_answer_regression`: `pass`; blockers=`none`
- `labs_surface_smoke`: `pass`; blockers=`none`

## Release Smoke

- `top_help`: `pass`
- `setup`: `pass`
- `advanced_help`: `pass`
- `labs_help`: `pass`
- `papers_help`: `pass`
- `hidden_paper_help`: `pass`
- `capture_help`: `pass`
- `status`: `pass`
- `doctor`: `pass`
- `invalid_command`: `pass`

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
