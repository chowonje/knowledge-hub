# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Research Preview Release Notes Review

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-research-preview-release-notes-review.v1`
- status: `ready`
- decision: `knowledge_hub_v01_rc_research_preview_release_notes_review_ready`
- nextRecommendedTranche: `knowledge_hub_v01_rc_release_package_handoff_or_corpus_scale_quality_gate`
- researchPreviewReleaseNotesReady: `True`
- publicDefaultDecision: `hold_public_default_promotion`
- defaultSurfaceDecision: `do_not_enable_default_ask_or_default_mcp`
- releaseNotesPathAllowedRows: `1`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- requiredPhrasePassRows: `11`
- forbiddenPhraseRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `promotion_decision_gate`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `release_notes_required_phrases`: `pass`; blockers=`none`
- `release_notes_forbidden_phrases`: `pass`; blockers=`none`

## Required Phrases

- `Knowledge Hub v0.1 RC Research Preview`: `True`
- `Research Preview`: `True`
- `discover -> index -> search/ask -> evidence review`: `True`
- `section/paragraph evidence`: `True`
- `public/default promotion remains held`: `True`
- `khub labs paper evidence-chunk-ask`: `True`
- `default MCP`: `True`
- `publicDefaultPromotionReadyRows=0`: `True`
- `publicDefaultPromotionHeldRows=1`: `True`
- `generalRcReadyRows=0`: `True`
- `corpusScaleClaimProvenRows=0`: `True`

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
- pushRows: `0`
- githubPrMutationRows: `0`
- branchDeletionRows: `0`
- rawGithubPayloadPersistedRows: `0`
- defaultMcpToolRows: `0`
- defaultKhubAskRouteRows: `0`
