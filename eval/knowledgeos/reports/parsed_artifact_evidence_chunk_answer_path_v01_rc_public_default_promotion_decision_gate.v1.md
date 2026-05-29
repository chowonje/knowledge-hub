# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Public/Default Promotion Decision Gate

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-public-default-promotion-decision-gate.v1`
- status: `ready`
- decision: `knowledge_hub_v01_rc_public_default_promotion_decision_gate_ready`
- nextRecommendedTranche: `knowledge_hub_v01_rc_research_preview_release_notes_or_corpus_scale_quality_gate`
- researchPreviewDecision: `release_notes_path_allowed`
- publicDefaultDecision: `hold_public_default_promotion`
- defaultSurfaceDecision: `do_not_enable_default_ask_or_default_mcp`
- corpusScaleDecision: `blocked_pending_corpus_scale_quality_gate`
- researchPreviewRcCandidateRows: `1`
- releaseNotesPathAllowedRows: `1`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- generalRcReadyRows: `0`
- corpusScaleClaimProvenRows: `0`
- qualityEvalCaseRows: `4`
- realAnswerSmokeCaseRows: `2`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `post_merge_convergence`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `public_default_promotion`: `pass`; blockers=`none`

## Remaining Blockers

- `P1` `public_default_promotion_held`: The evidence chunk answer path stays labs/research-preview only; public khub ask and default MCP are not promoted.
- `P1` `corpus_scale_quality_evidence_missing`: The current quality evidence covers a narrow gate, not corpus-scale answer quality for a broad default release.
- `P2` `default_surface_activation_requires_separate_gate`: Any default CLI/MCP activation must be implemented in a later explicit gate after corpus-scale evidence is available.

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
