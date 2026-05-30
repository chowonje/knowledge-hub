# KnowledgeOS v0.1 RC Research Preview Release Readiness Decision Gate

- schema: `knowledge-hub.product.knowledgeos-v01-rc-research-preview-release-readiness-decision-gate.v1`
- status: `ready`
- decision: `knowledgeos_v01_rc_research_preview_release_readiness_decision_gate_ready`
- nextRecommendedTranche: `operator_release_package_handoff_or_branch_cleanup_decision`
- researchPreviewDecision: `ready_for_controlled_research_preview_release`
- publicDefaultDecision: `hold_public_default_promotion`
- defaultSurfaceDecision: `do_not_enable_default_ask_or_default_mcp`
- generalReleaseDecision: `not_ready_for_general_release`
- strongestSupportedClaim: `section_paragraph_positive_slice_ready_with_conservative_no_answer_boundary`
- researchPreviewReleaseReadyRows: `1`
- positiveSectionParagraphQualityCompleteRows: `1`
- positiveAnswerPassRows: `7`
- provenancePassRows: `7`
- releaseSmokePassedRows: `10`
- publicHygieneIssueRows: `0`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- generalRcReadyRows: `0`
- corpusScaleClaimProvenRows: `0`
- tableEquationFigureDefaultEvidenceRows: `0`
- visualHintAnswerEvidenceRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `positive_section_paragraph_quality_complete_review`: `pass`; blockers=`none`
- `product_definition`: `pass`; blockers=`none`
- `release_notes_required_phrases`: `pass`; blockers=`none`
- `release_notes_forbidden_phrases`: `pass`; blockers=`none`
- `private_path_hygiene`: `pass`; blockers=`none`

## Next Actions

- `operator_release_package_handoff`: `recommended_not_applied`; requiresExplicitApproval=`True`; Prepare or review the Research Preview package handoff without enabling public/default surfaces.
- `branch_cleanup_decision`: `recommended_not_applied`; requiresExplicitApproval=`True`; Decide whether to clean up merged quality branches; do not delete branches in this gate.
- `public_default_promotion`: `held`; requiresExplicitApproval=`True`; Keep public/default `khub ask` and default MCP promotion held until corpus-scale quality evidence passes.

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
- mergeRows: `0`
- branchDeletionRows: `0`
- releaseTagRows: `0`
- packagePublishRows: `0`
- rawGithubPayloadPersistedRows: `0`
- defaultMcpToolRows: `0`
- defaultKhubAskRouteRows: `0`
