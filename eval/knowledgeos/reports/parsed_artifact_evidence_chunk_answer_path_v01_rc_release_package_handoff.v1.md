# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Release Package Handoff

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-release-package-handoff.v1`
- status: `ready`
- decision: `knowledge_hub_v01_rc_release_package_handoff_ready`
- nextRecommendedTranche: `operator_push_and_open_draft_pr_or_corpus_scale_quality_gate`
- releasePackageHandoff: `ready_for_operator_push_and_draft_pr`
- publicDefaultDecision: `hold_public_default_promotion`
- mergeDecision: `not_ready_for_merge`
- releaseTagDecision: `not_ready_for_release_tag`
- currentBranchAheadCommitRows: `3`
- currentBranchBehindCommitRows: `0`
- currentBlockingDirtyRows: `0`
- openPrRows: `0`
- draftPrBodyRows: `1`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- readyForMergeRows: `0`
- readyForReleaseTagRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Draft PR

- title: `KnowledgeOS v0.1 RC Research Preview package`
- base: `main`
- head: `codex/v01-rc-public-default-promotion-decision-gate-20260529`

## Operator Commands

- `push_branch`: `git push -u origin codex/v01-rc-public-default-promotion-decision-gate-20260529`
- `open_draft_pr`: `gh pr create --draft --base main --head codex/v01-rc-public-default-promotion-decision-gate-20260529 --title "KnowledgeOS v0.1 RC Research Preview package" --body-file <prepared-body-file>`

## Checks

- `release_notes_review`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `current_branch_state`: `pass`; blockers=`none`
- `github_pr_state`: `pass`; blockers=`none`

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
- releaseTagRows: `0`
- packagePublishRows: `0`
