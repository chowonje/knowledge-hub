# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Post-Merge Convergence

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-post-merge-convergence.v1`
- status: `ready`
- decision: `knowledge_hub_v01_rc_research_preview_post_merge_convergence_ready`
- nextRecommendedTranche: `knowledge_hub_v01_rc_public_default_promotion_decision_gate_or_release_notes`
- postMergeConvergence: `research_preview_rc_candidate`
- publicDefaultDecision: `hold_public_default_promotion`
- generalRcDecision: `blocked_pending_public_default_and_corpus_scale_evidence`
- prMergedRows: `1`
- mainMergeCommitMatchRows: `1`
- liveReleaseSmokePassedRows: `10`
- publicHygieneIssueRows: `0`
- noAnswerPassRows: `3`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- researchPreviewRcCandidateRows: `1`
- generalRcReadyRows: `0`
- corpusScaleClaimProvenRows: `0`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `release_gate`: `pass`; blockers=`none`
- `pre_merge_reports`: `pass`; blockers=`none`
- `git_main_state`: `pass`; blockers=`none`
- `github_pr_169`: `pass`; blockers=`none`
- `live_checks`: `pass`; blockers=`none`

## Remaining Blockers

- `P1` `public_default_promotion_held`: Evidence chunk answer path remains labs-only; public/default khub ask and default MCP promotion are still held.
- `P2` `corpus_scale_quality_evidence_missing`: Current quality evidence is tranche-scale; corpus-scale v0.1 claims still need a broader gate.
- `P2` `pre_merge_state_reports_historical_only`: Branch readiness and draft handoff reports are now historical snapshots after PR #169 merge.

## Historical Pre-Merge Reports

- `eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_branch_pr_readiness_review.v1.json`: `pre_merge_branch_pr_readiness_snapshot`
- `eval/knowledgeos/reports/parsed_artifact_evidence_chunk_answer_path_v01_rc_draft_pr_opening_or_handoff.v1.json`: `pre_merge_draft_pr_handoff_snapshot`

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
