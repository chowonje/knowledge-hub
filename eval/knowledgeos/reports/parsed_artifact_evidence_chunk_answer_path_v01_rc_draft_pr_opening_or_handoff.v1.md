# Parsed Artifact Evidence Chunk Answer Path v0.1 RC Draft PR Handoff

- schema: `knowledge-hub.paper.parsed-artifact-evidence-chunk-answer-path-v01-rc-draft-pr-opening-or-handoff.v1`
- status: `ready`
- decision: `knowledge_hub_v01_rc_draft_pr_opening_or_handoff_ready`
- nextRecommendedTranche: `operator_push_and_open_draft_pr_or_request_codex_pr_creation`
- operatorAction: `push_branch_and_open_draft_pr`
- readyForDraftPrCreation: `True`
- readyForMerge: `False`
- currentBranchAheadCommitRows: `34`
- currentBranchBehindCommitRows: `0`
- currentDirtyRows: `9`
- openPrRows: `0`
- publicDefaultPromotionReadyRows: `0`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Prepared Draft PR Body

# KnowledgeOS v0.1 RC labs evidence chunk preview

## Summary

- Defines the KnowledgeOS v0.1 RC product scope as a section/paragraph evidence-first paper QA and comparison Research Preview.
- Adds the parsed-artifact evidence chunk candidate path, opt-in runtime adapter, labs-only CLI/MCP preview surface, quality/user-test reports, and release/readiness gates.
- Keeps public/default `khub ask` and default MCP promotion held; labs activation remains explicit.

## Scope

- Base: `main`
- Head: `codex/next-implementation-20260528`
- Current branch commits ahead of origin/main: `34`
- Changed files recorded by readiness review: `184`

## Verification

- release smoke passed rows: `10`
- public hygiene issue rows: `0`
- no-answer pass rows: `3`
- public default promotion ready rows: `0`
- public default promotion held rows: `1`

## Non-Goals

- Does not make parsed-artifact evidence chunks default-on.
- Does not promote visual/table/equation/figure evidence to public/default answerability.
- Does not claim merge readiness; this is a draft PR review handoff.

## Reviewer Notes

- Review public/default surface boundaries first.
- Check schema-backed reports before discussing answer-quality expansion.
- Treat public/default promotion as a later explicit tranche.


## Operator Commands

- `push_branch`: `git push -u origin codex/next-implementation-20260528`; executedByThisReport=`False`
- `create_draft_pr`: `gh pr create --draft --base main --head codex/next-implementation-20260528 --title "KnowledgeOS v0.1 RC labs evidence chunk preview" --body-file <prepared-pr-body.md>`; executedByThisReport=`False`

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
