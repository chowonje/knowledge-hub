# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Live Runner Design

- schema: `knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-live-runner-design.v1`
- status: `ready`
- decision: `knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner_design_ready`
- nextRecommendedTranche: `corpus_scale_answer_quality_live_runner_dry_run`
- runnerId: `knowledgeos_v01_rc_corpus_scale_answer_quality_live_runner`
- mode: `design_only`
- executionSurface: `labs_internal_only`
- seedPaperRows: `20`
- seedQuestionRows: `50`
- plannedRunnerQuestionRows: `50`
- liveAnswerExecutionRows: `0`
- answerQualityMeasuredRows: `0`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Execution Phases

- `load_seed_cases`: `planned`; Load the fixed 20-paper / 50-question corpus-scale seed and preserve case ids.
- `invoke_labs_answer_path`: `planned`; Invoke only an explicit labs/internal answer path with paper ids and adapter opt-in.
- `capture_sanitized_outputs`: `planned`; Persist only schema-backed, sanitized per-case scores and redacted diagnostics.
- `score_deterministic_axes`: `planned`; Score answerability, no-answer safety, provenance, source coverage, and support without a judge model.
- `hold_public_default`: `planned`; Keep public khub ask and default MCP promotion blocked until the measured gate is green.

## Score Axes

- `answerability`: `planned`; minimumRows=`50`; Every answerable decision must be backed by eligible evidence; unsupported cases must abstain.
- `no_answer_safety`: `planned`; minimumRows=`50`; Expected no-answer cases must remain no-answer with no fallback-only elevation.
- `citation_provenance`: `planned`; minimumRows=`50`; Answer citations must carry source hash and chars/page/bbox provenance before scoring as supported.
- `source_coverage`: `planned`; minimumRows=`50`; Resolved paper ids and cited source ids must cover each case's expected source set.
- `answer_support`: `planned`; minimumRows=`50`; Claim support is measured against selected evidence terms and citation spans, not final-answer fluency.

## Checks

- `corpus_scale_answer_quality_gate`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `public_default_hold`: `pass`; blockers=`none`

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
- liveAnswerExecutionRows: `0`
- answerPathInvokedRows: `0`
- answerGeneratedRows: `0`
- answerQualityMeasuredRows: `0`
- answerQualityScoreComputedRows: `0`
- runnerDryRunRows: `0`
- llmCallRows: `0`
- githubPrMutationRows: `0`
- mergeRows: `0`
- branchDeletionRows: `0`
- releaseTagRows: `0`
- packagePublishRows: `0`
- rawPayloadPersistedRows: `0`
- defaultMcpToolRows: `0`
- defaultKhubAskRouteRows: `0`
