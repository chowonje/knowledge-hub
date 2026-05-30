# KnowledgeOS v0.1 RC Corpus-Scale Positive Section/Paragraph Seed

- schema: `knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-positive-section-paragraph-seed.v1`
- status: `ready`
- decision: `knowledgeos_v01_rc_corpus_scale_answer_quality_positive_section_paragraph_seed_ready`
- nextRecommendedTranche: `corpus_scale_answer_quality_positive_answer_execution_gate`
- inputCaseRows: `50`
- eligibleSectionParagraphProbeRows: `13`
- positiveSeedRows: `7`
- positiveMethodComparisonRows: `3`
- positiveLimitationRows: `4`
- positiveProbeHeldRows: `6`
- heldExpectedNoAnswerRows: `17`
- heldStructuredModalityRows: `34`
- controlledExecutionUnexpectedAnswerableRows: `0`
- controlledExecutionNoAnswerSafetyFailRows: `0`
- citationCount: `20`
- evidencePacketContractSpanRows: `20`
- localFakeLlmCallRows: `13`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `0`

## Checks

- `post_merge_convergence`: `pass`; blockers=`none`
- `controlled_execution_no_answer_safety`: `pass`; blockers=`none`
- `live_runner_dry_run`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `positive_seed_selection`: `pass`; blockers=`none`

## Positive Seed Rows

- `complex-paper-qa-seed-20260520-q028` category=`method_comparison_qa` papers=`['2404.16130', '2410.05779']` citations=`4` spans=`4` score=`1.0`
- `complex-paper-qa-seed-20260520-q029` category=`method_comparison_qa` papers=`['1706.03762', '2312.00752']` citations=`4` spans=`4` score=`1.0`
- `complex-paper-qa-seed-20260520-q034` category=`method_comparison_qa` papers=`['2005.11401', '2310.11511']` citations=`4` spans=`4` score=`1.0`
- `complex-paper-qa-seed-20260520-q035` category=`limitation_qa` papers=`['1706.03762']` citations=`2` spans=`2` score=`1.0`
- `complex-paper-qa-seed-20260520-q036` category=`limitation_qa` papers=`['2312.00752']` citations=`2` spans=`2` score=`1.0`
- `complex-paper-qa-seed-20260520-q037` category=`limitation_qa` papers=`['2010.11929']` citations=`2` spans=`2` score=`1.0`
- `complex-paper-qa-seed-20260520-q039` category=`limitation_qa` papers=`['2310.11511']` citations=`2` spans=`2` score=`1.0`

## Held Probe Rows

- `complex-paper-qa-seed-20260520-q027` category=`method_comparison_qa` qualityGrade=`partial` failures=`['support_term_gap']`
- `complex-paper-qa-seed-20260520-q030` category=`method_comparison_qa` qualityGrade=`fail` failures=`['citation_or_span_count_below_minimum', 'source_coverage_gap', 'support_term_gap']`
- `complex-paper-qa-seed-20260520-q031` category=`method_comparison_qa` qualityGrade=`partial` failures=`['support_term_gap']`
- `complex-paper-qa-seed-20260520-q032` category=`method_comparison_qa` qualityGrade=`partial` failures=`['support_term_gap']`
- `complex-paper-qa-seed-20260520-q033` category=`method_comparison_qa` qualityGrade=`partial` failures=`['support_term_gap']`
- `complex-paper-qa-seed-20260520-q038` category=`limitation_qa` qualityGrade=`partial` failures=`['support_term_gap']`

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
- githubPrMutationRows: `0`
- mergeRows: `0`
- branchDeletionRows: `0`
- releaseTagRows: `0`
- packagePublishRows: `0`
- rawPayloadPersistedRows: `0`
- defaultMcpToolRows: `0`
- defaultKhubAskRouteRows: `0`
