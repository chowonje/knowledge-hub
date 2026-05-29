# KnowledgeOS v0.1 RC Corpus-Scale Answer Quality Controlled Execution

- schema: `knowledge-hub.product.knowledgeos-v01-rc-corpus-scale-answer-quality-controlled-execution.v1`
- status: `blocked`
- decision: `knowledgeos_v01_rc_corpus_scale_answer_quality_controlled_execution_blocked`
- nextRecommendedTranche: `corpus_scale_answer_quality_answerability_gate_repair`
- attemptedCaseRows: `50`
- executionPassRows: `1`
- executionFailRows: `49`
- unexpectedAnswerableRows: `49`
- noAnswerSafetyFailRows: `49`
- localFakeLlmCallRows: `49`
- liveAnswerExecutionRows: `50`
- answerPathInvokedRows: `50`
- externalLlmCallRows: `0`
- modelApiCallRows: `0`
- judgeModelCallRows: `0`
- publicDefaultPromotionHeldRows: `1`
- privatePathLeakRows: `0`
- schemaViolationCount: `3`

## Checks

- `live_runner_dry_run`: `pass`; blockers=`none`
- `unsafe_counters`: `pass`; blockers=`none`
- `no_answer_safety`: `fail`; blockers=`no_answer_safety_fail_rows:49`
- `public_default_hold`: `pass`; blockers=`none`

## Failure Preview

- `complex-paper-qa-seed-20260520-q001` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q002` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q003` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q004` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q005` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q006` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q007` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q008` category=`table_numeric_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q009` category=`table_numeric_qa` expectation=`expected_no_answer` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q010` category=`table_numeric_qa` expectation=`expected_no_answer` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q011` category=`equation_citation_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`
- `complex-paper-qa-seed-20260520-q012` category=`equation_citation_qa` expectation=`blocked_until_structured_evidence` observedAnswerable=`True` failures=`['answer_support_failed', 'answerability_expectation_failed', 'citation_provenance_failed', 'no_answer_safety_failed', 'unexpected_answerable_for_no_answer_or_blocked_case', 'unexpected_llm_call_for_no_answer_or_blocked_case']`

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
