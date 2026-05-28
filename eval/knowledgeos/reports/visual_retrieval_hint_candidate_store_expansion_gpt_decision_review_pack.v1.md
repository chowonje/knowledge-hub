# Visual Retrieval Hint Candidate Store Expansion GPT Decision Review Pack

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-gpt-decision-review-pack.v1`
- status: `ready`
- decision: `ready_for_manual_gpt_decision_recommendation_run`
- generatedAt: `2026-05-27T02:15:56Z`
- packId: `visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001`
- gptReviewRows: `24`
- batchRows: `3`
- completedGptRecommendationRows: `0`
- finalHumanDecisionRows: `0`
- candidateStoreWriteRows: `0`

## Boundary

- writes: `report_and_operator_prompt_files_only`
- modelCalls: `False`
- webModelCalls: `False`
- manualOperatorWebModelRunRequired: `True`
- vectorIndexing: `False`
- runtimeAnswerVisibleExposureRows: `0`
- strictEvidencePromotionRows: `0`

## Batch Files

| batch | rows | prompt | recommendation template |
|---:|---:|---|---|
| 1 | 8 | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_01_prompt.md` | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_01_recommendation_template.v1.json` |
| 2 | 8 | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_02_prompt.md` | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_02_recommendation_template.v1.json` |
| 3 | 8 | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_03_prompt.md` | `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_gpt_decision_review_pack_001/batch_03_recommendation_template.v1.json` |

## Warnings

- `This pack is not GPT output and contains recommendation templates.`
- `GPT recommendations must not be promoted to final human decisions automatically.`
- `GPT recommendations are not strict evidence, citation-grade evidence, or runtime answer-visible text.`
- `This pack makes no in-repo model/API/web call.`
