# Visual Retrieval Hint Candidate Store Expansion Human/Product Decision Record

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-human-product-decision-record.v1`
- status: `decision_record_template_ready`
- decision: `manual_human_product_decisions_required`
- generatedAt: `2026-05-26T16:45:24Z`
- sourceReviewReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_review.v1.json`
- decisionRows: `24`
- defaultHoldRows: `24`
- humanDecisionRows: `0`
- approvedRows: `0`
- applyDesignCandidateRows: `0`
- candidateStoreWriteRows: `0`
- blockedRows: `0`
- privatePathLeakRows: `0`

## Decision Boundary

- defaultDecision: `hold_pending_human_product_review`
- applyAllowedByThisReport: `False`
- actualApprovalByThisReport: `False`
- plannedStoreRef: `papers_dir/visual_retrieval_hints/visual_retrieval_hint_candidates.v1.jsonl`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- indexMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- strictEvidencePromotionRows: `0`
- answerabilityGateBypassRows: `0`

## Decision Rows

| # | paperId | type | page | decision | accepted | applyDesignCandidate | hintCandidateId |
|---:|---|---|---:|---|---|---|---|
| 1 | clip-2021 | image_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:image_region:2:e25902a7066b85fb` |
| 2 | mae-2021 | image_region | 1 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:image_region:1:d54434235dc1f73e` |
| 3 | alexnet-2012 | image_region | 6 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:alexnet-2012:image_region:6:6bc72fab3308cc56` |
| 4 | clip-2021 | image_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:image_region:2:0d674a922cfdab51` |
| 5 | mae-2021 | image_region | 1 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:image_region:1:e3eb86ab7dd4815d` |
| 6 | alexnet-2012 | image_region | 8 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:alexnet-2012:image_region:8:aebea8a49393b7e9` |
| 7 | clip-2021 | image_region | 15 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:image_region:15:b2eef0626d7910cf` |
| 8 | mae-2021 | image_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:image_region:2:47e41fe986d634ac` |
| 9 | clip-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:figure_caption_region:2:791bd02c68004e94` |
| 10 | mae-2021 | figure_caption_region | 1 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:figure_caption_region:1:7f1d38a62ebc3ebe` |
| 11 | resnet-2015 | figure_caption_region | 6 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:resnet-2015:figure_caption_region:6:beeb9fb9ad38518d` |
| 12 | clip-2021 | figure_caption_region | 3 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:figure_caption_region:3:7de4aa3a082925a4` |
| 13 | mae-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:figure_caption_region:2:b1c1635f3ea9867d` |
| 14 | resnet-2015 | figure_caption_region | 8 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:resnet-2015:figure_caption_region:8:5983f29673800b6b` |
| 15 | clip-2021 | figure_caption_region | 5 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:figure_caption_region:5:aa1447f2ba9526ae` |
| 16 | mae-2021 | figure_caption_region | 2 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:figure_caption_region:2:eea2a43fb3574de9` |
| 17 | clip-2021 | table_region | 7 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:table_region:7:202c4c0be3960b79` |
| 18 | mae-2021 | table_region | 5 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:table_region:5:626e13443e1ee40c` |
| 19 | resnet-2015 | table_region | 6 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:resnet-2015:table_region:6:7adc39075ec4efe2` |
| 20 | clip-2021 | table_region | 17 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:table_region:17:0c93a8dbedbc0a03` |
| 21 | mae-2021 | table_region | 5 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:mae-2021:table_region:5:eeae9d1ecf9c1122` |
| 22 | clip-2021 | equation_region | 1 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:equation_region:1:1578f0d850ce7830` |
| 23 | clip-2021 | equation_region | 5 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:equation_region:5:a85a9e64dfa04951` |
| 24 | clip-2021 | equation_region | 5 | `hold_pending_human_product_review` | `False` | `False` | `visual-retrieval-hint:clip-2021:equation_region:5:2d6d39c3bbc50090` |

## Warnings

- `This report is a decision-record template/validation only; it is not an apply step.`
- `The default generated record holds every row pending human/product review.`
- `Approved rows, if supplied later, only become apply-design candidates and remain unindexed, not runtime-visible, and non-evidence.`
