# Visual Retrieval Hint Candidate Store Design

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_dry_run`
- generatedAt: `2026-05-26T13:20:29Z`
- sourceValidationReport: `eval/knowledgeos/reports/visual_annotation_web_output_001.validation.v1.json`
- sourceValidationRows: `18`
- candidateRows: `18`
- indexEligibleRows: `0`
- runtimeVisibleRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- candidateStoreMutationRows: `0`
- vectorIndexing: `False`
- indexEligibleRows: `0`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- answerabilityGateBypassRows: `0`

## Whole-Image Timing

- currentTrancheSendsWholeImagesToGpt: `False`
- earliestRecommendedTranche: `visual_full_image_annotation_pack_design`
- recommendedMaxRowsPerBatch: `24`
- rule: `Only after a separate pack-design gate, and only for candidates where context crops are insufficient, uncertainty is high, or image-region candidates need object/layout inspection.`

## Candidate Rows

| # | paperId | type | page | sourceCandidateId | indexEligible | runtimeVisible |
|---:|---|---|---:|---|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `visual-layout:alexnet-2012:figure_caption_region:3:5e4d346d4c7b2c53` | `False` | `False` |
| 2 | alexnet-2012 | equation_region | 4 | `visual-layout:alexnet-2012:equation_region:4:f9d505b3e6aabed6` | `False` | `False` |
| 3 | alexnet-2012 | equation_region | 4 | `visual-layout:alexnet-2012:equation_region:4:79cf21420d190e08` | `False` | `False` |
| 4 | alexnet-2012 | figure_caption_region | 5 | `visual-layout:alexnet-2012:figure_caption_region:5:03c450b40ade0263` | `False` | `False` |
| 5 | alexnet-2012 | figure_caption_region | 6 | `visual-layout:alexnet-2012:figure_caption_region:6:f8996246c800f4f1` | `False` | `False` |
| 6 | alexnet-2012 | equation_region | 6 | `visual-layout:alexnet-2012:equation_region:6:81ead85a04bce51a` | `False` | `False` |
| 7 | alexnet-2012 | table_region | 7 | `visual-layout:alexnet-2012:table_region:7:0777b7d11b932456` | `False` | `False` |
| 8 | alexnet-2012 | table_region | 7 | `visual-layout:alexnet-2012:table_region:7:b9f742e1da360378` | `False` | `False` |
| 9 | alexnet-2012 | figure_caption_region | 8 | `visual-layout:alexnet-2012:figure_caption_region:8:889b5135e8e9ab9e` | `False` | `False` |
| 10 | resnet-2015 | figure_caption_region | 1 | `visual-layout:resnet-2015:figure_caption_region:1:aba2bf94ef1625a2` | `False` | `False` |
| 11 | resnet-2015 | figure_caption_region | 2 | `visual-layout:resnet-2015:figure_caption_region:2:9687efdba9452012` | `False` | `False` |
| 12 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:0e80570b407b6793` | `False` | `False` |
| 13 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:69248b1db8c80503` | `False` | `False` |
| 14 | resnet-2015 | equation_region | 3 | `visual-layout:resnet-2015:equation_region:3:7068ad040e2d0243` | `False` | `False` |
| 15 | resnet-2015 | figure_caption_region | 4 | `visual-layout:resnet-2015:figure_caption_region:4:0b3754b79959d64e` | `False` | `False` |
| 16 | resnet-2015 | figure_caption_region | 5 | `visual-layout:resnet-2015:figure_caption_region:5:666c24606607fbbf` | `False` | `False` |
| 17 | resnet-2015 | table_region | 5 | `visual-layout:resnet-2015:table_region:5:749ede3c6c93e4fa` | `False` | `False` |
| 18 | resnet-2015 | table_region | 5 | `visual-layout:resnet-2015:table_region:5:6f67c711d387611a` | `False` | `False` |

## Warnings

- `This is a design report only; the planned candidate store is not written.`
- `All projected rows remain retrieval_hint_only, not evidence.`
- `Whole-image or whole-page GPT/VLM batches require a later pack-design gate.`
