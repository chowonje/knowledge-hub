# Visual Retrieval Hint Candidate Store Expansion Design

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run`
- generatedAt: `2026-05-27T07:44:25Z`
- sourceValidationReport: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.validation.v1.json`
- sourceValidationRows: `24`
- candidateRows: `24`
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
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:badaf33f92dc4c28` | `False` | `False` |
| 2 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` | `False` | `False` |
| 3 | resnet-2015 | table_region | 7 | `visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6` | `False` | `False` |
| 4 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d` | `False` | `False` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:a31e2f20793cd157` | `False` | `False` |
| 6 | resnet-2015 | table_region | 8 | `visual-layout:resnet-2015:table_region:8:499a16649cac8d1b` | `False` | `False` |
| 7 | clip-2021 | table_region | 25 | `visual-layout:clip-2021:table_region:25:795f0bd1367145d9` | `False` | `False` |
| 8 | mae-2021 | table_region | 11 | `visual-layout:mae-2021:table_region:11:229ba015a36e1bc6` | `False` | `False` |
| 9 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:f7695e0650ad9755` | `False` | `False` |
| 10 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:fb84bd6d006dd177` | `False` | `False` |
| 11 | clip-2021 | figure_caption_region | 11 | `visual-layout:clip-2021:figure_caption_region:11:d9042e6c82546c0e` | `False` | `False` |
| 12 | mae-2021 | figure_caption_region | 7 | `visual-layout:mae-2021:figure_caption_region:7:5e6020a5e57aa862` | `False` | `False` |
| 13 | clip-2021 | figure_caption_region | 12 | `visual-layout:clip-2021:figure_caption_region:12:36479d77a4a94f87` | `False` | `False` |
| 14 | mae-2021 | figure_caption_region | 13 | `visual-layout:mae-2021:figure_caption_region:13:d74e50f7fc0acd19` | `False` | `False` |
| 15 | clip-2021 | figure_caption_region | 13 | `visual-layout:clip-2021:figure_caption_region:13:c57a4c0a359c0b1e` | `False` | `False` |
| 16 | mae-2021 | figure_caption_region | 14 | `visual-layout:mae-2021:figure_caption_region:14:78c3aa6879bfbb7e` | `False` | `False` |
| 17 | clip-2021 | equation_region | 15 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` | `False` | `False` |
| 18 | clip-2021 | equation_region | 16 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` | `False` | `False` |
| 19 | clip-2021 | equation_region | 17 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` | `False` | `False` |
| 20 | clip-2021 | equation_region | 19 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` | `False` | `False` |
| 21 | clip-2021 | layout_region | 16 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` | `False` | `False` |
| 22 | mae-2021 | layout_region | 12 | `visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa` | `False` | `False` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` | `False` | `False` |
| 24 | resnet-2015 | layout_region | 6 | `visual-layout:resnet-2015:layout_region:6:99aa935003a72db9` | `False` | `False` |

## Warnings

- `This is a design report only; the planned candidate store is not written.`
- `All projected rows remain retrieval_hint_only, not evidence.`
- `Whole-image or whole-page GPT/VLM batches require a later pack-design gate.`
