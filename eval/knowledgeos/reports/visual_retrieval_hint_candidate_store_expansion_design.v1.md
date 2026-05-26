# Visual Retrieval Hint Candidate Store Expansion Design

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run`
- generatedAt: `2026-05-26T15:56:37Z`
- sourceValidationReport: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.validation.v1.json`
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
| 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` | `False` | `False` |
| 2 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` | `False` | `False` |
| 3 | alexnet-2012 | image_region | 6 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` | `False` | `False` |
| 4 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` | `False` | `False` |
| 5 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` | `False` | `False` |
| 6 | alexnet-2012 | image_region | 8 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` | `False` | `False` |
| 7 | clip-2021 | image_region | 15 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` | `False` | `False` |
| 8 | mae-2021 | image_region | 2 | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` | `False` | `False` |
| 9 | clip-2021 | figure_caption_region | 2 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` | `False` | `False` |
| 10 | mae-2021 | figure_caption_region | 1 | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` | `False` | `False` |
| 11 | resnet-2015 | figure_caption_region | 6 | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` | `False` | `False` |
| 12 | clip-2021 | figure_caption_region | 3 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` | `False` | `False` |
| 13 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` | `False` | `False` |
| 14 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` | `False` | `False` |
| 15 | clip-2021 | figure_caption_region | 5 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` | `False` | `False` |
| 16 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` | `False` | `False` |
| 17 | clip-2021 | table_region | 7 | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` | `False` | `False` |
| 18 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` | `False` | `False` |
| 19 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` | `False` | `False` |
| 20 | clip-2021 | table_region | 17 | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` | `False` | `False` |
| 21 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` | `False` | `False` |
| 22 | clip-2021 | equation_region | 1 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` | `False` | `False` |
| 23 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` | `False` | `False` |
| 24 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` | `False` | `False` |

## Warnings

- `This is a design report only; the planned candidate store is not written.`
- `All projected rows remain retrieval_hint_only, not evidence.`
- `Whole-image or whole-page GPT/VLM batches require a later pack-design gate.`
