# Visual Retrieval Hint Candidate Store Expansion Design

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-design.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_dry_run`
- generatedAt: `2026-05-27T03:17:05Z`
- sourceValidationReport: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.validation.v1.json`
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
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:4bad77d0856486fe` | `False` | `False` |
| 2 | mae-2021 | table_region | 7 | `visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e` | `False` | `False` |
| 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8` | `False` | `False` |
| 4 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98` | `False` | `False` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:33d1507d0cd15829` | `False` | `False` |
| 6 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5` | `False` | `False` |
| 7 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:71b8cf43d8079375` | `False` | `False` |
| 8 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b` | `False` | `False` |
| 9 | clip-2021 | figure_caption_region | 7 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` | `False` | `False` |
| 10 | mae-2021 | figure_caption_region | 3 | `visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa` | `False` | `False` |
| 11 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201` | `False` | `False` |
| 12 | clip-2021 | figure_caption_region | 8 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` | `False` | `False` |
| 13 | mae-2021 | figure_caption_region | 4 | `visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410` | `False` | `False` |
| 14 | clip-2021 | figure_caption_region | 9 | `visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f` | `False` | `False` |
| 15 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42` | `False` | `False` |
| 16 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` | `False` | `False` |
| 17 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` | `False` | `False` |
| 18 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` | `False` | `False` |
| 19 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` | `False` | `False` |
| 20 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` | `False` | `False` |
| 21 | clip-2021 | layout_region | 1 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` | `False` | `False` |
| 22 | mae-2021 | layout_region | 1 | `visual-layout:mae-2021:layout_region:1:aadf3033f0ed984e` | `False` | `False` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` | `False` | `False` |
| 24 | resnet-2015 | layout_region | 1 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` | `False` | `False` |

## Warnings

- `This is a design report only; the planned candidate store is not written.`
- `All projected rows remain retrieval_hint_only, not evidence.`
- `Whole-image or whole-page GPT/VLM batches require a later pack-design gate.`
