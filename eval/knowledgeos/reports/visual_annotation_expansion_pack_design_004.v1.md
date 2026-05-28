# Visual Annotation Expansion Pack Design

- schema: `knowledge-hub.paper.visual-annotation-expansion-pack-design.v1`
- status: `ready`
- decision: `ready_for_visual_annotation_expansion_attachment_pack`
- generatedAt: `2026-05-27T07:18:26Z`
- packId: `visual_annotation_expansion_pack_004`
- selectedExpansionRows: `24`
- imageCandidateRows: `0`
- figureCandidateRows: `8`
- tableCandidateRows: `8`
- equationCandidateRows: `4`
- wholeImageRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- modelCalls: `False`
- webModelCalls: `False`
- wholeImageGptRows: `0`
- cropWriteRows: `0`
- vectorIndexing: `False`
- candidateStoreMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- strictEvidencePromotionRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`

## Expansion Rows

| # | paperId | type | page | sourceCandidateId | reason | attachment |
|---:|---|---|---:|---|---|---|
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:badaf33f92dc4c28` | `new_paper_visual_coverage` | `context_crop_png` |
| 2 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` | `new_paper_visual_coverage` | `context_crop_png` |
| 3 | resnet-2015 | table_region | 7 | `visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 4 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d` | `new_paper_visual_coverage` | `context_crop_png` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:a31e2f20793cd157` | `new_paper_visual_coverage` | `context_crop_png` |
| 6 | resnet-2015 | table_region | 8 | `visual-layout:resnet-2015:table_region:8:499a16649cac8d1b` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 7 | clip-2021 | table_region | 25 | `visual-layout:clip-2021:table_region:25:795f0bd1367145d9` | `new_paper_visual_coverage` | `context_crop_png` |
| 8 | mae-2021 | table_region | 11 | `visual-layout:mae-2021:table_region:11:229ba015a36e1bc6` | `new_paper_visual_coverage` | `context_crop_png` |
| 9 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:f7695e0650ad9755` | `new_paper_visual_coverage` | `context_crop_png` |
| 10 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:fb84bd6d006dd177` | `new_paper_visual_coverage` | `context_crop_png` |
| 11 | clip-2021 | figure_caption_region | 11 | `visual-layout:clip-2021:figure_caption_region:11:d9042e6c82546c0e` | `new_paper_visual_coverage` | `context_crop_png` |
| 12 | mae-2021 | figure_caption_region | 7 | `visual-layout:mae-2021:figure_caption_region:7:5e6020a5e57aa862` | `new_paper_visual_coverage` | `context_crop_png` |
| 13 | clip-2021 | figure_caption_region | 12 | `visual-layout:clip-2021:figure_caption_region:12:36479d77a4a94f87` | `new_paper_visual_coverage` | `context_crop_png` |
| 14 | mae-2021 | figure_caption_region | 13 | `visual-layout:mae-2021:figure_caption_region:13:d74e50f7fc0acd19` | `new_paper_visual_coverage` | `context_crop_png` |
| 15 | clip-2021 | figure_caption_region | 13 | `visual-layout:clip-2021:figure_caption_region:13:c57a4c0a359c0b1e` | `new_paper_visual_coverage` | `context_crop_png` |
| 16 | mae-2021 | figure_caption_region | 14 | `visual-layout:mae-2021:figure_caption_region:14:78c3aa6879bfbb7e` | `new_paper_visual_coverage` | `context_crop_png` |
| 17 | clip-2021 | equation_region | 15 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` | `new_paper_visual_coverage` | `context_crop_png` |
| 18 | clip-2021 | equation_region | 16 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` | `new_paper_visual_coverage` | `context_crop_png` |
| 19 | clip-2021 | equation_region | 17 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` | `new_paper_visual_coverage` | `context_crop_png` |
| 20 | clip-2021 | equation_region | 19 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` | `new_paper_visual_coverage` | `context_crop_png` |
| 21 | clip-2021 | layout_region | 16 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` | `new_paper_visual_coverage` | `context_crop_png` |
| 22 | mae-2021 | layout_region | 12 | `visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa` | `new_paper_visual_coverage` | `context_crop_png` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 24 | resnet-2015 | layout_region | 6 | `visual-layout:resnet-2015:layout_region:6:99aa935003a72db9` | `remaining_high_value_visual_candidate` | `context_crop_png` |

## Warnings

- `This expansion design writes no crop files and sends no images to GPT/VLM.`
- `Image-region candidates are included only as bounded context-crop annotation candidates.`
- `Whole-image or whole-page annotation remains deferred to visual_full_image_annotation_pack_design.`
- `Any future derivedTextForRetrieval remains retrieval_hint_only and non-evidence.`
