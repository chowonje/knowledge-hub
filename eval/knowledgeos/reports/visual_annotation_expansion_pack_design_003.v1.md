# Visual Annotation Expansion Pack Design

- schema: `knowledge-hub.paper.visual-annotation-expansion-pack-design.v1`
- status: `ready`
- decision: `ready_for_visual_annotation_expansion_attachment_pack`
- generatedAt: `2026-05-27T02:49:23Z`
- packId: `visual_annotation_expansion_pack_003`
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
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:4bad77d0856486fe` | `new_paper_visual_coverage` | `context_crop_png` |
| 2 | mae-2021 | table_region | 7 | `visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e` | `new_paper_visual_coverage` | `context_crop_png` |
| 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 4 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98` | `new_paper_visual_coverage` | `context_crop_png` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:33d1507d0cd15829` | `new_paper_visual_coverage` | `context_crop_png` |
| 6 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 7 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:71b8cf43d8079375` | `new_paper_visual_coverage` | `context_crop_png` |
| 8 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b` | `new_paper_visual_coverage` | `context_crop_png` |
| 9 | clip-2021 | figure_caption_region | 7 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` | `new_paper_visual_coverage` | `context_crop_png` |
| 10 | mae-2021 | figure_caption_region | 3 | `visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa` | `new_paper_visual_coverage` | `context_crop_png` |
| 11 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 12 | clip-2021 | figure_caption_region | 8 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` | `new_paper_visual_coverage` | `context_crop_png` |
| 13 | mae-2021 | figure_caption_region | 4 | `visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410` | `new_paper_visual_coverage` | `context_crop_png` |
| 14 | clip-2021 | figure_caption_region | 9 | `visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f` | `new_paper_visual_coverage` | `context_crop_png` |
| 15 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42` | `new_paper_visual_coverage` | `context_crop_png` |
| 16 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` | `new_paper_visual_coverage` | `context_crop_png` |
| 17 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` | `new_paper_visual_coverage` | `context_crop_png` |
| 18 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` | `new_paper_visual_coverage` | `context_crop_png` |
| 19 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` | `new_paper_visual_coverage` | `context_crop_png` |
| 20 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` | `new_paper_visual_coverage` | `context_crop_png` |
| 21 | clip-2021 | layout_region | 1 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` | `new_paper_visual_coverage` | `context_crop_png` |
| 22 | mae-2021 | layout_region | 1 | `visual-layout:mae-2021:layout_region:1:aadf3033f0ed984e` | `new_paper_visual_coverage` | `context_crop_png` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 24 | resnet-2015 | layout_region | 1 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` | `remaining_high_value_visual_candidate` | `context_crop_png` |

## Warnings

- `This expansion design writes no crop files and sends no images to GPT/VLM.`
- `Image-region candidates are included only as bounded context-crop annotation candidates.`
- `Whole-image or whole-page annotation remains deferred to visual_full_image_annotation_pack_design.`
- `Any future derivedTextForRetrieval remains retrieval_hint_only and non-evidence.`
