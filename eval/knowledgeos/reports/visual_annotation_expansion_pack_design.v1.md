# Visual Annotation Expansion Pack Design

- schema: `knowledge-hub.paper.visual-annotation-expansion-pack-design.v1`
- status: `ready`
- decision: `ready_for_visual_annotation_expansion_attachment_pack`
- generatedAt: `2026-05-26T13:44:00Z`
- packId: `visual_annotation_expansion_pack_002`
- selectedExpansionRows: `24`
- imageCandidateRows: `8`
- figureCandidateRows: `8`
- tableCandidateRows: `5`
- equationCandidateRows: `3`
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
| 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 2 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 3 | alexnet-2012 | image_region | 6 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 4 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 5 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 6 | alexnet-2012 | image_region | 8 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 7 | clip-2021 | image_region | 15 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 8 | mae-2021 | image_region | 2 | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` | `first_image_region_probe_after_text_caption_batch` | `context_crop_png` |
| 9 | clip-2021 | figure_caption_region | 2 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` | `new_paper_visual_coverage` | `context_crop_png` |
| 10 | mae-2021 | figure_caption_region | 1 | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` | `new_paper_visual_coverage` | `context_crop_png` |
| 11 | resnet-2015 | figure_caption_region | 6 | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 12 | clip-2021 | figure_caption_region | 3 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` | `new_paper_visual_coverage` | `context_crop_png` |
| 13 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` | `new_paper_visual_coverage` | `context_crop_png` |
| 14 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 15 | clip-2021 | figure_caption_region | 5 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` | `new_paper_visual_coverage` | `context_crop_png` |
| 16 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` | `new_paper_visual_coverage` | `context_crop_png` |
| 17 | clip-2021 | table_region | 7 | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` | `new_paper_visual_coverage` | `context_crop_png` |
| 18 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` | `new_paper_visual_coverage` | `context_crop_png` |
| 19 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` | `remaining_high_value_visual_candidate` | `context_crop_png` |
| 20 | clip-2021 | table_region | 17 | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` | `new_paper_visual_coverage` | `context_crop_png` |
| 21 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` | `new_paper_visual_coverage` | `context_crop_png` |
| 22 | clip-2021 | equation_region | 1 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` | `new_paper_visual_coverage` | `context_crop_png` |
| 23 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` | `new_paper_visual_coverage` | `context_crop_png` |
| 24 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` | `new_paper_visual_coverage` | `context_crop_png` |

## Warnings

- `This expansion design writes no crop files and sends no images to GPT/VLM.`
- `Image-region candidates are included only as bounded context-crop annotation candidates.`
- `Whole-image or whole-page annotation remains deferred to visual_full_image_annotation_pack_design.`
- `Any future derivedTextForRetrieval remains retrieval_hint_only and non-evidence.`
