# Visual Annotation Expansion Web Output Template 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-template.v1`
- status: `ready`
- decision: `ready_for_manual_web_output_fill`
- generatedAt: `2026-05-26T14:22:54Z`
- templateId: `visual_annotation_expansion_web_output_template_002`
- sourceOperatorHandoff: `eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_002.v1.json`
- targetOutputRef: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json`
- validationCommand: `PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py`
- outputTemplateRows: `24`
- placeholderRows: `24`
- completedWebOutputRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- templateOnly: `True`
- completedWebOutputRows: `0`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- candidateStoreMutationRows: `0`
- wholeImageGptRows: `0`

## Instructions

1. Use this file as a fill template, not as completed web/VLM output.
2. For the final output, return only schema knowledge-hub.paper.visual-annotation-web-output.v1 with a rows array.
3. Replace every FILL_IN placeholder using only the attached context-crop image for that row.
4. Keep strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false for every row.
5. Save the completed output at eval/knowledgeos/reports/visual_annotation_expansion_web_output_002.manual.json and run: PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py

## Template Rows

| # | batch | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---:|---|---|---:|---|---|
| 1 | 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01-clip-2021-p2-image_region-29c8942b4b.png` |
| 2 | 1 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/02-mae-2021-p1-image_region-e3b2cace3b.png` |
| 3 | 1 | alexnet-2012 | image_region | 6 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/03-alexnet-2012-p6-image_region-bf79cf1040.png` |
| 4 | 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/04-clip-2021-p2-image_region-c453ed1cde.png` |
| 5 | 1 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/05-mae-2021-p1-image_region-4fec8160d0.png` |
| 6 | 1 | alexnet-2012 | image_region | 8 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/06-alexnet-2012-p8-image_region-0159a28636.png` |
| 7 | 1 | clip-2021 | image_region | 15 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/07-clip-2021-p15-image_region-ff64c7a303.png` |
| 8 | 1 | mae-2021 | image_region | 2 | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/08-mae-2021-p2-image_region-7fbc5df66c.png` |
| 9 | 2 | clip-2021 | figure_caption_region | 2 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/09-clip-2021-p2-figure_caption_region-289b56fe5d.png` |
| 10 | 2 | mae-2021 | figure_caption_region | 1 | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/10-mae-2021-p1-figure_caption_region-0fb21838ff.png` |
| 11 | 2 | resnet-2015 | figure_caption_region | 6 | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/11-resnet-2015-p6-figure_caption_region-082ba73f27.png` |
| 12 | 2 | clip-2021 | figure_caption_region | 3 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/12-clip-2021-p3-figure_caption_region-f49947f821.png` |
| 13 | 2 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/13-mae-2021-p2-figure_caption_region-8717d0f4b0.png` |
| 14 | 2 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/14-resnet-2015-p8-figure_caption_region-346008e3d9.png` |
| 15 | 2 | clip-2021 | figure_caption_region | 5 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/15-clip-2021-p5-figure_caption_region-d0e128afb9.png` |
| 16 | 2 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/16-mae-2021-p2-figure_caption_region-e44913da92.png` |
| 17 | 3 | clip-2021 | table_region | 7 | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/17-clip-2021-p7-table_region-c37aabf875.png` |
| 18 | 3 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/18-mae-2021-p5-table_region-25a22f5cc1.png` |
| 19 | 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/19-resnet-2015-p6-table_region-3639f35d2c.png` |
| 20 | 3 | clip-2021 | table_region | 17 | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/20-clip-2021-p17-table_region-911c2f4053.png` |
| 21 | 3 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/21-mae-2021-p5-table_region-b740715491.png` |
| 22 | 3 | clip-2021 | equation_region | 1 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/22-clip-2021-p1-equation_region-c3a34cc947.png` |
| 23 | 3 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/23-clip-2021-p5-equation_region-f51a092fa1.png` |
| 24 | 3 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/24-clip-2021-p5-equation_region-344eb60cd0.png` |

## Warnings

- `This template intentionally uses a different schema so it cannot be accepted as completed visual annotation output.`
- `A placeholder row is not a manual web/VLM observation.`
- `Do not upload whole pages or whole images for this gate.`
- `Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.`
