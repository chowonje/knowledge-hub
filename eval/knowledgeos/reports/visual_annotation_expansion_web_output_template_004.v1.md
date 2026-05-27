# Visual Annotation Expansion Web Output Template 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-web-output-template.v1`
- status: `ready`
- decision: `ready_for_manual_web_output_fill`
- generatedAt: `2026-05-27T07:19:02Z`
- templateId: `visual_annotation_expansion_web_output_template_004`
- sourceOperatorHandoff: `eval/knowledgeos/reports/visual_annotation_expansion_operator_handoff_004.v1.json`
- targetOutputRef: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.manual.json`
- validationCommand: `PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py --source-expansion-pack eval/knowledgeos/reports/visual_annotation_expansion_pack_design_004.v1.json --manual-output eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.manual.json --validation-json eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.validation.v1.json --validation-md eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.validation.v1.md`
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
5. Save the completed output at eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.manual.json and run: PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py --source-expansion-pack eval/knowledgeos/reports/visual_annotation_expansion_pack_design_004.v1.json --manual-output eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.manual.json --validation-json eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.validation.v1.json --validation-md eval/knowledgeos/reports/visual_annotation_expansion_web_output_004.validation.v1.md

## Template Rows

| # | batch | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---:|---|---|---:|---|---|
| 1 | 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:badaf33f92dc4c28` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/01-clip-2021-p21-table_region-c96d53c9d8.png` |
| 2 | 1 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/02-mae-2021-p8-table_region-889b867d5e.png` |
| 3 | 1 | resnet-2015 | table_region | 7 | `visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/03-resnet-2015-p7-table_region-0c644e5308.png` |
| 4 | 1 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/04-clip-2021-p22-table_region-b5e8cf4350.png` |
| 5 | 1 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:a31e2f20793cd157` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/05-mae-2021-p8-table_region-84ee50ebb1.png` |
| 6 | 1 | resnet-2015 | table_region | 8 | `visual-layout:resnet-2015:table_region:8:499a16649cac8d1b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/06-resnet-2015-p8-table_region-36ed3318d0.png` |
| 7 | 1 | clip-2021 | table_region | 25 | `visual-layout:clip-2021:table_region:25:795f0bd1367145d9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/07-clip-2021-p25-table_region-c7032928ee.png` |
| 8 | 1 | mae-2021 | table_region | 11 | `visual-layout:mae-2021:table_region:11:229ba015a36e1bc6` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/08-mae-2021-p11-table_region-4e16da1096.png` |
| 9 | 2 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:f7695e0650ad9755` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/09-clip-2021-p10-figure_caption_region-501a8f8860.png` |
| 10 | 2 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:fb84bd6d006dd177` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/10-mae-2021-p6-figure_caption_region-f224651812.png` |
| 11 | 2 | clip-2021 | figure_caption_region | 11 | `visual-layout:clip-2021:figure_caption_region:11:d9042e6c82546c0e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/11-clip-2021-p11-figure_caption_region-4a7c87f0b6.png` |
| 12 | 2 | mae-2021 | figure_caption_region | 7 | `visual-layout:mae-2021:figure_caption_region:7:5e6020a5e57aa862` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/12-mae-2021-p7-figure_caption_region-927e2cccf9.png` |
| 13 | 2 | clip-2021 | figure_caption_region | 12 | `visual-layout:clip-2021:figure_caption_region:12:36479d77a4a94f87` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/13-clip-2021-p12-figure_caption_region-0be5c97adb.png` |
| 14 | 2 | mae-2021 | figure_caption_region | 13 | `visual-layout:mae-2021:figure_caption_region:13:d74e50f7fc0acd19` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/14-mae-2021-p13-figure_caption_region-a306b56f89.png` |
| 15 | 2 | clip-2021 | figure_caption_region | 13 | `visual-layout:clip-2021:figure_caption_region:13:c57a4c0a359c0b1e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/15-clip-2021-p13-figure_caption_region-13fe437cbf.png` |
| 16 | 2 | mae-2021 | figure_caption_region | 14 | `visual-layout:mae-2021:figure_caption_region:14:78c3aa6879bfbb7e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/16-mae-2021-p14-figure_caption_region-c8a1968467.png` |
| 17 | 3 | clip-2021 | equation_region | 15 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/17-clip-2021-p15-equation_region-9221e49f2a.png` |
| 18 | 3 | clip-2021 | equation_region | 16 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/18-clip-2021-p16-equation_region-a16c8ba012.png` |
| 19 | 3 | clip-2021 | equation_region | 17 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/19-clip-2021-p17-equation_region-08ffb89a65.png` |
| 20 | 3 | clip-2021 | equation_region | 19 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/20-clip-2021-p19-equation_region-a0e8c8ec18.png` |
| 21 | 3 | clip-2021 | layout_region | 16 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/21-clip-2021-p16-layout_region-0cf094e719.png` |
| 22 | 3 | mae-2021 | layout_region | 12 | `visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/22-mae-2021-p12-layout_region-548f37868d.png` |
| 23 | 3 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/23-alexnet-2012-p1-layout_region-dbdd58c492.png` |
| 24 | 3 | resnet-2015 | layout_region | 6 | `visual-layout:resnet-2015:layout_region:6:99aa935003a72db9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/24-resnet-2015-p6-layout_region-36ee20cc9a.png` |

## Warnings

- `This template intentionally uses a different schema so it cannot be accepted as completed visual annotation output.`
- `A placeholder row is not a manual web/VLM observation.`
- `Do not upload whole pages or whole images for this gate.`
- `Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.`
