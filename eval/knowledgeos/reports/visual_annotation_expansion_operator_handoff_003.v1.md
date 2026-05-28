# Visual Annotation Expansion Operator Handoff 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-operator-handoff.v1`
- status: `ready`
- decision: `ready_for_operator_web_vlm_run`
- generatedAt: `2026-05-27T02:50:31Z`
- handoffId: `visual_annotation_expansion_operator_handoff_003`
- sourceManualRunPacket: `eval/knowledgeos/reports/visual_annotation_expansion_manual_run_packet_003.v1.json`
- expectedOutputRef: `eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.manual.json`
- validationCommand: `PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py --output eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.manual.json --source-expansion-pack eval/knowledgeos/reports/visual_annotation_expansion_pack_design_003.v1.json --source-attachment-pack eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003.v1.json --validation-json eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.validation.v1.json --validation-md eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.validation.v1.md`
- templateRows: `24`
- batchRows: `3`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- manualOperatorWebModelRunRequired: `True`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- candidateStoreMutationRows: `0`
- wholeImageGptRows: `0`

## Operator Steps

1. Open each batch in the manual run packet.
2. Upload only the listed context-crop PNG attachments for that batch.
3. Paste the batch prompt and row metadata into web GPT/Pro.
4. Ask for JSON only with schema knowledge-hub.paper.visual-annotation-web-output.v1.
5. Combine the returned rows into eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.manual.json.
6. Run validation with: PYTHONPATH=. python eval/knowledgeos/scripts/validate_visual_annotation_expansion_web_output.py --output eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.manual.json --source-expansion-pack eval/knowledgeos/reports/visual_annotation_expansion_pack_design_003.v1.json --source-attachment-pack eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003.v1.json --validation-json eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.validation.v1.json --validation-md eval/knowledgeos/reports/visual_annotation_expansion_web_output_003.validation.v1.md

## Template Rows

| # | batch | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---:|---|---|---:|---|---|
| 1 | 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:4bad77d0856486fe` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/01-clip-2021-p21-table_region-d4eb3193e7.png` |
| 2 | 1 | mae-2021 | table_region | 7 | `visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/02-mae-2021-p7-table_region-bfbbafd612.png` |
| 3 | 1 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/03-resnet-2015-p6-table_region-49e9afc133.png` |
| 4 | 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/04-clip-2021-p21-table_region-5d7e41a602.png` |
| 5 | 1 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:33d1507d0cd15829` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/05-mae-2021-p8-table_region-4ea2608fbe.png` |
| 6 | 1 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/06-resnet-2015-p6-table_region-d92ada1e73.png` |
| 7 | 1 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:71b8cf43d8079375` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/07-clip-2021-p22-table_region-9624b52f39.png` |
| 8 | 1 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/08-mae-2021-p8-table_region-0a9691a1bb.png` |
| 9 | 2 | clip-2021 | figure_caption_region | 7 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/09-clip-2021-p7-figure_caption_region-1252f24b4e.png` |
| 10 | 2 | mae-2021 | figure_caption_region | 3 | `visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/10-mae-2021-p3-figure_caption_region-04331d5d79.png` |
| 11 | 2 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/11-resnet-2015-p8-figure_caption_region-69bd31ffec.png` |
| 12 | 2 | clip-2021 | figure_caption_region | 8 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/12-clip-2021-p8-figure_caption_region-115dd1b617.png` |
| 13 | 2 | mae-2021 | figure_caption_region | 4 | `visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/13-mae-2021-p4-figure_caption_region-a4a60ffb47.png` |
| 14 | 2 | clip-2021 | figure_caption_region | 9 | `visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/14-clip-2021-p9-figure_caption_region-06b0a471fa.png` |
| 15 | 2 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/15-mae-2021-p6-figure_caption_region-7ded22699f.png` |
| 16 | 2 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/16-clip-2021-p10-figure_caption_region-007c581197.png` |
| 17 | 3 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/17-clip-2021-p5-equation_region-580eca5260.png` |
| 18 | 3 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/18-clip-2021-p5-equation_region-bfafd5e5ef.png` |
| 19 | 3 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/19-clip-2021-p10-equation_region-a20794562b.png` |
| 20 | 3 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/20-clip-2021-p10-equation_region-c7cc7b71d0.png` |
| 21 | 3 | clip-2021 | layout_region | 1 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/21-clip-2021-p1-layout_region-245d63628e.png` |
| 22 | 3 | mae-2021 | layout_region | 1 | `visual-layout:mae-2021:layout_region:1:aadf3033f0ed984e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/22-mae-2021-p1-layout_region-1c3eb161fd.png` |
| 23 | 3 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/23-alexnet-2012-p1-layout_region-fbdce7db66.png` |
| 24 | 3 | resnet-2015 | layout_region | 1 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/24-resnet-2015-p1-layout_region-7b46f50641.png` |

## Warnings

- `This handoff is not web/VLM output and must not be validated as completed output.`
- `The operator must replace every FILL_IN placeholder with observations from attached context crops.`
- `Do not upload whole pages or whole images for this gate.`
- `Do not treat derivedTextForRetrieval as strict, citation-grade, or answer-visible evidence.`
