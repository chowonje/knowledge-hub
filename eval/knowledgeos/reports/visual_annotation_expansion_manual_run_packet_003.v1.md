# Visual Annotation Expansion Manual Run Packet 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-manual-run-packet.v1`
- status: `ready`
- decision: `ready_for_manual_web_vlm_expansion_run`
- generatedAt: `2026-05-27T02:50:19Z`
- packetId: `visual_annotation_expansion_manual_run_packet_003`
- packetRows: `24`
- batchRows: `3`
- missingAttachmentRows: `0`
- privatePathLeakRows: `0`

## Batches

### Batch 1 of 3

Prompt:

```text
You are annotating visual retrieval hints for batch 1 of 3. Use only the attached context-crop images and the provided row metadata. Return one JSON object with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper claims from the image, do not infer hidden details, and do not include local file paths.
```

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:4bad77d0856486fe` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/01-clip-2021-p21-table_region-d4eb3193e7.png` |
| 2 | mae-2021 | table_region | 7 | `visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/02-mae-2021-p7-table_region-bfbbafd612.png` |
| 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/03-resnet-2015-p6-table_region-49e9afc133.png` |
| 4 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/04-clip-2021-p21-table_region-5d7e41a602.png` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:33d1507d0cd15829` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/05-mae-2021-p8-table_region-4ea2608fbe.png` |
| 6 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/06-resnet-2015-p6-table_region-d92ada1e73.png` |
| 7 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:71b8cf43d8079375` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/07-clip-2021-p22-table_region-9624b52f39.png` |
| 8 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/08-mae-2021-p8-table_region-0a9691a1bb.png` |

### Batch 2 of 3

Prompt:

```text
You are annotating visual retrieval hints for batch 2 of 3. Use only the attached context-crop images and the provided row metadata. Return one JSON object with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper claims from the image, do not infer hidden details, and do not include local file paths.
```

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | figure_caption_region | 7 | `visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/09-clip-2021-p7-figure_caption_region-1252f24b4e.png` |
| 2 | mae-2021 | figure_caption_region | 3 | `visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/10-mae-2021-p3-figure_caption_region-04331d5d79.png` |
| 3 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/11-resnet-2015-p8-figure_caption_region-69bd31ffec.png` |
| 4 | clip-2021 | figure_caption_region | 8 | `visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/12-clip-2021-p8-figure_caption_region-115dd1b617.png` |
| 5 | mae-2021 | figure_caption_region | 4 | `visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/13-mae-2021-p4-figure_caption_region-a4a60ffb47.png` |
| 6 | clip-2021 | figure_caption_region | 9 | `visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/14-clip-2021-p9-figure_caption_region-06b0a471fa.png` |
| 7 | mae-2021 | figure_caption_region | 6 | `visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/15-mae-2021-p6-figure_caption_region-7ded22699f.png` |
| 8 | clip-2021 | figure_caption_region | 10 | `visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/16-clip-2021-p10-figure_caption_region-007c581197.png` |

### Batch 3 of 3

Prompt:

```text
You are annotating visual retrieval hints for batch 3 of 3. Use only the attached context-crop images and the provided row metadata. Return one JSON object with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper claims from the image, do not infer hidden details, and do not include local file paths.
```

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:6936e394c7364af3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/17-clip-2021-p5-equation_region-580eca5260.png` |
| 2 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:7b890343228eec5b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/18-clip-2021-p5-equation_region-bfafd5e5ef.png` |
| 3 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:b13e1298e1e46bb8` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/19-clip-2021-p10-equation_region-a20794562b.png` |
| 4 | clip-2021 | equation_region | 10 | `visual-layout:clip-2021:equation_region:10:db8ab5cab587035e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/20-clip-2021-p10-equation_region-c7cc7b71d0.png` |
| 5 | clip-2021 | layout_region | 1 | `visual-layout:clip-2021:layout_region:1:7614f0e5b3451acd` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/21-clip-2021-p1-layout_region-245d63628e.png` |
| 6 | mae-2021 | layout_region | 1 | `visual-layout:mae-2021:layout_region:1:aadf3033f0ed984e` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/22-mae-2021-p1-layout_region-1c3eb161fd.png` |
| 7 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:0a3974d993387e3b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/23-alexnet-2012-p1-layout_region-fbdce7db66.png` |
| 8 | resnet-2015 | layout_region | 1 | `visual-layout:resnet-2015:layout_region:1:1442b7b9fe2f781a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/24-resnet-2015-p1-layout_region-7b46f50641.png` |

## Warnings

- `This packet is for manual web/VLM use only; this script makes no model calls.`
- `Return derivedTextForRetrieval as retrieval hints only, not evidence.`
- `Do not upload whole pages or whole images for this gate; use the listed context crops only.`
