# Visual Annotation Expansion Manual Run Packet 002

- schema: `knowledge-hub.paper.visual-annotation-expansion-manual-run-packet.v1`
- status: `ready`
- decision: `ready_for_manual_web_vlm_expansion_run`
- generatedAt: `2026-05-26T14:05:53Z`
- packetId: `visual_annotation_expansion_manual_run_packet_002`
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
| 1 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2616032b2a9d352d` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01-clip-2021-p2-image_region-29c8942b4b.png` |
| 2 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:258a60cc216d8130` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/02-mae-2021-p1-image_region-e3b2cace3b.png` |
| 3 | alexnet-2012 | image_region | 6 | `visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/03-alexnet-2012-p6-image_region-bf79cf1040.png` |
| 4 | clip-2021 | image_region | 2 | `visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/04-clip-2021-p2-image_region-c453ed1cde.png` |
| 5 | mae-2021 | image_region | 1 | `visual-layout:mae-2021:image_region:1:40b27e8b4073d98f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/05-mae-2021-p1-image_region-4fec8160d0.png` |
| 6 | alexnet-2012 | image_region | 8 | `visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/06-alexnet-2012-p8-image_region-0159a28636.png` |
| 7 | clip-2021 | image_region | 15 | `visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/07-clip-2021-p15-image_region-ff64c7a303.png` |
| 8 | mae-2021 | image_region | 2 | `visual-layout:mae-2021:image_region:2:0b66280633a636cc` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/08-mae-2021-p2-image_region-7fbc5df66c.png` |

### Batch 2 of 3

Prompt:

```text
You are annotating visual retrieval hints for batch 2 of 3. Use only the attached context-crop images and the provided row metadata. Return one JSON object with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper claims from the image, do not infer hidden details, and do not include local file paths.
```

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | figure_caption_region | 2 | `visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/09-clip-2021-p2-figure_caption_region-289b56fe5d.png` |
| 2 | mae-2021 | figure_caption_region | 1 | `visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/10-mae-2021-p1-figure_caption_region-0fb21838ff.png` |
| 3 | resnet-2015 | figure_caption_region | 6 | `visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/11-resnet-2015-p6-figure_caption_region-082ba73f27.png` |
| 4 | clip-2021 | figure_caption_region | 3 | `visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/12-clip-2021-p3-figure_caption_region-f49947f821.png` |
| 5 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/13-mae-2021-p2-figure_caption_region-8717d0f4b0.png` |
| 6 | resnet-2015 | figure_caption_region | 8 | `visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/14-resnet-2015-p8-figure_caption_region-346008e3d9.png` |
| 7 | clip-2021 | figure_caption_region | 5 | `visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/15-clip-2021-p5-figure_caption_region-d0e128afb9.png` |
| 8 | mae-2021 | figure_caption_region | 2 | `visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/16-mae-2021-p2-figure_caption_region-e44913da92.png` |

### Batch 3 of 3

Prompt:

```text
You are annotating visual retrieval hints for batch 3 of 3. Use only the attached context-crop images and the provided row metadata. Return one JSON object with schema knowledge-hub.paper.visual-annotation-web-output.v1 and a rows array. For every row, fill sourceCandidateId, visualObservationStatus, derivedTextForRetrieval, visibleText, retrievalKeywords, uncertainty, limitations, strictEvidence=false, citationGrade=false, and answerableWithoutTextEvidence=false. Do not create citation-grade evidence, do not answer paper claims from the image, do not infer hidden details, and do not include local file paths.
```

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | table_region | 7 | `visual-layout:clip-2021:table_region:7:74d3127c812d1e9b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/17-clip-2021-p7-table_region-c37aabf875.png` |
| 2 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:09cee49d2841b6a3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/18-mae-2021-p5-table_region-25a22f5cc1.png` |
| 3 | resnet-2015 | table_region | 6 | `visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/19-resnet-2015-p6-table_region-3639f35d2c.png` |
| 4 | clip-2021 | table_region | 17 | `visual-layout:clip-2021:table_region:17:21f8887ab432c389` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/20-clip-2021-p17-table_region-911c2f4053.png` |
| 5 | mae-2021 | table_region | 5 | `visual-layout:mae-2021:table_region:5:1179c62488dbb733` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/21-mae-2021-p5-table_region-b740715491.png` |
| 6 | clip-2021 | equation_region | 1 | `visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/22-clip-2021-p1-equation_region-c3a34cc947.png` |
| 7 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/23-clip-2021-p5-equation_region-f51a092fa1.png` |
| 8 | clip-2021 | equation_region | 5 | `visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/24-clip-2021-p5-equation_region-344eb60cd0.png` |

## Warnings

- `This packet is for manual web/VLM use only; this script makes no model calls.`
- `Return derivedTextForRetrieval as retrieval hints only, not evidence.`
- `Do not upload whole pages or whole images for this gate; use the listed context crops only.`
