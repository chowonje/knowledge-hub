# Visual Annotation Expansion Batch 02

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/09-clip-2021-p2-figure_caption_region-289b56fe5d.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/10-mae-2021-p1-figure_caption_region-0fb21838ff.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/11-resnet-2015-p6-figure_caption_region-082ba73f27.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/12-clip-2021-p3-figure_caption_region-f49947f821.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/13-mae-2021-p2-figure_caption_region-8717d0f4b0.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/14-resnet-2015-p8-figure_caption_region-346008e3d9.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/15-clip-2021-p5-figure_caption_region-d0e128afb9.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/16-mae-2021-p2-figure_caption_region-e44913da92.png`

## Rows To Fill

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

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:2:b354160446b3a89c",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:1:f3477cc8ce7ec8b1",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:6:6b3dd4d05dc69888",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:3:fd2db71adcd98ec4",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:2:460a5652014c6369",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:8:5cf998f6a7a3739f",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:5:87777b9ff7911bbe",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    },
    {
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:2:b35cefa766216e3f",
      "visualObservationStatus": "image_attached",
      "derivedTextForRetrieval": "Retrieval hint only: FILL_IN_FROM_ATTACHED_IMAGE.",
      "visibleText": "Visible fragments include: FILL_IN_VISIBLE_TEXT_ONLY.",
      "retrievalKeywords": [
        "FILL_IN_KEYWORD"
      ],
      "uncertainty": "FILL_IN_UNCERTAINTY.",
      "limitations": "Retrieval hint only, not evidence.",
      "strictEvidence": false,
      "citationGrade": false,
      "answerableWithoutTextEvidence": false
    }
  ]
}
```
