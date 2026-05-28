# Visual Annotation Expansion Batch 02

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/09-clip-2021-p7-figure_caption_region-1252f24b4e.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/10-mae-2021-p3-figure_caption_region-04331d5d79.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/11-resnet-2015-p8-figure_caption_region-69bd31ffec.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/12-clip-2021-p8-figure_caption_region-115dd1b617.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/13-mae-2021-p4-figure_caption_region-a4a60ffb47.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/14-clip-2021-p9-figure_caption_region-06b0a471fa.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/15-mae-2021-p6-figure_caption_region-7ded22699f.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/16-clip-2021-p10-figure_caption_region-007c581197.png`

## Rows To Fill

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

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:7:d3e128cd03f8c6bd",
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
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:3:534077a253e5a0fa",
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
      "sourceCandidateId": "visual-layout:resnet-2015:figure_caption_region:8:ddd82bd40f652201",
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
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:8:75613ea3ae6b4e06",
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
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:4:5260bb56bfc85410",
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
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:9:37cc4e2380f5208f",
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
      "sourceCandidateId": "visual-layout:mae-2021:figure_caption_region:6:dc979079d8bc6a42",
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
      "sourceCandidateId": "visual-layout:clip-2021:figure_caption_region:10:967a1812d18f9d62",
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
