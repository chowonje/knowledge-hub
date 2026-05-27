# Visual Annotation Expansion Batch 01

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/01-clip-2021-p21-table_region-c96d53c9d8.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/02-mae-2021-p8-table_region-889b867d5e.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/03-resnet-2015-p7-table_region-0c644e5308.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/04-clip-2021-p22-table_region-b5e8cf4350.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/05-mae-2021-p8-table_region-84ee50ebb1.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/06-resnet-2015-p8-table_region-36ed3318d0.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/07-clip-2021-p25-table_region-c7032928ee.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/08-mae-2021-p11-table_region-4e16da1096.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | table_region | 21 | `visual-layout:clip-2021:table_region:21:badaf33f92dc4c28` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/01-clip-2021-p21-table_region-c96d53c9d8.png` |
| 2 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:71f56f58347cc664` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/02-mae-2021-p8-table_region-889b867d5e.png` |
| 3 | resnet-2015 | table_region | 7 | `visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/03-resnet-2015-p7-table_region-0c644e5308.png` |
| 4 | clip-2021 | table_region | 22 | `visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/04-clip-2021-p22-table_region-b5e8cf4350.png` |
| 5 | mae-2021 | table_region | 8 | `visual-layout:mae-2021:table_region:8:a31e2f20793cd157` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/05-mae-2021-p8-table_region-84ee50ebb1.png` |
| 6 | resnet-2015 | table_region | 8 | `visual-layout:resnet-2015:table_region:8:499a16649cac8d1b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/06-resnet-2015-p8-table_region-36ed3318d0.png` |
| 7 | clip-2021 | table_region | 25 | `visual-layout:clip-2021:table_region:25:795f0bd1367145d9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/07-clip-2021-p25-table_region-c7032928ee.png` |
| 8 | mae-2021 | table_region | 11 | `visual-layout:mae-2021:table_region:11:229ba015a36e1bc6` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/08-mae-2021-p11-table_region-4e16da1096.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:table_region:21:badaf33f92dc4c28",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:8:71f56f58347cc664",
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
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:7:7e079100aed8aaf6",
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
      "sourceCandidateId": "visual-layout:clip-2021:table_region:22:cd7ebed3203dab3d",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:8:a31e2f20793cd157",
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
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:8:499a16649cac8d1b",
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
      "sourceCandidateId": "visual-layout:clip-2021:table_region:25:795f0bd1367145d9",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:11:229ba015a36e1bc6",
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
