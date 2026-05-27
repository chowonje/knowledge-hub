# Visual Annotation Expansion Batch 01

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/01-clip-2021-p21-table_region-d4eb3193e7.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/02-mae-2021-p7-table_region-bfbbafd612.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/03-resnet-2015-p6-table_region-49e9afc133.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/04-clip-2021-p21-table_region-5d7e41a602.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/05-mae-2021-p8-table_region-4ea2608fbe.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/06-resnet-2015-p6-table_region-d92ada1e73.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/07-clip-2021-p22-table_region-9624b52f39.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_003/assets/08-mae-2021-p8-table_region-0a9691a1bb.png`

## Rows To Fill

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

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:table_region:21:4bad77d0856486fe",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:7:8f36af5d66d8b78e",
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
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:6:b1729ae3f7fd50f8",
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
      "sourceCandidateId": "visual-layout:clip-2021:table_region:21:ba742e93b9b8eb98",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:8:33d1507d0cd15829",
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
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:6:e1c63e0b5921b1e5",
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
      "sourceCandidateId": "visual-layout:clip-2021:table_region:22:71b8cf43d8079375",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:8:5f83ffc6688fce4b",
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
