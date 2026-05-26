# Visual Annotation Expansion Batch 01

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/01-clip-2021-p2-image_region-29c8942b4b.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/02-mae-2021-p1-image_region-e3b2cace3b.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/03-alexnet-2012-p6-image_region-bf79cf1040.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/04-clip-2021-p2-image_region-c453ed1cde.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/05-mae-2021-p1-image_region-4fec8160d0.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/06-alexnet-2012-p8-image_region-0159a28636.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/07-clip-2021-p15-image_region-ff64c7a303.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/08-mae-2021-p2-image_region-7fbc5df66c.png`

## Rows To Fill

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

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:image_region:2:2616032b2a9d352d",
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
      "sourceCandidateId": "visual-layout:mae-2021:image_region:1:258a60cc216d8130",
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
      "sourceCandidateId": "visual-layout:alexnet-2012:image_region:6:508eb98021de0fd7",
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
      "sourceCandidateId": "visual-layout:clip-2021:image_region:2:2642c8db8bd29d0a",
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
      "sourceCandidateId": "visual-layout:mae-2021:image_region:1:40b27e8b4073d98f",
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
      "sourceCandidateId": "visual-layout:alexnet-2012:image_region:8:219e7594a10f3a3f",
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
      "sourceCandidateId": "visual-layout:clip-2021:image_region:15:1da1f1cf8ce81bd5",
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
      "sourceCandidateId": "visual-layout:mae-2021:image_region:2:0b66280633a636cc",
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
