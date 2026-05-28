# Visual Annotation Expansion Batch 03

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/17-clip-2021-p7-table_region-c37aabf875.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/18-mae-2021-p5-table_region-25a22f5cc1.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/19-resnet-2015-p6-table_region-3639f35d2c.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/20-clip-2021-p17-table_region-911c2f4053.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/21-mae-2021-p5-table_region-b740715491.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/22-clip-2021-p1-equation_region-c3a34cc947.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/23-clip-2021-p5-equation_region-f51a092fa1.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_002/assets/24-clip-2021-p5-equation_region-344eb60cd0.png`

## Rows To Fill

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

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:table_region:7:74d3127c812d1e9b",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:5:09cee49d2841b6a3",
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
      "sourceCandidateId": "visual-layout:resnet-2015:table_region:6:8b15bd4cbaa96ede",
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
      "sourceCandidateId": "visual-layout:clip-2021:table_region:17:21f8887ab432c389",
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
      "sourceCandidateId": "visual-layout:mae-2021:table_region:5:1179c62488dbb733",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:1:dd31b4b813b73af2",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:5:1c4738912cb4bdf9",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:5:1fa7893d5e5d7b10",
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
