# Visual Annotation Expansion Batch 03

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/17-clip-2021-p15-equation_region-9221e49f2a.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/18-clip-2021-p16-equation_region-a16c8ba012.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/19-clip-2021-p17-equation_region-08ffb89a65.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/20-clip-2021-p19-equation_region-a0e8c8ec18.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/21-clip-2021-p16-layout_region-0cf094e719.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/22-mae-2021-p12-layout_region-548f37868d.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/23-alexnet-2012-p1-layout_region-dbdd58c492.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/24-resnet-2015-p6-layout_region-36ee20cc9a.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | clip-2021 | equation_region | 15 | `visual-layout:clip-2021:equation_region:15:36e881ed149ab580` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/17-clip-2021-p15-equation_region-9221e49f2a.png` |
| 2 | clip-2021 | equation_region | 16 | `visual-layout:clip-2021:equation_region:16:ae86cff7e225293a` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/18-clip-2021-p16-equation_region-a16c8ba012.png` |
| 3 | clip-2021 | equation_region | 17 | `visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/19-clip-2021-p17-equation_region-08ffb89a65.png` |
| 4 | clip-2021 | equation_region | 19 | `visual-layout:clip-2021:equation_region:19:95c566a959a29f80` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/20-clip-2021-p19-equation_region-a0e8c8ec18.png` |
| 5 | clip-2021 | layout_region | 16 | `visual-layout:clip-2021:layout_region:16:0850e569d769717b` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/21-clip-2021-p16-layout_region-0cf094e719.png` |
| 6 | mae-2021 | layout_region | 12 | `visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/22-mae-2021-p12-layout_region-548f37868d.png` |
| 7 | alexnet-2012 | layout_region | 1 | `visual-layout:alexnet-2012:layout_region:1:e3b7474586959627` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/23-alexnet-2012-p1-layout_region-dbdd58c492.png` |
| 8 | resnet-2015 | layout_region | 6 | `visual-layout:resnet-2015:layout_region:6:99aa935003a72db9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_004/assets/24-resnet-2015-p6-layout_region-36ee20cc9a.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:15:36e881ed149ab580",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:16:ae86cff7e225293a",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:17:65b08b3d7099f2e0",
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
      "sourceCandidateId": "visual-layout:clip-2021:equation_region:19:95c566a959a29f80",
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
      "sourceCandidateId": "visual-layout:clip-2021:layout_region:16:0850e569d769717b",
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
      "sourceCandidateId": "visual-layout:mae-2021:layout_region:12:c7e7678d2ec73caa",
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
      "sourceCandidateId": "visual-layout:alexnet-2012:layout_region:1:e3b7474586959627",
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
      "sourceCandidateId": "visual-layout:resnet-2015:layout_region:6:99aa935003a72db9",
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
