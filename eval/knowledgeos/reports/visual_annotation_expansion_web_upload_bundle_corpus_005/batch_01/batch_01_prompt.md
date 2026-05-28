# Visual Annotation Expansion Batch 01

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/01-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p2-table_region-1fecf35c16.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/02-emu3.5-native-multimodal-models-are-world-learners-p6-table_region-ea8e904c4c.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/03-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p6-table_region-00a96171af.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/04-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p15-table_region-5cd6344640.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/05-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p8-table_region-c45ef0401e.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/06-high-resolution-image-synthesis-with-latent-diffusion-models-p4-table_region-c388a21db1.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/07-qwen-image-technical-report-p9-table_region-7889ebd3d8.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/08-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p7-table_region-0bafbfcfff.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | table_region | 2 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:43ba20bbd82c40cc` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/01-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p2-table_region-1fecf35c16.png` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | table_region | 6 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/02-emu3.5-native-multimodal-models-are-world-learners-p6-table_region-ea8e904c4c.png` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | table_region | 6 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:bf4da7f980d217f7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/03-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p6-table_region-00a96171af.png` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | table_region | 15 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/04-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p15-table_region-5cd6344640.png` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | table_region | 8 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/05-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p8-table_region-c45ef0401e.png` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | table_region | 4 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:cdab6c275bd25d05` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/06-high-resolution-image-synthesis-with-latent-diffusion-models-p4-table_region-c388a21db1.png` |
| 7 | qwen-image-technical-report | table_region | 9 | `visual-layout:qwen-image-technical-report:table_region:9:47a89c68476bd4bb` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/07-qwen-image-technical-report-p9-table_region-7889ebd3d8.png` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | table_region | 7 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:c8163ca9120a3550` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/08-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p7-table_region-0bafbfcfff.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:table_region:2:43ba20bbd82c40cc",
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
      "sourceCandidateId": "visual-layout:emu3.5-native-multimodal-models-are-world-learners:table_region:6:7e5ddeb596848dc5",
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
      "sourceCandidateId": "visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:table_region:6:bf4da7f980d217f7",
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
      "sourceCandidateId": "visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:table_region:15:bf53567890d6c9c9",
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
      "sourceCandidateId": "visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:table_region:8:96e7bcf8b6c5c548",
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
      "sourceCandidateId": "visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:table_region:4:cdab6c275bd25d05",
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
      "sourceCandidateId": "visual-layout:qwen-image-technical-report:table_region:9:47a89c68476bd4bb",
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
      "sourceCandidateId": "visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:table_region:7:c8163ca9120a3550",
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
