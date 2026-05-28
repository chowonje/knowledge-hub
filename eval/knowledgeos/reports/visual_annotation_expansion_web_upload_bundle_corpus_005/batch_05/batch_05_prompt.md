# Visual Annotation Expansion Batch 05

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/33-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-image_region-35e5f7c70e.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/34-emu3.5-native-multimodal-models-are-world-learners-p1-image_region-3adbc14e69.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/35-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p2-image_region-8473d7fc66.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/36-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p1-image_region-2c98531c19.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/37-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p1-image_region-2b79f24bcc.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/38-high-resolution-image-synthesis-with-latent-diffusion-models-p1-image_region-079b7b928c.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/39-qwen-image-technical-report-p1-image_region-ea864a8b48.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/40-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p2-image_region-4c8abd6fd5.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | image_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/33-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-image_region-35e5f7c70e.png` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | image_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:image_region:1:03dc21098ebf92d6` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/34-emu3.5-native-multimodal-models-are-world-learners-p1-image_region-3adbc14e69.png` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | image_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:2ea5949dc69e0e4c` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/35-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p2-image_region-8473d7fc66.png` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | image_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/36-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p1-image_region-2c98531c19.png` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | image_region | 1 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/37-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p1-image_region-2b79f24bcc.png` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | image_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:2360b13807a5f368` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/38-high-resolution-image-synthesis-with-latent-diffusion-models-p1-image_region-079b7b928c.png` |
| 7 | qwen-image-technical-report | image_region | 1 | `visual-layout:qwen-image-technical-report:image_region:1:50985e4339ab85a3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/39-qwen-image-technical-report-p1-image_region-ea864a8b48.png` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | image_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:204aa4cc291f768f` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/40-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p2-image_region-4c8abd6fd5.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:image_region:4:04ae14ad971d2f06",
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
      "sourceCandidateId": "visual-layout:emu3.5-native-multimodal-models-are-world-learners:image_region:1:03dc21098ebf92d6",
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
      "sourceCandidateId": "visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:image_region:2:2ea5949dc69e0e4c",
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
      "sourceCandidateId": "visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:image_region:1:0a83cc7b8ce39324",
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
      "sourceCandidateId": "visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:image_region:1:0792bc4980e9c1ad",
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
      "sourceCandidateId": "visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:image_region:1:2360b13807a5f368",
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
      "sourceCandidateId": "visual-layout:qwen-image-technical-report:image_region:1:50985e4339ab85a3",
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
      "sourceCandidateId": "visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:image_region:2:204aa4cc291f768f",
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
