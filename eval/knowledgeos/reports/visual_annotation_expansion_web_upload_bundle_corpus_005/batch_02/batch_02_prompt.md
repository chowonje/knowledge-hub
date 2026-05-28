# Visual Annotation Expansion Batch 02

Use only the attached context-crop PNG files listed below. Do not use whole pages, whole images, or outside sources.
Return JSON only with top-level schema `knowledge-hub.paper.visual-annotation-web-output.v1` and a `rows` array.
Every row must keep `strictEvidence=false`, `citationGrade=false`, and `answerableWithoutTextEvidence=false`.
`derivedTextForRetrieval` is a retrieval hint only; it is not evidence and not answer-visible text.

## Attachments To Upload

- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/09-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-figure_caption_region-a4ccacf781.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/10-emu3.5-native-multimodal-models-are-world-learners-p1-figure_caption_region-decd8c9078.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/11-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p2-figure_caption_region-599517831d.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/12-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p1-figure_caption_region-0de8098f7e.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/13-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p2-figure_caption_region-d7d10670a5.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/14-high-resolution-image-synthesis-with-latent-diffusion-models-p1-figure_caption_region-24a73d2eda.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/15-qwen-image-technical-report-p2-figure_caption_region-6b728766ad.png`
- `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/16-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p2-figure_caption_region-d7fcd3761f.png`

## Rows To Fill

| # | paperId | type | page | sourceCandidateId | attachmentRef |
|---:|---|---|---:|---|---|
| 1 | newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents | figure_caption_region | 4 | `visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:92b8890073250699` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/09-newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents-p4-figure_caption_region-a4ccacf781.png` |
| 2 | emu3.5-native-multimodal-models-are-world-learners | figure_caption_region | 1 | `visual-layout:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:4281f17d9edca1b7` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/10-emu3.5-native-multimodal-models-are-world-learners-p1-figure_caption_region-decd8c9078.png` |
| 3 | faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks | figure_caption_region | 2 | `visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:053df2c84c04975d` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/11-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-p2-figure_caption_region-599517831d.png` |
| 4 | autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation | figure_caption_region | 1 | `visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/12-autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-p1-figure_caption_region-0de8098f7e.png` |
| 5 | arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts | figure_caption_region | 2 | `visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/13-arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts-p2-figure_caption_region-d7d10670a5.png` |
| 6 | high-resolution-image-synthesis-with-latent-diffusion-models | figure_caption_region | 1 | `visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:bd010932c7e494a5` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/14-high-resolution-image-synthesis-with-latent-diffusion-models-p1-figure_caption_region-24a73d2eda.png` |
| 7 | qwen-image-technical-report | figure_caption_region | 2 | `visual-layout:qwen-image-technical-report:figure_caption_region:2:a2a6bdca9d8f65d3` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/15-qwen-image-technical-report-p2-figure_caption_region-6b728766ad.png` |
| 8 | photorealistic-text-to-image-diffusion-models-with-deep-language-understanding | figure_caption_region | 2 | `visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:16dc56c857fd0641` | `eval/knowledgeos/reports/visual_annotation_expansion_attachment_pack_005/assets/16-photorealistic-text-to-image-diffusion-models-with-deep-language-understanding-p2-figure_caption_region-d7fcd3761f.png` |

## JSON Shape To Return

```json
{
  "schema": "knowledge-hub.paper.visual-annotation-web-output.v1",
  "rows": [
    {
      "sourceCandidateId": "visual-layout:newtonbench-benchmarking-generalizable-scientific-law-discovery-in-llm-agents:figure_caption_region:4:92b8890073250699",
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
      "sourceCandidateId": "visual-layout:emu3.5-native-multimodal-models-are-world-learners:figure_caption_region:1:4281f17d9edca1b7",
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
      "sourceCandidateId": "visual-layout:faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks:figure_caption_region:2:053df2c84c04975d",
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
      "sourceCandidateId": "visual-layout:autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation:figure_caption_region:1:3db7901a05fe4358",
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
      "sourceCandidateId": "visual-layout:arithmetic-in-the-wild-llama-uses-base-10-addition-to-reason-about-cyclic-concepts:figure_caption_region:2:9768341fe65e7ef2",
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
      "sourceCandidateId": "visual-layout:high-resolution-image-synthesis-with-latent-diffusion-models:figure_caption_region:1:bd010932c7e494a5",
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
      "sourceCandidateId": "visual-layout:qwen-image-technical-report:figure_caption_region:2:a2a6bdca9d8f65d3",
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
      "sourceCandidateId": "visual-layout:photorealistic-text-to-image-diffusion-models-with-deep-language-understanding:figure_caption_region:2:16dc56c857fd0641",
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
