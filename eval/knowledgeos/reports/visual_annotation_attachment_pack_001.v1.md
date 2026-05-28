# Visual Annotation Attachment Pack 001

- schema: `knowledge-hub.paper.visual-annotation-attachment-pack.v1`
- status: `ready`
- decision: `ready_for_manual_web_vlm_run`
- generatedAt: `2026-05-26T12:42:37Z`
- attachmentPackId: `visual_annotation_attachment_pack_001`
- sourceWebPack: `eval/knowledgeos/reports/visual_annotation_web_pack_001.v1.json`
- assetRootRef: `eval/knowledgeos/reports/visual_annotation_attachment_pack_001`
- cropAttachmentRows: `18`
- blockedRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_and_attachment_files_only`
- apiCalls: `False`
- modelCalls: `False`
- webModelCalls: `False`
- vectorIndexing: `False`
- strictEvidencePromotionRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- indexMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- answerabilityGateBypassRows: `0`
- cropWriteRows: `18`
- pageImageWriteRows: `0`

## Upload Set

Upload the web pack Markdown and the PNG refs below to the manual web/VLM session.

| # | paperId | type | page | pixels | bytes | attachmentRef | blockerReason |
|---:|---|---|---:|---|---:|---|---|
| 1 | alexnet-2012 | figure_caption_region | 3 | `509x942` | 86378 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/01-alexnet-2012-p3-figure_caption_region-ecd99be999.png` | `` |
| 2 | alexnet-2012 | equation_region | 4 | `383x449` | 50794 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/02-alexnet-2012-p4-equation_region-0eb9465eea.png` | `` |
| 3 | alexnet-2012 | equation_region | 4 | `458x468` | 65632 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/03-alexnet-2012-p4-equation_region-d56b6c29af.png` | `` |
| 4 | alexnet-2012 | figure_caption_region | 5 | `960x700` | 121720 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/04-alexnet-2012-p5-figure_caption_region-baf84f3a74.png` | `` |
| 5 | alexnet-2012 | figure_caption_region | 6 | `480x810` | 176224 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/05-alexnet-2012-p6-figure_caption_region-7ed515b1e7.png` | `` |
| 6 | alexnet-2012 | equation_region | 6 | `1072x478` | 71657 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/06-alexnet-2012-p6-equation_region-11fb4e7d5b.png` | `` |
| 7 | alexnet-2012 | table_region | 7 | `715x762` | 117115 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/07-alexnet-2012-p7-table_region-48cfaeb9c3.png` | `` |
| 8 | alexnet-2012 | table_region | 7 | `533x745` | 109585 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/08-alexnet-2012-p7-table_region-51e96969b4.png` | `` |
| 9 | alexnet-2012 | figure_caption_region | 8 | `960x756` | 640451 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/09-alexnet-2012-p8-figure_caption_region-e3f030c188.png` | `` |
| 10 | resnet-2015 | figure_caption_region | 1 | `642x765` | 86849 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/10-resnet-2015-p1-figure_caption_region-088b3fb4a6.png` | `` |
| 11 | resnet-2015 | figure_caption_region | 2 | `495x495` | 40580 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/11-resnet-2015-p2-figure_caption_region-189f3fcaf7.png` | `` |
| 12 | resnet-2015 | equation_region | 3 | `607x456` | 67947 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/12-resnet-2015-p3-equation_region-7ca87b4ec2.png` | `` |
| 13 | resnet-2015 | equation_region | 3 | `614x456` | 85798 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/13-resnet-2015-p3-equation_region-08dbaccfe9.png` | `` |
| 14 | resnet-2015 | equation_region | 3 | `713x467` | 63642 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/14-resnet-2015-p3-equation_region-1c10f6a61b.png` | `` |
| 15 | resnet-2015 | figure_caption_region | 4 | `641x809` | 97530 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/15-resnet-2015-p4-figure_caption_region-4ba6c7fcef.png` | `` |
| 16 | resnet-2015 | figure_caption_region | 5 | `1159x743` | 139458 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/16-resnet-2015-p5-figure_caption_region-11e2f07238.png` | `` |
| 17 | resnet-2015 | table_region | 5 | `1195x721` | 124338 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/17-resnet-2015-p5-table_region-eb72c89d44.png` | `` |
| 18 | resnet-2015 | table_region | 5 | `677x742` | 123873 | `eval/knowledgeos/reports/visual_annotation_attachment_pack_001/assets/18-resnet-2015-p5-table_region-1f17bfd7d0.png` | `` |

## Warnings

- `Attachment PNGs are eval artifacts for manual web/VLM calibration only.`
- `Generated visual descriptions must remain derivedTextForRetrieval retrieval hints, not evidence.`
- `Full page PNGs are intentionally not written in pack 001 to keep the artifact set small.`
