# Visual Retrieval Hint Candidate Store Expansion Dry Run

- schema: `knowledge-hub.paper.visual-retrieval-hint-candidate-store-expansion-dry-run.v1`
- status: `ready`
- decision: `ready_for_visual_retrieval_hint_candidate_store_expansion_review`
- generatedAt: `2026-05-27T07:44:31Z`
- sourceDesignReport: `eval/knowledgeos/reports/visual_retrieval_hint_candidate_store_expansion_design_004.v1.json`
- dryRunRows: `24`
- plannedWriteRows: `24`
- candidateStoreWriteRows: `0`
- indexEligibleRows: `0`
- runtimeVisibleRows: `0`
- privatePathLeakRows: `0`

## Mutation Guarantees

- writes: `report_only`
- candidateStoreWriteRows: `0`
- vectorIndexing: `False`
- indexMutationRows: `0`
- runtimeAnswerVisibleExposureRows: `0`
- databaseMutationRows: `0`
- reindexOrReembedRows: `0`
- vaultScanRows: `0`
- externalDownloadRows: `0`
- strictEvidencePromotionRows: `0`
- answerabilityGateBypassRows: `0`

## Dry Run Rows

| # | paperId | type | page | hintCandidateId | wouldWriteOnApply | recordSha256 |
|---:|---|---|---:|---|---|---|
| 1 | clip-2021 | table_region | 21 | `visual-retrieval-hint:clip-2021:table_region:21:a8f65a0582f89042` | `True` | `sha256:8b003c5ac91606e75330c58c109d094d910c3ddee3b303173dafd5984543dca2` |
| 2 | mae-2021 | table_region | 8 | `visual-retrieval-hint:mae-2021:table_region:8:8a7d640f19165642` | `True` | `sha256:d4299d7329c3252b5ddca03cf26346f9dfa324f17366bc31f567386d53d03921` |
| 3 | resnet-2015 | table_region | 7 | `visual-retrieval-hint:resnet-2015:table_region:7:2f4406059ee9d734` | `True` | `sha256:66be81e0ad075523b7fb32c5a51de5457d6de9264e4574e1e6eae0e0fa1678c6` |
| 4 | clip-2021 | table_region | 22 | `visual-retrieval-hint:clip-2021:table_region:22:567a15ee4200c9cb` | `True` | `sha256:1547c767af6d3ab4bdc5e8ee889e2cdc19a0eb2c6e574093ba072dc17dfda188` |
| 5 | mae-2021 | table_region | 8 | `visual-retrieval-hint:mae-2021:table_region:8:df694cac46885c9d` | `True` | `sha256:2c5e2656686ec6940cf792ea592b624e19d2b28373a3ce4cbbfc7292a3e154e0` |
| 6 | resnet-2015 | table_region | 8 | `visual-retrieval-hint:resnet-2015:table_region:8:92c46b3c0c9ff6f0` | `True` | `sha256:1738caa18296267922bcc7ae4a94392dd7b1022a9bd5e5ca1036daafc37abc06` |
| 7 | clip-2021 | table_region | 25 | `visual-retrieval-hint:clip-2021:table_region:25:923f7e62652bdc04` | `True` | `sha256:419d386afc0adf3678b88c390a9a2eced7fa62edfd2fd12b9ca51ba6a8fdba73` |
| 8 | mae-2021 | table_region | 11 | `visual-retrieval-hint:mae-2021:table_region:11:6d9f3d0fb86bd453` | `True` | `sha256:95e989529eda98018a277f99f08187dcc5d1c95e991fc0a94f3c453a1cf87265` |
| 9 | clip-2021 | figure_caption_region | 10 | `visual-retrieval-hint:clip-2021:figure_caption_region:10:5625d2a79b026d71` | `True` | `sha256:723c220ec412ff0add29a8cd278dfb266dd28ec236698d0360d305474320408e` |
| 10 | mae-2021 | figure_caption_region | 6 | `visual-retrieval-hint:mae-2021:figure_caption_region:6:5e5f04d3298373f0` | `True` | `sha256:16efad57a2cda9f449b741122a2db6a5c291bc17d92d4d1bc5a0622d3020c722` |
| 11 | clip-2021 | figure_caption_region | 11 | `visual-retrieval-hint:clip-2021:figure_caption_region:11:ac2460a7ef6608c2` | `True` | `sha256:06e88f8e4b86cc744e8d06ccb7e0158e65bd80a82adefc7e6f7903d2edb94d5e` |
| 12 | mae-2021 | figure_caption_region | 7 | `visual-retrieval-hint:mae-2021:figure_caption_region:7:16b38e609ca3a26b` | `True` | `sha256:abcb881d0b5c6da341e8a2878807587053f45166c75ecceb292653d41203b3d7` |
| 13 | clip-2021 | figure_caption_region | 12 | `visual-retrieval-hint:clip-2021:figure_caption_region:12:ad31e942d2da6eb2` | `True` | `sha256:eda0a7e49a4629b6b8ff1d5936c82a80ef12951871acc20ffefa2bdfab74f795` |
| 14 | mae-2021 | figure_caption_region | 13 | `visual-retrieval-hint:mae-2021:figure_caption_region:13:e6f0577f97226d44` | `True` | `sha256:bfcb531e8667fec85ac4d4bab7524920c88790a55d54c09c2791a63f288edf02` |
| 15 | clip-2021 | figure_caption_region | 13 | `visual-retrieval-hint:clip-2021:figure_caption_region:13:c3864ff56609eddb` | `True` | `sha256:7e41e6bc98faf2a5578fea5c00e68956b4f8318e082485c348daed1075007b03` |
| 16 | mae-2021 | figure_caption_region | 14 | `visual-retrieval-hint:mae-2021:figure_caption_region:14:98a19ddfacee24a2` | `True` | `sha256:ca303fbe697fd99afd39733fdd371853fce38b2d25da04ba4ee153500f515632` |
| 17 | clip-2021 | equation_region | 15 | `visual-retrieval-hint:clip-2021:equation_region:15:607f1fd22d46a182` | `True` | `sha256:c68dacc4c54040d886c8f495242cc4aab09425bfe8eae2636d57200adb2af7d4` |
| 18 | clip-2021 | equation_region | 16 | `visual-retrieval-hint:clip-2021:equation_region:16:ab0704d8f387ba7a` | `True` | `sha256:61a54156223c82037eb6f4db636739f2435445b90c8ab44c663d75ea1aebc17e` |
| 19 | clip-2021 | equation_region | 17 | `visual-retrieval-hint:clip-2021:equation_region:17:e7f8b095a772716f` | `True` | `sha256:dad8aef44967e379e042c69423f1e1e416feb9620b863f88eb551cf4b49f3dcc` |
| 20 | clip-2021 | equation_region | 19 | `visual-retrieval-hint:clip-2021:equation_region:19:39cabc8d02aa5f3c` | `True` | `sha256:4f120014622de7f82e0d253d41c07fecf76861a7064d1b8d19521c12a74a3ac0` |
| 21 | clip-2021 | layout_region | 16 | `visual-retrieval-hint:clip-2021:layout_region:16:d67e516254c25dd0` | `True` | `sha256:1bf687a2ffd37b20b6f55c11f6a81c15bf0aa067bff522d18e9d68956ed86ed0` |
| 22 | mae-2021 | layout_region | 12 | `visual-retrieval-hint:mae-2021:layout_region:12:75f6ed0836e0cbad` | `True` | `sha256:22ec114654009dbb9ff5554b5bd2e5a133a1d8f7106f5563457b9308a9c8927a` |
| 23 | alexnet-2012 | layout_region | 1 | `visual-retrieval-hint:alexnet-2012:layout_region:1:afaec200e0ecb239` | `True` | `sha256:e5958c0773c554a4adf6eab0d35ca7154097711b7022d49967b683ea89f935a8` |
| 24 | resnet-2015 | layout_region | 6 | `visual-retrieval-hint:resnet-2015:layout_region:6:4b0a95bbcaa73546` | `True` | `sha256:8e5032c1ed2ee571a327a4943a223d6e8eea3096c2661a6411459f898ef33225` |

## Warnings

- `This dry-run previews future JSONL records but writes no candidate store.`
- `All rows remain unindexed and not runtime-visible after dry-run.`
- `A separate apply tranche is required before any store mutation, and a later gate is required before indexing.`
