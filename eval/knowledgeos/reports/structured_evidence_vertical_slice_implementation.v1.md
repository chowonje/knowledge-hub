# Structured Evidence Vertical Slice Implementation

- generatedAt: 2026-05-21T05:28:07Z
- runId: structured-evidence-vertical-slice-20260521
- apply: false
- paperRows: 5
- pilotReadbackPassRows: 3
- greenfieldGeneratedRows: 2
- generatedSourceSpanRecords: 2
- generatedStrictEvidenceRecords: 2

## Policy

- runtime answer integration: false
- table/equation parser: false
- sourceContentHash authority: corpus_manifest.expectedSourceContentHash

## Deferred Evidence Types

- table_cell_numeric
- equation_citation
- appendix_table_lookup

## Papers

### 1706.03762

- mode: pilot_readback
- status: pass
- artifactId: paper_1706_03762
- expectedSourceContentHash: sha256:bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697
- parsedArtifactLocator: papers_dir/parsed/1706.03762/document.json
- existing: sourceSpan=24 strictEvidence=23
- generated: sourceSpan=0 strictEvidence=0 applied=None
- trace pass: True

### 2005.11401

- mode: greenfield_section
- status: generated
- artifactId: paper_2005_11401
- expectedSourceContentHash: sha256:23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233
- parsedArtifactLocator: papers_dir/parsed/2005.11401/document.json
- existing: sourceSpan=0 strictEvidence=0
- generated: sourceSpan=1 strictEvidence=1 applied=False
- trace pass: True

### 1512.03385

- mode: greenfield_section
- status: generated
- artifactId: paper_1512_03385
- expectedSourceContentHash: sha256:1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a
- parsedArtifactLocator: papers_dir/parsed/1512.03385/document.json
- existing: sourceSpan=0 strictEvidence=0
- generated: sourceSpan=1 strictEvidence=1 applied=False
- trace pass: True

### 1506.02640

- mode: figure_caption_readback
- status: pass
- artifactId: paper_1506_02640
- expectedSourceContentHash: sha256:54bcd2dd05dc618849e8a94d8b88fe3eeb37f80e96e200600d38f1f733931678
- parsedArtifactLocator: papers_dir/parsed/1506.02640/document.json
- existing: sourceSpan=16 strictEvidence=16
- generated: sourceSpan=0 strictEvidence=0 applied=None
- trace pass: True

### 2005.14165

- mode: pilot_readback
- status: pass
- artifactId: paper_2005_14165
- expectedSourceContentHash: sha256:97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83
- parsedArtifactLocator: papers_dir/parsed/2005.14165/document.json
- existing: sourceSpan=62 strictEvidence=60
- generated: sourceSpan=0 strictEvidence=0 applied=None
- trace pass: True
