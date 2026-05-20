"""Report-only complex-paper QA seed pack helper.

This helper creates a first seed pack for later complex paper QA evaluation.
It only emits local report files.  It does not generate answers, call models,
scan the vault, mutate databases or indexes, reembed, create runtime evidence,
or change the answer path.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


COMPLEX_QA_SEED_PACK_SCHEMA_ID = "knowledge-hub.paper.complex-qa-seed-pack.v1"
DEFAULT_REPORT_DIR = (
    "~/.khub/reports/complex-paper-qa/2026-05-20/complex-paper-qa-seed-pack"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_MANIFEST = PROJECT_ROOT / "eval" / "knowledgeos" / "fixtures" / "corpus_manifest.json"


QUESTION_CATEGORIES = (
    "table_numeric_qa",
    "equation_citation_qa",
    "figure_caption_qa",
    "method_comparison_qa",
    "limitation_qa",
    "appendix_table_lookup_qa",
)
ANSWERABILITY_EXPECTATIONS = (
    "answerable",
    "expected_no_answer",
    "blocked_until_structured_evidence",
)

_SUPPLEMENTAL_PAPER_ROWS: tuple[dict[str, Any], ...] = (
    {
        "paperId": "2501.12948",
        "artifactId": "paper_2501_12948",
        "sourceIds": ["2501.12948"],
        "paperLabel": "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
        "seedSource": "repo_test_fixture_reference",
        "corpusTier": "supplemental_seed_pending_manifest",
        "sourceContentHashAvailable": False,
        "riskNotes": [
            "not_declared_in_corpus_manifest_on_origin_main",
            "must_verify_local_source_artifact_before_marking_any_seed_question_answerable",
        ],
    },
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _label_from_filename(value: Any) -> str:
    text = _clean_text(value)
    return text[:-4] if text.lower().endswith(".pdf") else text


def _paper_row_from_manifest(artifact: dict[str, Any]) -> dict[str, Any] | None:
    source_ids = [str(item) for item in list(artifact.get("sourceIds") or []) if str(item or "").strip()]
    artifact_id = _clean_text(artifact.get("artifactId"))
    paper_id = source_ids[0] if source_ids else artifact_id
    if not paper_id:
        return None
    return {
        "paperId": paper_id,
        "artifactId": artifact_id,
        "sourceIds": source_ids,
        "paperLabel": _label_from_filename(artifact.get("expectedFilename")) or paper_id,
        "seedSource": "eval_corpus_manifest",
        "corpusTier": _clean_text(artifact.get("corpusTier")) or "unknown",
        "sourceContentHashAvailable": bool(artifact.get("expectedSourceContentHash")),
        "riskNotes": [
            "corpus_manifest_declares_source_artifact_but_seed_pack_does_not_verify_local_presence",
            "answerability_requires_strict_structured_evidence_in_a_later_runner",
        ],
    }


def _load_seed_papers(corpus_manifest: str | Path, *, target_count: int = 20) -> list[dict[str, Any]]:
    manifest = _read_json(corpus_manifest)
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in list(manifest.get("artifacts") or []):
        if not isinstance(artifact, dict):
            continue
        row = _paper_row_from_manifest(artifact)
        if row is None:
            continue
        paper_id = str(row.get("paperId") or "")
        if paper_id in seen:
            continue
        papers.append(row)
        seen.add(paper_id)
        if len(papers) >= target_count:
            return papers[:target_count]

    for row in _SUPPLEMENTAL_PAPER_ROWS:
        paper_id = str(row.get("paperId") or "")
        if paper_id and paper_id not in seen:
            papers.append(dict(row))
            seen.add(paper_id)
        if len(papers) >= target_count:
            break
    return papers[:target_count]


def _contract(
    contract_id: str,
    *,
    must_have: list[str],
    blocked_if_missing: list[str],
) -> dict[str, Any]:
    return {
        "contractId": contract_id,
        "mustHave": must_have,
        "blockedIfMissing": blocked_if_missing,
    }


def _question(
    index: int,
    *,
    question_category: str,
    paper_ids: list[str],
    question: str,
    expected_evidence_type: str,
    answerability_expectation: str,
    required_evidence_contract: dict[str, Any],
    risk_notes: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "questionId": f"complex-paper-qa-seed-20260520-q{index:03d}",
        "questionCategory": question_category,
        "paperIds": paper_ids,
        "question": _clean_text(question),
        "expectedEvidenceType": expected_evidence_type,
        "answerabilityExpectation": answerability_expectation,
        "requiredEvidenceContract": required_evidence_contract,
        "baselineExpectedBehavior": (
            "abstain_or_no_answer"
            if answerability_expectation != "answerable"
            else "answer_only_if_strict_evidence_available"
        ),
        "riskNotes": risk_notes,
    }
    if len(paper_ids) == 1:
        row["paperId"] = paper_ids[0]
    return row


def _question_specs() -> list[dict[str, Any]]:
    table_contract = _contract(
        "strict_table_cell_numeric_evidence_v1",
        must_have=[
            "paperId",
            "tableLabel",
            "rowLabel",
            "columnLabel",
            "numericCellValue",
            "page",
            "sourceContentHash",
            "cellOrRegionLocator",
        ],
        blocked_if_missing=[
            "table_cell_value",
            "row_column_header_binding",
            "sourceContentHash",
            "cell_or_region_locator",
        ],
    )
    equation_contract = _contract(
        "strict_equation_citation_evidence_v1",
        must_have=[
            "paperId",
            "equationLabelOrAnchor",
            "equationText",
            "page",
            "sourceContentHash",
            "chars:start-end locator",
        ],
        blocked_if_missing=[
            "equation_text_alignment",
            "equation_label_or_anchor",
            "sourceContentHash",
            "chars_locator",
        ],
    )
    figure_contract = _contract(
        "strict_figure_caption_evidence_v1",
        must_have=[
            "paperId",
            "figureLabel",
            "captionText",
            "page",
            "sourceContentHash",
            "captionLocator",
        ],
        blocked_if_missing=[
            "caption_text",
            "figure_label",
            "sourceContentHash",
            "caption_locator",
        ],
    )
    method_contract = _contract(
        "strict_cross_paper_method_comparison_v1",
        must_have=[
            "paperIds",
            "methodSectionOrClaimPerPaper",
            "pagePerPaper",
            "sourceContentHashPerPaper",
            "strictSpanPerPaper",
        ],
        blocked_if_missing=[
            "strict_span_for_each_paper",
            "sourceContentHashPerPaper",
            "comparison_dimension_grounding",
        ],
    )
    limitation_contract = _contract(
        "strict_limitation_section_evidence_v1",
        must_have=[
            "paperId",
            "limitationOrDiscussionSpan",
            "page",
            "sourceContentHash",
            "chars:start-end locator",
        ],
        blocked_if_missing=[
            "explicit_limitation_span",
            "sourceContentHash",
            "chars_locator",
        ],
    )
    appendix_contract = _contract(
        "strict_appendix_table_lookup_evidence_v1",
        must_have=[
            "paperId",
            "appendixOrSupplementAnchor",
            "tableLabel",
            "rowLabel",
            "columnLabel",
            "cellValue",
            "page",
            "sourceContentHash",
            "cellOrRegionLocator",
        ],
        blocked_if_missing=[
            "appendix_anchor",
            "table_cell_value",
            "row_column_header_binding",
            "sourceContentHash",
        ],
    )
    return [
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["1706.03762"],
            "question": "In Attention Is All You Need, what BLEU value is reported for the big Transformer model on WMT 2014 English-to-German in the main results table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["numeric answer must not be inferred from memory or prose summary"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["1810.04805"],
            "question": "In BERT, what GLUE average score is reported for the large model in the reported results table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["requires row and column binding, not just a retrieved paragraph mentioning GLUE"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["2005.14165"],
            "question": "In Language Models are Few-Shot Learners, what value does the benchmark table report for the largest model under few-shot evaluation?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["large table extraction is error-prone without structured cell provenance"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["1512.03385"],
            "question": "In Deep Residual Learning for Image Recognition, what top-5 error value is reported for the relevant ResNet row in the ImageNet results table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["must preserve metric name, dataset, and model row together"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["2010.11929"],
            "question": "In An Image is Worth 16x16 Words, what ImageNet top-1 value is reported for the selected ViT model in the comparison table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["do not answer from model-name recall without table-cell evidence"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["1312.5602"],
            "question": "In Playing Atari with Deep Reinforcement Learning, what score or normalized metric is reported for the selected game in the results table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["Atari tables need game, metric, and row disambiguation"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["1707.06347"],
            "question": "In Proximal Policy Optimization Algorithms, what numeric result is reported for the chosen MuJoCo task in the comparison table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["plot-derived values are not acceptable unless structured as evidence"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["2006.11239"],
            "question": "In Denoising Diffusion Probabilistic Models, what FID value is reported for the selected dataset in the quantitative results table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": table_contract,
            "risk_notes": ["metric and dataset must be cell-grounded"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["2404.16130"],
            "question": "In the GraphRAG paper, what numeric result is reported for a global question answering comparison in the evaluation table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": table_contract,
            "risk_notes": ["baseline should abstain if the exact table cell is not represented as structured evidence"],
        },
        {
            "question_category": "table_numeric_qa",
            "paper_ids": ["2501.12948"],
            "question": "In DeepSeek-R1, what numeric score is shown for a selected language row in the appendix evaluation table?",
            "expected_evidence_type": "table_cell_numeric",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": table_contract,
            "risk_notes": ["supplemental seed paper is not corpus-manifest verified in this tranche"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["1706.03762"],
            "question": "Quote the scaled dot-product attention equation from Attention Is All You Need with its source span.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["quote-only task; interpretation without equation span is a failure"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["1406.2661"],
            "question": "Quote the minimax objective equation from Generative Adversarial Nets with citation-grade provenance.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["common equation is easy to hallucinate from memory"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["2006.11239"],
            "question": "Quote the diffusion training objective equation from Denoising Diffusion Probabilistic Models.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["requires equation extraction, not prose summary"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["1707.06347"],
            "question": "Quote the PPO clipped surrogate objective equation with a strict source locator.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["symbol-heavy equation must be preserved exactly enough for citation"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["1502.03167"],
            "question": "Quote the batch normalization transform equation from Batch Normalization with page and source hash.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["inline and display math may be split by parsers"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["2312.00752"],
            "question": "Quote the selective state space recurrence or update equation from Mamba with its source span.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["must distinguish architecture equation from explanatory prose"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["2201.11903"],
            "question": "Quote an equation-like formal prompt or scoring expression from Chain-of-Thought Prompting, if strict equation evidence exists.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["expected to abstain if no equation artifact is available"],
        },
        {
            "question_category": "equation_citation_qa",
            "paper_ids": ["2005.11401"],
            "question": "Quote a formal equation from Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks only if equation evidence is available.",
            "expected_evidence_type": "equation_source_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": equation_contract,
            "risk_notes": ["do not manufacture an equation from RAG prose"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["alexnet-2012"],
            "question": "According to the caption for the main AlexNet architecture figure, what components does the figure show?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["figure text and caption can be conflated without region-linked evidence"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2010.11929"],
            "question": "According to the relevant ViT figure caption, how are image patches represented in the model diagram?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["visual architecture summary must be caption-grounded"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["1706.03762"],
            "question": "According to the Transformer model architecture figure caption, what does the figure depict?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["do not use memorized architecture if caption span is absent"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2312.00752"],
            "question": "According to a Mamba figure caption, what system or block is illustrated?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["caption must be tied to a figure label and page"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2005.11401"],
            "question": "According to a RAG figure caption, what retrieval-generation flow is shown?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["workflow claims need caption provenance"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2410.05779"],
            "question": "According to a LightRAG figure caption, what graph or retrieval structure is depicted?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["baseline should abstain if figure-caption extraction is missing"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2404.16130"],
            "question": "According to a GraphRAG figure caption, what local-to-global summarization flow is shown?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["graph diagrams are easy to over-describe without caption evidence"],
        },
        {
            "question_category": "figure_caption_qa",
            "paper_ids": ["2006.11239"],
            "question": "According to the DDPM figure caption, what denoising or sampling process is illustrated?",
            "expected_evidence_type": "figure_caption_source_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": figure_contract,
            "risk_notes": ["must separate figure caption from surrounding explanatory text"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["2005.11401", "2007.01282"],
            "question": "Compare how RAG and FiD combine retrieval with generation, using strict method evidence from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["requires two-sided evidence; one paper alone is insufficient"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["2404.16130", "2410.05779"],
            "question": "Compare GraphRAG and LightRAG on graph construction and retrieval strategy using strict spans from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["do not answer from title-level similarity only"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["1706.03762", "2312.00752"],
            "question": "Compare the sequence modeling mechanism in Transformer and Mamba using strict method evidence from each paper.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["comparison must identify grounded dimensions, not generic architecture claims"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["alexnet-2012", "2010.11929"],
            "question": "Compare AlexNet and ViT on image representation and model architecture using strict spans from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["legacy strict-source repairs do not by themselves create structured figure/table evidence"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["1312.5602", "1707.06347"],
            "question": "Compare DQN and PPO on the learning objective and update mechanism using strict evidence from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["method comparison should not collapse algorithm families into generic RL summaries"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["1406.2661", "2006.11239"],
            "question": "Compare GANs and DDPMs on the generative training process using strict method evidence from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["equations and method spans need source-specific provenance"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["1810.04805", "2005.14165"],
            "question": "Compare BERT and GPT-3 on pretraining or prompting setup using strict evidence from both papers.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["avoid broad model-family claims without paper-specific spans"],
        },
        {
            "question_category": "method_comparison_qa",
            "paper_ids": ["2005.11401", "2310.11511"],
            "question": "Compare RAG and Self-RAG on retrieval use and critique or reflection mechanisms using strict method evidence.",
            "expected_evidence_type": "cross_paper_method_spans",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": method_contract,
            "risk_notes": ["must distinguish retrieval architecture from training or critique behavior"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["1706.03762"],
            "question": "What limitation or caveat does Attention Is All You Need state about the proposed model or experiments?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["limitation must be explicit in the paper, not inferred by the evaluator"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2312.00752"],
            "question": "What limitation or boundary condition does the Mamba paper state for selective state spaces?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["requires an explicit limitation span"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2010.11929"],
            "question": "What limitation does the ViT paper state about data scale, compute, or model use?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["answer must not be a generic transformer limitation"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2005.14165"],
            "question": "What limitation or risk does the GPT-3 paper state about few-shot language models?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["policy or societal risks need paper-specific citation"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2310.11511"],
            "question": "What limitation or failure mode does Self-RAG state about retrieval or self-reflection?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "blocked_until_structured_evidence",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["must avoid assuming limitations from broader RAG literature"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2404.16130"],
            "question": "What limitation does the GraphRAG paper state for local-to-global summarization or graph construction?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["expected to abstain until an explicit limitation span is available"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2410.05779"],
            "question": "What limitation does the LightRAG paper state about graph retrieval or update cost?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["do not answer from plausible system tradeoffs"],
        },
        {
            "question_category": "limitation_qa",
            "paper_ids": ["2501.12948"],
            "question": "What limitation does DeepSeek-R1 state about reasoning capability, language coverage, or evaluation setup?",
            "expected_evidence_type": "limitation_section_span",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": limitation_contract,
            "risk_notes": ["supplemental seed row remains unverified by the corpus manifest"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2005.14165"],
            "question": "In the GPT-3 appendix tables, what value is reported for the selected benchmark under the chosen prompting setting?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["appendix table extraction is not guaranteed by current runtime evidence"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2501.12948"],
            "question": "In the DeepSeek-R1 appendix table, what score is reported for the selected language row and benchmark column?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["requires appendix table-cell provenance; supplemental corpus status is pending"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["1810.04805"],
            "question": "In a BERT appendix or supplementary table, what value is reported for the selected ablation row?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["baseline should not answer when appendix table cell is not materialized"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["1706.03762"],
            "question": "In a Transformer appendix table, what exact value is reported for a selected hyperparameter or ablation?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["hyperparameter details need row-column evidence"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2005.11401"],
            "question": "In a RAG appendix table, what exact score is reported for the selected dataset and retrieval setting?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["answer must remain blocked without appendix table extraction"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2007.01282"],
            "question": "In a FiD appendix table, what exact score is reported for the selected open-domain QA dataset?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["do not use leaderboard recall without structured evidence"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2201.11903"],
            "question": "In a Chain-of-Thought appendix table, what result is reported for the selected reasoning benchmark?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["appendix lookup must preserve benchmark and prompting condition"],
        },
        {
            "question_category": "appendix_table_lookup_qa",
            "paper_ids": ["2310.11511"],
            "question": "In a Self-RAG appendix table, what result is reported for the selected retrieval or critique setting?",
            "expected_evidence_type": "appendix_table_cell",
            "answerability_expectation": "expected_no_answer",
            "required_evidence_contract": appendix_contract,
            "risk_notes": ["baseline should abstain until appendix evidence is strict"],
        },
    ]


def _semantic_violations(papers: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    question_ids = [str(item.get("questionId") or "") for item in questions]
    if len(question_ids) != len(set(question_ids)):
        violations.append("duplicate_question_id")
    paper_ids = {str(item.get("paperId") or "") for item in papers}
    for item in questions:
        category = str(item.get("questionCategory") or "")
        expectation = str(item.get("answerabilityExpectation") or "")
        if category not in QUESTION_CATEGORIES:
            violations.append(f"unknown_question_category:{category}")
        if expectation not in ANSWERABILITY_EXPECTATIONS:
            violations.append(f"unknown_answerability_expectation:{expectation}")
        for paper_id in list(item.get("paperIds") or []):
            if str(paper_id) not in paper_ids:
                violations.append(f"question_references_missing_paper:{item.get('questionId')}:{paper_id}")
    return sorted(set(violations))


def build_complex_qa_seed_pack(
    *,
    corpus_manifest: str | Path = DEFAULT_CORPUS_MANIFEST,
    target_paper_count: int = 20,
) -> dict[str, Any]:
    """Build the report-only complex QA seed pack."""

    papers = _load_seed_papers(corpus_manifest, target_count=target_paper_count)
    questions = [
        _question(index, **spec)
        for index, spec in enumerate(_question_specs(), start=1)
    ]
    category_counts = Counter(str(item.get("questionCategory") or "") for item in questions)
    answerability_counts = Counter(str(item.get("answerabilityExpectation") or "") for item in questions)
    evidence_counts = Counter(str(item.get("expectedEvidenceType") or "") for item in questions)
    expected_no_answer_or_blocked = sum(
        1
        for item in questions
        if item.get("answerabilityExpectation") in {"expected_no_answer", "blocked_until_structured_evidence"}
    )
    semantic_violations = _semantic_violations(papers, questions)
    counts = {
        "paperRows": len(papers),
        "questionRows": len(questions),
        "tableNumericQuestionRows": category_counts.get("table_numeric_qa", 0),
        "equationCitationQuestionRows": category_counts.get("equation_citation_qa", 0),
        "figureCaptionQuestionRows": category_counts.get("figure_caption_qa", 0),
        "methodComparisonQuestionRows": category_counts.get("method_comparison_qa", 0),
        "limitationQuestionRows": category_counts.get("limitation_qa", 0),
        "appendixTableLookupQuestionRows": category_counts.get("appendix_table_lookup_qa", 0),
        "answerableRows": answerability_counts.get("answerable", 0),
        "expectedNoAnswerRows": answerability_counts.get("expected_no_answer", 0),
        "blockedUntilStructuredEvidenceRows": answerability_counts.get("blocked_until_structured_evidence", 0),
        "expectedNoAnswerOrBlockedRows": expected_no_answer_or_blocked,
        "byQuestionCategory": dict(sorted(category_counts.items())),
        "byAnswerabilityExpectation": {
            key: answerability_counts.get(key, 0)
            for key in ANSWERABILITY_EXPECTATIONS
        },
        "byExpectedEvidenceType": dict(sorted(evidence_counts.items())),
        "schemaViolationCount": len(semantic_violations),
        "llmCallRows": 0,
        "databaseMutationRows": 0,
        "vaultScanRows": 0,
        "runtimeEvidenceCreatedRows": 0,
        "citationEvidenceCreatedRows": 0,
    }
    return {
        "schema": COMPLEX_QA_SEED_PACK_SCHEMA_ID,
        "status": "ok" if not semantic_violations else "blocked",
        "generatedAt": _now(),
        "seedPack": {
            "name": "complex-paper-qa-seed-pack",
            "version": "2026-05-20",
            "purpose": "Measure later whether structured paper evidence improves complex QA answer quality while preserving abstention when strict evidence is missing.",
            "nextRecommendedTranche": "complex QA abstain baseline runner",
        },
        "inputs": {
            "corpusManifest": str(Path(str(corpus_manifest)).expanduser()),
            "targetPaperCount": target_paper_count,
            "questionSeedSource": "static_repo_seed_pack",
        },
        "counts": counts,
        "policy": {
            "reportOnly": True,
            "answerGenerationRun": False,
            "llmCalls": False,
            "answerPathChanged": False,
            "databaseMutation": False,
            "indexMutation": False,
            "reindexOrReembed": False,
            "vaultScan": False,
            "runtimeEvidenceCreated": False,
            "citationEvidenceCreated": False,
            "strictEvidenceCreated": False,
            "parserRoutingChanged": False,
            "canonicalParsedArtifactsWritten": False,
            "questionExecutionRun": False,
        },
        "warnings": [
            "seed_pack_contains_questions_only_not_answers",
            "answerability_expectations_are_baseline_policy_inputs_not_answer_results",
            "do_not_mark_questions_answerable_without_strict_structured_evidence",
        ],
        "semanticViolations": semantic_violations,
        "papers": papers,
        "questions": questions,
    }


def render_complex_qa_seed_pack_markdown(report: dict[str, Any]) -> str:
    counts = dict(report.get("counts") or {})
    lines = [
        "# Complex Paper QA Seed Pack",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Paper rows: `{counts.get('paperRows', 0)}`",
        f"- Question rows: `{counts.get('questionRows', 0)}`",
        f"- Expected no-answer or blocked rows: `{counts.get('expectedNoAnswerOrBlockedRows', 0)}`",
        f"- Next recommended tranche: `{(report.get('seedPack') or {}).get('nextRecommendedTranche', '')}`",
        "",
        "## Policy",
        "",
        "Report-only seed pack. It does not generate answers, call LLMs, change the answer path, mutate DB/index state, reindex/reembed, scan the vault, or create runtime/citation evidence.",
        "",
        "## Counts",
        "",
        f"- By question category: `{json.dumps(counts.get('byQuestionCategory') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By answerability: `{json.dumps(counts.get('byAnswerabilityExpectation') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- By expected evidence type: `{json.dumps(counts.get('byExpectedEvidenceType') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Questions",
        "",
    ]
    for item in list(report.get("questions") or []):
        lines.extend(
            [
                f"### `{item.get('questionId', '')}`",
                "",
                f"- Category: `{item.get('questionCategory', '')}`",
                f"- Papers: `{', '.join(list(item.get('paperIds') or []))}`",
                f"- Expected evidence type: `{item.get('expectedEvidenceType', '')}`",
                f"- Answerability expectation: `{item.get('answerabilityExpectation', '')}`",
                f"- Question: {item.get('question', '')}",
                "",
            ]
        )
    return "\n".join(lines)


def write_complex_qa_seed_pack_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    root = Path(str(output_dir)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "complex-paper-qa-seed-pack.json"
    summary_path = root / "complex-paper-qa-seed-pack-summary.json"
    markdown_path = root / "complex-paper-qa-seed-pack.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": COMPLEX_QA_SEED_PACK_SCHEMA_ID,
        "status": report.get("status"),
        "generatedAt": report.get("generatedAt"),
        "counts": report.get("counts"),
        "policy": report.get("policy"),
        "warnings": report.get("warnings"),
        "nextRecommendedTranche": (report.get("seedPack") or {}).get("nextRecommendedTranche"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_complex_qa_seed_pack_markdown(report), encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Generate a report-only complex-paper QA seed pack.")
    parser.add_argument("--corpus-manifest", default=str(DEFAULT_CORPUS_MANIFEST), help="Path to corpus_manifest.json.")
    parser.add_argument("--target-paper-count", type=int, default=20, help="Number of paper rows to seed.")
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR, help="Directory for local seed-pack reports.")
    parser.add_argument("--json", action="store_true", help="Print seed-pack payload as JSON.")
    args = parser.parse_args(argv)

    report = build_complex_qa_seed_pack(
        corpus_manifest=args.corpus_manifest,
        target_paper_count=args.target_paper_count,
    )
    paths: dict[str, str] = {}
    if args.output_dir:
        paths = write_complex_qa_seed_pack_reports(report, args.output_dir)
    if paths:
        report = {**report, "reportPaths": paths}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPLEX_QA_SEED_PACK_SCHEMA_ID",
    "DEFAULT_REPORT_DIR",
    "build_complex_qa_seed_pack",
    "render_complex_qa_seed_pack_markdown",
    "write_complex_qa_seed_pack_reports",
]
