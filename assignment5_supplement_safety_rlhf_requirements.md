# CS336 Assignment 5 Supplement: Instruction Tuning And RLHF Requirements

Source PDF: `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`

This is a lookup-oriented summary of the optional supplement, not a solution guide. The local file is
the Spring 2025 supplement included in this repository.

## Snapshot

- Title: `CS336 Assignment 5 Supplement (alignment): Instruction Tuning and RLHF`
- Version: `1.0.1`
- Term: Spring 2025
- Status: Optional supplement
- Main model family: `Llama 3.1 8B`
- Annotator model for AlpacaEval/Safety: `Llama 3.3 70B Instruct`
- Adapter hook file: `tests/adapters.py`
- Relevant optional test files:
  - `tests/test_data.py`
  - `tests/test_dpo.py`
  - `tests/test_metrics.py`
  - `tests/test_sft.py`

## What To Implement

- Zero-shot prompting baselines for multiple evaluation datasets.
- Supervised fine-tuning, abbreviated `SFT`, from instruction-response demonstrations.
- Direct Preference Optimization, abbreviated `DPO`, from pairwise preference data.

## What To Run

- Measure `Llama 3.1 8B` zero-shot prompting performance.
- Instruction-tune `Llama 3.1 8B`.
- Fine-tune `Llama 3.1 8B` on pairwise preference data.
- Evaluate baseline, SFT, and DPO models on helpfulness, safety, math, and knowledge benchmarks.

## Evaluation Datasets

- `MMLU`: multiple-choice knowledge evaluation.
- `GSM8K`: grade-school math word problems.
- `AlpacaEval`: instruction-following preference evaluation.
- `SimpleSafetyTests`: safety-oriented prompts judged by an annotator model.
- `Anthropic HH`: helpful/harmless preference data for DPO.

## Problem Checklist

| Problem label | Points | What it asks for |
| --- | ---: | --- |
| `mmlu_baseline` | 4 | Parse MMLU predictions, run zero-shot evaluation, report parse failures, throughput, metrics, and error analysis. |
| `gsm8k_baseline` | 4 | Parse numeric GSM8K predictions, run zero-shot evaluation, report parse failures, throughput, metrics, and error analysis. |
| `alpaca_eval_baseline` | 4 | Generate AlpacaEval outputs, estimate throughput, run evaluator, report win rates and qualitative failures. |
| `sst_baseline` | 4 | Generate SimpleSafetyTests outputs, run safety evaluator, report safe-output proportion and unsafe examples. |
| `look_at_sft` | 4 | Inspect random instruction-tuning examples and comment on task/data characteristics. |
| `data_loading` | 3 | Load the instruction-tuning dataset into a convenient training data structure. |
| `sft_script` | 4 | Write an instruction-tuning training script and track training behavior. |
| `sft` | 6 | Fine-tune `Llama 3 8B` on the provided instruction-tuning data. |
| `mmlu_sft` | 4 | Evaluate the SFT model on MMLU and compare with the zero-shot baseline. |
| `gsm8k_sft` | 4 | Evaluate the SFT model on GSM8K and compare with the zero-shot baseline. |
| `alpaca_eval_sft` | 4 | Evaluate the SFT model on AlpacaEval and compare win rates with the baseline. |
| `sst_sft` | 4 | Evaluate the SFT model on SimpleSafetyTests and compare with the baseline. |
| `red_teaming` | 4 | Discuss possible misuse cases and report red-teaming attempts/results for the SFT model. |
| `look_at_hh` | 2 | Load and inspect Anthropic HH helpful/harmless preference examples. |
| `dpo_loss` | 2 | Implement per-instance DPO loss and connect the adapter test. |
| `dpo_training` | 4 | Train with DPO on HH, evaluate on AlpacaEval, SimpleSafetyTests, GSM8K, and MMLU. |

Total: 61 optional points.

## Evaluation Commands Mentioned

AlpacaEval uses the local Llama 3.3 70B Instruct annotator configuration and compares against GPT-4
Turbo reference outputs. The handout notes this requires two GPUs with more than 80GB memory each.

SimpleSafetyTests evaluation uses:

```sh
uv run python scripts/evaluate_safety.py
```

The exact input/output paths are part of the experiment setup in the handout.

## Experiment Deliverables To Track

- Baseline outputs, parse failures, throughput, metrics, and short error analyses.
- SFT dataset inspection notes.
- SFT training curve/screenshot and downstream evaluation comparisons.
- AlpacaEval win rate and length-controlled win rate for baseline, SFT, and DPO.
- SimpleSafetyTests safe-output proportion for baseline, SFT, and DPO.
- Red-teaming methodology and qualitative results.
- DPO validation curve and post-DPO benchmark comparisons.
- Notes on possible alignment tax from DPO by comparing AlpacaEval/Safety with GSM8K/MMLU.

## Useful Search Terms

- `MMLU`
- `GSM8K`
- `AlpacaEval`
- `SimpleSafetyTests`
- `Anthropic HH`
- `SFT`
- `DPO`
- `RLHF`
- `Llama 3.1 8B`
- `Llama 3.3 70B Instruct`
- `safe outputs`
- `winrate`
- `length-controlled winrate`
- `alignment tax`

