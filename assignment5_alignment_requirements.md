# CS336 Assignment 5 Alignment: Reasoning RL Requirements

Source PDF: `cs336_spring2026_assignment5_alignment.pdf`

This is a lookup-oriented summary of the handout, not a solution guide. It lists the required tasks,
deliverables, and useful search terms so you can quickly navigate the assignment.

## Snapshot

- Title: `CS336 Assignment 5 (alignment): Reasoning RL`
- Version: `26.0.0`
- Term: Spring 2026
- Main model: `OLMo-2-0425-1B`
- Main dataset: `GSM8K`
- Main required test file: `tests/test_grpo.py`
- Adapter hook file: `tests/adapters.py`
- Submission files:
  - `writeup.pdf`
  - `code.zip`, generated with `test_and_make_submission.sh`

## What To Implement

- Zero-shot, few-shot, and chain-of-thought prompting.
- Group Relative Policy Optimization, abbreviated `GRPO`.
- Policy-gradient estimator variants for variance reduction and importance-weight clipping.

## What To Run

- Measure prompting performance for `OLMo-2-0425-1B` on `GSM8K`.
- Run on-policy GRPO on `OLMo-2-0425-1B` to improve `GSM8K` performance.
- Run RL variants including `RFT`, `Dr. GRPO`, and `MaxRL`.
- Run off-policy GRPO variants to study speed and clipping/reweighting strategies.

## Required Tests

From `README.md`:

```sh
uv run pytest tests/test_grpo.py
```

The handout also mentions targeted tests for individual components. Use the problem labels below as
search terms in `tests/test_grpo.py` and `tests/adapters.py`.

## Problem Checklist

| Problem label | Points | What it asks for |
| --- | ---: | --- |
| `prompting_baselines` | 5 | Evaluate prompting variants for `OLMo-2-0425-1B` on `GSM8K`; report metrics and qualitative behavior. |
| `baseline_calcs` | 5 | Derive variance expressions for simple policy-gradient estimators. |
| `tokenize_prompt_and_output` | 1 | Tokenize prompt/output separately, concatenate without special tokens, and construct `response_mask`. |
| `get_response_log_probs` | 1 | Compute response token log-probabilities, optionally with next-token entropy. |
| `compute_rollout_rewards` | 1 | Compute raw rewards for rollout responses. |
| `compute_group_normalized_rewards_grpo` | 1 | Normalize rewards within each rollout group for standard GRPO. |
| `compute_policy_gradient_loss_on_policy` | 1 | Compute on-policy per-token policy-gradient loss. |
| `aggregate_loss_across_microbatch_sequence` | 0.5 | Aggregate per-token loss with sequence-level normalization. |
| `grpo_train_step_standard_on_policy` | 5 | Implement one standard on-policy GRPO batch update. |
| `grpo_experiments_standard_on_policy` | 10 | Run standard on-policy GRPO experiments and report learning curves, examples, and final validation accuracy. |
| `grpo_learning_rate` | 3 | Sweep learning rates around the recommended default and report final validation reward. |
| `grpo_prompt_ablation` | 3 | Compare `question_only`, `r1_zero`, and `r1_zero_three_shot` prompts. |
| `think_about_length_normalization` | 1 | Discuss sequence-length normalization versus constant normalization. |
| `compute_group_normalized_rewards_drgrpo` | 0.5 | Extend reward normalization for Dr. GRPO settings. |
| `aggregate_loss_across_microbatch_constant` | 0.5 | Extend loss aggregation to constant normalization. |
| `think_about_rft` | 2 | Discuss the RFT objective and behavior. |
| `derive_difficulty_reweightings` | 6 | Derive prompt difficulty reweightings induced by advantage normalizers. |
| `think_about_advantage_normalization` | 2 | Discuss std, mean, and no advantage normalization. |
| `compute_group_normalized_rewards_maxrl` | 0.5 | Extend reward normalization for MaxRL settings. |
| `grpo_train_step_variants_on_policy` | 2.5 | Support the full set of on-policy variants. |
| `grpo_experiments_variants_on_policy` | 10 | Compare fixed-hyperparameter on-policy RL variants with multiple seeds. |
| `derive_surrogate_objectives` | 2 | Derive surrogate objectives for importance-reweighting approaches. |
| `compute_policy_gradient_loss_off_policy` | 1 | Add token-level off-policy importance reweighting. |
| `think_about_importance_reweighting` | 2 | Discuss no reweighting, clipped token-level reweighting, and GSPO-style sequence-level reweighting. |
| `compute_policy_gradient_loss_off_policy_gspo` | 1 | Add sequence-level GSPO reweighting. |
| `grpo_train_step_off_policy` | 2.5 | Extend GRPO train step to support off-policy arguments. |
| `grpo_experiments_off_policy` | 10 | Compare off-policy algorithms using fixed hyperparameters and multiple seeds. |
| `try_your_own` | 10 | Propose and evaluate your own policy-gradient estimator. |

Total: 90 points.

## Experiment Deliverables To Track

- Prompting baseline metrics and qualitative examples.
- Standard GRPO learning curves and rollout examples before/after training.
- Learning-rate sweep plot.
- Prompt ablation plots/commentary.
- On-policy variant comparison plots/commentary.
- Off-policy algorithm comparison plots/commentary.
- Final writeup discussion for derivation and "think about" questions.

## Useful Search Terms

- `OLMo-2-0425-1B`
- `GSM8K`
- `GRPO`
- `Dr. GRPO`
- `MaxRL`
- `RFT`
- `GSPO`
- `response_mask`
- `importance_reweighting_method`
- `advantage_normalizer`
- `loss_normalization`
- `r1_zero`
- `question_only`
- `r1_zero_three_shot`

