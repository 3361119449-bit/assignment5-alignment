from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.optim import AdamW

# from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.checkpoint import get_model_and_tokenizer

# 因为这个脚本也放在 scripts/ 目录下，所以可以直接这样 import。
from prompting_baselines import (
    apply_prompt_template,
    get_reward_fn,
    load_gsm8k_jsonl,
    load_prompt_template,
    save_json,
    save_jsonl,
)

from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Run standard on-policy GRPO on GSM8K with OLMo-2-0425-1B."
    )

    # ------------------------------------------------------------------
    # Model / data / prompt
    # ------------------------------------------------------------------

    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="Hugging Face model name or local model path.",
    )

    parser.add_argument(
        "--prompt-name",
        type=str,
        default="r1_zero",
        choices=["question_only", "r1_zero", "r1_zero_three_shot"],
    )

    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help="Path to prompt template. If omitted, inferred from --prompt-name.",
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/gsm8k/train.jsonl"),
        help="Path to GSM8K training jsonl file.",
    )

    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
        help="Path to GSM8K validation jsonl file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_standard_on_policy"),
        help="Directory for saving logs, rollouts, and checkpoints.",
    )

    # ------------------------------------------------------------------
    # Device / reproducibility
    # ------------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--train-gpu",
        type=int,
        default=0,
        help="GPU id for Hugging Face model training.",
    )

    parser.add_argument(
        "--vllm-gpu",
        type=int,
        default=1,
        help="GPU id for vLLM rollout generation.",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
    )

    # ------------------------------------------------------------------
    # vLLM sampling hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for rollout generation.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--rollout-generation-batch-size",
        type=int,
        default=None,
        help="Internal vLLM batch size for rollout generation.",
    )

    parser.add_argument(
        "--eval-generation-batch-size",
        type=int,
        default=128,
        help="Internal vLLM batch size for validation generation.",
    )

    # ------------------------------------------------------------------
    # GRPO training hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Optimizer learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Optimizer weight decay.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Number of rollout batches / GRPO update steps.",
    )

    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
        help="Number of rollout responses per rollout batch.",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="Number of rollout responses used for one training update.",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of responses sampled per prompt.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Number of microbatches for gradient accumulation.",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Use negative value to disable.",
    )

    parser.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-6,
        help="Small epsilon for advantage normalization.",
    )

    # ------------------------------------------------------------------
    # Evaluation / rollout logging / saving
    # ------------------------------------------------------------------

    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate on validation set every N rollout batches.",
    )

    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=1024,
        help="Number of validation examples to evaluate.",
    )

    parser.add_argument(
        "--train-rollout-log-every",
        type=int,
        default=40,
        help="Save current training rollouts every N rollout batches.",
    )

    parser.add_argument(
        "--num-rollouts-to-log",
        type=int,
        default=16,
        help="Number of rollouts to save for qualitative inspection.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save model checkpoint every N rollout batches.",
    )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cs336-assignment5",
        help="Weights & Biases project name.",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name.",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )

    args = parser.parse_args()

    if args.prompt_path is None:
        prompt_paths = {
            "question_only": Path("cs336_alignment/prompts/question_only.prompt"),
            "r1_zero": Path("cs336_alignment/prompts/r1_zero.prompt"),
            "r1_zero_three_shot": Path(
                "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
            ),
        }
        args.prompt_path = prompt_paths[args.prompt_name]

    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError(
            "--rollout-batch-size must be divisible by --group-size."
        )

    if args.train_batch_size % args.group_size != 0:
        raise ValueError(
            "--train-batch-size must be divisible by --group-size."
        )

    if args.train_batch_size > args.rollout_batch_size:
        raise ValueError(
            "--train-batch-size must be <= --rollout-batch-size."
        )

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "--train-batch-size must be divisible by "
            "--gradient-accumulation-steps."
        )

    if args.max_grad_norm < 0:
        args.max_grad_norm = None

    return args


def set_seed(seed: int) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    """将 Python 对象保存为 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """将多个字典记录逐行保存为 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def to_loggable_value(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value

    value = value.detach().cpu()

    return value.item() if value.numel() == 1 else value.tolist()


def to_loggable_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: to_loggable_value(v) for k, v in metadata.items()}


def evaluate_prompt(
    reward_fn,
    completions,
    examples,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:

    total_reward_sum = 0.0
    format_reward_sum = 0.0
    answer_reward_sum = 0.0
    response_length_sum = 0.0

    category_1_count = 0
    category_2_count = 0
    category_3_count = 0
    other_count = 0

    records: list[dict[str, Any]] = []

    # zip(examples, completions) 会一一配对:
    # 第 i 个 example 对应第 i 个 completion。
    for example, completion in zip(examples, completions):
        response = completion.text
        ground_truth = example.final_answer

        reward_dict = reward_fn(response, ground_truth)

        reward = float(reward_dict["reward"])
        format_reward = float(reward_dict["format_reward"])
        answer_reward = float(reward_dict["answer_reward"])
        response_length = len(completion.token_ids)

        # 累加，用于后面计算平均值。
        total_reward_sum += reward
        format_reward_sum += format_reward
        answer_reward_sum += answer_reward
        response_length_sum += response_length

        if format_reward == 1.0 and answer_reward == 1.0:
            category = 1
            category_1_count += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            category = 2
            category_2_count += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            category = 3
            category_3_count += 1
        else:
            category = -1
            other_count += 1

        records.append(
            {
                "question": example.question,
                "ground_truth": ground_truth,
                "response": response,
                "finish_reason": completion.finish_reason,
                "response_length": response_length,
                "reward": reward,
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "category": category,
            }
        )

    n = len(records)

    if n == 0:
        raise ValueError("No evaluation records were produced.")

    # 汇总指标。
    metrics: dict[str, float | int] = {
        "num_examples": n,
        "mean_reward": total_reward_sum / n,
        "mean_format_reward": format_reward_sum / n,
        "mean_answer_reward": answer_reward_sum / n,
        "avg_response_length": response_length_sum / n,
        "category_1_count": category_1_count,
        "category_2_count": category_2_count,
        "category_3_count": category_3_count,
        "other_count": other_count,
    }

    return metrics, records


def make_prompts_ground_truths(
    examples,
    prompt_template: str,
    num_prompts,
    group_size: int = 1,
):
    if num_prompts <= len(examples):
        sampled_examples = random.sample(examples, num_prompts)
    else:
        sampled_examples = random.choices(examples, k=num_prompts)
    # 正常训练时无放回抽样；小数据调试或数据不足时允许重复抽样。


    prompts = []
    ground_truths = []
    questions = []

    for example in sampled_examples:
        prompt = apply_prompt_template(
            prompt_template,
            example.question,
        )

        for _ in range(group_size):
            prompts.append(prompt)
            ground_truths.append(example.final_answer)
            questions.append(example.question)

    return prompts, ground_truths, questions


def make_train_rollout_records(
    step: int,
    prompts: list[str],
    questions: list[str],
    ground_truths: list[str],
    rollout_responses: list[str],
    completions,
    reward_fn,
    num_rollouts_to_log: int,
) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []

    n = min(num_rollouts_to_log, len(rollout_responses))

    for idx in range(n):
        reward_dict = reward_fn(rollout_responses[idx], ground_truths[idx])

        records.append(
            {
                "step": step,
                "index": idx,
                "question": questions[idx],
                "prompt": prompts[idx],
                "ground_truth": ground_truths[idx],
                "response": rollout_responses[idx],
                "finish_reason": completions[idx].finish_reason,
                "response_length": len(completions[idx].token_ids),
                "reward": float(reward_dict["reward"]),
                "format_reward": float(reward_dict["format_reward"]),
                "answer_reward": float(reward_dict["answer_reward"]),
            }
        )

    return records


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    output_dir: Path,
    step: int,
) -> None:

    checkpoint_dir = output_dir / f"checkpoint_step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the model weights.
    model.save_pretrained(save_directory=checkpoint_dir)
    tokenizer.save_pretrained(save_directory=checkpoint_dir)

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "iteration": step,
        },
        checkpoint_dir / "training_state.pt",
    )


def main() -> None:

    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_rollouts").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval").mkdir(parents=True, exist_ok=True)

    # 读取 GSM8K 数据。
    examples = load_gsm8k_jsonl(args.train_path)
    val_examples = load_gsm8k_jsonl(args.val_path)

    prompt_template = load_prompt_template(args.prompt_path)

    # 用 get_reward_fn("r1_zero") 选择 reward function。
    reward_fn = get_reward_fn(args.prompt_name)

    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "prompt_path": str(args.prompt_path),
                "train_path": str(args.train_path),
                "val_path": str(args.val_path),
                "output_dir": str(args.output_dir),
            },
        )

    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": 1,
    }

    if args.prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        # 当模型生成 </answer> 时停止。
        sampling_params["stop"] = ["</answer>"]

        # include_stop_str_in_output=True 表示输出文本中保留 </answer>。
        # 这对 r1_zero_reward_fn 解析格式可能有帮助。
        sampling_params["include_stop_str_in_output"] = True

    # 加载 tokenizer 和 HF training model。
    train_device = f"cuda:{args.train_gpu}"
    model, tokenizer = get_model_and_tokenizer(
        model_id_or_dir=args.model_name,
        device=train_device,
    )

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 初始化 vLLM server。
    server = VLLMServer(
        model_id=args.model_name,
        gpu=args.vllm_gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 启动 vLLM server。
    server.start()

    server.init_weight_sync(policy_device=train_device)

    # 调用 vLLM 批量生成。
    # 返回值是 list[VLLMCompletion]。
    step = 0

    while step < args.max_steps:

        # 同步模型权重到 vLLM，保证 rollout 来自当前 policy。
        server.sync_policy_weights(model)

        # 用 vLLM 生成 rollouts。
        # 每个 prompt 生成重复 group_size 个 responses。
        num_prompts = args.rollout_batch_size // args.group_size

        repeated_prompts, repeated_ground_truths, repeated_questions = make_prompts_ground_truths(
            examples,
            prompt_template,
            num_prompts,
            args.group_size,
        )

        completions = server.generate_completions(
            prompts=repeated_prompts,
            sampling_params=sampling_params,
            batch_size=args.rollout_generation_batch_size,
        )
        # batch_size=一次送多少个 prompt 给 vLLM server 生成。

        rollout_responses = [completion.text for completion in completions]



        # 展开成 GRPO 需要的 batch，然后调用 grpo_train_step。
        # 注意：loss、gradient norm、token entropy、train reward 等应由 grpo_train_step 的 metadata 返回。
        total_loss, metadata = grpo_train_step(
            model,
            tokenizer,
            optimizer,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            reward_fn,
            repeated_prompts,
            rollout_responses,
            repeated_ground_truths,
            args.group_size,
            baseline="mean",
            advantage_eps=args.advantage_eps,
            advantage_normalizer="std",
            importance_reweighting_method="none",
            old_log_probs=None,
            cliprange=None,
            loss_normalization="sequence",
            normalization_constant=None,
        )

        metadata = to_loggable_dict(metadata)

        train_log = {
            "step": step,
            "train/total_loss": float(total_loss),
            **{
                f"train/{key}": value
                for key, value in metadata.items()
            },
        }

        print(train_log)

        if not args.disable_wandb:
            wandb.log(train_log, step=step)

        if step % args.train_rollout_log_every == 0:
            train_rollout_records = make_train_rollout_records(
                step=step,
                prompts=repeated_prompts,
                questions=repeated_questions,
                ground_truths=repeated_ground_truths,
                rollout_responses=rollout_responses,
                completions=completions,
                reward_fn=reward_fn,
                num_rollouts_to_log=args.num_rollouts_to_log,
            )

            write_jsonl(
                args.output_dir / "train_rollouts" / f"step_{step:06d}.jsonl",
                train_rollout_records,
            )

        if step % args.eval_every == 0:

            # 对每个问题套用 prompt 模板。prompts 的长度和 examples 一样。
            server.sync_policy_weights(model)
            model.eval()

            with torch.no_grad():
                num_eval_examples = min(args.max_eval_examples, len(val_examples))

                val_sampled_examples = random.sample(
                    val_examples,
                    num_eval_examples,
                )

                val_sampled_prompts = [
                    apply_prompt_template(
                        prompt_template,
                        example.question,
                    )
                    for example in val_sampled_examples
                ]

                val_sampled_completions = server.generate_completions(
                    prompts=val_sampled_prompts,
                    sampling_params=sampling_params,
                    batch_size=args.eval_generation_batch_size,
                )

                eval_metrics, eval_records = evaluate_prompt(
                    reward_fn,
                    val_sampled_completions,
                    val_sampled_examples,
                )

            eval_log = {
                "step": step,
                **{
                    f"eval/{key}": value
                    for key, value in eval_metrics.items()
                },
            }

            print(eval_log)

            write_json(
                args.output_dir / "eval" / f"metrics_step_{step:06d}.json",
                eval_log,
            )

            write_jsonl(
                args.output_dir / "eval" / f"records_step_{step:06d}.jsonl",
                eval_records,
            )

            if not args.disable_wandb:
                wandb.log(eval_log, step=step)

            model.train()

        if step % args.save_every == 0:
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=args.output_dir,
                step=step,
            )

        step += 1

    # 最后再保存一次。
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        output_dir=args.output_dir,
        step=step,
    )

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()



# 注意：这个脚本会自动把 grpo_train_step 返回的所有 metadata 记录成 train/... 指标。所以你需要在 grpo_train_step 里面把这些 key 放进 metadata，例如：

# metadata["grad_norm"] = float(grad_norm)
# metadata["token_entropy"] = float(mean_token_entropy)
# metadata["mean_reward"] = float(mean_reward)
# metadata["mean_format_reward"] = float(mean_format_reward)

# 这样 wandb 里就会自动出现：

# train/grad_norm
# train/token_entropy
# train/mean_reward
# train/mean_format_reward
# eval/mean_reward
# eval/mean_format_reward
# eval/avg_response_length

# 定期 rollout 样例会保存在：

# outputs/grpo_standard_on_policy/train_rollouts/

# validation 结果会保存在：

# outputs/grpo_standard_on_policy/eval/