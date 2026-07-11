import torch



n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0



你这份脚本主要问题是：EvalRecord/GSM8KExample 类型没定义、evaluate_prompt 返回值不一致、args.save_every 没有 parser 参数、
保存路径是目录却直接 torch.save(..., out)、wandb/logging 部分没接上、eval 采样可能超过验证集长度。我尽量保留了你原来的结构、变量名和注释，只做必要补全和修正。

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
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
        help="Hugging Face 模型名称或本地模型路径。",
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
        default=Path("cs336_alignment/prompts/r1_zero.prompt"),
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/gsm8k/train.jsonl"),
        help="GSM8K 训练集 jsonl 文件路径。",
    )

    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
        help="GSM8K 验证集 jsonl 文件路径。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_standard_on_policy"),
        help="保存日志、rollout 结果和模型检查点的目录。",
    )

    # ------------------------------------------------------------------
    # Device / reproducibility
    # ------------------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。",
    )

    parser.add_argument(
        "--train-gpu",
        type=int,
        default=0,
        help="用于 Hugging Face 模型训练的 GPU 编号。",
    )

    parser.add_argument(
        "--vllm-gpu",
        type=int,
        default=1,
        help="用于 vLLM 生成 rollout 的 GPU 编号。",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM 使用的 GPU 显存比例。",
    )

    # ------------------------------------------------------------------
    # vLLM sampling hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="rollout 生成时使用的采样温度。",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p 采样参数。",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最多生成的 token 数量。",
    )

    parser.add_argument(
        "--rollout-generation-batch-size",
        type=int,
        default=None,
        help="rollout 生成时 vLLM 内部使用的 batch size。",
    )

    parser.add_argument(
        "--eval-generation-batch-size",
        type=int,
        default=128,
        help="验证集生成时 vLLM 内部使用的 batch size。",
    )

    # ------------------------------------------------------------------
    # GRPO training hyperparameters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="优化器学习率。",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="优化器权重衰减系数。",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="rollout batch 的数量，也就是 GRPO 更新步数。",
    )

    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
        help="每个 rollout batch 中生成的回答数量。",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
        help="一次训练更新中使用的 rollout 回答数量。",
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="每个 prompt 采样生成的回答数量。",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="梯度累积使用的 microbatch 数量。",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="梯度裁剪的范数上限。设为负数表示关闭梯度裁剪。",
    )

    parser.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-6,
        help="advantage 归一化时使用的小常数 epsilon。",
    )

    # ------------------------------------------------------------------
    # Evaluation / rollout logging
    # ------------------------------------------------------------------

    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="每隔多少个 rollout batch 在验证集上评估一次。",
    )

    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=1024,
        help="用于验证评估的最大样本数量。",
    )

    parser.add_argument(
        "--train-rollout-log-every",
        type=int,
        default=40,
        help="每隔多少个 rollout batch 保存一次当前训练 rollout。",
    )

    parser.add_argument(
        "--num-rollouts-to-log",
        type=int,
        default=16,
        help="保存用于定性检查的 rollout 数量。",
    )

    # ------------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cs336-assignment5",
        help="Weights & Biases 项目名称。",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases 运行名称。",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="关闭 wandb 日志记录。",
    )

    args = parser.parse_args()

    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError(
            "--rollout-batch-size must be divisible by --group-size."
        )

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "--train-batch-size must be divisible by "
            "--gradient-accumulation-steps."
        )

    if args.max_grad_norm < 0:
        args.max_grad_norm = None

    return args





def evaluate_prompt(
    reward_fn,
    completions,
    examples: list[GSM8KExample],
) -> tuple[dict[str, float | int], list[EvalRecord]]:

    total_reward_sum = 0.0
    format_reward_sum = 0.0
    answer_reward_sum = 0.0

    # zip(examples, completions) 会一一配对:
    # 第 i 个 example 对应第 i 个 completion。
    for example, completion in zip(examples, completions):
        response = completion.text
        ground_truth = example.final_answer

        reward_dict = reward_fn(response, ground_truth)

        reward = float(reward_dict["reward"])
        format_reward = float(reward_dict["format_reward"])
        answer_reward = float(reward_dict["answer_reward"])


        # 累加，用于后面计算平均值。
        total_reward_sum += reward
        format_reward_sum += format_reward
        answer_reward_sum += answer_reward

        n = len(examples)

        # 汇总指标。
        metrics: dict[str, float | int] = {
            "num_examples": n,
            "mean_reward": total_reward_sum / n,
            "mean_format_reward": format_reward_sum / n,
            "mean_answer_reward": answer_reward_sum / n,
        }
    return metrics

def make_prompts_ground_truths(
    examples,
    prompt_template: str,
    num_prompts,
    group_size: int = 1,
):
    sampled_examples = random.sample(examples, num_prompts)
    prompts = []
    ground_truths = []
    # questions = []

    for example in sampled_examples:
        prompt = apply_prompt_template(
            prompt_template,
            example.question,
        )

        for _ in range(group_size):
            prompts.append(prompt)
            ground_truths.append(example.final_answer)
            # questions.append(example.question)

    return prompts, ground_truths



def main() -> None:

    

    args = parse_args()
    # 读取 GSM8K 数据。
    examples = load_gsm8k_jsonl(args.train_path)
    val_examples = load_gsm8k_jsonl(args.val_path)



    prompt_template=load_prompt_template(args.prompt_path)
    
    # 用 get_reward_fn("r1_zero") 选择 reward function。
    reward_fn = get_reward_fn(args.prompt_name)
    
    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p ,
        "max_tokens": args.max_tokens,
        "n": 1,
    }

    if args.prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        # 当模型生成 </answer> 时停止。
        sampling_params["stop"] = ["</answer>"]

        # include_stop_str_in_output=True 表示输出文本中保留 </answer>。这对 r1_zero_reward_fn 解析格式可能有帮助。
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



    num_prompts = args.rollout_batch_size // args.group_size
    out=args.output_dir
    step=0
    while step<=args.max_steps:
        #同步模型权重到 vLLM 保证 rollout 来自当前 policy。
        server.sync_policy_weights(model) 

        #预处理后的 train set 抽一批 prompts假设抽 rollout_batch_size 个问题。


        # 用 vLLM 生成 rollouts每个 prompt 生成重复 group_size 个 responses。
        num_prompts = args.rollout_batch_size // args.group_size
        repeated_prompts,repeated_ground_truths=make_prompts_ground_truths(examples,
                        prompt_template,
                        num_prompts,
                        args.group_size)
        
        completions = server.generate_completions(
            prompts=repeated_prompts,
            sampling_params=sampling_params,
            batch_size=args.rollout_generation_batch_size,
        )
        rollout_responses = [completion.text for completion in completions]
        # 展开成 GRPO 需要的 batch

        total_loss, metadata=grpo_train_step(
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






        if step % args.train_rollout_log_every==0:

            args.num_rollouts_to_log


        # 调用 grpo_train_step
        if step % args.eval_every == 0:
                # 对每个问题套用 prompt 模板。prompts 的长度和 examples 一样。
            server.sync_policy_weights(model) 
            model.eval()
            with torch.no_grad():
                val_sampled_examples = random.sample(val_examples, args.max_eval_examples)
                val_sampled_prompts = [apply_prompt_template(
                        prompt_template,
                        example.question,
                    ) for example in val_sampled_examples]


                val_sampled_completions = server.generate_completions(
                prompts=val_sampled_prompts ,
                sampling_params=sampling_params,
                batch_size=args.eval_generation_batch_size)

                rollout_responses = [completion.text for completion in val_sampled_completions]
                metrics, records=evaluate_prompt(reward_fn,
                                val_sampled_completions,
                                val_sampled_examples)
  

            model.train()
# [:args.max_eval_examples]



        if step % args.save_every == 0:
            # Save the model weights
            model.save_pretrained(save_directory=args.output_dir)
            tokenizer.save_pretrained(save_directory=args.output_dir)
            torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iteration": step,
                    },
                    out,
                )


        step+=1
