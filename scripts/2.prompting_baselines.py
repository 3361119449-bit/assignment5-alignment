from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cs336_alignment.vllm_utils import VLLMServer

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)


@dataclass
class GSM8KExample:
    """一条 GSM8K 样本。"""

    question: str
    full_answer: str
    final_answer: str


@dataclass
class EvalRecord:
    """一条评估记录，包含生成、评分和分类结果。"""

    prompt_name: str
    question: str
    ground_truth: str
    response: str
    reward: float
    format_reward: float
    answer_reward: float
    category: str
    finish_reason: str | None


def load_gsm8k_jsonl(path: str | Path) -> list[GSM8KExample]:
    """读取 GSM8K jsonl，并提取最终答案。"""

    path = Path(path)

    examples: list[GSM8KExample] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            question = obj["question"]
            full_answer = obj["answer"]

            if "####" in full_answer:
                final_answer = full_answer.split("####")[-1].strip()
            else:
                final_answer = full_answer.strip()

            examples.append(
                GSM8KExample(
                    question=question,
                    full_answer=full_answer,
                    final_answer=final_answer,
                )
            )

    return examples


def load_prompt_template(path: str | Path) -> str:
    """读取 prompt 模板文件。"""

    return Path(path).read_text(encoding="utf-8")


def apply_prompt_template(template: str, question: str) -> str:
    """替换模板中的 {question}。"""

    return template.replace("{question}", question)


def load_all_prompt_templates(prompt_dir: str | Path) -> dict[str, str]:
    """读取三个 prompt 模板。"""

    prompt_dir = Path(prompt_dir)

    prompt_files = {
        "question_only": prompt_dir / "question_only.prompt",
        "r1_zero": prompt_dir / "r1_zero.prompt",
        "r1_zero_three_shot": prompt_dir / "r1_zero_three_shot_gsm8k.prompt",
    }

    return {
        name: load_prompt_template(path)
        for name, path in prompt_files.items()
    }


def get_reward_fn(prompt_name: str) -> Callable[[str, str], dict[str, float]]:
    """根据 prompt 名称选择 reward function。"""

    if prompt_name == "question_only":
        return question_only_reward_fn

    if prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        return r1_zero_reward_fn

    raise ValueError(f"Unknown prompt_name: {prompt_name}")


def get_sampling_params(prompt_name: str) -> dict[str, Any]:
    """构造 vLLM sampling 参数。"""

    sampling_params: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
    }

    if prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        sampling_params["stop"] = ["</answer>"]

        sampling_params["include_stop_str_in_output"] = True

    return sampling_params


def categorize_reward(format_reward: float, answer_reward: float) -> str:
    """根据格式和答案 reward 分类。"""

    if format_reward == 1 and answer_reward == 1:
        return "correct_format_and_answer"

    if format_reward == 1 and answer_reward == 0:
        return "format_correct_answer_wrong"

    if format_reward == 0 and answer_reward == 0:
        return "format_wrong_answer_wrong"

    return "answer_correct_but_format_wrong"


def evaluate_one_prompt(
    server: VLLMServer,
    prompt_name: str,
    template: str,
    examples: list[GSM8KExample],
    batch_size: int | None = None,
) -> tuple[dict[str, float | int], list[EvalRecord]]:
    """用一个 prompt 模板评估所有样本。"""

    reward_fn = get_reward_fn(prompt_name)
    sampling_params = get_sampling_params(prompt_name)

    prompts = [
        apply_prompt_template(template, example.question)
        for example in examples
    ]

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )


    records: list[EvalRecord] = []

    counts = {
        "correct_format_and_answer": 0,
        "format_correct_answer_wrong": 0,
        "format_wrong_answer_wrong": 0,
        "answer_correct_but_format_wrong": 0,
    }

    total_reward_sum = 0.0
    format_reward_sum = 0.0
    answer_reward_sum = 0.0

    for example, completion in zip(examples, completions):
        response = completion.text
        ground_truth = example.final_answer

        reward_dict = reward_fn(response, ground_truth)

        reward = float(reward_dict["reward"])
        format_reward = float(reward_dict["format_reward"])
        answer_reward = float(reward_dict["answer_reward"])

        category = categorize_reward(format_reward, answer_reward)
        counts[category] += 1

        total_reward_sum += reward
        format_reward_sum += format_reward
        answer_reward_sum += answer_reward

        records.append(
            EvalRecord(
                prompt_name=prompt_name,
                question=example.question,
                ground_truth=ground_truth,
                response=response,
                reward=reward,
                format_reward=format_reward,
                answer_reward=answer_reward,
                category=category,
                finish_reason=completion.finish_reason,
            )
        )

    n = len(examples)

    metrics: dict[str, float | int] = {
        "num_examples": n,
        "mean_reward": total_reward_sum / n,
        "mean_format_reward": format_reward_sum / n,
        "mean_answer_reward": answer_reward_sum / n,
        **counts,
    }

    return metrics, records


def save_json(path: str | Path, obj: Any) -> None:
    """保存 JSON 文件。"""

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, records: list[EvalRecord]) -> None:
    """保存 EvalRecord 为 jsonl。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def print_metrics(prompt_name: str, metrics: dict[str, float | int]) -> None:
    """打印评估指标。"""

    print("=" * 80)
    print(f"Prompt: {prompt_name}")
    print("=" * 80)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()


def print_some_examples(
    records: list[EvalRecord],
    category: str,
    max_examples: int = 3,
) -> None:
    """打印指定类别的若干样例。"""

    selected = [
        record for record in records
        if record.category == category
    ]

    print("-" * 80)
    print(f"Examples for category: {category}")
    print("-" * 80)

    for record in selected[:max_examples]:
        print("Question:")
        print(record.question)
        print()

        print("Ground truth:")
        print(record.ground_truth)
        print()

        print("Response:")
        print(record.response)
        print()

        print(
            f"reward={record.reward}, "
            f"format_reward={record.format_reward}, "
            f"answer_reward={record.answer_reward}, "
            f"finish_reason={record.finish_reason}"
        )
        print("-" * 80)

    print()


def main() -> None:
    """解析命令行参数并运行完整评估。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
    )

    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("cs336_alignment/prompts"),
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="allenai/OLMo-2-0425-1B",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="For debugging. If omitted, evaluate the full file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prompting_baselines"),
    )

    args = parser.parse_args()

    examples = load_gsm8k_jsonl(args.data_path)

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    templates = load_all_prompt_templates(args.prompt_dir)

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    server.start()

    all_metrics: dict[str, dict[str, float | int]] = {}

    for prompt_name, template in templates.items():
        metrics, records = evaluate_one_prompt(
            server=server,
            prompt_name=prompt_name,
            template=template,
            examples=examples,
            batch_size=args.batch_size,
        )

        all_metrics[prompt_name] = metrics

        print_metrics(prompt_name, metrics)

        save_jsonl(
            args.output_dir / f"{prompt_name}_records.jsonl",
            records,
        )

        for category in [
            "correct_format_and_answer",
            "format_correct_answer_wrong",
            "format_wrong_answer_wrong",
            "answer_correct_but_format_wrong",
        ]:
            print_some_examples(
                records,
                category,
                max_examples=3,
            )

    save_json(
        args.output_dir / "metrics.json",
        all_metrics,
    )


if __name__ == "__main__":
    main()
