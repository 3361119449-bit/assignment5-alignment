from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# vLLM 推理服务器封装类。
# 作业提供的 cs336_alignment/vllm_utils.py 里定义了 VLLMServer。
from cs336_alignment.vllm_utils import VLLMServer

# 作业提供的 GSM8K 打分函数。
# question_only_prompt 和 r1_zero_prompt 的输出格式不同，所以要用不同 reward function。
from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)


# ---------------------------------------------------------------------
# 1. 数据结构定义
# ---------------------------------------------------------------------

@dataclass
class GSM8KExample:
    """
    表示 GSM8K 数据集中的一条样本。

    GSM8K 每条数据大概长这样:
        {
            "question": "...",
            "answer": "... reasoning ... #### 72"
        }

    Attributes:
        question:
            str，原始数学题题目。

        full_answer:
            str，GSM8K 原始 answer 字段。
            通常包含推理过程和最终答案，例如：
                "... reasoning ... #### 72"

        final_answer:
            str，从 full_answer 中提取出的最终答案。
            通常是 #### 后面的内容。
    """

    question: str
    full_answer: str
    final_answer: str


@dataclass
class EvalRecord:
    """
    表示一次模型生成和评分的完整记录。

    每个 prompt 对每个 GSM8K 样本都会生成一个 EvalRecord。
    后面会把这些记录保存成 jsonl，方便你之后人工检查模型输出。

    Attributes:
        prompt_name:
            str，使用的 prompt 名称，例如 question_only、r1_zero。

        question:
            str，GSM8K 原始题目。

        ground_truth:
            str，标准最终答案。

        response:
            str，模型生成的回答。

        reward:
            float，总 reward。

        format_reward:
            float，格式 reward。

        answer_reward:
            float，答案 reward。

        category:
            str，根据 format_reward 和 answer_reward 得到的分类。

        finish_reason:
            str 或 None，vLLM 返回的停止原因。
    """

    prompt_name: str
    question: str
    ground_truth: str
    response: str
    reward: float
    format_reward: float
    answer_reward: float
    category: str
    finish_reason: str | None


# ---------------------------------------------------------------------
# 2. 读取 GSM8K 数据
# ---------------------------------------------------------------------

def load_gsm8k_jsonl(path: str | Path) -> list[GSM8KExample]:
    """
    读取 GSM8K jsonl 文件，并把每一行转换成 GSM8KExample。

    Args:
        path:
            str 或 Path，GSM8K jsonl 文件路径。
            文件中每一行应是一个 JSON 对象，格式通常为：
                {
                    "question": str,
                    "answer": str
                }

            answer 通常形如：
                "... reasoning ... #### 72"

    Returns:
        list[GSM8KExample]:
            解析后的 GSM8K 样本列表。
            每个元素包含：
                question:
                    原始题目文本。
                full_answer:
                    原始 answer 字段，包括推理过程和最终答案。
                final_answer:
                    从 full_answer 中提取出的最终答案。
    """

    # 保证 path 一定是 Path 对象。
    # 这样后面可以用 path.open(), path.exists() 等 pathlib 方法。
    path = Path(path)

    examples: list[GSM8KExample] = []

    # 用 with 打开文件，可以保证读完后自动关闭文件。
    with path.open("r", encoding="utf-8") as f:
        # jsonl 是一行一个 JSON，所以这里逐行读取。
        for line in f:
            # 去掉每行开头和结尾的空白字符，比如 \n、空格、\t。
            line = line.strip()

            # 如果有空行，跳过。
            # 否则 json.loads("") 会报错。
            if not line:
                continue

            # 把一行 JSON 字符串解析成 Python 字典。
            obj = json.loads(line)

            # GSM8K 每行一般有 question 和 answer 两个字段。
            question = obj["question"]
            full_answer = obj["answer"]

            # GSM8K 的 answer 通常形如:
            #   "... reasoning ... #### 72"
            # 这里我们只取 #### 后面的最终答案。
            if "####" in full_answer:
                final_answer = full_answer.split("####")[-1].strip()
            else:
                # 理论上 GSM8K 应该有 ####。
                # 这里写 else 是为了代码更健壮。
                final_answer = full_answer.strip()

            # 把这一条样本保存成 GSM8KExample 对象。
            examples.append(
                GSM8KExample(
                    question=question,
                    full_answer=full_answer,
                    final_answer=final_answer,
                )
            )

    return examples


# ---------------------------------------------------------------------
# 3. 读取和应用 prompt 模板
# ---------------------------------------------------------------------

def load_prompt_template(path: str | Path) -> str:
    """
    读取 prompt 模板文件。

    Args:
        path:
            str 或 Path，prompt 模板文件路径。
            例如：
                cs336_alignment/prompts/question_only.prompt
                cs336_alignment/prompts/r1_zero.prompt
                cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt

    Returns:
        str:
            prompt 模板文件的完整文本内容。
            模板中通常包含 {question} 占位符。

    Notes:
        这里不要用 template.format(question=question)。

        原因是 prompt 里可能包含 LaTeX 或其他大括号，比如 \\boxed{}。
        format 会把所有 {} 都当成格式化占位符，容易报错。

        所以后面用：
            template.replace("{question}", question)

        只替换精确的 {question}。
    """

    return Path(path).read_text(encoding="utf-8")


def apply_prompt_template(template: str, question: str) -> str:
    """
    将 prompt 模板中的 {question} 替换成实际题目。

    Args:
        template:
            str，prompt 模板文本。
            其中应包含精确占位符 {question}。

        question:
            str，GSM8K 的具体题目文本。

    Returns:
        str:
            替换完成后的最终 prompt，可直接传给模型生成。
    """

    return template.replace("{question}", question)


def load_all_prompt_templates(prompt_dir: str | Path) -> dict[str, str]:
    """
    从 prompt 目录中读取作业要求的三个 prompt 模板。

    Args:
        prompt_dir:
            str 或 Path，存放 prompt 文件的目录。
            该目录下应包含：
                question_only.prompt
                r1_zero.prompt
                r1_zero_three_shot_gsm8k.prompt

    Returns:
        dict[str, str]:
            从 prompt 名称到模板内容的映射：
                {
                    "question_only": str,
                    "r1_zero": str,
                    "r1_zero_three_shot": str,
                }
    """

    prompt_dir = Path(prompt_dir)

    # prompt_dir / "xxx.prompt" 是 pathlib 的路径拼接写法。
    # 例如:
    #   Path("cs336_alignment/prompts") / "r1_zero.prompt"
    # 得到:
    #   Path("cs336_alignment/prompts/r1_zero.prompt")
    prompt_files = {
        "question_only": prompt_dir / "question_only.prompt",
        "r1_zero": prompt_dir / "r1_zero.prompt",
        "r1_zero_three_shot": prompt_dir / "r1_zero_three_shot_gsm8k.prompt",
    }

    # 字典推导式:
    # 对 prompt_files 里的每个 name, path，
    # 读取文件内容，并返回 name -> 模板内容。
    return {
        name: load_prompt_template(path)
        for name, path in prompt_files.items()
    }


# ---------------------------------------------------------------------
# 4. 根据 prompt 选择 reward function 和 sampling 参数
# ---------------------------------------------------------------------

def get_reward_fn(prompt_name: str) -> Callable[[str, str], dict[str, float]]:
    """
    根据 prompt 名称选择对应的 reward function。

    Args:
        prompt_name:
            str，prompt 类型名称。
            支持：
                "question_only"
                "r1_zero"
                "r1_zero_three_shot"

    Returns:
        Callable[[str, str], dict[str, float]]:
            对应的 reward function。
            输入为：
                response:
                    模型生成文本。
                ground_truth:
                    标准最终答案。
            输出为包含 reward 统计的字典，例如：
                {
                    "reward": float,
                    "format_reward": float,
                    "answer_reward": float,
                }

    Raises:
        ValueError:
            当 prompt_name 不属于支持的 prompt 类型时抛出。
    """

    # question_only 不要求 <think> / <answer> 标签，
    # 所以用 question_only_reward_fn。
    if prompt_name == "question_only":
        return question_only_reward_fn

    # r1_zero 和 r1_zero_three_shot 都要求 <think> / <answer> 格式，
    # 所以用 r1_zero_reward_fn。
    if prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        return r1_zero_reward_fn

    raise ValueError(f"Unknown prompt_name: {prompt_name}")


def get_sampling_params(prompt_name: str) -> dict[str, Any]:
    """
    根据 prompt 类型构造 vLLM sampling 参数。

    Args:
        prompt_name:
            str，prompt 类型名称。
            支持：
                "question_only"
                "r1_zero"
                "r1_zero_three_shot"

    Returns:
        dict[str, Any]:
            vLLM 使用的 sampling 参数。
            至少包含：
                temperature:
                    float，采样温度。
                top_p:
                    float，nucleus sampling 参数。
                max_tokens:
                    int，最大生成长度。
                n:
                    int，每个 prompt 生成几个 completion。

            对于 r1_zero 和 r1_zero_three_shot，还会包含：
                stop:
                    list[str]，停止字符串，这里是 ["</answer>"]。
                include_stop_str_in_output:
                    bool，是否在返回文本中保留 stop string。
    """

    # 作业要求:
    # - temperature = 1.0
    # - top_p = 1.0
    # - max_tokens = 512
    # - n = 1
    sampling_params: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
    }

    # 注意:
    # stop = ["</answer>"] 只用于 r1_zero 和 r1_zero_three_shot。
    # question_only 不应该使用这个 stop string。
    if prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        # 当模型生成 </answer> 时停止。
        sampling_params["stop"] = ["</answer>"]

        # include_stop_str_in_output=True 表示输出文本中保留 </answer>。
        # 这对 r1_zero_reward_fn 解析格式可能有帮助。
        sampling_params["include_stop_str_in_output"] = True

    return sampling_params


# ---------------------------------------------------------------------
# 5. 按作业要求分类模型输出
# ---------------------------------------------------------------------

def categorize_reward(format_reward: float, answer_reward: float) -> str:
    """
    根据 format_reward 和 answer_reward 给模型输出分类。

    Args:
        format_reward:
            float，格式 reward。
            通常为 1.0 或 0.0。
            1.0 表示输出格式正确。

        answer_reward:
            float，答案 reward。
            通常为 1.0 或 0.0。
            1.0 表示最终答案正确。

    Returns:
        str:
            输出所属类别：
                "correct_format_and_answer":
                    格式正确，答案正确。
                "format_correct_answer_wrong":
                    格式正确，答案错误。
                "format_wrong_answer_wrong":
                    格式错误，答案错误。
                "answer_correct_but_format_wrong":
                    答案正确，但格式错误。
    """

    # 作业主要要求统计三类:
    #   1. format_reward = 1 且 answer_reward = 1
    #   2. format_reward = 1 且 answer_reward = 0
    #   3. format_reward = 0 且 answer_reward = 0
    #
    # 这里额外加了一类:
    #   answer_correct_but_format_wrong
    #
    # 这是为了防止出现 format_reward = 0 但 answer_reward = 1 的情况。
    if format_reward == 1 and answer_reward == 1:
        return "correct_format_and_answer"

    if format_reward == 1 and answer_reward == 0:
        return "format_correct_answer_wrong"

    if format_reward == 0 and answer_reward == 0:
        return "format_wrong_answer_wrong"

    return "answer_correct_but_format_wrong"


# ---------------------------------------------------------------------
# 6. 评估某一个 prompt
# ---------------------------------------------------------------------

def evaluate_one_prompt(
    server: VLLMServer,
    prompt_name: str,
    template: str,
    examples: list[GSM8KExample],
    batch_size: int | None = None,
) -> tuple[dict[str, float | int], list[EvalRecord]]:
    """
    使用某一种 prompt 模板在 GSM8K 样本上进行评估。

    Args:
        server:
            VLLMServer，已经启动的 vLLM 推理服务器。

        prompt_name:
            str，当前评估的 prompt 名称。
            用于选择 reward function 和 sampling 参数。

        template:
            str，prompt 模板文本。
            其中通常包含 {question} 占位符。

        examples:
            list[GSM8KExample]，要评估的 GSM8K 样本列表。

        batch_size:
            int 或 None，vLLM 生成时的 batch size。
            如果为 None，则由 VLLMServer 内部决定。

    Returns:
        tuple[dict[str, float | int], list[EvalRecord]]:

            metrics:
                dict[str, float | int]，整体评估指标，包括：
                    num_examples:
                        int，评估样本数。
                    mean_reward:
                        float，平均总 reward。
                    mean_format_reward:
                        float，平均格式 reward。
                    mean_answer_reward:
                        float，平均答案 reward。
                    correct_format_and_answer:
                        int，格式正确且答案正确的数量。
                    format_correct_answer_wrong:
                        int，格式正确但答案错误的数量。
                    format_wrong_answer_wrong:
                        int，格式错误且答案错误的数量。
                    answer_correct_but_format_wrong:
                        int，答案正确但格式错误的数量。

            records:
                list[EvalRecord]，每个样本的详细生成和评分记录。

    Raises:
        RuntimeError:
            当 vLLM 返回的 completion 数量和 examples 数量不一致时抛出。
    """

    reward_fn = get_reward_fn(prompt_name)
    sampling_params = get_sampling_params(prompt_name)

    # 对每个问题套用 prompt 模板。
    # prompts 的长度和 examples 一样。
    prompts = [
        apply_prompt_template(template, example.question)
        for example in examples
    ]

    # 调用 vLLM 批量生成。
    # 返回值是 list[VLLMCompletion]。
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    # 理论上一个 prompt 对应一个 completion。
    # 如果数量不一致，说明生成过程出了问题。
    if len(completions) != len(examples):
        raise RuntimeError(
            f"Number of completions does not match number of examples: "
            f"{len(completions)} vs {len(examples)}"
        )

    records: list[EvalRecord] = []

    # 初始化四种类别的计数器。
    counts = {
        "correct_format_and_answer": 0,
        "format_correct_answer_wrong": 0,
        "format_wrong_answer_wrong": 0,
        "answer_correct_but_format_wrong": 0,
    }

    total_reward_sum = 0.0
    format_reward_sum = 0.0
    answer_reward_sum = 0.0

    # zip(examples, completions) 会一一配对:
    # 第 i 个 example 对应第 i 个 completion。
    for example, completion in zip(examples, completions):
        response = completion.text
        ground_truth = example.final_answer

        # 调用作业提供的 reward function。
        # 返回类似:
        # {
        #     "reward": 0.0 or 1.0,
        #     "format_reward": 0.0 or 1.0,
        #     "answer_reward": 0.0 or 1.0,
        # }
        reward_dict = reward_fn(response, ground_truth)

        reward = float(reward_dict["reward"])
        format_reward = float(reward_dict["format_reward"])
        answer_reward = float(reward_dict["answer_reward"])

        # 按 format_reward 和 answer_reward 分类。
        category = categorize_reward(format_reward, answer_reward)
        counts[category] += 1

        # 累加，用于后面计算平均值。
        total_reward_sum += reward
        format_reward_sum += format_reward
        answer_reward_sum += answer_reward

        # 保存这一条具体样本的完整信息。
        # 后面可以人工检查模型输出。
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

    # 汇总指标。
    metrics: dict[str, float | int] = {
        "num_examples": n,
        "mean_reward": total_reward_sum / n,
        "mean_format_reward": format_reward_sum / n,
        "mean_answer_reward": answer_reward_sum / n,
        **counts,
    }

    return metrics, records


# ---------------------------------------------------------------------
# 7. 保存结果
# ---------------------------------------------------------------------

def save_json(path: str | Path, obj: Any) -> None:
    """
    将 Python 对象保存为格式化 JSON 文件。

    Args:
        path:
            str 或 Path，输出 JSON 文件路径。
            如果父目录不存在，会自动创建。

        obj:
            Any，要保存的 Python 对象。
            需要能被 json.dump 序列化。

    Returns:
        None
    """

    path = Path(path)

    # 如果输出目录不存在，就自动创建。
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, records: list[EvalRecord]) -> None:
    """
    将 EvalRecord 列表保存为 jsonl 文件。

    Args:
        path:
            str 或 Path，输出 jsonl 文件路径。
            如果父目录不存在，会自动创建。

        records:
            list[EvalRecord]，要保存的评估记录列表。
            每个 EvalRecord 会被转换成一个 JSON 对象，占一行。

    Returns:
        None
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # asdict(record) 会把 dataclass 对象转成普通 dict。
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# 8. 打印指标和样例
# ---------------------------------------------------------------------

def print_metrics(prompt_name: str, metrics: dict[str, float | int]) -> None:
    """
    在终端打印某个 prompt 的评估指标。

    Args:
        prompt_name:
            str，prompt 名称。

        metrics:
            dict[str, float | int]，评估指标字典。
            例如：
                {
                    "num_examples": int,
                    "mean_reward": float,
                    "mean_format_reward": float,
                    ...
                }

    Returns:
        None
    """

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
    """
    从指定类别中打印若干模型输出样例。

    Args:
        records:
            list[EvalRecord]，完整评估记录列表。

        category:
            str，要筛选的类别名称。
            例如：
                "correct_format_and_answer"
                "format_correct_answer_wrong"
                "format_wrong_answer_wrong"
                "answer_correct_but_format_wrong"

        max_examples:
            int，最多打印多少条样例。

    Returns:
        None
    """

    # 从所有 records 中筛选出指定类别。
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


# ---------------------------------------------------------------------
# 9. 主函数：解析命令行参数并运行完整评估
# ---------------------------------------------------------------------

def main() -> None:
    """
    命令行入口函数，运行完整 prompting baseline 评估流程。

    Args:
        None:
            该函数不直接接收 Python 参数。
            参数通过 argparse 从命令行读取，包括：
                --data-path:
                    GSM8K jsonl 数据路径。
                --prompt-dir:
                    prompt 模板目录。
                --model-id:
                    Hugging Face 模型名或本地模型路径。
                --gpu:
                    vLLM 使用的 GPU 编号。
                --seed:
                    随机种子。
                --gpu-memory-utilization:
                    vLLM 使用 GPU 显存的比例。
                --batch-size:
                    vLLM 生成时的 batch size。
                --max-examples:
                    调试时最多评估多少条样本。
                --output-dir:
                    评估结果输出目录。

    Returns:
        None:
            该函数会启动 vLLM server，运行三个 prompt 的评估，
            并将 metrics 和 records 保存到 output_dir。
    """

    parser = argparse.ArgumentParser()

    # GSM8K 数据路径。
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
    )

    # prompt 文件夹路径。
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("cs336_alignment/prompts"),
    )

    # Hugging Face 模型名或本地模型路径。
    parser.add_argument(
        "--model-id",
        type=str,
        default="allenai/OLMo-2-0425-1B",
    )

    # vLLM 使用哪张 GPU。
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    # 随机种子。
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # vLLM 最多使用多少比例的 GPU 显存。
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )

    # vLLM 内部生成时的 batch size。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    # 调试用。
    # 如果设置 --max-examples 20，就只跑前 20 条。
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="For debugging. If omitted, evaluate the full file.",
    )

    # 输出目录。
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/prompting_baselines"),
    )

    args = parser.parse_args()

    # 读取 GSM8K 数据。
    examples = load_gsm8k_jsonl(args.data_path)

    # 调试时可以只截取前 max_examples 条，避免一上来跑完整测试集。
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    # 读取三个 prompt 模板。
    templates = load_all_prompt_templates(args.prompt_dir)

    # 初始化 vLLM server。
    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 启动 vLLM server。
    server.start()

    # 保存所有 prompt 的指标。
    all_metrics: dict[str, dict[str, float | int]] = {}

    # 依次评估:
    #   question_only
    #   r1_zero
    #   r1_zero_three_shot
    for prompt_name, template in templates.items():
        metrics, records = evaluate_one_prompt(
            server=server,
            prompt_name=prompt_name,
            template=template,
            examples=examples,
            batch_size=args.batch_size,
        )

        all_metrics[prompt_name] = metrics

        # 在终端打印指标。
        print_metrics(prompt_name, metrics)

        # 保存每条样本的详细生成结果。
        save_jsonl(
            args.output_dir / f"{prompt_name}_records.jsonl",
            records,
        )

        # 打印每类样例，方便写作业 commentary。
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

    # 保存整体 metrics。
    save_json(
        args.output_dir / "metrics.json",
        all_metrics,
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# 调试运行示例
# ---------------------------------------------------------------------

# 调试时建议先跑小样本：
#
# uv run python scripts/eval_prompting_baselines.py \
#   --max-examples 20 \
#   --batch-size 8

# 确认没有路径、vLLM、reward function 的错误后，再跑完整测试集：
#
# uv run python scripts/eval_prompting_baselines.py \
#   --data-path data/gsm8k/test.jsonl \
#   --prompt-dir cs336_alignment/prompts \
#   --model-id allenai/OLMo-2-0425-1B \
#   --gpu 0 \
#   --batch-size 32