评分函数内部会自动从 ground_truth 中提取 #### 后面的 18 作为正确答案，你不需要手动解析。




#############################################################################
with path.open("r", encoding="utf-8") as f:
    for line in f:
        print(line)
适合大文件，因为可以一行一行处理，不必一次性加载进内存。
with 结束后文件会自动关闭。
    



#############################################################################
text = Path(path).read_text(encoding="utf-8")

等价于“打开文件 → 读取全部内容 → 关闭文件”。
直接返回整个文件内容，类型是 str。





#############################################################################
path = "data.txt"

这是一个字符串，类型是 str。

执行：

from pathlib import Path

path = Path(path)

之后，path 就变成了 Path 对象：






#############################################################################
line = line.strip()

意思是：去掉字符串 line 开头和结尾的空白字符，然后重新赋值给 line。




#############################################################################

obj = json.loads(line)

意思是：把一行 JSON 格式的字符串解析成 Python 对象。

这里通常配合 .jsonl 文件使用。.jsonl 是 JSON Lines 格式：每一行都是一个独立的 JSON 对象。

比如文件里某一行是：

{"question": "1+1=?", "answer": "2"}

读出来时，line 是字符串：

line = '{"question": "1+1=?", "answer": "2"}'

执行：

import json

obj = json.loads(line)

之后，obj 就变成 Python 字典：

{
    "question": "1+1=?",
    "answer": "2"
}


#############################################################################

@dataclass
class GSM8KExample:
    question: str
    full_answer: str
    final_answer: str


@dataclass 的作用是：自动帮你生成初始化函数 __init__ 等常用方法。

没有 @dataclass 的话，你可能要这样写：

class GSM8KExample:
    def __init__(self, question, full_answer, final_answer):
        self.question = question
        self.full_answer = full_answer
        self.final_answer = final_answer




#############################################################################
template.replace("{question}", question)

意思是：把字符串 template 里面所有的 "{question}" 替换成变量 question 的内容。

#############################################################################

    prompt_files = {
        "question_only": prompt_dir / "question_only.prompt",
        "r1_zero": prompt_dir / "r1_zero.prompt",
        "r1_zero_three_shot": prompt_dir / "r1_zero_three_shot_gsm8k.prompt",
    }#这里的 /  pathlib.Path 里的路径拼接。




#############################################################################
prompt_dir 里的 dir 是 directory 的缩写，意思是：目录 / 文件夹。


#############################################################################

3.1 Using vLLM for inference 使用 vLLM 推理
    


@dataclass
class VLLMCompletion:#vLLM 生成结果
text: str
token_ids: list[int]
finish_reason: str | None
@dataclass
class VLLMServer:#vLLM 服务器配置
model_id: str#要加载的模型名字或路径。
gpu: int = 0
seed: int = 0
gpu_memory_utilization: float = 0.9#表示 vLLM 最多可以使用 GPU 显存的比例。


VLLMCompletion 这个类表示模型生成的一条回答。
completion.text

就是：

"The answer is 72."

finish_reason: str | None

表示模型为什么停止生成。

常见可能值类似：

"stop"   表示遇到了 stop string，比如 </answer>
"length" 表示达到最大生成长度 max_tokens
None     表示没有明确停止原因，或接口没有返回



def start(self) -> None: ...
def generate_completions(
        

          server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", ...)
    server.start()
    server.generate_completions(prompts, sampling_params)
    #你传进去的是多个 prompt，所以模型会给每个 prompt 生成一个 completion。


在这个作业中，r1_zero prompt 会设置 stop string：

sampling_params["stop"] = ["</answer>"]

所以如果模型生成到 </answer>，就可能停止。


#############################################################################

    question_only_reward_fn(response, ground_truth)
    r1_zero_reward_fn(response, ground_truth)
    #两个函数都返回 dict：{"reward": float, "format_reward": float, "answer_reward": float}



#############################################################################
    如果我想得到000111这样的列表呢
已思考若干秒

可以用 列表重复 + 拼接：

lst = [0] * 3 + [1] * 3
print(lst)

#############################################################################

Hugging Face 的模型调用结果通常不是直接返回一个 Tensor，而是返回一个 输出对象。

也就是说：

outputs = model(input_ids)

得到的 outputs 不是单纯的 logits，而是类似这样的对象：

CausalLMOutputWithPast(
    loss=None,
    logits=...,
    past_key_values=...,
    hidden_states=None,
    attentions=None,
)

#############################################################################

实际实现时一般直接用 log_softmax，不要先 softmax 再 log，数值更稳定。
log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

#############################################################################

# 模型 next-token distribution 的 entropy是什么
# 但如果模型认为：

# Paris: 0.25
# London: 0.20
# Berlin: 0.18
# Rome: 0.17
# 其他: 0.20

# 概率分布很分散，模型不确定下一个 token 是什么，所以 entropy 很高。
# 数学公式是：

# H = - sum(p * log p)

# 其中 p 是每个可能 token 的概率。

# 如果概率集中在一个选项上，entropy 小。

# 如果概率分散在很多选项上，entropy 大。
#############################################################################

而交叉熵就是它取负号：

- log pθ(x_t | x_<t)
log_probs 只取 label 对应 token 的概率；token_entropy 要衡量整个预测分布的不确定性，
所以必须用 vocab_size 维度上的所有 token 概率来计算，但最终每个位置只返回一个 entropy 数值。

#############################################################################

4.2.2 Using vLLM in a reinforcement learning loop 在 RL 循环中使用 vLLM

把 Hugging Face model 和 optimizer 放在一个
GPU 上用于训练，把 vLLM（包含模型和 KV cache）放在另一个 GPU 上为什么要这么做

因为这个作业里同时要做两件很耗 GPU 资源的事：

训练模型
用当前模型批量生成 rollout



#############################################################################
def init_weight_sync(self, policy_device: str): ...
# 建立 cuda:0 上的训练模型 和 cuda:1 上的 vLLM 之间的权重同步通道
# 为什么需要 NCCL？

# NCCL 是 NVIDIA 提供的 GPU 通信库，常用于多 GPU 之间高速传输数据，比如：

# GPU 0 -> GPU 1

# 在这个作业里，它用于把训练后的 policy 权重快速复制给 vLLM。

#############################################################################

server.sync_policy_weights(policy)

把最新 policy 权重同步给 vLLM。

#############################################################################


sampling_params["stop"] = ["</answer>"]

意思是：当模型生成到 </answer> 时，就停止继续生成。

sampling_params["include_stop_str_in_output"] = True

意思是：最终返回的 response 里保留这个停止字符串 </answer>。
#############################################################################

4.2.3 GRPO components 组件
#############################################################################
reward_fn: Callable[[str, str], dict[str, float]],
reward_fn这是一个打分函数。


enumerate(zip(...)) 每次返回的是：

i, (rollout_response, repeated_ground_truth)


另外，reward 是 float，建议不要用 torch.long，而用：

dtype=torch.float32



还有 metadata 里最好放 Python float，而不是 tensor，方便 logging：

float(raw_rewards.mean().item())
#############################################################################

Problem (compute_group_normalized_rewards_grpo): Group normalization (1 point)

#############################################################################

2. Literal["mean", "none"]

这是类型标注，表示这个参数只能推荐传这两个字符串之一：

"mean"
"none"
#############################################################################
Problem (compute_policy_gradient_loss_on_policy): On-policy policy gradient (1 point)

#############################################################################
 policy_log_probs = policy_log_probs.masked_fill(~response_mask, float("zero"))/len_y/(B*G)

float("zero") 是错的，应该是 0.0。
~response_mask 只有在 response_mask 是 bool tensor 时才合理；如果你的 mask 是 0/1 long tensor，要先 .bool()。

#############################################################################
Problem (aggregate_loss_across_microbatch_sequence): Aggregate loss across tokens and
sequences (0.5 points)

#############################################################################
4.2.4 GRPO training step
Problem (grpo_train_step_standard_on_policy): GRPO train step (5 points)
#############################################################################
意识到，需要先rollout一些回答后，再进行训练，也就是说奖励可以提前先计算，然后再把输入放到
model里，得到需要用来计算的log

#############################################################################
4.3 Experiments 实验
相当于我们输入的每个提示词都被套上了一层模板，也就是意味着推理的时候agent会识别我们的提示词的类别，然后套上对应的提示词模板对吗
不一定只有需要cot


不是说“我们输入的每个提示词都会先被 agent 分类，然后机械地套一个固定模板”，而是很多 AI 系统会在用户输入外面叠加一层或多层隐含上下文 / 指令框架 / 任务流程。

可以理解成：

系统级规则
+ 开发者规则
+ 工具/插件说明
+ 安全策略
+ 用户画像/记忆
+ 当前对话历史
+ 用户这次输入
+ 可能检索到的外部资料
= 模型最终看到的完整上下文

#############################################################################
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )其中batch_size是什么
    batch_size = 32

那就是 vLLM 每次处理 32 条 prompt，大约分 8 批完成。




GRPO 训练里通常有两份模型：

GPU 0: Hugging Face model，用来训练，需要 optimizer
GPU 1: vLLM model，用来快速生成 rollout，不需要 optimizer

#############################################################################

torch.no_grad()和optimizer.zero_grad()的区别
#############################################################################
统计验证集的奖励信息之后还要统计损失吗
validation 集没有“固定 target response”用于标准 supervised loss。#轨迹不一定要相同
#############################################################################
args.group_size是个啥



args.rollout_generation_batch_size 表示：调用 vLLM 生成 rollout 时，vLLM 内部一次处理多少条 prompt。

它控制的是推理生成的分批大小，不是 GRPO 的 group size，也不是训练 batch size。


        args.group_size 表示：每个 prompt 要生成多少个 rollout responses。

#############################################################################
注意训练的随机采样用随机起点的方式
这里的随机采样prompt不一样

最好是后套模板，因为要随机采样example######################

#############################################################################
args.n传入vllm这是什么
args.n 传给 vLLM 的意思是：

每个 prompt 生成几个 completion / response。

#############################################################################

wandb 是 Weights & Biases 的简称，是一个实验记录和可视化工具。

在训练模型时，它可以帮你自动记录这些东西：

loss 曲线
reward 曲线
format_reward / answer_reward
learning rate
gradient norm
eval accuracy
训练超参数
生成样例
checkpoint 信息

你可以把它理解成：

一个在线实验日志面板，用来追踪训练过程和对比不同实验。


原因是：

wandb 主要适合记录数字曲线；
模型生成的完整 response、prompt、ground truth 这种长文本，更适合存成本地 .jsonl；

#############################################################################

重点是不要这样写：

probs_all = torch.exp(log_probs_all)
token_entropy = -(probs_all * log_probs_all).sum(dim=-1)

因为这样 token_entropy 会带计算图，虽然你后面 logging 时 detach 也行，但会多占一点显存。




