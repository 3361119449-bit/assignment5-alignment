

# 主要需要改 5 点：

# prompt_strs 和 output_strs 是 list[str]，所以不能直接 tokenizer.encode(prompt_strs)。要对每个样本循环处理。
# 作业要求返回的是 dict[str, torch.Tensor]，不是 tuple。
# 每条样本长度可能不同，所以要 padding 到同一长度。
# tokenizer.encode(...) 要加 add_special_tokens=False。
# response_mask 要和 labels 对齐，长度是 len(prompt_and_output) - 1。


#这里发现训练的时候的数据是不需要padding，只需要按需要的长度生成



# input_ids[i] += torch.tensor(prompt_id_list[i])

# 因为 input_ids[i] 是长度 max_len - 1 的整行，但 prompt_id_list[i] 可能更短，不能直接相加。应该用切片赋值：


# dtype=torch.long是为什么
# dtype=torch.long 的意思是：这个 Tensor 里面存的是整数，而且是 PyTorch 常用的 64 位整数类型


# torch.full 是 PyTorch 里用来创建 Tensor 的函数。

# 它的作用是：创建一个指定形状的 Tensor，并把里面所有元素都填成同一个值。

# def tokenize_prompt_and_output(
# prompt_strs: list[str],
# output_strs: list[str],
# tokenizer: PreTrainedTokenizer,
# ) -> dict[str, torch.Tensor]:
    
#     prompt_id_list=[]
#     output_id_list=[]
#     response_mask_list=[]
#     max_len=0
#     batch_size=0
#     for  prompt_str,output_str in zip(prompt_strs,output_strs):
#         batch_size+=1
#         prompt=tokenizer.encode(prompt_str,add_special_tokens=False)
#         output=tokenizer.encode(output_str,add_special_tokens=False)



#         prompt_and_output=prompt+output
#         max_len=max(len(prompt_and_output),max_len)
  
#         response_mask = [0] *(len(prompt)-1)  + [1] * len(output)#当对应 label token 属于 response 时为 1，否则为 0。
#         response_mask_list.append(response_mask)
#         prompt_id_list.append(prompt_and_output[:-1])
#         output_id_list.append(prompt_and_output[1:])
#     input_ids=torch.zeros(batch_size,max_len-1, dtype=torch.long)
#     labels=torch.zeros(batch_size,max_len-1, dtype=torch.long)
#     response_mask=torch.zeros(batch_size,max_len-1, dtype=torch.long)
#     for i in range(batch_size):
#         input_ids[i, :len(prompt_id_list[i])] = torch.tensor(prompt_id_list[i], dtype=torch.long)
#         labels[i, :len(output_id_list[i])] = torch.tensor(output_id_list[i], dtype=torch.long)
#         response_mask[i, :len(response_mask_list[i])] = torch.tensor(response_mask_list[i], dtype=torch.long)


#     return {"input_ids":input_ids,"labels":labels,"response_mask":response_mask}


# # 现在 padding 用的是 0

# # input_ids = torch.zeros(batch_size, max_len-1, dtype=torch.long)
# # labels = torch.zeros(batch_size, max_len-1, dtype=torch.long)

# # 这不一定错，但更规范的是用 tokenizer 的 pad_token_id：

# # pad_token_id = tokenizer.pad_token_id
# # if pad_token_id is None:
# #     pad_token_id = tokenizer.eos_token_id
# # if pad_token_id is None:
# #     pad_token_id = 0

# # 然后：

# # input_ids = torch.full((batch_size, max_len - 1), pad_token_id, dtype=torch.long)
# # labels = torch.full((batch_size, max_len - 1), pad_token_id, dtype=torch.long)


import torch
from transformers import PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """
    将 prompt 和 output 分别 tokenize，然后拼接，最后构造：

        input_ids     = prompt_and_output[:-1]
        labels        = prompt_and_output[1:]
        response_mask = 和 labels 对齐，output 部分为 1，其余为 0

    返回的三个 tensor shape 都是：
        (batch_size, max(prompt_and_output_lens) - 1)
    """

    # prompt_strs 和 output_strs 应该一一对应
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, "
            f"got {len(prompt_strs)} and {len(output_strs)}"
        )

    prompt_id_list = []
    output_id_list = []
    response_mask_list = []

    max_len = 0
    batch_size = 0

    prompt_cache: dict[str, list[int]] = {}
    for prompt_str, output_str in zip(prompt_strs, output_strs):
##############加上了缓存，因为prompt会重复，不需要重复编码
        # prompt_cache: dict[str, list[int]] = {}

        if prompt_str in prompt_cache:
            prompt = prompt_cache[prompt_str]
        else:
            prompt = tokenizer.encode(prompt_str, add_special_tokens=False)
            prompt_cache[prompt_str] = prompt



        batch_size += 1

        # 分别 tokenize prompt 和 output
        # 作业要求 add_special_tokens=False
        
        # prompt = tokenizer.encode(
        #     prompt_str,
        #     add_special_tokens=False,
        # )

        output = tokenizer.encode(
            output_str,
            add_special_tokens=False,
        )

        # 直接拼接，中间不加任何特殊 token
        prompt_and_output = prompt + output

        # 记录 batch 中最长的 prompt + output 长度
        max_len = max(len(prompt_and_output), max_len)

        # input_ids 是完整序列去掉最后一个 token
        prompt_id_list.append(prompt_and_output[:-1])

        # labels 是完整序列去掉第一个 token
        output_id_list.append(prompt_and_output[1:])

        # response_mask 要和 labels 对齐

        response_mask = [0] * (len(prompt) - 1) + [1] * len(output)
        response_mask_list.append(response_mask)

    # 防御性处理：空 batch
    if batch_size == 0:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "labels": torch.empty((0, 0), dtype=torch.long),
            "response_mask": torch.empty((0, 0), dtype=torch.long),
        }

    # 最终序列长度是 max_len - 1
    # 因为 input_ids 和 labels 都会比 prompt_and_output 少一个 token
    seq_len = max_len - 1

    # padding token id
    # 你的原版用 torch.zeros，等价于 pad token id = 0。
    # 这里稍微更稳妥：优先用 tokenizer.pad_token_id。
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        pad_token_id = 0

    # input_ids 和 labels 是 token id，必须是 torch.long
    #
    # 这里用 torch.full 而不是 torch.zeros，
    # 是为了 padding 位置填 pad_token_id，而不一定是 0。
    input_ids = torch.full(
        (batch_size, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, seq_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    # response_mask 按作业要求返回 0/1
    response_mask = torch.zeros(
        (batch_size, seq_len),
        dtype=torch.long,
    )

    for i in range(batch_size):
        # 每条样本真实长度
        cur_len = len(prompt_id_list[i])

        # 检查三者长度是否一致
        # 这有助于发现 response_mask 对齐错误
        assert len(prompt_id_list[i]) == len(output_id_list[i])
        assert len(output_id_list[i]) == len(response_mask_list[i])

        # 把第 i 条样本填进 batch tensor 的前 cur_len 个位置
        input_ids[i, :cur_len] = torch.tensor(
            prompt_id_list[i],
            dtype=torch.long,
        )

        labels[i, :cur_len] = torch.tensor(
            output_id_list[i],
            dtype=torch.long,
        )

        response_mask[i, :cur_len] = torch.tensor(
            response_mask_list[i],
            dtype=torch.long,
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }



# input_ids = train_batch["input_ids"].to(device)
# labels = train_batch["labels"].to(device)
# logits = model(input_ids).logits

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

import torch
from transformers import PreTrainedModel


# log_probs 只取 label 对应 token 的概率；token_entropy 要衡量整个预测分布的不确定性，
# 所以必须用 vocab_size 维度上的所有 token 概率来计算，但最终每个位置只返回一个 entropy 数值。

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    从 causal language model 中获取每个 label token 的 log probability。

    Args:
        model:
            Hugging Face causal language model。

        input_ids:
            shape = (batch_size, sequence_length)
            输入给模型的 token ids。

        labels:
            shape = (batch_size, sequence_length)
            每个位置要计算 log probability 的目标 token ids。

        return_token_entropy:
            如果为 True，额外返回每个位置的 next-token distribution entropy。

    Returns:
        dict[str, torch.Tensor]

        "log_probs":
            shape = (batch_size, sequence_length)
            表示每个 labels[b, t] 对应 token 的 conditional log-probability：

                log p_theta(x_t | x_<t)

        "token_entropy":
            可选，shape = (batch_size, sequence_length)
            表示每个位置的 per-token entropy。
            只有 return_token_entropy=True 时存在。
    """

    logits = model(input_ids).logits

    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        # shape = (batch_size, sequence_length)
        # 每个位置 labels[b, t] 这个 token 的 conditional log-probability:
        #     log p_theta(x_t | x_<t)
        "log_probs": selected_log_probs,
    }

    if return_token_entropy:
        probs_all = torch.exp(log_probs_all)

        token_entropy = -(probs_all * log_probs_all).sum(dim=-1)

        result["token_entropy"] = token_entropy

    return result



# def compute_rollout_rewards(
# reward_fn: Callable[[str, str], dict[str, float]],
# rollout_responses: list[str],
# repeated_ground_truths: list[str],
# ) -> tuple[torch.Tensor, dict[str, float]]:
#     rollout_batch_size=len(rollout_responses)
#     raw_rewards=torch.empty(rollout_batch_size,dtype=torch.float32)
#     format_rewards=torch.empty(rollout_batch_size,dtype=torch.float32)
#     for i, (rollout_response, repeated_ground_truth)  in enumerate(zip(rollout_responses,repeated_ground_truths)):
#         reward_dict=reward_fn(rollout_response,repeated_ground_truth )
#         raw_rewards[i]=reward_dict["reward"]
#         format_rewards[i]=reward_dict["format_reward"]

#     metadata={"mean total reward":float(raw_rewards.mean().item()),"mean format reward":float(format_rewards.mean().item())}
#     return raw_rewards,metadata


from typing import Callable

import torch


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    对一批 rollout responses 计算 reward。

    Args:
        reward_fn:
            打分函数，输入 response 和 ground_truth，
            返回 dict，例如：
                {
                    "reward": 1.0,
                    "format_reward": 1.0,
                    "answer_reward": 1.0,
                }

        rollout_responses:
            模型生成的一批回答。
            长度是 rollout_batch_size。

        repeated_ground_truths:
            和 rollout_responses 一一对应的标准答案。
            因为每个 prompt 可能生成 group_size 个 response，
            所以 ground truth 会被重复 group_size 次。

    Returns:
        raw_rewards:
            shape = (rollout_batch_size,)
            每个 rollout 的原始 reward。

        metadata:
            一些用于 logging 的平均 reward 统计。
    """

    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"rollout_responses and repeated_ground_truths must have same length, "
            f"got {len(rollout_responses)} and {len(repeated_ground_truths)}"
        )

    rollout_batch_size = len(rollout_responses)

    # reward 是 float，所以用 float32 更合适。
    raw_rewards = torch.empty(
        rollout_batch_size,
        dtype=torch.float32,
    )

    format_rewards = torch.empty(
        rollout_batch_size,
        dtype=torch.float32,
    )

    answer_rewards = torch.empty(
        rollout_batch_size,
        dtype=torch.float32,
    )

    for i, (rollout_response, repeated_ground_truth) in enumerate(
        zip(rollout_responses, repeated_ground_truths)
    ):
        reward_dict = reward_fn(
            rollout_response,
            repeated_ground_truth,
        )

        raw_rewards[i] = float(reward_dict["reward"])
        format_rewards[i] = float(reward_dict["format_reward"])
        answer_rewards[i] = float(reward_dict["answer_reward"])

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "mean_format_reward": float(format_rewards.mean().item()),
        "mean_answer_reward": float(answer_rewards.mean().item()),
    }

    return raw_rewards, metadata
#每个 rollout response 的 unnormalized_reward



# def compute_group_normalized_rewards(
# raw_rewards: torch.Tensor,
# group_size: int,
# baseline: Literal["mean", "none"] = "mean",
# advantage_eps: float = 1e-6,
# advantage_normalizer: Literal["std", "none", "mean"] = "std",
# ):
    
#     rewards=raw_rewards.reshape(-1,group_size)
#     group_mean=rewards.mean(dim=-1,keepdim=True)
#     if baseline == "mean":
#         rewards-=group_mean
#     if advantage_normalizer=="std":
#         rewards=rewards/(torch.std(rewards,dim=-1,keepdim=True)+advantage_eps)
#     elif advantage_normalizer=="mean":
#         rewards=rewards/(group_mean+advantage_eps)
from typing import Literal
import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    """
    Args:
        raw_rewards:
            shape = (rollout_batch_size,)
            每个 rollout response 的原始 reward。

        group_size:
            每个 prompt 采样出的 response 数量。
            rollout_batch_size 必须能被 group_size 整除。

        baseline:
            "mean":
                每个 group 内减去该 group 的平均 reward。
            "none":
                不减 baseline。

        advantage_eps:
            防止除以 0 的小常数。

        advantage_normalizer:
            "std":
                除以每个 group 的 reward 标准差。
            "mean":
                除以每个 group 的平均 reward。
            "none":
                不做 normalization。

    Returns:
        advantages:
            torch.Tensor, shape = (rollout_batch_size,)
            group-normalized 后的一维 advantage。

        metadata:
            dict[str, float]
            用于 logging 的 reward 统计信息。
    """

    rewards = raw_rewards.reshape(-1, group_size)

    group_mean = rewards.mean(dim=-1, keepdim=True)
    group_std = rewards.std(dim=-1, keepdim=True)

    if baseline == "mean":
        advantages = rewards - group_mean
    elif baseline == "none":
        advantages = rewards

    if advantage_normalizer == "std":
        advantages = advantages / (group_std + advantage_eps)

    elif advantage_normalizer == "mean":
        advantages = advantages / (group_mean + advantage_eps)

    elif advantage_normalizer == "none":
        pass

    advantages = advantages.reshape(-1)

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()),
        "std_reward": float(raw_rewards.std().item()),
        "max_reward": float(raw_rewards.max().item()),
        "min_reward": float(raw_rewards.min().item()),
    }

    return advantages, metadata

# def compute_policy_gradient_loss(
# raw_rewards_or_advantages: torch.Tensor,
# policy_log_probs: torch.Tensor,
# importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
# old_log_probs: torch.Tensor | None = None,
# cliprange: float | None = None,
# response_mask: torch.Tensor | None = None,
# ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
#     len_y=response_mask.sum(dim=-1,keepdim=True)
#     B,G=response_mask.shape[0],response_mask.shape[-1]
#     policy_log_probs = policy_log_probs.masked_fill(~response_mask, float("zero"))/len_y/(B*G)
#     per_token_policy_gradient_loss=policy_log_probs*raw_rewards_or_advantages
#     result={"per_token_policy_gradient_loss":per_token_policy_gradient_loss}
#     return result

from typing import Literal
import torch


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        raw_rewards_or_advantages:
            shape = (batch_size,) 或 (batch_size, 1)，每条 rollout 的 reward/advantage。

        policy_log_probs:
            shape = (batch_size, sequence_length)，当前 policy 的 per-token log probability。

        importance_reweighting_method:
            当前 on-policy 版本只支持 "none"。

        old_log_probs:
            off-policy 时使用；当前不用。

        cliprange:
            off-policy clipping 时使用；当前不用。

        response_mask:
            shape = (batch_size, sequence_length)，当前函数不用，后续聚合 loss 时使用。

    Returns:
        per_token_policy_gradient_loss:
            shape = (batch_size, sequence_length)，每个 token 的 policy-gradient loss。

        metadata:
            dict[str, torch.Tensor]，当前 on-policy 版本返回空 dict。
    """

    if importance_reweighting_method != "none":
        raise NotImplementedError

    advantages = raw_rewards_or_advantages.reshape(-1, 1)

    per_token_policy_gradient_loss = -advantages * policy_log_probs

    metadata = {}

    return per_token_policy_gradient_loss, metadata

# Problem (aggregate_loss_across_microbatch_sequence): Aggregate loss across tokens and
# sequences (0.5 points)
from typing import Literal
import torch

def aggregate_loss_across_microbatch(
per_token_policy_gradient_loss: torch.Tensor,
mask: torch.Tensor,
loss_normalization: Literal["sequence", "constant"] = "sequence",
normalization_constant: int | None = None,
) -> torch.Tensor:
    """
    Args:
        per_token_policy_gradient_loss:
            shape = (batch_size, sequence_length)
            每个 token 位置上的 policy-gradient loss。

        mask:
            shape = (batch_size, sequence_length)
            response token 位置为 1，其余位置为 0。

        loss_normalization:
            "sequence":
                每条 sequence 内先对 response token 求平均，
                再对 batch 求平均。

            "constant":
                后续 Dr. GRPO 会用到；当前 sequence 版本可以先不实现。

        normalization_constant:
            loss_normalization="constant" 时使用。

    Returns:
        loss:
            scalar tensor，聚合后的标量 loss。
    """
    len_y=mask.sum(dim=-1,keepdim=True)
    BG=mask.shape[0]
    per_token_policy_gradient_loss = (per_token_policy_gradient_loss.masked_fill(~mask.bool(), float(0))/len_y).sum(dim=-1)
    per_token_policy_gradient_loss=per_token_policy_gradient_loss.sum()/BG

    return per_token_policy_gradient_loss



# def aggregate_loss_across_microbatch(
#     per_token_policy_gradient_loss: torch.Tensor,
#     mask: torch.Tensor,
#     loss_normalization: Literal["sequence", "constant"] = "sequence",
#     normalization_constant: int | None = None,
# ) -> torch.Tensor:
#     """
#     Args:
#         per_token_policy_gradient_loss:
#             shape = (batch_size, sequence_length)
#             每个 token 位置上的 policy-gradient loss。

#         mask:
#             shape = (batch_size, sequence_length)
#             response token 位置为 1，其余位置为 0。

#         loss_normalization:
#             "sequence":
#                 每条 sequence 内先对 response token 求平均，
#                 再对 batch 求平均。

#             "constant":
#                 后续 Dr. GRPO 会用到；当前 sequence 版本可以先不实现。

#         normalization_constant:
#             loss_normalization="constant" 时使用。

#     Returns:
#         loss:
#             scalar tensor，聚合后的标量 loss。
#     """

#     len_y = mask.sum(dim=-1, keepdim=True)

#     batch_size = mask.shape[0]

#     loss = (
#         per_token_policy_gradient_loss.masked_fill(
#             ~mask.bool(),
#             0.0,
#         )
#         / len_y
#     ).sum(dim=-1)

#     loss = loss.sum() / batch_size

#     return loss



# def grpo_train_step(
# model: PreTrainedModel,
# tokenizer: PreTrainedTokenizer,
# optimizer: Optimizer,
# gradient_accumulation_steps: int,
# max_grad_norm: float | None,
# reward_fn: Callable[[str, str], dict[str, float]],
# repeated_prompts: list[str],
# rollout_responses: list[str],
# repeated_ground_truths: list[str],
# group_size: int,
# # Reward normalization
# baseline: Literal["mean", "none"] = "mean",
# advantage_eps: float = 1e-6,
# advantage_normalizer: Literal["std", "none", "mean"] = "std",
# # Importance reweighting and clipping
# importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
# old_log_probs: torch.Tensor | None = None,
# cliprange: float | None = None,
# # Loss normalization
# loss_normalization: Literal["sequence", "constant"] = "sequence",
# normalization_constant: int | None = None,
# ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    

#     prompt_and_output=tokenize_prompt_and_output(
#         repeated_prompts,
#         rollout_responses,
#         tokenizer,
#     )
#     inputs=prompt_and_output["input_ids"]
#     labels=prompt_and_output["labels"]
#     mask=prompt_and_output["response_mask"]
#     # return {
#     #     "input_ids": input_ids,
#     #     "labels": labels,
#     #     "response_mask": response_mask,
#     # }



# #打分函数，输入 response 和 ground_truth， 返回 dict，例如：
# #  { "reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0, }
#     raw_rewards,_=compute_rollout_rewards(reward_fn,rollout_responses ,repeated_ground_truths) 


#     advantages,_=compute_group_normalized_rewards(
#         raw_rewards,
#         group_size,
#         baseline= baseline,
#         advantage_eps = advantage_eps,
#         advantage_normalizer= advantage_normalizer,
#     )

#     # gradient_accumulation_steps = 4
#     microbatch_size = len(inputs) // gradient_accumulation_steps
#     optimizer.zero_grad()
#     for i in range(0, len(inputs), microbatch_size):
#         inputs_microbatch = inputs[i:i+microbatch_size]
#         labels_microbatch = labels[i:i+microbatch_size]
#         # Forward pass.
#         # logits = model(inputs_microbatch)
#         log_probs=get_response_log_probs(
#         model,
#         inputs_microbatch,
#         labels_microbatch,
#         return_token_entropy=False,
#     )["log_probs"]

#         # loss = loss_fn(logits, labels_microbatch) * (len(inputs_microbatch) / len(inputs))
#         per_token_policy_gradient_loss,_=compute_policy_gradient_loss(
#         advantages[i:i+microbatch_size],
#         log_probs,
#         importance_reweighting_method = importance_reweighting_method,
#         old_log_probs= old_log_probs,
#         cliprange = cliprange,
#         response_mask= None) 

#         loss=aggregate_loss_across_microbatch(
#         per_token_policy_gradient_loss,
#         mask[i:i+microbatch_size],
#         loss_normalization= loss_normalization,
#         normalization_constant =normalization_constant)* (len(inputs_microbatch) / len(inputs))

        
#         # Backward pass.
#         loss.backward()
#     if max_grad_norm is not None:
#         torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
#     # Update weights once across entire batch.
#     optimizer.step()
#     # Zero gradients once across entire batch.
#     optimizer.zero_grad()
#     metadata = {}
#     return(loss,metadata)


from typing import Callable, Literal
import torch
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    model.train()

    device = next(model.parameters()).device

    train_batch = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )

    inputs = train_batch["input_ids"].to(device)
    labels = train_batch["labels"].to(device)
    mask = train_batch["response_mask"].to(device)

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )

    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    advantages = advantages.to(device)

    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    batch_size = inputs.shape[0]

    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"gradient_accumulation_steps={gradient_accumulation_steps}"
        )

    microbatch_size = batch_size // gradient_accumulation_steps

    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    total_entropy = torch.tensor(0.0, device=device)
    total_response_tokens = torch.tensor(0.0, device=device)

    loss_metadata_all: dict[str, torch.Tensor] = {}

    for i in range(0, batch_size, microbatch_size):
        inputs_microbatch = inputs[i : i + microbatch_size]
        labels_microbatch = labels[i : i + microbatch_size]
        mask_microbatch = mask[i : i + microbatch_size]
        advantages_microbatch = advantages[i : i + microbatch_size]

        if old_log_probs is None:
            old_log_probs_microbatch = None
        else:
            old_log_probs_microbatch = old_log_probs[i : i + microbatch_size]

        logprob_output = get_response_log_probs(
            model,
            inputs_microbatch,
            labels_microbatch,
            return_token_entropy=True,
        )

        log_probs = logprob_output["log_probs"]
        token_entropy = logprob_output["token_entropy"]

        per_token_policy_gradient_loss, loss_metadata = compute_policy_gradient_loss(
            advantages_microbatch,
            log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs_microbatch,
            cliprange=cliprange,
            response_mask=mask_microbatch,
        )

        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss,
            mask_microbatch,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        if loss_normalization == "sequence":
            scaled_loss = microbatch_loss * (
                inputs_microbatch.shape[0] / batch_size
            )
        elif loss_normalization == "constant":
            scaled_loss = microbatch_loss
        else:
            raise NotImplementedError(
                f"Unsupported loss_normalization: {loss_normalization}"
            )

        scaled_loss.backward()

        total_loss = total_loss + scaled_loss.detach()

        total_entropy = total_entropy + (
            token_entropy.masked_fill(~mask_microbatch.bool(), 0.0).sum().detach()
        )
        total_response_tokens = total_response_tokens + mask_microbatch.sum().detach()
#把每个 microbatch loss 计算过程中产生的额外统计信息，收集到最终 metadata 里，方便 logging。
        loss_metadata_all.update(loss_metadata)

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
    else:#我们仍然想记录一下当前梯度有多大
        parameters_with_grad = [
            p for p in model.parameters()
            if p.grad is not None
        ]

        if len(parameters_with_grad) == 0:
            grad_norm = torch.tensor(0.0, device=device)
        else:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([
                    torch.linalg.vector_norm(p.grad.detach(), 2)
                    for p in parameters_with_grad
                ]),
                2,
            )

    optimizer.step()
    optimizer.zero_grad()

    mean_token_entropy = total_entropy / total_response_tokens.clamp_min(1)

    metadata: dict[str, torch.Tensor | float] = {
        "loss": float(total_loss.item()),
        "grad_norm": float(grad_norm.item()),
        "token_entropy": float(mean_token_entropy.item()),
    }

    metadata.update(reward_metadata)
    metadata.update(advantage_metadata)

    for key, value in loss_metadata_all.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = value.detach()
        else:
            metadata[key] = value

    return total_loss.detach(), metadata