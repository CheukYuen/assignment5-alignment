# CS336 作业 5 (对齐): 对齐与推理强化学习

版本 1.0.2 | CS336 教学团队 | 2025 年春季

---

## 1 作业概述

在本次作业中，你将获得训练语言模型进行数学推理的实践经验。

### 你将实现的内容

1. MATH 数据集（竞赛数学题）的零样本提示基线 (Hendrycks et al., 2021)
2. 监督微调 (SFT)：使用来自更强推理模型（DeepSeek R1）的推理轨迹进行微调
3. 专家迭代 (Expert Iteration)：使用验证奖励提高推理性能
4. 组相对策略优化 (GRPO)：使用验证奖励提高推理性能

此外还有一个**完全可选**的部分，关于将语言模型对齐到人类偏好（RLHF/DPO）。

### 你将运行的实验

1. 测量 Qwen 2.5 Math 1.5B 的零样本提示性能（我们的基线）
2. 在 Qwen 2.5 Math 1.5B 上用 R1 推理轨迹运行 SFT
3. 在 Qwen 2.5 Math 1.5B 上运行专家迭代
4. 在 Qwen 2.5 Math 1.5B 上运行 GRPO

### 代码结构

1. `cs336_alignment/*`：你将在此编写作业 5 的代码。这里几乎没有预设代码，你可以从零开始。
2. `cs336_alignment/prompts/*`：提供了文本文件形式的提示模板，避免从 PDF 复制粘贴出错。
3. `tests/*.py`：你必须通过的所有测试。**你只需要通过 `tests/test_sft.py` 和 `tests/test_grpo.py` 中的测试**——其余测试是针对非必做部分的。测试通过 `tests/adapters.py` 中定义的钩子调用你的代码。
4. `README.md`：环境设置的基本说明。

### 可以使用的工具

我们期望你从零构建大部分 RL 相关组件。你可以：
- 使用 vLLM 从语言模型生成文本（§3.1）
- 使用 HuggingFace Transformers 加载 Qwen 2.5 Math 1.5B 模型和分词器并运行前向传播（§4.1）
- **不能**使用任何训练工具类（如 `Trainer` 类）

### 提交方式

提交以下文件到 Gradescope：
- `writeup.pdf`：回答所有书面问题
- `code.zip`：包含所有编写的代码

---

## 2 使用语言模型进行推理

### 2.1 动机

语言模型的一个重要用例是构建能处理广泛自然语言处理任务的通用系统。本次作业聚焦于语言模型的数学推理能力，以此作为设置评估、执行监督微调和尝试用强化学习教 LM 推理的试验场。

与之前的作业有两个区别：

- **模型方面**：我们不再使用之前作业中从零训练的模型（它们太弱了，无法进行数学推理），而是切换到 Qwen 2.5 Math 1.5B Base 这个现代高性能语言模型。
- **评估方面**：我们将引入新的评估基准。之前我们用交叉熵作为下游任务的替代指标，但本次作业的重点是弥合基础模型与下游任务之间的差距，使用独立于交叉熵的评估方式。我们将使用 MATH 12K 数据集（包含有挑战性的高中竞赛数学题），通过将模型输出与参考答案比较来评估。

### 2.2 链式思维推理与推理 RL

**链式思维推理**：早期方法通过微调语言模型使用"草稿本"将简单数学任务分解为中间步骤。另一种方法是提示强大模型"一步步思考"，这显著提高了数学推理任务的表现。

**通过专家迭代学习推理**：Self-Taught Reasoner (STaR, Zelikman et al., 2022) 将推理视为一个自举循环：预训练模型先采样多样的链式思维，只保留导致正确答案的轨迹，然后在这些"专家"轨迹上微调。迭代此过程可以提高 LM 的推理能力和正确率。

**使用验证奖励的推理 RL（o1 和 R1）**：近期工作探索了使用更强大的强化学习算法和验证奖励来提升推理性能。OpenAI 的 o1/o3、DeepSeek R1、Moonshot 的 kimi k1.5 使用策略梯度方法在数学和代码任务上进行训练，其中通过字符串匹配或单元测试来验证正确性。后续工作如 Open-R1、SimpleRL-Zoo 和 TinyZero 确认了即使在 1.5B 参数的小模型上，纯强化学习训练也能提升推理性能。

### 我们的设置：模型和数据集

我们将逐步考虑越来越复杂的方法来训练基础语言模型进行逐步推理以解决数学问题。

- **模型**：Qwen 2.5 Math 1.5B Base（在高质量合成数学预训练数据上持续预训练的 Qwen 2.5 1.5B 模型）
- **数据集**：MATH 数据集，位于 Together 集群的 `/data/a5-alignment/MATH`

---

## 3 测量零样本 MATH 性能

首先测量基础语言模型在 5K 示例 MATH 测试集上的性能，建立基线。

除非另行说明，MATH 实验将使用 DeepSeek R1-Zero 提示模板（`r1_zero` 提示）：

```
A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

该提示位于 `cs336_alignment/prompts/r1_zero.prompt`。

**关于提示选择的说明**：`r1_zero` 提示并不是最大化 RL 后下游性能的最佳选择（因为提示与 Qwen 2.5 Math 1.5B 的预训练方式不匹配）。我们选择它是因为用 RL 训练时能看到明确的准确率提升，让我们快速走通 RL 的机制。后面你还将与 `question_only` 提示进行对比。

### 3.1 使用 vLLM 进行离线语言模型推理

我们推荐使用 vLLM 进行离线批量推理。vLLM 是一个高吞吐量、内存高效的推理引擎，包含优化的 CUDA 核心、PagedAttention 等技术。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

llm = LLM(model=<path to model>)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Together 集群上的预训练模型路径：
- Qwen 2.5 Math 1.5B Base：`/data/a5-alignment/models/Qwen2.5-Math-1.5B`
- Llama 3.1 8B Base（可选部分）：`/data/a5-alignment/models/Llama-3.1-8B`
- Llama 3.3 70B Instruct（可选部分）：`/data/a5-alignment/models/Llama-3.3-70B-Instruct`

### 3.2 零样本 MATH 基线

**提示设置**：加载 MATH 测试集示例，使用 `r1_zero` 提示格式化问题并让语言模型生成回答。

**评估指标**：对于数学题，我们需要处理语义等价的答案匹配问题（如 `0.5` 与 `1/2`）。我们使用 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` 作为答案解析和奖励函数。

**生成超参数**：温度 1.0，top-p 1.0，最大生成长度 1024。使用 `</answer>` 作为停止字符串：

```python
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
```

### 问题 (math_baseline): 4 分

**(a)** 编写脚本评估 Qwen 2.5 Math 1.5B 在 MATH 上的零样本表现。脚本应：
1. 从 `/data/a5-alignment/MATH/validation.jsonl` 加载 MATH 验证集
2. 使用 `r1_zero` 提示格式化为字符串
3. 为每个示例生成输出
4. 计算评估指标
5. 将示例、模型生成和评估分数序列化到磁盘

建议实现一个 `evaluate_vllm` 方法以便后续复用。

**交付物**：评估零样本 MATH 性能的脚本。

**(b)** 运行评估脚本。有多少模型生成落入以下类别：
1. 格式奖励和答案奖励都为 1（完全正确）
2. 格式奖励 1 但答案奖励 0
3. 格式奖励 0 但答案奖励 0

对于格式奖励为 0 的情况（至少 10 例），你认为问题出在基础模型输出还是解析器上？格式奖励为 1 但答案奖励为 0 的情况（至少 10 例）呢？

**交付物**：对模型和奖励函数性能的评论，包括各类别的示例。

**(c)** Qwen 2.5 Math 1.5B 零样本基线在 MATH 上表现如何？

**交付物**：1-2 句话加评估指标。

---

## 4 MATH 的监督微调 (SFT)

### 算法 1：监督微调 (SFT)

```
输入：初始策略模型 π_θ_init；SFT 数据集 D
1: 策略模型 π_θ ← π_θ_init
2: for step = 1, ..., n_sft_steps do
3:     从 D 中采样一批 question-response 对 D_b
4:     使用模型 π_θ 计算响应在给定问题下的交叉熵损失
5:     对交叉熵损失求梯度，更新模型参数 θ
6: end for
输出：π_θ
```

**监督微调用于推理**：我们将在 MATH 数据集上微调基础模型。目标不是直接预测正确答案，而是先生成链式思维推理轨迹再给出答案。SFT 数据来自 DeepSeek R1 的推理轨迹，位于：

```
/data/a5-alignment/MATH/sft.jsonl
```

SFT 通常作为 RL 微调前的热启动步骤，原因有二：
1. SFT 需要高质量标注数据（带推理轨迹），而 RL 只需要正确答案作为反馈
2. 即使有标注数据，RL 仍能通过找到更好的策略来进一步提升性能

### 4.1 使用 HuggingFace 模型

**加载模型和分词器**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
```

**前向传播**：

```python
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)
logits = model(input_ids).logits
loss = F.cross_entropy(...)
```

**保存模型**：使用 `.save_pretrained()` 保存到 `/data/yourusername` 目录下（模型文件较大）。

**梯度累积**：即使使用 bfloat16 和 FlashAttention-2，80GB GPU 也可能不够支持合理的 batch size。通过梯度累积，我们在多个 batch 上累积梯度后再执行梯度更新步骤。

```python
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs)
    loss = loss_fn(logits, labels) / gradient_accumulation_steps
    loss.backward()

    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.2 SFT 辅助方法

#### 问题 (tokenize_prompt_and_output): 提示和输出的分词 (2 分)

实现 `tokenize_prompt_and_output` 方法：将问题和输出分别分词，拼接后构建 `response_mask`（响应 token 为 True，问题和填充 token 为 False）。

```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
```

返回字典包含：
- `input_ids`：形状 `(batch_size, max(prompt_and_output_lens) - 1)`，分词后的提示和输出，去掉最后一个 token
- `labels`：形状同上，移位的 input_ids（去掉第一个 token）
- `response_mask`：形状同上，labels 中响应 token 的掩码

测试：实现 `adapters.run_tokenize_prompt_and_output`，然后运行 `uv run pytest -k test_tokenize_prompt_and_output`。

#### 问题 (compute_entropy): 逐 token 熵 (1 分)

实现 `compute_entropy` 方法，计算下一个 token 预测的逐 token 熵。

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
```

- 输入 `logits`：形状 `(batch_size, sequence_length, vocab_size)`
- 输出：形状 `(batch_size, sequence_length)`

注意：应使用数值稳定的方法（如 `logsumexp`）避免溢出。

测试：实现 `adapters.run_compute_entropy`，运行 `uv run pytest -k test_compute_entropy`。

#### 问题 (get_response_log_probs): 响应对数概率（和熵）(2 分)

实现 `get_response_log_probs`：从因果语言模型获取逐 token 条件对数概率，可选返回 token 熵。

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
```

返回：
- `"log_probs"`：形状 `(batch_size, sequence_length)`，条件对数概率
- `"token_entropy"`：可选，形状 `(batch_size, sequence_length)`，token 熵

实现提示：通过 `model(input_ids).logits` 获取 logits。

测试：实现 `adapters.run_get_response_log_probs`，运行 `uv run pytest -k test_get_response_log_probs`。

#### 问题 (masked_normalize): 掩码归一化 (1 分)

实现 `masked_normalize`：在某个维度上求和并除以常数，只考虑 mask==1 的元素。

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
```

测试：实现 `adapters.run_masked_normalize`，运行 `uv run pytest -k test_masked_normalize`。

#### 问题 (sft_microbatch_train_step): 微批次训练步骤 (3 分)

实现单个 SFT 微批次训练步骤，包括交叉熵损失、掩码求和、梯度缩放。

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

返回：
- `loss`：标量张量，经梯度累积调整的微批次损失
- `metadata`：包含统计信息的字典

实现提示：应在此函数中调用 `loss.backward()`，并确保针对梯度累积进行调整。

测试：实现 `adapters.run_sft_microbatch_train_step`，运行 `uv run pytest -k test_sft_microbatch_train_step`。

#### 记录生成内容

实现 `log_generations` 函数，用于记录模型的在线生成内容。建议记录：
1. 输入提示
2. SFT/RL 模型生成的响应
3. 标准答案
4. 奖励信息（格式、答案、总奖励）
5. 响应的平均 token 熵
6. 正确/错误响应的平均响应长度

#### 问题 (log_generations): 记录生成内容 (1 分)

**交付物**：实现一个可用于记录模型生成内容的 `log_generations` 函数。

### 4.3 SFT 实验

使用上述组件实现完整的 SFT 流程，在 MATH 数据集上微调 Qwen 2.5 Math 1.5B Base。

SFT 数据位于 `/data/a5-alignment/MATH/sft.jsonl`，每个示例是 `{"prompt": str, "response": str}` 格式的 JSON。

训练过程中应定期在 MATH 验证集上评估。脚本应使用 2 个 GPU：一个用于策略模型训练，一个用于 vLLM 评估。

初始化 vLLM 和加载策略权重的启动代码：

```python
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    # 使用 monkeypatch 将 vLLM 模型放到指定设备上
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
```

建议使用梯度裁剪（clip value 1.0）。

#### 问题 (sft_experiment): 在 MATH 数据集上运行 SFT (2 分) (2 H100 小时)

1. 使用 Qwen 2.5 Math 1.5B Base 在推理 SFT 数据上运行 SFT，变化用于 SFT 的唯一示例数量（范围 {128, 256, 512, 1024}），同时使用完整数据集。在使用完整数据集时，调整学习率和 batch size 以达到至少 15% 的验证准确率。

   **交付物**：不同数据集大小对应的验证准确率曲线。

2. 过滤推理 SFT 示例，只保留产生正确答案的示例。在（完整）过滤后数据集上运行 SFT，报告过滤后数据集大小和验证准确率。

   **交付物**：报告数据集大小和验证准确率曲线。与之前的 SFT 实验进行对比。

---

## 5 MATH 的专家迭代

上一节我们通过过滤 SFT 数据中的错误示例来提升 SFT 模型性能。本节我们更进一步：将过滤过程应用到基础模型自己生成的推理轨迹上。这就是**专家迭代** (Expert Iteration)。

### 算法 2：专家迭代 (EI)

```
输入：初始策略模型 π_θ_init；奖励函数 R；任务问题集 D
1: 策略模型 π_θ ← π_θ_init
2: for step = 1, ..., n_ei_steps do
3:     从 D 中采样一批问题 D_b
4:     设旧策略 π_θ_old ← π_θ
5:     对 D_b 中每个问题 q，采样 G 个输出 {o^(i)} ~ π_θ_old(·|q)
6:     对每个采样输出运行奖励函数 R(q, o^(i)) 计算奖励
7:     过滤掉错误输出（r^(i) = 0 的），得到正确 question-response 对的 SFT 数据集 D_sft
8:     π_θ = SFT(π_θ, D_sft)（使用算法 1）
9: end for
输出：π_θ
```

提示：给 vLLM 的 `SamplingParams` 传入 `min_tokens` 值，确保不会生成空字符串。

```python
sampling_min_tokens = 4
sampling_params = SamplingParams(
    temperature=sampling_temperature,
    max_tokens=sampling_max_tokens,
    min_tokens=sampling_min_tokens,
    n=G,
    seed=seed,
)
```

同样建议使用梯度裁剪（clip value 1.0）。

#### 问题 (expert_iteration_experiment): 在 MATH 上运行专家迭代 (2 分) (6 H100 小时)

使用 Qwen 2.5 Math 1.5B Base 在 MATH 训练集（`/data/a5-alignment/MATH/train.jsonl`）上运行专家迭代：
- 变化每个问题的 rollout 数 G 和 SFT 步骤中的 epoch 数
- 使用 `n_ei_steps = 5`
- 每个 EI 步骤的 batch size（D_b 大小）在 {512, 1024, 2048} 中变化
- 记录模型响应的熵随训练变化
- 确保 vLLM 在第二个 `</answer>` 标签处停止生成

**交付物**：
- 不同 rollout 配置的验证准确率曲线（至少尝试 2 种不同的 rollout 数和 epoch 数）
- 在 MATH 上达到至少 15% 验证准确率的模型
- 与 SFT 性能的 2 句话对比讨论，以及跨 EI 步骤的性能变化
- 模型响应熵随训练变化的图

---

## 6 策略梯度入门

RL 对验证奖励的训练可以显著提升基础模型的推理能力和性能。最强的开放推理模型（如 DeepSeek R1）都是用策略梯度训练的。

### 6.1 语言模型作为策略

因果语言模型（参数 θ）定义了给定当前文本前缀 s_t（状态/观测）下一个 token a_t ∈ V 的概率分布。在 RL 语境中，下一个 token 是**动作**，当前文本前缀是**状态**。因此 LM 是一个*分类随机策略*：

$$a_t \sim \pi_\theta(\cdot | s_t), \quad \pi_\theta(a_t | s_t) = [\text{softmax}(f_\theta(s_t))]_{a_t}$$

两个基本操作：
1. **从策略采样**：从分类分布中抽取动作 a_t
2. **动作的对数似然**：评估 log π_θ(a_t | s_t)

### 6.2 轨迹

（有限水平）轨迹是智能体经历的状态和动作的交替序列：

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$$

在 LLM 的 RL 中，环境是确定性的：下一个状态就是旧前缀拼接上生成的 token，即 s_{t+1} = s_t || a_t。轨迹也称为 episode 或 rollout。

### 6.3 奖励和回报

标量奖励 r_t = R(s_t, a_t) 评估在状态 s_t 采取动作 a_t 的即时质量。对于验证域的 RL，标准做法是给中间步骤零奖励，只在终止动作给**验证奖励**：

$$r_T = R(s_T, a_T) = \begin{cases} 1 & \text{如果轨迹匹配标准答案} \\ 0 & \text{否则} \end{cases}$$

回报 R(τ) 沿轨迹聚合奖励。我们使用**有限水平无折扣回报**：

$$R(\tau) = \sum_{t=0}^{T} r_t$$

智能体的目标是最大化期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

### 6.4 Vanilla 策略梯度

使用梯度上升学习策略参数 θ：

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)$$

核心恒等式是 REINFORCE 策略梯度：

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]$$

**推导**：利用对数导数技巧 ∇_θ P = P ∇_θ log P，以及环境项（初始状态分布、转移概率、奖励函数）相对于 θ 是常数的事实。

**梯度的样本估计**：给定 N 个 rollout 的批次 D = {τ^(i)}，策略梯度的无偏估计为：

$$\hat{g} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$$

### 6.5 策略梯度基线

Vanilla 策略梯度的主要问题是梯度估计的高方差。常见技术是从奖励中减去仅依赖于状态的**基线**函数 b：

$$B = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)(R(\tau) - b(s_t))\right]$$

只要基线仅依赖于状态，带基线的策略梯度就是无偏的。

**关于策略梯度"损失"的说明**：在 PyTorch 中，我们定义 `pg_loss` 使得调用 `pg_loss.backward()` 会将模型参数的梯度缓冲区填充为策略梯度的近似值。`pg_loss` 不是传统意义上的损失——在 RL 中应始终记录和报告训练/验证**奖励**作为有意义的评估指标。

### 6.6 离策略策略梯度

REINFORCE 是在策略 (on-policy) 算法：训练数据由当前策略收集，采样大量 rollout 只为进行一步梯度更新，非常低效。

离策略策略梯度使用旧策略 π_θ_old 的 rollout 来优化当前策略 π_θ：

$$\hat{g}_{\text{off-policy}} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T} \frac{\pi_\theta(a_t^{(i)}|s_t^{(i)})}{\pi_{\theta_\text{old}}(a_t^{(i)}|s_t^{(i)})} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) R(\tau^{(i)})$$

这可以通过重要性采样推导得出。

---

## 7 组相对策略优化 (GRPO)

### 7.1 GRPO 算法

**优势估计**：GRPO 的核心思想是对每个问题从策略 π_θ 采样多个输出，用它们来计算基线。对于问题 q 和组输出 {o^(i)}，组归一化奖励为：

$$A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, \ldots, r^{(G)}) + \text{advantage\_eps}}$$

GRPO 目标结合三个思想：
1. 离策略策略梯度
2. 使用组归一化计算优势 A^(i)
3. PPO 风格的裁剪机制，防止策略偏离旧策略太远

### 算法 3：GRPO

```
输入：初始策略模型 π_θ_init；奖励函数 R；任务问题集 D
1: 策略模型 π_θ ← π_θ_init
2: for step = 1, ..., n_grpo_steps do
3:     从 D 中采样一批问题 D_b
4:     设旧策略 π_θ_old ← π_θ
5:     对 D_b 中每个问题 q，从 π_θ_old 采样 G 个输出
6:     计算每个输出的奖励
7:     用组归一化计算优势 A^(i)
8:     for train step = 1, ..., n_train_steps_per_rollout_batch do
9:         通过最大化 GRPO-Clip 目标更新策略 π_θ
10:    end for
11: end for
输出：π_θ
```

**GRPO-Clip 目标**（公式 29）：对每个 token 取裁剪比率与未裁剪比率乘以优势的最小值，这限制了策略更新的幅度。

**Dr. GRPO 变体**（Liu et al., 2025）：提出移除标准差归一化，只减去组均值：

$$A^{(i)} = r^{(i)} - \text{mean}(r^{(1)}, \ldots, r^{(G)})$$

### 7.2 实现

#### 问题 (compute_group_normalized_rewards): 组归一化 (2 分)

实现 `compute_group_normalized_rewards`：计算每组 rollout 响应的原始奖励，在组内归一化，返回归一化后的奖励和原始奖励。

```python
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
```

返回 `tuple[torch.Tensor, torch.Tensor, dict[str, float]]`：
- `advantages`：形状 `(rollout_batch_size,)`，组归一化奖励
- `raw_rewards`：形状 `(rollout_batch_size,)`，未归一化奖励
- `metadata`：你选择记录的统计信息

测试：`uv run pytest -k test_compute_group_normalized_rewards`

#### 问题 (compute_naive_policy_gradient_loss): 朴素策略梯度 (1 分)

实现朴素策略梯度损失：将优势乘以对数概率（取负）。

$$-A_t \cdot \log p_\theta(o_t | q, o_{<t})$$

```python
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # (batch_size, 1)
    policy_log_probs: torch.Tensor,           # (batch_size, sequence_length)
) -> torch.Tensor:                            # (batch_size, sequence_length)
```

实现提示：将 `raw_rewards_or_advantages` 广播到 `sequence_length` 维度。

测试：`uv run pytest -k test_compute_naive_policy_gradient_loss`

#### 问题 (compute_grpo_clip_loss): GRPO-Clip 损失 (2 分)

实现 GRPO-Clip 损失（公式 33）：

$$-\min\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon\right) A_t\right)$$

```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,      # (batch_size, 1)
    policy_log_probs: torch.Tensor, # (batch_size, sequence_length)
    old_log_probs: torch.Tensor,    # (batch_size, sequence_length)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

返回：
- `loss`：形状 `(batch_size, sequence_length)` 的逐 token 裁剪损失
- `metadata`：建议记录每个 token 是否被裁剪

测试：`uv run pytest -k test_compute_grpo_clip_loss`

#### 问题 (compute_policy_gradient_loss): 策略梯度损失包装器 (1 分)

实现 `compute_policy_gradient_loss`，一个方便的包装器，根据 `loss_type` 分发到正确的损失函数：

- `"no_baseline"`：朴素策略梯度，优势 A = R(q, o)（原始奖励）
- `"reinforce_with_baseline"`：朴素策略梯度，使用组归一化奖励作为优势
- `"grpo_clip"`：GRPO-Clip 损失

```python
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

测试：`uv run pytest -k test_compute_policy_gradient_loss`

#### 问题 (masked_mean): 掩码均值 (1 分)

实现 `masked_mean`：在给定维度上计算张量的均值，只考虑 mask==1 的元素。

```python
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
```

测试：`uv run pytest -k test_masked_mean`

#### 问题 (grpo_microbatch_train_step): GRPO 微批次训练步骤 (3 分)

实现单个 GRPO 微批次更新，包括策略梯度损失、掩码均值、梯度缩放和反向传播。

```python
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

返回：
- `loss`：经梯度累积调整的标量损失
- `metadata`：包含统计信息的字典

实现提示：应在此函数中调用 `loss.backward()`。

测试：`uv run pytest -k test_grpo_microbatch_train_step`

#### 问题 (grpo_train_loop): GRPO 训练循环 (5 分)

实现完整的 GRPO 训练循环。开始在 MATH 上训练策略，确认验证奖励随时间提升，rollout 合理。

**交付物**：验证奖励随步数变化的图，以及若干示例 rollout。

**默认超参数**：

```python
n_grpo_steps = 200
learning_rate = 1e-5
advantage_eps = 1e-6
rollout_batch_size = 256
group_size = 8
sampling_temperature = 1.0
sampling_min_tokens = 4
sampling_max_tokens = 1024
epochs_per_rollout_batch = 1    # On-policy
train_batch_size = 256          # On-policy
gradient_accumulation_steps = 128  # microbatch size = 2
loss_type = "reinforce_with_baseline"
use_std_normalization = True
optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),
)
```

**额外提示**：
- 使用 `r1_zero` 提示，在第二个 `</answer>` 标签处停止生成
- 建议使用 `typer` 进行参数解析
- 使用梯度裁剪（clip value 1.0）
- 定期记录验证奖励（每 5-10 步），至少评估 1024 个验证样本
- GRPO-Clip 只应在离策略时使用（需要旧对数概率）
- 离策略设置中，旧对数概率只需在每个 rollout batch 后计算一次，多个 epoch 复用
- 不应对旧对数概率求导
- 建议记录：损失、梯度范数、token 熵、裁剪比例（离策略时）、训练奖励（总奖励、格式奖励、答案奖励）

**健全性检查断言**：
```python
assert train_batch_size % gradient_accumulation_steps == 0
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0
n_prompts_per_rollout_batch = rollout_batch_size // group_size
assert train_batch_size >= group_size
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
```

---

## 8 GRPO 实验

每个实验需要 2 个 GPU（一个 vLLM，一个策略）。

**提前停止说明**：如果在 200 步之前就看到超参数间的显著差异（如某个配置发散），可以提前停止以节省时间和计算资源。

#### 问题 (grpo_learning_rate): 调整学习率 (2 分) (6 H100 小时)

使用默认超参数，对学习率进行扫描，报告最终验证答案奖励（或注意发散情况）。

**交付物**：
- 不同学习率的验证奖励曲线
- 在 MATH 上达到至少 25% 验证准确率的模型
- 关于其他记录指标趋势的 2 句话讨论

后续实验使用此处找到的最佳学习率。

#### 问题 (grpo_baselines): 基线的效果 (2 分) (2 H100 小时)

在在策略设置下，比较两种损失类型：`no_baseline` 和 `reinforce_with_baseline`。注意默认超参数中 `use_std_normalization = True`。

**交付物**：
- 各损失类型的验证奖励曲线
- 关于其他记录指标趋势的 2 句话讨论

后续实验使用最佳损失类型。

#### 长度归一化

`masked_mean`（按序列长度平均）和 `masked_normalize`（求和后除以常数）两种方法对梯度的影响不同。

通过一个示例说明（batch_size=2，第一个响应 4 个 token，第二个响应 7 个 token）：
- `masked_mean`：每个 token 的梯度贡献与其所在序列长度成反比
- `masked_normalize`：所有 token 的梯度贡献相同

#### 问题 (think_about_length_normalization): 思考长度归一化 (1 分)

在不运行实验的情况下，比较两种方法的优缺点。是否有某些特定设置或示例中一种方法明显更好？

#### 问题 (grpo_length_normalization): 长度归一化的效果 (2 分) (2 H100 小时)

比较 `masked_mean` 和 `masked_normalize` 的端到端 GRPO 训练效果。报告验证答案奖励曲线，评论发现。

提示：关注稳定性相关指标，如梯度范数。

固定较好的长度归一化方法用于后续实验。

#### 问题 (grpo_group_standard_deviation): 标准差归一化的效果 (2 分) (2 H100 小时)

比较 `use_std_normalization == True` 和 `use_std_normalization == False` 的性能。报告验证答案奖励曲线，评论发现。

提示：关注稳定性相关指标，如梯度范数。

固定较好的组归一化方法用于后续实验。

#### 离策略 vs 在策略

目前所有实验都是在策略的（每个 rollout batch 只做一步梯度更新）。在策略方法虽然理论上有据且稳定，但效率低下——rollout 是瓶颈，只做一步梯度更新浪费了大量计算。

#### 问题 (grpo_off_policy): 实现离策略 GRPO

实现离策略 GRPO 训练：
- 每个 rollout batch 可以进行多个 epoch 的梯度步
- epoch 和优化器更新次数由 `rollout_batch_size`、`epochs_per_rollout_batch` 和 `train_batch_size` 控制
- 在每个 rollout batch 生成后、内层循环前，从策略获取响应对数概率作为 `old_log_probs`
- 建议使用 `torch.inference_mode()`
- 使用 `"GRPO-Clip"` 损失类型

#### 问题 (grpo_off_policy_sweep): 离策略 GRPO 超参数扫描 (4 分) (12 H100 小时)

固定 `rollout_batch_size = 256`，选择 `epochs_per_rollout_batch` 和 `train_batch_size` 的范围进行扫描。先做少量步数（<50）的粗扫描了解性能分布，再做更多步数（200）的精细扫描。

与在策略运行（`epochs_per_rollout_batch = 1`，`train_batch_size = 256`）对比，报告验证步数和墙钟时间两个维度的验证答案奖励曲线。评论发现，包括熵和响应长度等趋势。

提示：需要调整 `gradient_accumulation_steps` 以保持内存使用恒定。

#### 裁剪消融

裁剪的目的是在对单个 rollout batch 进行多步梯度更新时防止策略偏离旧策略太远。

#### 问题 (grpo_off_policy_clip_ablation): 离策略 GRPO-Clip 消融 (2 分) (2 H100 小时)

实现无裁剪版本的逐 token 损失作为新损失类型 `"GRPO-No-Clip"`：

$$-\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})} A_t$$

使用前一个问题中最佳的离策略超参数运行。报告验证答案奖励曲线，与 GRPO-Clip 对比。

#### 提示的效果

RL 中使用的提示对模型性能有重大影响。我们将使用一个极简提示替代 R1-Zero 提示：

```
{question}
```

位于 `cs336_alignment/prompts/question_only.prompt`。同时将奖励函数改为 `cs336_alignment/drgrpo_grader.py` 中的 `question_only_reward_fn`。

#### 问题 (grpo_prompt_ablation): 提示消融 (2 分) (2 H100 小时)

报告 R1-Zero 提示和 question-only 提示的验证答案奖励曲线。指标对比如何？包括熵、响应长度、梯度范数等。尝试解释你的发现。

---

## 9 排行榜：GRPO on MATH

作为（必做）作业的最后部分，你将在 2 个 H100 GPU 上 4 小时训练内，尝试获得尽可能高的验证奖励。

**模型**：继续使用 Qwen 2.5 Math 1.5B Base。

**数据集**：MATH 训练集和验证集（`/data/a5-alignment/MATH/train.jsonl` 和 `/data/a5-alignment/MATH/validation.jsonl`）。不允许使用其他数据或强模型的推理链进行 SFT。验证准确率必须在完整验证集（5K 示例）上报告。

**约束**：
1. 验证准确率为完整 MATH 验证集（5K 示例）的平均准确率
2. 验证时必须使用 R1-Zero 提示
3. 使用温度 1.0、最大 token 1024 进行 vLLM 评估
4. 使用 `r1_zero_reward_fn` 奖励函数计算验证准确率

**算法**：你可以自由调整超参数或完全改变训练算法，只要不使用外部数据或其他模型（可以使用多份模型拷贝）。

**系统优化建议**：
- 考虑降低精度进行 rollout 或训练
- 使用 `torch.compile`
- 不必将 vLLM 和训练策略限制在各自一个设备上，可以探索更好的并行化方式

**参考项目**：veRL、trl、torchtune、oat

**关于 KL 散度**：上述实验未包含对参考模型的 KL 散度项。实验表明省略 KL 项对性能没有影响且节省 GPU 内存。但你也可以尝试加入 KL 或其他正则化形式。

#### 问题 (leaderboard): 排行榜 (16 分) (16 H100 小时)

**交付物**：在 2 个 H100 GPU 上 4 小时训练内获得的验证准确率，以及验证准确率随墙钟时间变化的截图（x 轴截止 ≤ 4 小时）。

---

## 10 结语

恭喜你完成了本课程的最后一个作业！你应该为自己的辛勤工作感到自豪。我们希望你享受了从零构建现代语言模型核心组件的过程。
