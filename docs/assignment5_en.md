# CS336 Assignment 5 (Alignment): Alignment and Reasoning RL

Version 1.0.2 | CS336 Staff | Spring 2025

---

## 1 Assignment Overview

In this assignment, you will gain hands-on experience with training language models to reason when solving math problems.

### What you will implement

1. Zero-shot prompting baseline for the MATH dataset of competition math problems (Hendrycks et al., 2021)
2. Supervised finetuning (SFT), given reasoning traces from a stronger reasoning model (DeepSeek R1)
3. Expert Iteration for improving reasoning performance with verified rewards
4. Group-Relative Policy Optimization (GRPO) for improving reasoning performance with verified rewards

There is also an **entirely optional** part on aligning language models to human preferences.

### What you will run

1. Measure Qwen 2.5 Math 1.5B zero-shot prompting performance (our baseline)
2. Run SFT on Qwen 2.5 Math 1.5B with reasoning traces from R1
3. Run Expert Iteration on Qwen 2.5 Math 1.5B with verified rewards
4. Run GRPO on Qwen 2.5 Math 1.5B with verified rewards

### Code structure

1. `cs336_alignment/*`: Where you'll write your code for assignment 5. Note that there's no code in here (aside from a little starter code), so you should be able to do whatever you want from scratch.
2. `cs336_alignment/prompts/*`: Text files with prompts to minimize copy-paste errors from PDF.
3. `tests/*.py`: All the tests that you must pass. **You are only expected to pass the tests in `tests/test_sft.py` and `tests/test_grpo.py`** — the rest of the tests are for the non-mandatory parts. Tests invoke hooks defined in `tests/adapters.py`.
4. `README.md`: Basic instructions on setting up your environment.

### What you can use

We expect you to build most of the RL related components from scratch. You may:
- Use vLLM to generate text from language models (Section 3.1)
- Use HuggingFace Transformers to load the Qwen 2.5 Math 1.5B model and tokenizer and run forward passes (Section 4.1)
- You may **not** use any of the training utilities (e.g., the `Trainer` class)

### How to submit

Submit the following files to Gradescope:
- `writeup.pdf`: Answer all the written questions. Please typeset your responses.
- `code.zip`: Contains all the code you've written.

---

## 2 Reasoning with Language Models

### 2.1 Motivation

One of the remarkable use cases of language models is in building generalist systems that can handle a wide range of natural language processing tasks. In this assignment, we focus on mathematical reasoning as a testbed for setting up evaluations, performing supervised finetuning, and experimenting with teaching LMs to reason using reinforcement learning (RL).

Two key differences from past assignments:

- **Model**: We switch to Qwen 2.5 Math 1.5B Base, a modern high-performance language model, since our earlier models are too weak for non-trivial mathematical reasoning.
- **Evaluation**: We introduce a new benchmark. Rather than using cross-entropy as a surrogate, we use the MATH 12K dataset of challenging high-school competition mathematics problems, evaluating by comparing model outputs against reference answers.

### 2.2 Chain-of-Thought Reasoning and Reasoning RL

**Chain-of-thought reasoning with LLMs**: Early approaches finetuned models to use a "scratchpad" to break problems into intermediate steps. Other work prompts a strong model to "think step by step", significantly improving performance on mathematical reasoning tasks.

**Learning to reason with expert iteration**: The Self-Taught Reasoner (STaR, Zelikman et al., 2022) frames reasoning as a bootstrapping loop: a pretrained model first samples diverse chains-of-thought, keeps only those that lead to correct answers, and then finetunes on these "expert" traces.

**Reasoning RL with verified rewards, o1, and R1**: Recent work by OpenAI (o1/o3/o4), DeepSeek (R1), and Moonshot (kimi k1.5) uses policy gradient methods to train on math and code tasks where string matching or unit tests verify correctness. Later works like Open-R1, SimpleRL-Zoo, and TinyZero confirm that pure RL with verified rewards can improve reasoning performance even on models as small as 1.5B parameters.

### Our setup: model and dataset

- **Model**: Qwen 2.5 Math 1.5B Base (continually pretrained from Qwen 2.5 1.5B on high-quality synthetic math pretraining data)
- **Dataset**: The MATH dataset, available on the Together cluster at `/data/a5-alignment/MATH`

---

## 3 Measuring Zero-Shot MATH Performance

We start by measuring the performance of our base language model on the 5K example test set of MATH.

Unless otherwise specified, we use the DeepSeek R1-Zero prompt (the `r1_zero` prompt):

```
A conversation between User and Assistant. The User asks a question, and the Assistant
solves it. The Assistant first thinks about the reasoning process in the mind and
then provides the User with the answer. The reasoning process is enclosed within
<think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

Located in `cs336_alignment/prompts/r1_zero.prompt`.

**Note on prompt choice**: The `r1_zero` prompt is not the best choice for maximizing downstream performance after RL, due to a mismatch with how Qwen 2.5 Math 1.5B was pretrained. We choose it because RL with it shows clear accuracy improvements in a short number of steps. You will later compare to the `question_only` prompt.

### 3.1 Using vLLM for offline language model inference

We recommend using vLLM for offline batched inference. vLLM is a high-throughput and memory-efficient inference engine incorporating optimized CUDA kernels, PagedAttention for efficient attention KV caching, etc.

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

Pre-trained model paths on the Together cluster:
- Qwen 2.5 Math 1.5B Base: `/data/a5-alignment/models/Qwen2.5-Math-1.5B`
- Llama 3.1 8B Base (optional): `/data/a5-alignment/models/Llama-3.1-8B`
- Llama 3.3 70B Instruct (optional): `/data/a5-alignment/models/Llama-3.3-70B-Instruct`

### 3.2 Zero-shot MATH Baseline

**Prompting setup**: Load the MATH test set examples, format them as string prompts using the `r1_zero` prompt, and generate outputs.

**Evaluation metric**: We use `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` as the answer parser and reward function, which handles semantically equivalent answer matching.

**Generation hyperparameters**: Temperature 1.0, top-p 1.0, max generation length 1024. Use `</answer>` as stop string:

```python
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
```

### Problem (math_baseline): 4 points

**(a)** Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script should:
1. Load the MATH validation examples from `/data/a5-alignment/MATH/validation.jsonl`
2. Format them as string prompts using the `r1_zero` prompt
3. Generate outputs for each example
4. Calculate evaluation metrics
5. Serialize the examples, model generations, and evaluation scores to disk

It might be helpful to implement a method `evaluate_vllm` for reuse later.

**Deliverable**: A script to evaluate baseline zero-shot MATH performance.

**(b)** Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each of the following categories:
1. Correct with both format reward and answer reward 1
2. Format reward 1 and answer reward 0
3. Format reward 0 and answer reward 0

Observing at least 10 cases where format reward is 0, do you think the issue is with the base model's output, or the parser? What about in (at least 10) cases where format reward is 1 but answer reward is 0?

**Deliverable**: Commentary on the model and reward function performance, including examples of each category.

**(c)** How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?

**Deliverable**: 1-2 sentences with evaluation metrics.

---

## 4 Supervised Finetuning for MATH

### Algorithm 1: Supervised Finetuning (SFT)

```
Input: initial policy model π_θ_init; SFT dataset D
1: policy model π_θ ← π_θ_init
2: for step = 1, ..., n_sft_steps do
3:     Sample a batch of question-response pairs D_b from D
4:     Compute the cross-entropy loss of the responses given the questions using model π_θ
5:     Update the model parameters θ by taking a gradient step w.r.t. the cross-entropy loss
6: end for
Output: π_θ
```

**Supervised finetuning for reasoning**: We finetune our base model on the MATH dataset. Rather than finetuning to directly predict correct answers, we finetune it to first generate a chain-of-thought reasoning trace followed by an answer. The SFT data consists of reasoning traces from DeepSeek R1, located at:

```
/data/a5-alignment/MATH/sft.jsonl
```

SFT is often used as a warm-start for a second RL finetuning step because:
1. SFT requires high-quality annotated data (with reasoning traces), whereas RL requires only the correct answer as feedback
2. Even with abundant annotated data, RL can still unlock performance gains by finding better policies

### 4.1 Using HuggingFace Models

**Loading a HuggingFace model and tokenizer**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
```

**Forward pass**:

```python
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)
logits = model(input_ids).logits
loss = F.cross_entropy(...)
```

**Saving a trained model**: Use `.save_pretrained()` to save under `/data/yourusername` since models can be quite large.

**Gradient accumulation**: Even with bfloat16 and FlashAttention-2, an 80GB GPU may not have enough memory for reasonable batch sizes. We accumulate gradients over several batches before taking a gradient step:

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

### 4.2 SFT Helper Methods

#### Problem (tokenize_prompt_and_output): Prompt and output tokenization (2 points)

Implement a method `tokenize_prompt_and_output` that tokenizes the question and output separately, concatenates them together, and constructs a `response_mask`.

```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
```

Returns a dict with keys:
- `input_ids`: shape `(batch_size, max(prompt_and_output_lens) - 1)`, tokenized prompt and output strings with the final token sliced off
- `labels`: shape `(batch_size, max(prompt_and_output_lens) - 1)`, shifted input_ids (without the first token)
- `response_mask`: shape `(batch_size, max(prompt_and_output_lens) - 1)`, a mask on the response tokens in the labels

Test: Implement `adapters.run_tokenize_prompt_and_output`, then run `uv run pytest -k test_tokenize_prompt_and_output`.

#### Problem (compute_entropy): Per-token entropy (1 point)

Implement a method `compute_entropy` that computes the per-token entropy of next-token predictions.

```python
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
```

- Input `logits`: shape `(batch_size, sequence_length, vocab_size)`
- Output: shape `(batch_size, sequence_length)`

Note: you should use a numerically stable method (e.g., using `logsumexp`) to avoid overflow.

Test: Implement `adapters.run_compute_entropy`, then run `uv run pytest -k test_compute_entropy`.

#### Problem (get_response_log_probs): Response log-probs (and entropy) (2 points)

Implement a method `get_response_log_probs` that gets per-token conditional log-probabilities from a causal language model, and optionally the entropy of the model's next-token distribution.

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
```

Returns:
- `"log_probs"`: shape `(batch_size, sequence_length)`, conditional log-probabilities
- `"token_entropy"`: optional, shape `(batch_size, sequence_length)`, per-token entropy

Implementation tips: Obtain logits with `model(input_ids).logits`.

Test: Implement `adapters.run_get_response_log_probs`, then run `uv run pytest -k test_get_response_log_probs`.

#### Problem (masked_normalize): Masked normalize (1 point)

Implement a method `masked_normalize` that sums over tensor elements and normalizes by a constant while respecting a boolean mask.

```python
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
```

Test: Implement `adapters.run_masked_normalize`, then run `uv run pytest -k test_masked_normalize`.

#### Problem (sft_microbatch_train_step): Microbatch train step (3 points)

Implement a single micro-batch update for SFT, including cross-entropy loss, summing with a mask, and gradient scaling.

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

Returns:
- `loss`: scalar tensor, the microbatch loss adjusted for gradient accumulation
- `metadata`: dict with metadata from the underlying loss call

Implementation tips: You should call `loss.backward()` in this function. Make sure to adjust for gradient accumulation.

Test: Implement `adapters.run_sft_microbatch_train_step`, then run `uv run pytest -k test_sft_microbatch_train_step`.

#### Logging generations in-the-loop

Write a function `log_generations` that will prompt your model to generate responses for some given prompts (e.g., sampled from the validation set). It's a good idea to log at least:

1. The input prompt
2. The response generated by the SFT/RL model
3. The ground-truth answer
4. The reward information, including format, answer, and total reward
5. The average token entropy of the response
6. The average response length for correct and incorrect responses

#### Problem (log_generations): Logging generations (1 point)

**Deliverable**: Implement a function `log_generations` that can be used to log generations from your model.

### 4.3 SFT Experiment

Using the pieces above, implement the full SFT procedure (Algorithm 1) to finetune the Qwen 2.5 Math 1.5B Base model on the MATH dataset.

Each example in `/data/a5-alignment/MATH/sft.jsonl` consists of a formatted prompt and a target response, where the target response includes a chain-of-thought reasoning trace and the final answer. Each example is a JSON element of type `{"prompt": str, "response": str}`.

You should run your script with 2 GPUs, using one GPU for the policy model and the other for the vLLM instance to evaluate the policy.

Starter code for initializing vLLM and loading policy weights:

```python
from vllm.model_executor import set_random_seed as vllm_set_random_seed

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
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

Suggest using gradient clipping with clip value 1.0.

#### Problem (sft_experiment): Run SFT on the MATH dataset (2 points) (2 H100 hrs)

1. Run SFT on the reasoning SFT examples (in `/data/a5-alignment/MATH/sft.jsonl`) using the Qwen 2.5 Math 1.5B base model, varying the number of unique examples for SFT in the range {128, 256, 512, 1024}, along with using the full dataset. Tune the learning rate and batch size to achieve at least 15% validation accuracy when using the full dataset.

   **Deliverable**: Validation accuracy curves associated with different dataset sizes.

2. Filter the reasoning SFT examples to only include examples that produce the correct answer. Run SFT on the (full) filtered dataset and report the size of the filtered dataset and the validation accuracy you achieve.

   **Deliverable**: Report the size of the dataset and the validation accuracy curve. Compare your findings to the previous SFT experiment.

---

## 5 Expert Iteration for MATH

In the previous section, we observed that we can improve the performance of our SFT model by filtering out bad examples from the SFT data. In this section, we go one step further: we apply this filtering procedure to reasoning traces we generate from our base model itself. This process is known as **expert iteration**.

### Algorithm 2: Expert Iteration (EI)

```
Input: initial policy model π_θ_init; reward function R; task questions D
1: policy model π_θ ← π_θ_init
2: for step = 1, ..., n_ei_steps do
3:     Sample a batch of questions D_b from D
4:     Set the old policy model π_θ_old ← π_θ
5:     Sample G outputs {o^(i)} ~ π_θ_old(·|q) for each question q ∈ D_b
6:     Compute rewards {r^(i)} for each sampled output by running reward function R(q, o^(i))
7:     Filter out wrong outputs (i.e., o^(i) with r^(i) = 0) to obtain D_sft of correct question-response pairs
8:     π_θ = SFT(π_θ, D_sft) (Algorithm 1)
9: end for
Output: π_θ
```

Tip: pass a `min_tokens` value to your vLLM `SamplingParams` to ensure you don't generate an empty string:

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

Use gradient clipping with clip value 1.0.

#### Problem (expert_iteration_experiment): Run expert iteration on the MATH dataset (2 points) (6 H100 hrs)

Run expert iteration on the MATH dataset (at `/data/a5-alignment/MATH/train.jsonl`) using the Qwen 2.5 Math 1.5B Base model:
- Vary the number of rollouts G per question and the number of epochs used in the SFT step
- Use `n_ei_steps = 5`
- Vary the batch size for each expert iteration step (i.e., the size of D_b) in {512, 1024, 2048}
- Log the entropy of the model's responses over training
- Make sure to have vLLM terminate generations at the second answer tag `</answer>`

**Deliverables**:
- Validation accuracy curves associated with different rollout configurations. Try at least 2 different rollout counts and epoch counts.
- A model that achieves validation accuracy of at least 15% on MATH.
- A brief 2 sentence discussion comparing to your SFT performance, as well as performance across EI steps.
- A plot of the entropy of the model's responses over training.

---

## 6 Primer on Policy Gradients

Performing RL against verified rewards can lead to significant improvements in reasoning capabilities and performance of base models.

### 6.1 Language Models as Policies

A causal language model with parameters θ defines a probability distribution over the next token a_t ∈ V given the current text prefix s_t (the state/observation). In RL context, the next token is an **action** and the current text prefix is the **state**. The LM is a *categorical stochastic policy*:

$$a_t \sim \pi_\theta(\cdot | s_t), \quad \pi_\theta(a_t | s_t) = [\text{softmax}(f_\theta(s_t))]_{a_t}$$

Two primitive operations are needed:
1. **Sampling from the policy**: drawing an action a_t from the categorical distribution
2. **Scoring the log-likelihood of an action**: evaluating log π_θ(a_t | s_t)

### 6.2 Trajectories

A (finite-horizon) trajectory is the interleaved sequence of states and actions:

$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$$

In RL with LLMs, the environment is deterministic: s_{t+1} = s_t || a_t. Trajectories are also called episodes or rollouts.

### 6.3 Rewards and Return

A scalar reward r_t = R(s_t, a_t) judges the quality of the action at state s_t. For RL on verified domains, we assign zero reward to intermediate steps and a **verified reward** to the terminal action:

$$r_T = R(s_T, a_T) = \begin{cases} 1 & \text{if the trajectory matches the ground-truth} \\ 0 & \text{otherwise} \end{cases}$$

The return R(τ) aggregates rewards. We use **finite-horizon undiscounted returns**:

$$R(\tau) = \sum_{t=0}^{T} r_t$$

The objective is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

### 6.4 Vanilla Policy Gradient

Learn policy parameters θ with gradient ascent on the expected return:

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)$$

The REINFORCE policy gradient:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]$$

**Derivation** uses the log-derivative trick ∇_θ P = P ∇_θ log P and the fact that environment terms are constant in θ.

**Sample estimate**: Given a batch of N rollouts D = {τ^(i)}, the unbiased estimator is:

$$\hat{g} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$$

### 6.5 Policy Gradient Baselines

The main issue with vanilla policy gradient is high variance. A common technique is to subtract a **baseline** function b that depends only on the state:

$$B = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)(R(\tau) - b(s_t))\right]$$

As long as the baseline depends only on the state, the baselined policy gradient is unbiased.

**A note on policy gradient "losses"**: In PyTorch, we define `pg_loss` such that `pg_loss.backward()` populates gradient buffers with the approximate policy gradient. `pg_loss` is not a loss in the canonical sense — for RL, always log and report train and validation **rewards** as the meaningful evaluation metrics.

### 6.6 Off-Policy Policy Gradient

REINFORCE is on-policy: the training data is collected by the same policy. This is inefficient.

The off-policy policy gradient estimate uses rollouts from a previous policy π_θ_old:

$$\hat{g}_{\text{off-policy}} = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T} \frac{\pi_\theta(a_t^{(i)}|s_t^{(i)})}{\pi_{\theta_\text{old}}(a_t^{(i)}|s_t^{(i)})} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) R(\tau^{(i)})$$

This can be derived via importance sampling.

---

## 7 Group Relative Policy Optimization (GRPO)

### 7.1 GRPO Algorithm

**Advantage estimation**: The core idea of GRPO is to sample multiple outputs for each question from the policy π_θ and use them to compute a baseline. For question q and group outputs {o^(i)}, the group-normalized reward is:

$$A^{(i)} = \frac{r^{(i)} - \text{mean}(r^{(1)}, \ldots, r^{(G)})}{\text{std}(r^{(1)}, \ldots, r^{(G)}) + \text{advantage\_eps}}$$

The GRPO objective combines three ideas:
1. Off-policy policy gradient
2. Computing advantages A^(i) with group normalization
3. A clipping mechanism, as in Proximal Policy Optimization (PPO, Schulman et al., 2017)

### Algorithm 3: Group Relative Policy Optimization (GRPO)

```
Input: initial policy model π_θ_init; reward function R; task questions D
1: policy model π_θ ← π_θ_init
2: for step = 1, ..., n_grpo_steps do
3:     Sample a batch of questions D_b from D
4:     Set the old policy model π_θ_old ← π_θ
5:     Sample G outputs ~ π_θ_old(·|q) for each question q ∈ D_b
6:     Compute rewards for each sampled output by running reward function R(q, o^(i))
7:     Compute A^(i) with group normalization
8:     for train step = 1, ..., n_train_steps_per_rollout_batch do
9:         Update the policy model π_θ by maximizing the GRPO-Clip objective
10:    end for
11: end for
Output: π_θ
```

**GRPO-Clip objective** (Eq. 29): For each token, take the minimum of the clipped and unclipped ratio times the advantage, which limits the magnitude of policy updates.

**Dr. GRPO variant** (Liu et al., 2025): Proposes removing the standard deviation normalization, computing simply:

$$A^{(i)} = r^{(i)} - \text{mean}(r^{(1)}, \ldots, r^{(G)})$$

### 7.2 Implementation

#### Problem (compute_group_normalized_rewards): Group normalization (2 points)

Implement `compute_group_normalized_rewards`: calculate raw rewards for each group of rollout responses, normalize them within their groups, and return both.

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

Returns `tuple[torch.Tensor, torch.Tensor, dict[str, float]]`:
- `advantages`: shape `(rollout_batch_size,)`, group-normalized rewards
- `raw_rewards`: shape `(rollout_batch_size,)`, unnormalized rewards
- `metadata`: your choice of statistics to log

Test: `uv run pytest -k test_compute_group_normalized_rewards`

#### Problem (compute_naive_policy_gradient_loss): Naive policy gradient (1 point)

Implement the naive policy gradient loss, which multiplies the advantage by the log-probability (and negates):

$$-A_t \cdot \log p_\theta(o_t | q, o_{<t})$$

```python
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # (batch_size, 1)
    policy_log_probs: torch.Tensor,           # (batch_size, sequence_length)
) -> torch.Tensor:                            # (batch_size, sequence_length)
```

Implementation tip: Broadcast `raw_rewards_or_advantages` over the `sequence_length` dimension.

Test: `uv run pytest -k test_compute_naive_policy_gradient_loss`

#### Problem (compute_grpo_clip_loss): GRPO-Clip loss (2 points)

Implement the per-token GRPO-Clip loss (Eq. 33):

$$-\min\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})} A_t, \text{clip}\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})}, 1-\epsilon, 1+\epsilon\right) A_t\right)$$

```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,      # (batch_size, 1)
    policy_log_probs: torch.Tensor, # (batch_size, sequence_length)
    old_log_probs: torch.Tensor,    # (batch_size, sequence_length)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

Returns:
- `loss`: shape `(batch_size, sequence_length)`, per-token clipped loss
- `metadata`: suggest logging whether each token was clipped

Test: `uv run pytest -k test_compute_grpo_clip_loss`

#### Problem (compute_policy_gradient_loss): Policy-gradient wrapper (1 point)

Implement `compute_policy_gradient_loss`, a convenience wrapper that dispatches to the correct loss routine:

- `"no_baseline"`: Naive policy gradient loss, advantage is raw rewards A = R(q, o)
- `"reinforce_with_baseline"`: Naive policy gradient loss using group-normalized rewards as advantage
- `"grpo_clip"`: GRPO-Clip loss

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

Test: `uv run pytest -k test_compute_policy_gradient_loss`

#### Problem (masked_mean): Masked mean (1 point)

Implement `masked_mean`: average tensor elements along a dimension while respecting a boolean mask.

```python
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
```

Test: `uv run pytest -k test_masked_mean`

#### Problem (grpo_microbatch_train_step): Microbatch train step (3 points)

Implement a single micro-batch update for GRPO, including policy-gradient loss, averaging with a mask, and gradient scaling.

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

Returns:
- `loss`: scalar tensor, microbatch loss adjusted for gradient accumulation
- `metadata`: dict with metadata

Implementation tips: You should call `loss.backward()` in this function.

Test: `uv run pytest -k test_grpo_microbatch_train_step`

#### Problem (grpo_train_loop): GRPO train loop (5 points)

Implement a complete train loop for GRPO. Begin training a policy on MATH and confirm that you see validation rewards improving over time, along with sensible rollouts.

**Deliverable**: A plot with the validation rewards with respect to steps, and a few example rollouts over time.

**Default hyperparameters**:

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
gradient_accumulation_steps = 128  # microbatch size is 2, will fit on H100
loss_type = "reinforce_with_baseline"
use_std_normalization = True
optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),
)
```

**Additional tips**:
- Remember to use the `r1_zero` prompt, and direct vLLM to stop at the second answer tag `</answer>`
- We suggest using `typer` for argument parsing
- Use gradient clipping with clip value 1.0
- You should routinely log validation rewards (e.g., every 5 or 10 steps). Evaluate on at least 1024 validation examples, as CoT/RL evaluations can be noisy
- GRPO-Clip should only be used when off-policy (since it requires the old log-probabilities)
- In the off-policy setting, compute old log-probabilities once and reuse them for each epoch
- You should not differentiate with respect to the old log-probabilities
- You should log: the loss, gradient norm, token entropy, clip fraction (if off-policy), train rewards (total, format, and answer)

**Sanity check asserts**:
```python
assert train_batch_size % gradient_accumulation_steps == 0
micro_train_batch_size = train_batch_size // gradient_accumulation_steps
assert rollout_batch_size % group_size == 0
n_prompts_per_rollout_batch = rollout_batch_size // group_size
assert train_batch_size >= group_size
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
```

---

## 8 GRPO Experiments

Each experiment takes 2 GPUs, one for the vLLM instance and one for the policy.

**Note on stopping runs early**: If you see significant differences between hyperparameters before 200 GRPO steps, feel free to stop early to save time and compute.

#### Problem (grpo_learning_rate): Tune the learning rate (2 points) (6 H100 hrs)

Starting with the suggested hyperparameters, perform a sweep over the learning rates and report the final validation answer rewards (or note divergence).

**Deliverables**:
- Validation reward curves associated with multiple learning rates
- A model that achieves validation accuracy of at least 25% on MATH
- A brief 2 sentence discussion on any other trends you notice on other logged metrics

Use the best learning rate for the rest of the experiments.

#### Problem (grpo_baselines): Effect of baselining (2 points) (2 H100 hrs)

In the on-policy setting, compare the two loss types: `no_baseline` and `reinforce_with_baseline`. Note that `use_std_normalization` is `True` in the default hyperparameters.

**Deliverables**:
- Validation reward curves associated with each loss type
- A brief 2 sentence discussion on any other trends

Use the best loss type for subsequent experiments.

#### Length normalization

Two approaches for aggregating per-token losses:
- `masked_mean`: averages over the unmasked tokens in each sequence
- `masked_normalize`: sums over unmasked tokens and divides by a constant (supports `constant_normalizer != 1.0`)

These approaches affect the gradient differently. With `masked_mean`, each token's gradient contribution is inversely proportional to its sequence length. With `masked_normalize`, all tokens contribute equally.

#### Problem (think_about_length_normalization): Think about length normalization (1 point)

Compare the two approaches (without running experiments). What are the pros and cons of each approach? Are there any specific settings or examples where one approach seems better?

#### Problem (grpo_length_normalization): Effect of length normalization (2 points) (2 H100 hrs)

Compare normalization with `masked_mean` and `masked_normalize` with an end-to-end GRPO training run. Report the validation answer reward curves. Comment on the findings, including any other metrics that have a noticeable trend.

Hint: consider metrics related to stability, such as the gradient norm.

Fix to the better performing approach for subsequent experiments.

#### Problem (grpo_group_standard_deviation): Effect of standard deviation normalization (2 points) (2 H100 hrs)

Compare the performance of `use_std_normalization == True` and `use_std_normalization == False`. Report the validation answer reward curves. Comment on the findings.

Hint: consider metrics related to stability, such as the gradient norm.

Fix to the better performing approach for subsequent experiments.

#### Off-policy vs on-policy

So far all experiments have been on-policy (one gradient step per rollout batch). While theoretically justified and stable, this is inefficient since rollouts are the bottleneck.

#### Problem (grpo_off_policy): Implement off-policy GRPO

Implement off-policy GRPO training:
- You should be able to take multiple epochs of gradient steps per rollout batch
- The number of epochs and optimizer updates are controlled by `rollout_batch_size`, `epochs_per_rollout_batch`, and `train_batch_size`
- Edit your main training loop to get response logprobs from the policy after each rollout batch generation phase — these will be the `old_log_probs`
- We suggest using `torch.inference_mode()`
- You should use the `"GRPO-Clip"` loss type

#### Problem (grpo_off_policy_sweep): Off-policy GRPO hyperparameter sweep (4 points) (12 H100 hrs)

Fixing `rollout_batch_size = 256`, choose a range over `epochs_per_rollout_batch` and `train_batch_size` to sweep over. First do a broad sweep for a limited number of GRPO steps (<50), and then a more focused sweep for a larger number of GRPO steps (200).

Compare to your on-policy run with `epochs_per_rollout_batch = 1` and `train_batch_size = 256`, reporting plots with respect to number of validation steps as well as with respect to wall-clock time.

Report the validation answer reward curves. Comment on the findings, including any other metrics that have a noticeable trend such as entropy and response length. Compare the entropy of the model's responses over training to what you observed in the EI experiment.

Hint: you will need to change `gradient_accumulation_steps` to keep memory usage constant.

#### Ablating clipping in the off-policy setting

Recall that clipping prevents the policy from moving too far from the old policy when taking many gradient steps on a single rollout batch.

#### Problem (grpo_off_policy_clip_ablation): Off-policy GRPO-Clip ablation (2 points) (2 H100 hrs)

Implement the unclipped per-token loss as a new loss type `"GRPO-No-Clip"`:

$$-\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_\text{old}}(o_t|q, o_{<t})} A_t$$

Take your best performing off-policy hyperparameters and run the unclipped version. Report the validation answer reward curves. Comment on the findings compared to your GRPO-Clip run.

#### Effect of prompt

The prompt used during RL can have a dramatic effect on the model's performance. Instead of the R1-Zero prompt, we use a simple prompt at `cs336_alignment/prompts/question_only.prompt`:

```
{question}
```

You will also change your reward function to `question_only_reward_fn` in `cs336_alignment/drgrpo_grader.py`.

#### Problem (grpo_prompt_ablation): Prompt ablation (2 points) (2 H100 hrs)

Report the validation answer reward curves for both the R1-Zero prompt and the question-only prompt. How do metrics compare, including any noticeable trend such as entropy, response length, and gradient norm? Try to explain your findings.

---

## 9 Leaderboard: GRPO on MATH

As the last part of the (mandatory) assignment, you will experiment with approaches to obtain the highest validation rewards possible within 4 hours of training on 2 H100 GPUs.

**Model**: Continue using the Qwen 2.5 Math 1.5B Base model.

**Dataset**: MATH train and validation datasets at `/data/a5-alignment/MATH/train.jsonl` and `/data/a5-alignment/MATH/validation.jsonl`. You are not allowed to use any other data or do SFT on reasoning chains from stronger models.

**Constraints on evaluation**:
1. Validation accuracy should be the average accuracy over the entire MATH validation set (all 5K examples)
2. You must use the R1-Zero prompt at validation time
3. You must use temperature 1.0 and max tokens 1024 with vLLM for evaluation
4. You must calculate validation accuracy by averaging the answer rewards produced by the `r1_zero_reward_fn` reward function

**Algorithm**: You are free to tune hyperparameters or change the training algorithm entirely, as long as you do not use any extraneous data or another model (you are free to use more copies of the model if you want).

**Systems optimizations**: You might observe that at least one GPU is always idle. Consider lower precision for rollouts or training, `torch.compile`, and other systems optimizations. You are not constrained to placing vLLM on a single device and the train policy on another device.

**Reference projects**: veRL, trl, torchtune, oat

**On KL divergence**: The above experiments did not include a KL divergence term with respect to a reference model. In experiments, omitting the KL term had no impact on performance while saving GPU memory. However, you are encouraged to experiment with KL or other forms of regularization, **as long as you use Qwen 2.5 Math 1.5B Base or some model obtained through your algorithm for it**.

#### Problem (leaderboard): Leaderboard (16 points) (16 H100 hrs)

**Deliverable**: Report a validation accuracy obtained within 4 hours of training on 2 H100 GPUs and a screenshot of your validation accuracy with respect to wall-clock time, where the x-axis ends at ≤ 4 hours.

---

## 10 Epilogue

Congratulations on finishing the last assignment of the class! You should be proud of your hard work. We hope you enjoyed learning the foundations underlying modern language models by building their main components from scratch.
