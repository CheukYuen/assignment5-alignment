# CLAUDE.md

## Project Overview

Stanford CS336 Spring 2025 Assignment 5: Alignment. This assignment covers SFT (Supervised Fine-Tuning), Expert Iteration, and GRPO (Group Relative Policy Optimization) with verified rewards on MATH.

There is also an optional supplemental assignment on safety alignment, instruction tuning, and RLHF (DPO).

## Repository Structure

- `cs336_alignment/` — Main module with alignment code (implementation goes here)
  - `drgrpo_grader.py` — Dr. GRPO grading utilities
  - `prompts/` — Prompt templates (alpaca_sft, question_only, r1_zero, zero_shot_system_prompt)
- `tests/` — Test suite
  - `adapters.py` — Adapter functions connecting implementations to tests (implement here)
  - `test_sft.py`, `test_grpo.py`, `test_dpo.py`, `test_metrics.py`, `test_data.py` — Test files
  - `_snapshots/` — Test snapshots
  - `fixtures/` — Test fixtures
- `scripts/` — Evaluation scripts (alpaca_eval, safety evaluation)
- `data/` — Datasets (alpaca_eval, gsm8k, mmlu, simple_safety_tests)
- `docs/` — Assignment documentation in markdown format
  - `assignment5_en.md` — English markdown version of the assignment PDF
  - `assignment5_zh.md` — Chinese translation of the assignment

## Key Commands

```bash
# Install dependencies
uv sync --no-install-package flash-attn && uv sync

# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sft.py -v
uv run pytest tests/test_grpo.py -v
uv run pytest tests/test_dpo.py -v
uv run pytest tests/test_metrics.py -v
uv run pytest tests/test_data.py -v

# Run a specific test
uv run pytest tests/test_sft.py::test_name -v

# Make submission
bash test_and_make_submission.sh
```

## Implementation Notes

- All adapter functions in `tests/adapters.py` start as `raise NotImplementedError` — implement them to pass the tests.
- The implementation code should go in `cs336_alignment/` module.
- Key areas to implement:
  1. **Tokenization**: `run_tokenize_prompt_and_output` — tokenize prompt+output with response mask
  2. **SFT**: `run_sft_microbatch_train_step` — supervised fine-tuning loss and backprop
  3. **GRPO**: Group-normalized rewards, policy gradient losses (naive, REINFORCE with baseline, GRPO-clip)
  4. **Utilities**: `run_masked_mean`, `run_masked_normalize`, `run_compute_entropy`
  5. **Optional (DPO)**: `run_compute_per_instance_dpo_loss`
  6. **Optional (Data)**: `get_packed_sft_dataset`, `run_iterate_batches`
  7. **Optional (Metrics)**: `run_parse_mmlu_response`, `run_parse_gsm8k_response`

## Tech Stack

- Python 3.11-3.12
- PyTorch, Transformers, Flash-Attention 2
- vLLM 0.7.2 for inference
- accelerate for distributed training
- alpaca-eval for evaluation
- uv for dependency management

## Style

- Follow existing code patterns in the repo
- Use type hints consistent with existing signatures
- Keep implementations clean and well-documented where logic is non-trivial
