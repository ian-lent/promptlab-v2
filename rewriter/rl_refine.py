#!/usr/bin/env python3
"""
Optional RLDF: refine rewriter with PPO + detector reward (trl PPOTrainer).

Not wired by default — reward is noisy; warm-start from supervised LoRA first.
"""

from __future__ import annotations

# Skeleton: instantiate PPOTrainer with a reward function that
#   1) runs rewriter → prompt'
#   2) target LLM → essay
#   3) detector.score(essay) + semantic similarity → scalar reward
#
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# See https://huggingface.co/docs/trl/ppo_trainer

def main() -> None:
    raise SystemExit(
        "RL refinement is optional; implement when supervised rewriter plateaus. "
        "See module docstring and trl PPO docs."
    )


if __name__ == "__main__":
    main()
