"""Multi-armed bandit over mutation operators (simple epsilon-greedy)."""

from __future__ import annotations

import random
from collections import defaultdict

from deslop.mutator import MUTATION_OPS, mutate_prompt
from deslop.prompt_bank import PromptCandidate


class MutationBandit:
    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        self.counts: dict[str, int] = defaultdict(int)
        self.rewards: dict[str, float] = defaultdict(float)

    def choose_op(self) -> str:
        if random.random() < self.epsilon or not self.counts:
            return random.choice(list(MUTATION_OPS.keys()))
        means = {op: self.rewards[op] / max(1, self.counts[op]) for op in MUTATION_OPS}
        return max(means, key=means.get)

    def update(self, op: str, reward: float) -> None:
        self.counts[op] += 1
        self.rewards[op] += reward

    def mutate(
        self,
        parent: PromptCandidate,
        *,
        mutator_groq_model: str,
        other: PromptCandidate | None = None,
    ) -> PromptCandidate:
        op = self.choose_op()
        if op == "crossover" and other is None:
            op = "paraphrase"
        return mutate_prompt(parent, op, mutator_groq_model=mutator_groq_model, other=other)
