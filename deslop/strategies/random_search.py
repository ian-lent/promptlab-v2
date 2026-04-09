"""Baseline: random mutations only."""

from __future__ import annotations

import random

from deslop.mutator import mutate_prompt, random_op
from deslop.prompt_bank import PromptCandidate, seed_prompt_bank


def random_search_step(
    population: list[PromptCandidate],
    target_size: int,
    *,
    mutator_groq_model: str,
    cotrain_round: int = 0,
) -> list[PromptCandidate]:
    seeds = seed_prompt_bank()
    out: list[PromptCandidate] = []
    while len(out) < target_size:
        parent = random.choice(population if population else seeds)
        child = mutate_prompt(parent, random_op(), mutator_groq_model=mutator_groq_model)
        child.cotrain_round = cotrain_round
        out.append(child)
    return out
