"""Baseline: random mutations only."""

from __future__ import annotations

import logging
import random
import uuid

from deslop.mutator import mutate_prompt, random_op
from deslop.prompt_bank import PromptCandidate, seed_prompt_bank

logger = logging.getLogger(__name__)

_MUTATE_ATTEMPTS = 3


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
        child = None
        for _ in range(_MUTATE_ATTEMPTS):
            child = mutate_prompt(parent, random_op(), mutator_groq_model=mutator_groq_model)
            if child is not None:
                break
        if child is None:
            logger.warning(
                "random_search: mutate_prompt failed after %d attempts; injecting seed variant",
                _MUTATE_ATTEMPTS,
            )
            s = random.choice(seeds)
            child = PromptCandidate(
                id=str(uuid.uuid4())[:12] + "_inj",
                system_prompt=s.system_prompt,
                user_template=s.user_template,
                style_instructions=s.style_instructions,
                generation=0,
                cotrain_round=cotrain_round,
                mutation_op="parse_fallback_seed",
            )
        else:
            child.cotrain_round = cotrain_round
        out.append(child)
    return out
