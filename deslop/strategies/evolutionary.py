"""Tournament selection and population refill (40% mutate, 30% crossover, 20% elitism, 10% seeds)."""

from __future__ import annotations

import random
import uuid

from deslop.mutator import mutate_prompt, random_op
from deslop.prompt_bank import PromptCandidate, seed_prompt_bank


def tournament_select(population: list[PromptCandidate], k: int, fitness_key: str = "fitness") -> list[PromptCandidate]:
    """Keep top k by fitness (higher is better)."""
    ranked = sorted(population, key=lambda c: getattr(c, fitness_key), reverse=True)
    return ranked[:k]


def refill_population(
    survivors: list[PromptCandidate],
    target_size: int,
    *,
    seed_bank: list[PromptCandidate] | None = None,
    mutator_groq_model: str,
    cotrain_round: int = 0,
) -> list[PromptCandidate]:
    """Expand survivors to target_size using elitism, mutation, crossover, random seeds."""
    seed_bank = seed_bank or seed_prompt_bank()
    out: list[PromptCandidate] = []
    # 20% elitism: keep top unchanged
    n_elite = max(1, int(0.2 * target_size))
    elite = survivors[:n_elite]
    for e in elite:
        c = PromptCandidate(
            id=str(uuid.uuid4())[:12],
            system_prompt=e.system_prompt,
            user_template=e.user_template,
            style_instructions=e.style_instructions,
            fitness=e.fitness,
            generation=e.generation,
            cotrain_round=cotrain_round,
            mutation_op="elitism",
            parent_id=e.id,
        )
        out.append(c)

    while len(out) < target_size:
        r = random.random()
        if r < 0.40 and survivors:
            parent = random.choice(survivors)
            op = random_op()
            if op == "crossover":
                op = "paraphrase"
            child = mutate_prompt(parent, op, mutator_groq_model=mutator_groq_model)
            child.cotrain_round = cotrain_round
            out.append(child)
        elif r < 0.70 and len(survivors) >= 2:
            a, b = random.sample(survivors, 2)
            child = mutate_prompt(a, "crossover", mutator_groq_model=mutator_groq_model, other=b)
            child.cotrain_round = cotrain_round
            out.append(child)
        elif r < 0.90 and survivors:
            parent = random.choice(survivors)
            child = mutate_prompt(parent, random_op(), mutator_groq_model=mutator_groq_model)
            child.cotrain_round = cotrain_round
            out.append(child)
        else:
            s = random.choice(seed_bank)
            fresh = PromptCandidate(
                id=s.id + "_inj",
                system_prompt=s.system_prompt,
                user_template=s.user_template,
                style_instructions=s.style_instructions,
                generation=0,
                cotrain_round=cotrain_round,
                mutation_op="random_seed_injection",
            )
            out.append(fresh)
    return out[:target_size]
