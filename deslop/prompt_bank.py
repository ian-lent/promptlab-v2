"""Structured prompt candidates and seed bank (see also repo root slop_configs/)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def apply_user_template_slots(user_template: str, topic: str, style_instructions: str = "") -> str:
    """
    Fill only ``{topic}`` and ``{style_instructions}``.

    Do not use ``str.format()`` — LLM-mutated templates may contain JSON ``{...}`` which would
    raise KeyError or corrupt the prompt.
    """
    return (
        user_template.replace("{topic}", topic).replace(
            "{style_instructions}", style_instructions or ""
        )
    )


@dataclass
class PromptCandidate:
    id: str
    system_prompt: str
    user_template: str
    style_instructions: str = ""
    fitness: float = 0.0
    generation: int = 0
    cotrain_round: int = 0
    mutation_op: str = "seed"
    parent_id: str | None = None

    def format_user(self, topic: str) -> str:
        return apply_user_template_slots(
            self.user_template, topic, self.style_instructions or ""
        )

    def full_text(self) -> str:
        display_user = apply_user_template_slots(
            self.user_template, "<topic>", self.style_instructions or ""
        )
        parts = [self.system_prompt.strip(), display_user]
        if self.style_instructions.strip():
            parts.append(self.style_instructions.strip())
        return "\n\n".join(p for p in parts if p)

    def with_updates(
        self,
        *,
        system_prompt: str | None = None,
        user_template: str | None = None,
        style_instructions: str | None = None,
        mutation_op: str = "mutate",
        parent_id: str | None = None,
        generation: int | None = None,
        cotrain_round: int | None = None,
    ) -> PromptCandidate:
        return PromptCandidate(
            id=str(uuid.uuid4())[:12],
            system_prompt=system_prompt if system_prompt is not None else self.system_prompt,
            user_template=user_template if user_template is not None else self.user_template,
            style_instructions=(
                style_instructions if style_instructions is not None else self.style_instructions
            ),
            fitness=self.fitness,
            generation=generation if generation is not None else self.generation,
            cotrain_round=cotrain_round if cotrain_round is not None else self.cotrain_round,
            mutation_op=mutation_op,
            parent_id=parent_id or self.id,
        )


def seed_prompt_bank() -> list[PromptCandidate]:
    """Hand-written seeds from naive to heavily instructed (anti-slop)."""
    seeds: list[tuple[str, str, str]] = [
        (
            "You are a helpful assistant.",
            "Write an essay about {topic}.",
            "",
        ),
        (
            "You are a thoughtful writer.",
            "Write a short essay on {topic}. Be clear and direct.",
            "",
        ),
        (
            "You write for a general audience.",
            "Draft an essay about {topic}. {style_instructions}",
            "Use concrete examples. Avoid cliché openings.",
        ),
        (
            "You are an essayist who values clarity over polish.",
            "Write about {topic}. {style_instructions}",
            "Vary sentence length. It's fine if one paragraph is a single short sentence.",
        ),
        (
            "You are a tired graduate student explaining ideas without hype.",
            "Essay topic: {topic}. {style_instructions}",
            "No bullet lists unless the topic truly needs them. Skip moralizing summaries.",
        ),
        (
            "You are a skeptical journalist writing an opinion piece.",
            "Address {topic}. {style_instructions}",
            "Include one counterargument you take seriously. End when the argument feels complete.",
        ),
        (
            "You are a passionate hobbyist, not a corporate blog.",
            "Write on {topic}. {style_instructions}",
            "Let some sentences run long; let others be fragments. Use one specific anecdote or detail.",
        ),
        (
            "You prefer plain language and honest uncertainty.",
            "Essay on {topic}. {style_instructions}",
            "If you are unsure about a claim, say so briefly. Avoid 'delve', 'landscape', 'leverage'.",
        ),
        (
            "You write literary nonfiction.",
            "Compose an essay exploring {topic}. {style_instructions}",
            "Allow a small digression if it serves the thread. No title line; start with the essay body.",
        ),
        (
            "You are an AI asked to sound human — do it by being specific, not performative.",
            "Topic: {topic}. {style_instructions}",
            "Do not mention being an AI. No preamble or closing 'hope this helps'.",
        ),
        (
            "You value voice and rhythm over template structure.",
            "Write about {topic}. {style_instructions}",
            "You may start with a conjunction in one paragraph. Include one minor self-correction.",
        ),
        (
            "You write for a small magazine.",
            "Piece on {topic}. {style_instructions}",
            "Avoid five-paragraph-essay symmetry. Let sections breathe.",
        ),
    ]
    out: list[PromptCandidate] = []
    for sys_p, user_t, style in seeds:
        out.append(
            PromptCandidate(
                id=str(uuid.uuid4())[:12],
                system_prompt=sys_p,
                user_template=user_t,
                style_instructions=style,
                mutation_op="seed",
            )
        )
    return out
