"""LLM-based prompt mutations (Groq)."""

from __future__ import annotations

import json
import logging
import os
import re

from deslop.prompt_bank import PromptCandidate

logger = logging.getLogger(__name__)

MUTATION_OPS: dict[str, str] = {
    "paraphrase": (
        "Rewrite this prompt to convey the same instructions but with completely different wording."
    ),
    "add_constraint": (
        "Add one specific stylistic constraint to this prompt that would make the output sound "
        "more human-written. Examples: 'start one paragraph with a conjunction', "
        "'include a minor self-correction', 'use at least one sentence fragment'."
    ),
    "remove_bloat": (
        "Simplify this prompt by removing any instruction that sounds like it came from a prompt "
        "engineering guide. Keep only the most essential, natural-sounding directions."
    ),
    "inject_voice": (
        "Add a sentence to this prompt that defines a specific authorial voice — e.g., a tired "
        "grad student, a passionate hobbyist, a skeptical journalist."
    ),
    "restructure": (
        "Reorganize this prompt so the essay structure instructions come before the topic, or "
        "vice versa. Change the ordering of constraints."
    ),
    "crossover": (
        "Given two prompts A and B, produce a child prompt that combines the best stylistic "
        "instructions from each."
    ),
}

_SYSTEM_PARSE = re.compile(r"SYSTEM:\s*(.*?)(?=USER:|$)", re.DOTALL | re.IGNORECASE)
_USER_PARSE = re.compile(r"USER:\s*(.*?)(?=STYLE:|$)", re.DOTALL | re.IGNORECASE)
_STYLE_PARSE = re.compile(r"STYLE:\s*(.*)$", re.DOTALL | re.IGNORECASE)


def _groq_complete(system: str, user: str, model: str, max_tokens: int = 1024) -> str:
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.85,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_triplet(text: str) -> tuple[str, str, str] | None:
    """
    Parse SYSTEM:/USER:/STYLE: blocks or JSON.

    When the model returns JSON (leading ``{``), malformed or truncated output yields ``None``
    so callers can retry or fall back instead of crashing the evolutionary loop.
    """
    text = text.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return (
            str(data.get("system_prompt", "")).strip(),
            str(data.get("user_template", "")).strip(),
            str(data.get("style_instructions", "")).strip(),
        )
    sm = _SYSTEM_PARSE.search(text)
    um = _USER_PARSE.search(text)
    st = _STYLE_PARSE.search(text)
    system = sm.group(1).strip() if sm else ""
    user = um.group(1).strip() if um else text
    style = st.group(1).strip() if st else ""
    if not system:
        system = "You are a thoughtful writer."
    if "{topic}" not in user:
        user = user + " Topic: {topic}."
    return system, user, style


def mutate_prompt(
    parent: PromptCandidate,
    op: str,
    *,
    mutator_groq_model: str = "llama-3.1-8b-instant",
    other: PromptCandidate | None = None,
) -> PromptCandidate | None:
    op_text = MUTATION_OPS.get(op, MUTATION_OPS["paraphrase"])
    base = (
        f"SYSTEM PROMPT:\n{parent.system_prompt}\n\nUSER TEMPLATE:\n{parent.user_template}\n\n"
        f"STYLE INSTRUCTIONS:\n{parent.style_instructions}\n"
    )
    if op == "crossover" and other is not None:
        user_msg = (
            base
            + "\n---\nPROMPT B:\n"
            + f"SYSTEM:\n{other.system_prompt}\nUSER:\n{other.user_template}\nSTYLE:\n{other.style_instructions}\n"
            + "\nProduce ONE child. Respond ONLY with valid JSON keys: "
            "system_prompt, user_template, style_instructions. "
            "user_template MUST contain the literal substring {topic}."
        )
    else:
        user_msg = (
            base
            + "\nApply the following mutation goal:\n"
            + op_text
            + "\n\nRespond ONLY with valid JSON keys: system_prompt, user_template, style_instructions. "
            "user_template MUST contain the literal substring {topic}."
        )

    raw = _groq_complete(
        "You rewrite LLM system/user prompts for essay generation. Output only JSON.",
        user_msg,
        mutator_groq_model,
    )
    stripped = raw.strip()
    raw_fixed = stripped
    if not stripped.startswith("{"):
        raw_fixed = "{" + raw.split("{", 1)[-1] if "{" in raw else raw

    triplet: tuple[str, str, str] | None = None
    for cand in dict.fromkeys((raw_fixed.strip(), stripped)):
        triplet = _parse_triplet(cand)
        if triplet is not None:
            break

    if triplet is None:
        logger.warning(
            "mutator: failed to parse LLM output (JSON decode error or empty JSON segment); "
            "op=%r preview=%r",
            op,
            raw[:500],
        )
        return None

    system, user, style = triplet
    return parent.with_updates(
        system_prompt=system,
        user_template=user,
        style_instructions=style,
        mutation_op=op,
        parent_id=parent.id,
        generation=parent.generation + 1,
    )


def random_op() -> str:
    import random

    return random.choice(list(MUTATION_OPS.keys()))
