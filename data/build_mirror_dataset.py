#!/usr/bin/env python3
"""
Generate AI mirror essays from human source CSV (refactored from notebooks/build_mirror_dataset).

Providers: gemini, groq, ollama. Requires API keys in environment (see README).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

_BOILERPLATE_PREFIXES = [
    r"^Sure[,!.]?\s*",
    r"^Here(?:'s| is) (?:a|an|the|my) .{0,60}?:\s*",
    r"^Title:\s*.+?\n",
    r"^(?:Of course|Certainly|Absolutely)[,!.]?\s*(?:Here .{0,60}?:\s*)?",
    r"^I(?:'m| am) happy to help.{0,80}?:\s*",
    r"^As requested,.{0,80}?:\s*",
    r"^Below (?:is|are) .{0,60}?:\s*",
]
_BOILERPLATE_SUFFIXES = [
    r"\nI hope (?:this|that) .{0,120}?$",
    r"\nLet me know if .{0,120}?$",
    r"\nFeel free to .{0,120}?$",
    r"\n\(Word count: \d+\s*words?\)\s*$",
    r"\n---+\s*$",
]
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
    flags=re.UNICODE,
)
_QUOTE_MAP = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
})


def clean_llm_output(text: str, min_words: int = 50) -> str | None:
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = _EMOJI_RE.sub("", text)
    text = text.translate(_QUOTE_MAP)
    for pat in _BOILERPLATE_PREFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    for pat in _BOILERPLATE_SUFFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text.split()) >= min_words else None


def clean_title(raw: str) -> str:
    t = raw.strip()
    t = re.sub(r'^["\']|["\']$', "", t).strip()
    t = re.sub(r"^(?:Title|Subject):\s*", "", t, flags=re.IGNORECASE).strip()
    return t


def make_title_prompt(essay_text: str) -> str:
    excerpt = " ".join(essay_text.split()[:800])
    return (
        "What is a concise, descriptive title for the following essay?\n\n"
        f"{excerpt}\n\n"
        "Respond with ONLY the title — no quotation marks, no explanation, no preamble."
    )


def make_mirror_prompt(title: str, target_word_count: int) -> str:
    low = int(target_word_count * 0.85)
    high = int(target_word_count * 1.15)
    return (
        f"Write an essay with the following title:\n\nTitle: {title}\n\n"
        "Requirements:\n"
        f"- Length: between {low} and {high} words\n"
        "- Write in a thoughtful, personal voice as if written by a knowledgeable human author\n"
        "- Do NOT include the title at the top of your response\n"
        "- Do NOT include any preamble, word count, headers, or closing remarks\n"
        "- Begin directly with the first sentence of the essay body"
    )


def build_clients(provider: str, gemini_model: str, groq_model: str, ollama_host: str, ollama_model: str):
    if provider == "gemini":
        import google.generativeai as genai

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY for Gemini.")
        genai.configure(api_key=key)
        return ("gemini", genai.GenerativeModel(gemini_model), None, None, None)

    if provider == "groq":
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Set GROQ_API_KEY for Groq.")
        return ("groq", Groq(api_key=key), groq_model, None, None)

    if provider == "ollama":
        return ("ollama", None, None, ollama_host, ollama_model)

    raise ValueError(provider)


def call_llm(
    state: tuple,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_base_delay: float,
    inter_call_sleep: float,
) -> str | None:
    provider = state[0]
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                model = state[1]
                cfg = {"max_output_tokens": max_tokens, "temperature": temperature}
                resp = model.generate_content(prompt, generation_config=cfg)
                if not resp.candidates:
                    return None
                return resp.text

            if provider == "groq":
                client, groq_model = state[1], state[2]
                resp = client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content

            if provider == "ollama":
                import urllib.request as ur

                _, _, _, host, ollama_model = state
                payload = json.dumps(
                    {
                        "model": ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": temperature},
                    }
                ).encode()
                req = ur.Request(
                    f"{host}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with ur.urlopen(req, timeout=180) as r:
                    return json.loads(r.read()).get("response", "")

        except Exception as e:
            exc_type = type(e).__name__.lower()
            msg = str(e).lower()
            full = f"{exc_type} {msg}"
            if any(kw in full for kw in ["404", "not found", "invalid", "authentication", "permission"]):
                print(f"Fatal: {e}")
                return None
            is_retryable = any(
                kw in full
                for kw in [
                    "429",
                    "toomany",
                    "rate",
                    "quota",
                    "resource exhausted",
                    "500",
                    "502",
                    "503",
                    "unavailable",
                    "timeout",
                    "connection",
                ]
            )
            if is_retryable:
                wait = retry_base_delay * (2**attempt) + random.uniform(0, 2)
                print(f"Retryable [{type(e).__name__}] attempt {attempt + 1}/{max_retries}; sleep {wait:.1f}s")
                time.sleep(wait)
            else:
                print(f"No retry: {e}")
                return None
        time.sleep(inter_call_sleep)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate human/AI essay pairs (mirrors). Each source essay yields 2 JSONL rows "
            "(score 0.0 human, 1.0 AI). For a strong binary detector, target ≥2000 successful "
            "pairs (~≥2000 source essays after filtering); use --max-samples accordingly."
        )
    )
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with essay column")
    ap.add_argument("--text-col", type=str, default="essay")
    ap.add_argument("--output-jsonl", type=Path, default=Path("data/mirror/mirrors.jsonl"))
    ap.add_argument("--provider", choices=("gemini", "groq", "ollama"), default="groq")
    ap.add_argument("--gemini-model", default="gemini-2.0-flash")
    ap.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--ollama-model", default="llama3")
    ap.add_argument("--min-char-len", type=int, default=500)
    ap.add_argument("--max-char-len", type=int, default=12_000)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens-title", type=int, default=60)
    ap.add_argument("--max-tokens-essay", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--inter-call-sleep", type=float, default=None)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-base-delay", type=float, default=5.0)
    args = ap.parse_args()

    limits = {
        "gemini": (4.5, 5, 10.0),
        "groq": (2.0, 4, 5.0),
        "ollama": (0.1, 3, 2.0),
    }
    inter_sleep, retries, retry_delay = limits[args.provider]
    if args.inter_call_sleep is not None:
        inter_sleep = args.inter_call_sleep

    random.seed(args.seed)
    df_raw = pd.read_csv(args.csv)
    if args.text_col not in df_raw.columns:
        raise SystemExit(f"Column {args.text_col!r} not in {list(df_raw.columns)}")

    df_h = (
        df_raw[[args.text_col]]
        .copy()
        .rename(columns={args.text_col: "text"})
        .assign(text=lambda d: d["text"].astype(str).str.strip())
        .loc[lambda d: d["text"].str.len().between(args.min_char_len, args.max_char_len)]
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )
    if args.max_samples and len(df_h) > args.max_samples:
        df_h = df_h.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    state = build_clients(
        args.provider, args.gemini_model, args.groq_model, args.ollama_host, args.ollama_model
    )
    model_label = args.groq_model if args.provider == "groq" else args.gemini_model
    if args.provider == "ollama":
        model_label = args.ollama_model

    with args.output_jsonl.open("a", encoding="utf-8") as out_f:
        for idx, row in tqdm(df_h.iterrows(), total=len(df_h), desc=f"mirrors [{args.provider}]"):
            text = row["text"]
            wc = len(text.split())
            title_raw = call_llm(
                state,
                make_title_prompt(text),
                max_tokens=args.max_tokens_title,
                temperature=0.3,
                max_retries=retries,
                retry_base_delay=retry_delay,
                inter_call_sleep=inter_sleep,
            )
            if not title_raw:
                continue
            title = clean_title(title_raw)
            raw_mirror = call_llm(
                state,
                make_mirror_prompt(title, wc),
                max_tokens=args.max_tokens_essay,
                temperature=args.temperature,
                max_retries=retries,
                retry_base_delay=retry_delay,
                inter_call_sleep=inter_sleep,
            )
            mirror = clean_llm_output(raw_mirror or "", min_words=50)
            if not mirror:
                continue
            ts = datetime.now(timezone.utc).isoformat()
            human_rec = {
                "text": text,
                "score": 0.0,
                "domain": "mirror_essay",
                "source": "mirror_human",
                "split": "train",
                "round": 0,
                "mirror_idx": int(idx),
                "timestamp": ts,
            }
            ai_rec = {
                "text": mirror,
                "score": 1.0,
                "domain": "mirror_essay",
                "source": f"mirror_ai_{model_label}",
                "split": "train",
                "round": 0,
                "mirror_idx": int(idx),
                "timestamp": ts,
            }
            out_f.write(json.dumps(human_rec, ensure_ascii=False) + "\n")
            out_f.write(json.dumps(ai_rec, ensure_ascii=False) + "\n")

    print(f"Appended mirror pairs to {args.output_jsonl}")


if __name__ == "__main__":
    main()
