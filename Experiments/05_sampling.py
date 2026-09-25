"""
Phase 1 / Topic 1.2 — Sampling.

Run one experiment at a time:
    python 05_sampling.py 1   # what defaults is the model actually using?
    python 05_sampling.py 2   # see the probability list for one token
    python 05_sampling.py 3   # temperature: same question, many runs
    python 05_sampling.py 4   # seed: reproducible randomness
    python 05_sampling.py 5   # top_k / top_p / min_p: cutting off the garbage
"""

import math
import sys

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"
BASE = {"num_ctx": 16384, "num_thread": 12}   # from config.py — keep constant


def chat(content: str, options: dict, logprobs: bool = False, top_logprobs: int = 0) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "keep_alive": "10m",
        "options": {**BASE, **options},
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def text(d: dict) -> str:
    return d["message"]["content"].strip().replace("\n", " ")


# ---------------------------------------------------------------------------
# 1 — Which sampling settings are you getting when you set nothing?
# ---------------------------------------------------------------------------
def exp1_defaults():
    show = requests.post(f"{HOST}/api/show", json={"model": MODEL}, timeout=30).json()
    params = show.get("parameters")
    print("Parameters shipped inside this model's Modelfile:")
    print(params if params else "  (none — Ollama's built-in defaults apply)")
    print("\nOllama's built-in defaults, used for anything not listed above:")
    print("  temperature 0.8   top_k 40   top_p 0.9   min_p 0.0   repeat_penalty 1.1")
    print("\nIf the two lists disagree, the Modelfile wins.")
    print("Either way: every call you made in 1.1 without setting temperature")
    print("was NOT greedy. Only the ones with temperature 0.0 were.")


# ---------------------------------------------------------------------------
# 2 — The probability list behind a single token
# ---------------------------------------------------------------------------
def exp2_distribution():
    prompt = "Reply with only one word: a colour."
    for temp in (0.0, 1.5):
        d = chat(prompt, {"temperature": temp, "seed": 42, "num_predict": 4},
                 logprobs=True, top_logprobs=10)
        lp = d.get("logprobs")
        if not lp:
            sys.exit("No logprobs returned. Your Ollama is too old for this. "
                     "Update Ollama to the latest version and rerun.")

        first = lp[0]
        print(f"\ntemperature={temp}   answer: {text(d)!r}")
        print(f"  chosen first token: {first['token']!r}")
        print("  top 10 candidates for that position:")
        shown = 0.0
        for alt in first.get("top_logprobs", []):
            p = math.exp(alt["logprob"])
            shown += p
            bar = "#" * int(p * 50)
            print(f"    {alt['token']!r:<14} {p * 100:6.2f}%  {bar}")
        print(f"  these 10 cover {shown * 100:.1f}% — the other ~150,000 tokens share the rest")

    print("\nCompare the two tables above.")
    print("If the percentages are IDENTICAL at both temperatures, Ollama is reporting")
    print("the model's raw probabilities, before temperature is applied.")
    print("If they differ, it is reporting the reshaped ones. Tell me which you see.")


# ---------------------------------------------------------------------------
# 3 — Temperature: same question, six runs, different seeds
# ---------------------------------------------------------------------------
def exp3_temperature():
    prompt = "Suggest a name for a pet cat. Reply with only the name."
    for temp in (0.0, 0.7, 1.5):
        outs = []
        for seed in range(1, 7):
            d = chat(prompt, {"temperature": temp, "seed": seed, "num_predict": 10,
                              "top_k": 1000, "top_p": 1.0, "min_p": 0.0})
            outs.append(text(d))
        print(f"\ntemperature={temp}  ({len(set(outs))} different answers out of 6)")
        for o in outs:
            print(f"    {o}")
    print("\nAt 0.0 the seed does nothing — there is no randomness for it to control.")


# ---------------------------------------------------------------------------
# 4 — Seed: randomness you can repeat
# ---------------------------------------------------------------------------
def exp4_seed():
    prompt = "Write a four-word slogan for a coffee shop."
    for seed in (7, 7, 7, 8, 9):
        d = chat(prompt, {"temperature": 1.0, "seed": seed, "num_predict": 16})
        print(f"  seed={seed}:  {text(d)}")
    print("\nSame seed -> same output, even at temperature 1.0.")
    print("Caveat: only on the same machine, same Ollama version, same settings.")
    print("Changing num_thread or the model file can change the result even with the same seed.")


# ---------------------------------------------------------------------------
# 5 — High temperature, with and without cutting the tail
# ---------------------------------------------------------------------------
def exp5_truncation():
    prompt = "Describe a pothole on a highway in one sentence."
    configs = [
        ("almost no cut-off   (top_k=1000, top_p=1.0, min_p=0)", {"top_k": 1000, "top_p": 1.0, "min_p": 0.0}),
        ("Ollama default      (top_k=40,   top_p=0.9, min_p=0)", {"top_k": 40, "top_p": 0.9, "min_p": 0.0}),
        ("min_p only          (top_k=1000, top_p=1.0, min_p=0.1)", {"top_k": 1000, "top_p": 1.0, "min_p": 0.1}),
    ]
    for label, cut in configs:
        print(f"\n{label}   temperature=1.8")
        for seed in (1, 2):
            d = chat(prompt, {"temperature": 1.8, "seed": seed, "num_predict": 40, **cut})
            print(f"    seed {seed}: {text(d)}")
    print("\nSame high temperature in every block. Only the cut-off changes.")
    print("The cut-off is what decides whether high temperature means 'creative'")
    print("or 'broken'.")


EXPERIMENTS = {"1": exp1_defaults, "2": exp2_distribution, "3": exp3_temperature,
               "4": exp4_seed, "5": exp5_truncation}

if __name__ == "__main__":
    try:
        requests.get(HOST, timeout=5).raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"Ollama is not running at {HOST}. Start it with: ollama serve")

    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key not in EXPERIMENTS:
        sys.exit(f"Pick one of: {', '.join(EXPERIMENTS)}")
    EXPERIMENTS[key]()
