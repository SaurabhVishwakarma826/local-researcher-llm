"""
Phase 1 / Topic 1.1 — corrected throughput benchmark.

The first version measured fixed overhead, not throughput, because the
prompts were tiny. This sweeps prompt length with the model warm and a
constant output length, so per-token cost is actually visible.

    python 02_benchmark.py
"""

import time
import uuid

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"
NUM_CTX = 16384
NUM_PREDICT = 24          # constant, so decode cost is comparable across rows
TARGETS = [128, 512, 1024, 2048, 4096, 8192]

# ~1 token per word for this filler; we oversupply and let the tokenizer decide.
FILLER = ("The quarterly maintenance report notes routine wear on the surface "
          "layer near the northern junction and recommends inspection. ")


def call(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "10m",          # keep weights resident between rows
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=1800)
    r.raise_for_status()
    return r.json()


def build_prompt(target_tokens: int) -> str:
    # Unique prefix defeats Ollama's prompt-prefix cache, so every row is a
    # genuine cold prefill rather than a cache hit on the previous row.
    reps = max(1, target_tokens // 16)
    return f"[run {uuid.uuid4().hex}]\n" + (FILLER * reps) + "\nSummarize in one line."


def main():
    print("Warming up (loading weights)...")
    t0 = time.time()
    warm = call("hi")
    print(f"  load_duration was {warm.get('load_duration', 0) / 1e9:.2f}s, "
          f"wall {time.time() - t0:.1f}s\n")

    print(f"{'prompt tok':>11} {'prefill s':>10} {'prefill t/s':>12} "
          f"{'out tok':>8} {'decode s':>9} {'decode t/s':>11} {'load s':>7}")
    print("-" * 74)

    rows = []
    for target in TARGETS:
        d = call(build_prompt(target))
        p_tok = d["prompt_eval_count"]
        p_s = d.get("prompt_eval_duration", 0) / 1e9
        o_tok = d["eval_count"]
        o_s = d["eval_duration"] / 1e9
        load_s = d.get("load_duration", 0) / 1e9

        p_rate = p_tok / p_s if p_s else float("nan")
        o_rate = o_tok / o_s if o_s else float("nan")
        rows.append((p_tok, p_rate, o_rate))

        print(f"{p_tok:>11} {p_s:>10.2f} {p_rate:>12.1f} "
              f"{o_tok:>8} {o_s:>9.2f} {o_rate:>11.1f} {load_s:>7.2f}")

    print("\nWhat to look for:")
    print("  1. prefill t/s should RISE sharply from the first row as the fixed")
    print("     overhead is amortised over more tokens, then flatten.")
    print("  2. decode t/s should stay roughly FLAT. It does not care how long")
    print("     the prompt was; it re-reads the weights once per output token.")
    print("  3. At the longest rows prefill t/s starts to sag. That sag is the")
    print("     quadratic attention term becoming visible. It is a slope, not")
    print("     a cliff, at this model size.")
    print("  4. load s must be ~0 on every row. If it is not, the model is being")
    print("     evicted between calls and the numbers are contaminated.")

    if len(rows) >= 2:
        best = max(r[1] for r in rows)
        print(f"\n  Peak prefill: {best:.1f} tok/s")
        print(f"  Decode:       {sum(r[2] for r in rows) / len(rows):.1f} tok/s average")
        print(f"  True ratio:   {best / (sum(r[2] for r in rows) / len(rows)):.1f}x")


if __name__ == "__main__":
    try:
        requests.get(HOST, timeout=5).raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"Ollama is not running at {HOST}. Start it with: ollama serve")
    main()
