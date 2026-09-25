"""
Phase 1 / Topic 1.1 — thread count diagnostic.

21 tok/s prefill on a 3B Q4 model is roughly 4x below what this class of
CPU should do. The most common cause on Intel hybrid CPUs (P-cores +
E-cores + low-power E-cores) is llama.cpp spreading work evenly across
all cores, so every layer waits on the slowest thread.

num_thread is a per-request option, so this needs no Ollama restart.

    python 03_threads.py
"""

import os
import uuid

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"

# ~1500-token prompt: long enough that fixed overhead is negligible,
# short enough that a bad thread count does not take ten minutes.
FILLER = ("The quarterly maintenance report notes routine wear on the surface "
          "layer near the northern junction and recommends inspection. ")
PROMPT_BODY = FILLER * 95

THREADS = [4, 6, 8, 10, 12, 14, 16, 0]   # 0 = let Ollama decide (your current default)


def run(num_thread: int) -> dict:
    opts = {
        "temperature": 0.0,
        "seed": 42,
        "num_ctx": 4096,
        "num_predict": 32,
    }
    if num_thread:
        opts["num_thread"] = num_thread

    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            # unique prefix defeats Ollama's prompt cache
            "content": f"[{uuid.uuid4().hex}]\n{PROMPT_BODY}\nSummarize in one line.",
        }],
        "stream": False,
        "keep_alive": "10m",
        "options": opts,
    }
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=1800)
    r.raise_for_status()
    return r.json()


def main():
    print(f"Logical processors reported by Python: {os.cpu_count()}\n")

    print("Warming up...")
    run(0)

    print(f"\n{'num_thread':>11} {'prompt tok':>11} {'prefill t/s':>12} {'decode t/s':>11}")
    print("-" * 48)

    results = []
    for nt in THREADS:
        d = run(nt)
        p_tok = d["prompt_eval_count"]
        p_rate = p_tok / (d["prompt_eval_duration"] / 1e9)
        o_rate = d["eval_count"] / (d["eval_duration"] / 1e9)
        label = str(nt) if nt else "default"
        results.append((label, p_rate, o_rate))
        print(f"{label:>11} {p_tok:>11} {p_rate:>12.1f} {o_rate:>11.1f}")

    best_p = max(results, key=lambda r: r[1])
    best_d = max(results, key=lambda r: r[2])
    default = next(r for r in results if r[0] == "default")

    print(f"\nBest prefill: num_thread={best_p[0]} at {best_p[1]:.1f} tok/s "
          f"({best_p[1] / default[1]:.2f}x vs default)")
    print(f"Best decode:  num_thread={best_d[0]} at {best_d[2]:.1f} tok/s "
          f"({best_d[2] / default[2]:.2f}x vs default)")
    print("\nIf a specific thread count clearly beats 'default', that count goes")
    print("into config.py and gets sent on every request from here on.")
    print("If nothing beats default, the bottleneck is elsewhere and we look at")
    print("flash attention and the GPU/CPU split next.")


if __name__ == "__main__":
    main()
