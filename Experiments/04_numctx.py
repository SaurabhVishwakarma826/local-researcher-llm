"""
Phase 1 / Topic 1.1 — does num_ctx itself slow prefill?

Same prompt, same threads, server already warm. Only num_ctx changes.
Each setting runs twice so single-run noise is visible.

    python 04_numctx.py
"""

import uuid

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"
NUM_THREAD = 12
CTX_VALUES = [4096, 16384, 4096, 16384]   # interleaved, so drift can't masquerade as an effect

FILLER = ("The quarterly maintenance report notes routine wear on the surface "
          "layer near the northern junction and recommends inspection. ")
PROMPT_BODY = FILLER * 95   # ~1870 tokens, fits in both windows


def run(num_ctx: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": f"[{uuid.uuid4().hex}]\n{PROMPT_BODY}\nSummarize in one line."}],
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx,
                    "num_thread": NUM_THREAD, "num_predict": 32},
    }
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=1800)
    r.raise_for_status()
    return r.json()


def main():
    try:
        requests.get(HOST, timeout=5).raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"Ollama is not running at {HOST}. Start it with: ollama serve")

    print("Warming up...")
    run(4096)

    print(f"\n{'num_ctx':>8} {'prompt tok':>11} {'prefill t/s':>12} {'decode t/s':>11} {'load s':>7}")
    print("-" * 53)
    for ctx in CTX_VALUES:
        d = run(ctx)
        p = d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9)
        o = d["eval_count"] / (d["eval_duration"] / 1e9)
        print(f"{ctx:>8} {d['prompt_eval_count']:>11} {p:>12.1f} {o:>11.1f} "
              f"{d.get('load_duration', 0) / 1e9:>7.2f}")

    print("\nIf 16384 rows are clearly slower: num_ctx is set per request, sized to the prompt.")
    print("If they match: the earlier 19.6 tok/s was server state, and 16384 is safe as a default.")
    print("Note load s: a non-zero value on a row where num_ctx changed means Ollama")
    print("reloaded the model to resize the KV cache. That reload is itself a cost.")


if __name__ == "__main__":
    main()
