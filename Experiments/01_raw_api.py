"""
Phase 1 / Topic 1 — What an LLM call actually is.

Run each experiment one at a time. Read the output before moving on.
    python 01_raw_api.py 1
    python 01_raw_api.py 2
    python 01_raw_api.py 3
    python 01_raw_api.py 4

No LangChain. No Ollama Python SDK. Just HTTP, because the whole point
is to see the actual payload that crosses the wire.
"""

import json
import sys

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"


def post(path: str, body: dict) -> dict:
    r = requests.post(f"{HOST}{path}", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# EXPERIMENT 1 — What does the server think it has?
# ---------------------------------------------------------------------------
def exp1_inspect_model():
    tags = requests.get(f"{HOST}/api/tags", timeout=30).json()
    print("Models installed:")
    for m in tags["models"]:
        print(f"  {m['name']:<28} {m['size'] / 1e9:.2f} GB  {m['details']['quantization_level']}")

    show = post("/api/show", {"model": MODEL})
    info = show.get("model_info", {})

    # The key is architecture-prefixed, e.g. "qwen2.context_length"
    ctx_key = next((k for k in info if k.endswith(".context_length")), None)
    emb_key = next((k for k in info if k.endswith(".embedding_length")), None)
    layer_key = next((k for k in info if k.endswith(".block_count")), None)
    head_key = next((k for k in info if k.endswith(".attention.head_count")), None)
    kv_head_key = next((k for k in info if k.endswith(".attention.head_count_kv")), None)

    print(f"\n{MODEL} architecture:")
    print(f"  max context length : {info.get(ctx_key)}")
    print(f"  hidden size (d)    : {info.get(emb_key)}")
    print(f"  transformer layers : {info.get(layer_key)}")
    print(f"  attention heads    : {info.get(head_key)}")
    print(f"  KV heads (GQA)     : {info.get(kv_head_key)}")
    print(f"  parameter count    : {show['details'].get('parameter_size')}")
    print(f"  quantization       : {show['details'].get('quantization_level')}")

    print("\n--- chat template the model was trained with ---")
    print(show.get("template", "(none)"))


# ---------------------------------------------------------------------------
# EXPERIMENT 2 — The full, unedited response object
# ---------------------------------------------------------------------------
def exp2_full_response():
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are terse. Answer in one sentence."},
            {"role": "user", "content": "What is ArcFace?"},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }

    print("--- REQUEST ---")
    print(json.dumps(body, indent=2))

    data = post("/api/chat", body)

    print("\n--- RESPONSE (everything the server sent back) ---")
    print(json.dumps(data, indent=2))

    ns = 1e9
    prompt_tok = data["prompt_eval_count"]
    out_tok = data["eval_count"]
    prefill_s = data.get("prompt_eval_duration", 0) / ns
    decode_s = data["eval_duration"] / ns

    print("\n--- WHAT THOSE NUMBERS MEAN ---")
    print(f"  prompt_eval_count = {prompt_tok} tokens in   (prefill: read your prompt)")
    print(f"  eval_count        = {out_tok} tokens out  (decode: write the answer)")
    print(f"  load_duration     = {data.get('load_duration', 0) / ns:.2f}s  (weights -> RAM, ~0 if warm)")
    print(f"  prefill speed     = {prompt_tok / prefill_s:.1f} tok/s" if prefill_s else "")
    print(f"  decode speed      = {out_tok / decode_s:.1f} tok/s")
    print(f"  total_duration    = {data['total_duration'] / ns:.2f}s")
    print("\n  Prefill is parallel (all prompt tokens at once).")
    print("  Decode is sequential (one token, feed it back, repeat).")
    print("  That asymmetry is why a long prompt is cheap and a long answer is slow.")


# ---------------------------------------------------------------------------
# EXPERIMENT 3 — The context window is a hard wall, and it fails SILENTLY
# ---------------------------------------------------------------------------
def exp3_context_truncation():
    secret = "The access code is MERIDIAN-7734."
    filler = "The quarterly maintenance report notes routine wear on the surface layer. " * 120
    prompt = f"{secret}\n\n{filler}\n\nWhat is the access code?"

    for num_ctx in (512, 2048, 8192, 16384):
        data = post(
            "/api/chat",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx},
            },
        )
        print(f"\nnum_ctx={num_ctx}")
        print(f"  tokens the server actually read: {data['prompt_eval_count']}")
        print(f"  answer: {data['message']['content'].strip()[:200]}")

    print("\nThe secret sits at the START of the prompt.")
    print("With a small window it is silently dropped and the model confabulates.")
    print("No error. No warning. This is the #1 way RAG pipelines break in week 3.")


# ---------------------------------------------------------------------------
# EXPERIMENT 4 — Streaming, and where the roles actually live
# ---------------------------------------------------------------------------
def exp4_streaming_and_roles():
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a pirate. Never break character."},
            {"role": "user", "content": "How do I center a div?"},
            {"role": "assistant", "content": "Arr, ye be wantin' flexbox, matey."},
            {"role": "user", "content": "What did I just ask about?"},
        ],
        "stream": True,
        "options": {"temperature": 0.0, "seed": 42},
    }

    print("Streaming (watch it arrive token by token):\n")
    with requests.post(f"{HOST}/api/chat", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        final = None
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                final = chunk
                break
            print(chunk["message"]["content"], end="", flush=True)

    print("\n\n--- final chunk (metrics arrive last) ---")
    print(json.dumps(final, indent=2))
    print("\nNote: the model has no memory. That 4-message list IS the memory.")
    print("Every turn you resend the whole conversation. Phase 7 is about")
    print("what to do when that list stops fitting in the context window.")


EXPERIMENTS = {
    "1": exp1_inspect_model,
    "2": exp2_full_response,
    "3": exp3_context_truncation,
    "4": exp4_streaming_and_roles,
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key not in EXPERIMENTS:
        sys.exit(f"Pick one of: {', '.join(EXPERIMENTS)}")
    EXPERIMENTS[key]()
