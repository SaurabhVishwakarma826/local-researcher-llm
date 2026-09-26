"""
app/llm.py — the ONLY file in the project that talks to Ollama.

Everything else (RAG, agents, API) calls these two functions, so settings
from config.py are applied everywhere and every call is measured.
"""

import json

import requests

from app import config


def is_up() -> bool:
    try:
        return requests.get(config.OLLAMA_HOST, timeout=5).ok
    except requests.exceptions.ConnectionError:
        return False


def _options(sampling, num_predict) -> dict:
    opts = {"num_ctx": config.NUM_CTX, "num_thread": config.NUM_THREAD}
    opts.update(sampling if sampling is not None else config.SAMPLING_DETERMINISTIC)
    if num_predict is not None:
        opts["num_predict"] = num_predict
    return opts


def _stats(d: dict) -> dict:
    ns = 1e9
    p_tok = d.get("prompt_eval_count", 0)
    p_s = d.get("prompt_eval_duration", 0) / ns
    o_tok = d.get("eval_count", 0)
    o_s = d.get("eval_duration", 0) / ns
    s = {
        "prompt_tokens": p_tok,          # FULL prompt size, including tokens served from cache (measured 1.4)
        "output_tokens": o_tok,
        "prefill_tps": p_tok / p_s if p_s else 0.0,
        "decode_tps": o_tok / o_s if o_s else 0.0,
        "load_s": d.get("load_duration", 0) / ns,
        "total_s": d.get("total_duration", 0) / ns,
        "done_reason": d.get("done_reason"),
    }
    w = []
    # A prefill rate is only meaningful on a long prompt (lesson from 01/02).
    if p_tok >= 300 and s["prefill_tps"] < config.PREFILL_TOK_S_FLOOR:
        w.append(f"slow prefill ({s['prefill_tps']:.0f} tok/s) - try restarting Ollama")
    if s["load_s"] > 1.0:
        w.append(f"model (re)loaded, took {s['load_s']:.1f}s")
    if s["done_reason"] == "length":
        w.append("answer was cut off by num_predict")
    s["warnings"] = w
    return s


def _body(messages, sampling, num_predict, stream, fmt=None) -> dict:
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": config.KEEP_ALIVE,
        "options": _options(sampling, num_predict),
    }
    if fmt is not None:
        body["format"] = fmt
    return body


def chat(messages, sampling=None, fmt=None, num_predict=None):
    """One complete answer. fmt can be a JSON schema to force the output shape.
    Returns (text, stats)."""
    r = requests.post(f"{config.OLLAMA_HOST}/api/chat",
                      json=_body(messages, sampling, num_predict, False, fmt), timeout=1800)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], _stats(d)


def stream_chat(messages, on_token, sampling=None, num_predict=None):
    """Streams the answer, calling on_token(piece) as each piece arrives.
    Returns (full_text, stats) at the end."""
    parts = []
    with requests.post(f"{config.OLLAMA_HOST}/api/chat",
                       json=_body(messages, sampling, num_predict, True),
                       stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                on_token(piece)
            if chunk.get("done"):
                return "".join(parts), _stats(chunk)
    raise RuntimeError("stream ended without a final 'done' message")
