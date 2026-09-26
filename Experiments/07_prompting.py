"""
Phase 1 / Topic 1.4 — Prompt engineering, measured.

Run one experiment at a time:
    python 07_prompting.py 1   # vague vs specific instructions
    python 07_prompting.py 2   # zero-shot vs few-shot, scored
    python 07_prompting.py 3   # asking for JSON vs forcing JSON
"""

import json
import sys

import requests

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"
BASE = {"num_ctx": 16384, "num_thread": 12}
DET = {"temperature": 0.0, "seed": 42}


def chat(messages: list, fmt=None, num_predict: int = 200) -> tuple:
    body = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "10m",
            "options": {**BASE, **DET, "num_predict": num_predict}}
    if fmt is not None:
        body["format"] = fmt
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"].strip(), d["prompt_eval_count"], d["eval_count"]


# ---------------------------------------------------------------------------
# 1 — Vague vs specific
# ---------------------------------------------------------------------------
def exp1_specific():
    prompts = [
        ("VAGUE", "Tell me about IRI."),
        ("SPECIFIC", "Explain IRI (International Roughness Index) as used in road condition surveys. "
                     "Audience: a junior engineer. Format: exactly 3 bullet points, each under 20 words. "
                     "No introduction and no conclusion."),
    ]
    for label, p in prompts:
        text, t_in, t_out = chat([{"role": "user", "content": p}], num_predict=300)
        print(f"===== {label}  ({t_in} tokens in, {t_out} tokens out) =====")
        print(text, "\n")
    print("Check the VAGUE answer: did it even pick the right 'IRI'?")
    print("Count the output tokens: at ~11 tok/s, every extra 100 tokens is ~9 seconds of waiting.")


# ---------------------------------------------------------------------------
# 2 — Zero-shot vs few-shot, with a score
# ---------------------------------------------------------------------------
LABELS = ["pothole", "crack", "rutting", "none"]

TESTS = [
    ("A bowl-shaped hole about 30 cm wide with loose gravel inside.", "pothole"),
    ("Long thin lines running along the lane edge.", "crack"),
    ("Two parallel grooves pressed into the wheel paths.", "rutting"),
    ("Fresh, smooth asphalt with clear lane markings.", "none"),
    ("A spider-web pattern of fine lines across the surface.", "crack"),
    ("The lane dips along where truck tyres usually run.", "rutting"),
    ("A deep hole has formed where the surface broke away.", "pothole"),
    ("Surface looks worn but has no breaks, holes or grooves.", "none"),
]

# Examples are DIFFERENT sentences from the tests, so we measure learning, not copying.
EXAMPLES = [
    ("Water collects in a sunken hole in the middle of the lane.", "pothole"),
    ("A single hairline line runs across the road.", "crack"),
    ("Deep ruts have formed in the tyre tracks of the slow lane.", "rutting"),
    ("The road surface is intact and even.", "none"),
]


def exp2_few_shot():
    system = (f"Classify the road description into exactly one label: {', '.join(LABELS)}. "
              "Reply with the label only, in lowercase, nothing else.")

    # Few-shot uses roles from 1.3: each example is a fake past user/assistant exchange.
    shots = []
    for desc, lab in EXAMPLES:
        shots += [{"role": "user", "content": desc}, {"role": "assistant", "content": lab}]

    def norm(s: str) -> str:
        return s.strip().strip(".").strip('"').lower()

    zero_ok = few_ok = 0
    print(f"{'expected':<9} {'zero-shot':<22} {'few-shot':<22} description")
    print("-" * 95)
    for desc, want in TESTS:
        z, _, _ = chat([{"role": "system", "content": system}, {"role": "user", "content": desc}], num_predict=12)
        f, _, _ = chat([{"role": "system", "content": system}, *shots, {"role": "user", "content": desc}], num_predict=12)
        z_hit, f_hit = norm(z) == want, norm(f) == want
        zero_ok += z_hit
        few_ok += f_hit
        zm = "OK " if z_hit else "XX "
        fm = "OK " if f_hit else "XX "
        print(f"{want:<9} {zm + z[:18]:<22} {fm + f[:18]:<22} {desc[:40]}")

    n = len(TESTS)
    print(f"\nzero-shot: {zero_ok}/{n}    few-shot: {few_ok}/{n}")
    print("This is a tiny version of Phase 4: fixed test cases, a correct answer for each,")
    print("and a score. You now compare prompts with a number instead of an opinion.")


# ---------------------------------------------------------------------------
# 3 — Asking for JSON vs forcing JSON with `format`
# ---------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {"defects": {"type": "array", "items": {"type": "string"}}},
    "required": ["defects"],
}

ASKS = [
    "List three common road defects.",
    "Name two defects you would see on an old highway.",
    "What damage does heavy truck traffic cause to roads? Give three.",
]


def check(text: str) -> str:
    """Checks the SHAPE, not just 'is it JSON' — the mistake from 1.3 experiment 3."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "FAIL: not JSON"
    if not isinstance(obj, dict) or "defects" not in obj:
        return "FAIL: wrong shape"
    if not isinstance(obj["defects"], list) or not all(isinstance(x, str) for x in obj["defects"]):
        return "FAIL: items are not strings"
    return "OK"


def exp3_structured():
    for ask in ASKS:
        asked, _, _ = chat([{"role": "user", "content":
                             ask + ' Reply with JSON only, in the form {"defects": ["...", "..."]}.'}])
        forced, _, _ = chat([{"role": "user", "content": ask + " Reply in JSON."}], fmt=SCHEMA)
        print(f"Q: {ask}")
        print(f"  asked  -> {check(asked):<28} {asked[:70]!r}")
        print(f"  forced -> {check(forced):<28} {forced[:70]!r}\n")
    print("'forced' uses Ollama's format option: the model is only ALLOWED to produce")
    print("tokens that fit the schema. It cannot add ```json fences, extra text, or the")
    print("wrong structure. Content can still be wrong — the shape cannot.")


EXPERIMENTS = {"1": exp1_specific, "2": exp2_few_shot, "3": exp3_structured}

if __name__ == "__main__":
    try:
        requests.get(HOST, timeout=5).raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit(f"Ollama is not running at {HOST}. Start it with: ollama serve")
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key not in EXPERIMENTS:
        sys.exit(f"Pick one of: {', '.join(EXPERIMENTS)}")
    EXPERIMENTS[key]()
