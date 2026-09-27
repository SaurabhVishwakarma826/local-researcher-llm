"""
Phase 2 / Topic 2.1 — Tokenisation.

Run one experiment at a time:
    python 08_tokens.py 1   # look inside: text -> pieces -> IDs
    python 08_tokens.py 2   # how your domain terms get split
    python 08_tokens.py 3   # can we count tokens exactly, BEFORE calling Ollama?
    python 08_tokens.py 4   # is "characters / 3.5" a good estimate? by text type

First run downloads the Qwen tokenizer (~10 MB) from Hugging Face.
"""

import sys

import requests
from transformers import AutoTokenizer

HOST = "http://localhost:11434"
MODEL = "qwen2.5:3b"
TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"   # same tokenizer as the Ollama model

_tok = None


def tok():
    global _tok
    if _tok is None:
        print("Loading tokenizer...")
        _tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _tok


def pieces(ids):
    """Show each token as readable text. '·' marks a space, '�' means the token
    is only PART of a character (the rest of that character is in the next token)."""
    t = tok()
    return [t.decode([i]).replace(" ", "·").replace("\n", "\\n") for i in ids]


# ---------------------------------------------------------------------------
# 1 — Text -> pieces -> IDs
# ---------------------------------------------------------------------------
def exp1_inside():
    t = tok()
    print(f"Vocabulary size: {len(t):,} tokens\n")

    text = "The pothole on NH-44 was 30 cm deep."
    ids = t.encode(text)
    print(f"Text:   {text}")
    print(f"Pieces: {pieces(ids)}")
    print(f"IDs:    {ids}")
    print(f"{len(text)} characters -> {len(ids)} tokens\n")

    print("Remember 'Blue', 'blue' and ' Blue' from 1.2? Three different tokens:")
    for w in ("Blue", "blue", " Blue", " blue"):
        print(f"  {w!r:<9} -> IDs {t.encode(w)}")
    print("\nA leading space is part of the token. ' Blue' (mid-sentence) and 'Blue'")
    print("(start of text) are different entries in the vocabulary.")


# ---------------------------------------------------------------------------
# 2 — Your domain terms
# ---------------------------------------------------------------------------
TERMS = [
    "pothole", "potholes", "rutting", "IRI", " IRI", "International Roughness Index",
    "ArcFace", "LiDAR", "NHAI", "num_ctx", "rut_iri", "Qwen2.5",
    "MERIDIAN-7734", "12345.678", "2026-09-27",
    "सड़क में गड्ढा",          # Hindi: "pothole in the road"
]


def exp2_domain():
    t = tok()
    print(f"{'text':<32} {'tokens':>6}   pieces")
    print("-" * 90)
    for term in TERMS:
        ids = t.encode(term)
        print(f"{term!r:<32} {len(ids):>6}   {pieces(ids)}")
    print("\nThings to notice:")
    print("  - numbers: are digits grouped, or one token per digit?")
    print("  - 'IRI' vs ' IRI': the space changes the split")
    print("  - Hindi: count tokens vs characters, and look for '�' (a split character)")


# ---------------------------------------------------------------------------
# 3 — Exact count before sending, checked against Ollama
# ---------------------------------------------------------------------------
def ollama_count(messages) -> int:
    body = {"model": MODEL, "messages": messages, "stream": False, "keep_alive": "10m",
            "options": {"num_ctx": 16384, "num_thread": 12, "temperature": 0.0, "num_predict": 1}}
    r = requests.post(f"{HOST}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["prompt_eval_count"]


def exp3_exact_count():
    t = tok()
    cases = [
        ("1.3 exp 1 (Ollama said 29)", [
            {"role": "system", "content": "You are terse. Answer in one sentence."},
            {"role": "user", "content": "What is a pothole?"}]),
        ("no system message", [
            {"role": "user", "content": "Who are you, and who created you? One sentence."}]),
        ("multi-turn with code", [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "Write is_prime(n) in Python."},
            {"role": "assistant", "content": "def is_prime(n):\n    if n < 2:\n        return False\n"
                                             "    i = 2\n    while i * i <= n:\n        if n % i == 0:\n"
                                             "            return False\n        i += 1\n    return True"},
            {"role": "user", "content": "Now explain the loop condition."}]),
    ]
    print(f"{'case':<30} {'tokenizer':>10} {'Ollama':>8}   match")
    print("-" * 60)
    for label, msgs in cases:
        # Apply the SAME chat template locally (1.3), then count.
        text = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ours = len(t.encode(text, add_special_tokens=False))
        theirs = ollama_count(msgs)
        print(f"{label:<30} {ours:>10} {theirs:>8}   {'YES' if ours == theirs else 'NO  (diff ' + str(theirs - ours) + ')'}")
    print("\nIf all three match, we can count tokens exactly before sending anything,")
    print("and chat.py can stop guessing with characters / 3.5.")


# ---------------------------------------------------------------------------
# 4 — How good is "characters / 3.5"?
# ---------------------------------------------------------------------------
SAMPLES = {
    "English prose": ("Road condition surveys measure surface roughness, cracking and rutting "
                      "so that maintenance budgets can be allocated to the sections that need it most."),
    "Python code": ("def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                    "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                    "        i += 1\n    return True"),
    "JSON": '{"section_id": "NH44-KM-123", "iri_m_per_km": 3.42, "defects": ["pothole", "rutting"]}',
    "numbers table": "km, iri, rut_mm\n123.0, 3.42, 12\n123.1, 3.58, 14\n123.2, 4.01, 19\n123.3, 2.97, 9",
    "Hindi": "सड़क की स्थिति का सर्वेक्षण सतह की खुरदरापन, दरारें और गड्ढों को मापता है।",
}


def exp4_chars_per_token():
    t = tok()
    print(f"{'type':<15} {'chars':>6} {'tokens':>7} {'chars/token':>12} {'estimate (/3.5)':>16} {'error':>7}")
    print("-" * 70)
    for label, text in SAMPLES.items():
        n = len(t.encode(text))
        est = int(len(text) / 3.5)
        err = (est - n) / n * 100
        print(f"{label:<15} {len(text):>6} {n:>7} {len(text) / n:>12.2f} {est:>16} {err:>+6.0f}%")
    print("\nNegative error = the estimate says FEWER tokens than reality.")
    print("That is the dangerous direction: you think it fits in num_ctx, and it does not.")


EXPERIMENTS = {"1": exp1_inside, "2": exp2_domain, "3": exp3_exact_count, "4": exp4_chars_per_token}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1"
    if key not in EXPERIMENTS:
        sys.exit(f"Pick one of: {', '.join(EXPERIMENTS)}")
    if key == "3":
        try:
            requests.get(HOST, timeout=5).raise_for_status()
        except requests.exceptions.ConnectionError:
            raise SystemExit(f"Ollama is not running at {HOST}. Start it with: ollama serve")
    EXPERIMENTS[key]()
