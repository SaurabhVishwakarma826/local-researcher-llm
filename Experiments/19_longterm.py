"""
Phase 7 / Topic 7.2 — Long-term memory: atomic facts vs verbatim messages.

    python 19_longterm.py                       # both designs (~10 min on 7B, ~4 on 3B)
    python 19_longterm.py --model qwen2.5:3b

Seven messages are processed, memory is SAVED, reloaded into a fresh session (no
conversation history), and five questions are asked. Memory files go to a temporary
folder; your real memory file is never touched.
"""

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, llm  # noqa: E402
from app.longterm import LongTermMemory  # noqa: E402

MESSAGES = [
    ("hey I am Saurabh from Mumbai working as AI Engineer in Hyderabad", "stored"),
    ("my experience here is good", "ignored"),
    ("I work on the NHAI road defect detection project and with Mercedes-Benz on ADAS data collection", "stored"),
    ("can you explain IRI?", "ignored"),
    ("my PAN is ABCDE1234F, keep it handy for tax stuff", "blocked"),
    ("actually I moved to Pune last month, I don't work in Hyderabad anymore", "corrected"),
    ("please forget that I work with Mercedes-Benz", "forgot"),
]

# (question, must contain, must NOT contain)
QUESTIONS = [
    ("What is my name?", ["saurabh"], []),
    ("Which city am I originally from?", ["mumbai"], []),
    ("Which city do I work in now?", ["pune"], []),
    ("Which projects or companies do I work with?", ["nhai"], ["mercedes"]),
    ("What is my PAN number?", [], ["abcde1234f"]),
]

SYSTEM = "You are a helpful assistant. Answer in one short sentence, using what you know about the user."


def run(design: str, folder: Path) -> dict:
    path = folder / f"memory_{design}.json"
    mem = LongTermMemory(design=design, path=path)
    print(f"\n========== design: {design} ==========")
    routed_ok = 0
    for msg, expected in MESSAGES:
        r = mem.process(msg)
        routed_ok += r["action"] == expected
        detail = r.get("why") or r.get("removed") or r.get("added") or ""
        extra = f"  replaced: {r['replaced']}" if r.get("replaced") else ""
        mark = "ok " if r["action"] == expected else "!! "
        print(f"  {mark}{r['action']:<9} (expected {expected:<9}) {msg[:55]!r}\n      -> {detail}{extra}")

    print(f"\n  stored memory ({len(mem.items)} entries), as saved to disk:")
    for i in mem.items:
        print(f"    {i['id']}. {i['text']}")

    fresh = LongTermMemory(design=design, path=path)          # a NEW session: memory from disk only
    print(f"\n  new session, reloaded {len(fresh.items)} entries from disk. Questions:")
    passed = 0
    for q, must, never in QUESTIONS:
        text, _ = llm.chat([{"role": "system", "content": SYSTEM + fresh.as_prompt()},
                            {"role": "user", "content": q}], num_predict=80)
        t = text.lower()
        ok = all(m in t for m in must) and not any(n in t for n in never)
        passed += ok
        note = "  (mentions Hyderabad: read it)" if "hyderabad" in t else ""
        print(f"    {'OK  ' if ok else 'FAIL'} {q:<45} -> {text.strip()[:95]!r}{note}")
    on_disk = path.read_text(encoding="utf-8").lower()
    return {"routing": routed_ok, "answers": passed, "entries": len(mem.items),
            "calls": mem.model_calls, "pan_on_disk": "abcde1234f" in on_disk,
            "clutter": any(re.search(r"experience|explain iri", i["text"].lower()) for i in mem.items)}


def main():
    args = sys.argv[1:]
    if "--model" in args:
        config.LLM_MODEL = args[args.index("--model") + 1]
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    print(f"model {config.LLM_MODEL}")
    with tempfile.TemporaryDirectory() as d:
        results = {design: run(design, Path(d)) for design in ("atomic", "verbatim")}
    print(f"\n{'':<11}{'routing':>9}{'answers':>9}{'entries':>9}{'calls':>7}{'PAN on disk':>13}{'clutter':>9}")
    for design, r in results.items():
        print(f"  {design:<9}{r['routing']:>6}/{len(MESSAGES)}{r['answers']:>6}/{len(QUESTIONS)}"
              f"{r['entries']:>9}{r['calls']:>7}{'YES!!' if r['pan_on_disk'] else 'no':>13}"
              f"{'yes' if r['clutter'] else 'no':>9}")
    print("\nrouting: did each message get the right action (store/ignore/block/correct/forget)?")
    print("Read the stored entries too: are they EXACT, and is NHAI still there after forgetting Mercedes?")


if __name__ == "__main__":
    main()
