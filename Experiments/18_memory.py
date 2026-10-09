"""
Phase 7 / Topic 7.1 — Short-term memory.

    python 18_memory.py 1                          # what does Ollama drop when a chat overflows num_ctx?
    python 18_memory.py 2                          # trim vs window vs summary on a 20-turn conversation
    python 18_memory.py 2 --model qwen2.5:3b       # same, faster (memory mechanics don't need 7B)
    python 18_memory.py 2 --only summary,facts     # run only some strategies

Needs Ollama running. Experiment 2 makes ~80 model calls: ~10 min with 3B, ~25 min with 7B.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, llm, tokens  # noqa: E402
from app.memory import Memory  # noqa: E402

FRENCH = {"je", "vous", "votre", "est", "le", "la", "les", "et", "nom", "une", "des", "pour", "avec", "c'est"}
ENGLISH = {"the", "is", "your", "and", "name", "you", "for", "with", "are", "it", "of"}


def language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    fr, en = sum(w in FRENCH for w in words), sum(w in ENGLISH for w in words)
    return "French" if fr > en else "English" if en > fr else "unclear"


# ---------------------------------------------------------------------------
# 1 — What gets dropped when a chat overflows?
# ---------------------------------------------------------------------------
def exp1_overflow():
    system = "You must ALWAYS reply in French, whatever language the user writes in."
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "My name is Ravi. Please remember it."},
            {"role": "assistant", "content": "Entendu, Ravi. Je m'en souviendrai."}]
    topics = ["dust control", "conveyor belts", "kiln maintenance", "fuel storage", "safety meetings",
              "machine downtime", "worker training", "quality checks", "spare parts", "shift handover",
              "noise control", "waste heat", "water use", "fire drills"]
    for t in topics:
        msgs += [{"role": "user", "content": f"Tell me briefly about {t} in a cement plant."},
                 {"role": "assistant", "content": f"Voici quelques points sur {t} : il est important de "
                  "planifier, de mesurer régulièrement et de former les équipes pour des résultats durables."}]
    msgs.append({"role": "user", "content": "What is my name? Answer in one sentence."})
    total = tokens.count_messages(msgs)

    for ctx in (512, config.NUM_CTX):
        config.NUM_CTX = ctx
        text, s = llm.chat(msgs, num_predict=60)
        print(f"\nnum_ctx={ctx:<6} conversation={total} tokens   Ollama read {s['prompt_tokens']} tokens")
        print(f"  reply: {text.strip()[:150]!r}")
        print(f"  -> language: {language(text)}   remembers 'Ravi': {'yes' if 'ravi' in text.lower() else 'NO'}")
    print("\nLanguage tells us whether the SYSTEM message survived; 'Ravi' tells us whether the")
    print("FIRST exchange survived. Changing num_ctx reloads the model, hence the pause.")


# ---------------------------------------------------------------------------
# 2 — Three strategies, one 20-turn conversation, a deliberately small budget
# ---------------------------------------------------------------------------
TURNS = [
    "Hi, I'm Ravi. I manage a cement plant near Pune.",
    "Our kiln burns about 2,400 tonnes of fuel a year.",
    "In one sentence, what is a cement kiln?", "Give one tip for reducing dust in a plant.",
    "What does PPE stand for?", "Name one benefit of preventive maintenance.",
    "In one sentence, what is a conveyor belt used for?", "What is a good way to start a safety meeting?",
    "Name one common cause of machine downtime.", "By the way, our plant manager is Meena Joshi.",
    "What does KPI stand for?", "Give one tip for writing a clear email.",
    "Name one renewable energy source.", "In one sentence, what is a supply chain?",
    "What is one benefit of recycling paper?", "Give one tip for time management.",
    "In one sentence, what is an audit?", "Who is our plant manager?",
    "What's my name, and where is my plant?",
    "How many tonnes of fuel does our kiln burn a year?",
]
SYSTEM = "You are a helpful assistant. Answer in at most two short sentences."
BUDGET = 450          # deliberately small: forces every strategy to discard something


def run_strategy(strategy: str) -> dict:
    mem = Memory(strategy=strategy, budget=BUDGET)
    secs, prompt_toks, replies = [], [], []
    t_all = time.time()
    for i, q in enumerate(TURNS, 1):
        msgs = mem.messages(SYSTEM, q)
        text, s = llm.chat(msgs, num_predict=80)
        text = text.strip()
        mem.add(q, text)
        secs.append(s["total_s"])
        prompt_toks.append(s["prompt_tokens"])
        replies.append(text)
        print(f"  {strategy:<7} turn {i:>2}  {s['prompt_tokens']:>4} tok  {s['total_s']:5.1f}s"
              + (f"   -> {text[:90]!r}" if i >= 18 else ""), flush=True)
    a18, a19, a20 = replies[17].lower(), replies[18].lower(), replies[19]
    return {
        "name": "ravi" in a19, "place": "pune" in a19, "manager": "meena" in a18,
        "fuel": bool(re.search(r"2[,\s]?400", a20)),
        "summary_calls": mem.summary_calls, "dropped": mem.dropped, "facts": mem.facts,
        "max_prompt": max(prompt_toks), "conv_seconds": sum(secs),
        "wall_seconds": time.time() - t_all, "summary": mem.summary,
    }


def exp2_strategies():
    print(f"20 turns, history budget {BUDGET} tokens, window {config.MEMORY_WINDOW_TURNS} exchanges, "
          f"model {config.LLM_MODEL}\n")
    args = sys.argv[1:]
    strategies = (args[args.index("--only") + 1].split(",") if "--only" in args
                  else ["trim", "window", "summary", "facts"])
    results = {}
    for st in strategies:
        results[st] = run_strategy(st)
        print()
    print(f"{'':<10}{'name':>6}{'Pune':>6}{'2,400':>7}{'Meena':>7}{'forgot':>8}{'extra calls':>13}"
          f"{'max prompt':>12}{'total time':>12}")
    for st, r in results.items():
        yn = lambda b: "yes" if b else "NO"
        print(f"  {st:<8}{yn(r['name']):>6}{yn(r['place']):>6}{yn(r['fuel']):>7}{yn(r['manager']):>7}"
              f"{r['dropped']:>8}{r['summary_calls']:>13}{r['max_prompt']:>12}{r['wall_seconds']:>11.0f}s")
    if "summary" in results:
        print(f"\nFinal running summary:\n{results['summary']['summary']}")
    if "facts" in results:
        print("\nFinal fact memory:\n" + "\n".join(f"- {f}" for f in results["facts"]["facts"]))
    print("\nTurn 10 states a fact in the MIDDLE (Meena). A strategy that only keeps the")
    print("beginning now fails visibly. Read the summary/facts: are the facts EXACT?")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--model" in args:
        config.LLM_MODEL = args[args.index("--model") + 1]
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    key = args[0] if args else ""
    if key == "1":
        exp1_overflow()
    elif key == "2":
        exp2_strategies()
    else:
        sys.exit(__doc__)
