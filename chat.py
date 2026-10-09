"""
chat.py — CLI chatbot with conversation memory (Phase 1, memory added in Phase 7.1).

Run from the project root:

    python chat.py

Commands:  /reset   /system [text]   /memory   /forget   /stats   /quit

Memory strategy comes from config.MEMORY_STRATEGY (7.1: "facts" was the only strategy
that remembered turn 2 AND turn 10 of a 20-turn conversation within budget).
"""

import requests

from app import config, llm, tokens
from app.memory import Memory

HELP = "Commands: /reset  /system [text]  /memory  /forget  /stats  /quit"


def main():
    if not llm.is_up():
        raise SystemExit(f"Ollama is not running at {config.OLLAMA_HOST}. Start it with: ollama serve")

    system = config.SYSTEM_PROMPT
    mem = Memory(strategy=config.MEMORY_STRATEGY)
    print(f"Chatting with {config.LLM_MODEL}, memory: {mem.strategy}. {HELP}")

    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        # Anything starting with "/" is a command and NEVER reaches the model.
        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            arg = arg.strip()
            if cmd == "/quit":
                break
            elif cmd == "/reset":
                mem = Memory(strategy=config.MEMORY_STRATEGY)
                print("(conversation and remembered facts cleared)")
            elif cmd == "/system" and arg:
                system = arg
                mem = Memory(strategy=config.MEMORY_STRATEGY)
                print("(system message replaced, conversation cleared)")
            elif cmd == "/system":
                print(f"current system message: {system!r}")
            elif cmd == "/memory":
                print(f"strategy {mem.strategy}: {len(mem.turns)} recent exchange(s) kept word for word")
                if mem.facts:
                    print("remembered facts:\n" + "\n".join(f"  - {f}" for f in mem.facts))
                if mem.summary:
                    print(f"summary:\n  {mem.summary}")
                if not (mem.facts or mem.summary):
                    print("(nothing remembered beyond the recent exchanges)")
            elif cmd == "/forget":
                mem.facts, mem.summary = [], ""
                print("(remembered facts and summary cleared; recent exchanges kept)")
            elif cmd == "/stats":
                msgs = mem.messages(system, "")
                print(f"recent exchanges: {len(mem.turns)}   facts: {len(mem.facts)}   "
                      f"context: {tokens.count_messages(msgs)} / budget {mem.budget} tokens   "
                      f"extra memory calls so far: {mem.summary_calls}")
            else:
                print(f"unknown command. {HELP}")
            continue

        msgs = mem.messages(system, user)
        expected = tokens.count_messages(msgs)          # counted BEFORE sending
        print("bot > ", end="", flush=True)
        try:
            text, s = llm.stream_chat(msgs, on_token=lambda t: print(t, end="", flush=True))
        except (requests.RequestException, RuntimeError) as e:
            print(f"\n(error: {e})")
            continue

        calls_before = mem.summary_calls
        mem.add(user, text)                             # may fold an old exchange into memory
        note = " | memory updated" if mem.summary_calls > calls_before else ""
        print(f"\n   [prompt {s['prompt_tokens']} tokens | out {s['output_tokens']} | "
              f"{s['total_s']:.1f}s | facts {len(mem.facts)}{note}]")
        if s["prompt_tokens"] != expected:
            print(f"   ! token count mismatch: predicted {expected}, Ollama read {s['prompt_tokens']}")
        for w in s["warnings"]:
            print(f"   ! {w}")


if __name__ == "__main__":
    main()
