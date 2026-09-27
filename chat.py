"""
chat.py — Phase 1 CLI chatbot, with exact token counting from Phase 2.

Run from the project root:

    python chat.py

Commands:  /reset   /system [text]   /stats   /quit
"""

import requests

from app import config, llm, tokens


def trim(messages) -> int:
    """Drop the oldest user+assistant pair until under budget. Never drops the system message."""
    dropped = 0
    while tokens.count_messages(messages) > config.CHAT_HISTORY_BUDGET and len(messages) > 3:
        del messages[1:3]
        dropped += 1
    return dropped


def main():
    if not llm.is_up():
        raise SystemExit(f"Ollama is not running at {config.OLLAMA_HOST}. Start it with: ollama serve")

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    print(f"Chatting with {config.LLM_MODEL}. Commands: /reset  /system [text]  /stats  /quit")

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
                messages = messages[:1]
                print("(history cleared, system message kept)")
            elif cmd == "/system" and arg:
                messages = [{"role": "system", "content": arg}]
                print("(system message replaced, history cleared)")
            elif cmd == "/system":
                print(f"current system message: {messages[0]['content']!r}")
            elif cmd == "/stats":
                print(f"messages: {len(messages)}   context: {tokens.count_messages(messages)} tokens "
                      f"/ budget {config.CHAT_HISTORY_BUDGET} / num_ctx {config.NUM_CTX}")
            else:
                print("unknown command. Commands: /reset  /system [text]  /stats  /quit")
            continue

        messages.append({"role": "user", "content": user})
        dropped = trim(messages)
        if dropped:
            print(f"(dropped {dropped} oldest exchange(s) to stay under budget - "
                  f"expect this turn to be slow: the prompt cache is lost)")

        expected = tokens.count_messages(messages)   # counted BEFORE sending

        print("bot > ", end="", flush=True)
        try:
            text, s = llm.stream_chat(messages, on_token=lambda t: print(t, end="", flush=True))
        except (requests.RequestException, RuntimeError) as e:
            messages.pop()          # don't keep a question that never got an answer
            print(f"\n(error: {e})")
            continue

        messages.append({"role": "assistant", "content": text})
        print(f"\n   [prompt {s['prompt_tokens']} tokens | out {s['output_tokens']} | "
              f"decode {s['decode_tps']:.1f} t/s | {s['total_s']:.1f}s | "
              f"context now {tokens.count_messages(messages)}/{config.CHAT_HISTORY_BUDGET}]")

        # Continuous self-check: our count before sending vs what Ollama actually read.
        if s["prompt_tokens"] != expected:
            print(f"   ! token count mismatch: we predicted {expected}, Ollama read {s['prompt_tokens']}")
        for w in s["warnings"]:
            print(f"   ! {w}")


if __name__ == "__main__":
    main()
