"""
chat.py — Phase 1 CLI chatbot. Run from the project root:

    python chat.py

Commands:  /reset   /system [text]   /stats   /quit
"""

import requests

from app import config, llm


def estimate_tokens(messages) -> int:
    """Rough count of the WHOLE conversation. Phase 2 replaces this with the real tokenizer.
    +4 per message covers the <|im_start|>role ... <|im_end|> markers from 1.3."""
    chars = sum(len(m["content"]) for m in messages)
    return int(chars / config.CHARS_PER_TOKEN_ESTIMATE) + 4 * len(messages)


def trim(messages) -> int:
    """Drop the oldest user+assistant pair until under budget. Never drops the system message."""
    dropped = 0
    while estimate_tokens(messages) > config.CHAT_HISTORY_BUDGET and len(messages) > 3:
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
                print(f"messages: {len(messages)}   estimated context: "
                      f"{estimate_tokens(messages)} / budget {config.CHAT_HISTORY_BUDGET} / num_ctx {config.NUM_CTX}")
            else:
                print("unknown command. Commands: /reset  /system [text]  /stats  /quit")
            continue

        messages.append({"role": "user", "content": user})
        dropped = trim(messages)
        if dropped:
            print(f"(dropped {dropped} oldest exchange(s) to stay under budget - "
                  f"expect this turn to be slow: the prompt cache is lost)")

        print("bot > ", end="", flush=True)
        try:
            text, s = llm.stream_chat(messages, on_token=lambda t: print(t, end="", flush=True))
        except (requests.RequestException, RuntimeError) as e:
            messages.pop()          # don't keep a question that never got an answer
            print(f"\n(error: {e})")
            continue

        messages.append({"role": "assistant", "content": text})
        print(f"\n   [prompt tokens {s['prompt_tokens']} (real) | out {s['output_tokens']} | "
              f"decode {s['decode_tps']:.1f} t/s | {s['total_s']:.1f}s | "
              f"est. context {estimate_tokens(messages)}/{config.CHAT_HISTORY_BUDGET}]")
        for w in s["warnings"]:
            print(f"   ! {w}")


if __name__ == "__main__":
    main()
