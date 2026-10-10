"""
chat.py — CLI chatbot with short-term AND long-term memory (Phase 1, memory added in Phase 7).

Run from the project root:

    python chat.py           # conversation + memory
    python chat.py --docs    # ... + answers from your PDFs (7.3)

Commands:  /memory   /remember TEXT   /forget N   /forget all   /reset   /system [text]   /stats   /quit

  short-term  the last few exchanges, word for word (config.MEMORY_WINDOW_TURNS)
  long-term   durable facts across sessions, saved to config.LONGTERM_MEMORY_PATH
              (7.2: verbatim beat atomic). Each message goes through: never-store
              check (code) -> classify -> store / correct / forget. This runs AFTER
              the reply is shown, and anything it does is printed.
"""

import re
import sys

import requests

from app import config, llm, retrieve, tokens
from app.longterm import LongTermMemory
from app.memory import Memory
from app.prompt import sanitize
from app.rewrite import concat, rewrite, union_search

# Docs mode answers from the documents OR from what it remembers about the user (rag.py's
# rules allow documents only, so "what's my name?" would be refused). rag.py is unchanged,
# so the Phase 4 golden-set results stay comparable.
DOC_RULES = ("Answer using the documents above and what you know about the user.\n"
             "- For questions about the user (name, work, city, projects), use <about_user>.\n"
             "- After every fact taken from a document, cite its number in square brackets, like [1].\n"
             "- Never use your own knowledge for facts about the documents' subjects.\n"
             "- If neither the documents nor what you know about the user contain the answer, reply "
             "with exactly: I don't know based on the provided documents.")


def search(previous: list, message: str) -> list:
    """7.3 experiment: concat 15/15 on follow-ups, but in the live chat it dragged a NEW topic
    ('hilly areas') back to the old one. 'union' hedges: best of raw and concat in turn."""
    mode = getattr(config, "FOLLOWUP_QUERY", "concat")
    if mode == "union":
        return union_search(previous, message, config.TOP_K)
    query = (rewrite(previous, message) if mode == "rewrite"
             else concat(previous, message) if mode == "concat" else message)
    return retrieve.search(query, k=config.TOP_K)


def docs_messages(system: str, turns: list, question: str, hits: list, about_user: str = "") -> list:
    """History stays plain text (old documents are NOT resent); only THIS turn carries documents.
    What the user told us goes in the FINAL turn, next to the documents (7.3: with memory only in
    the system message, 'where do I work?' was refused despite memory saying Hyderabad)."""
    msgs = [{"role": "system", "content": system}]
    for u, a in turns:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    docs = "\n\n".join(f"<doc>\n[{i}] {h.source}, page {h.page}\n{sanitize(h.text)}\n</doc>"
                        for i, h in enumerate(hits, 1)) or "(no documents)"
    about = f"<about_user>\n{about_user}\n</about_user>\n\n" if about_user else ""
    msgs.append({"role": "user",
                 "content": f"<documents>\n{docs}\n</documents>\n\n{about}Question: {question}\n\n{DOC_RULES}"})
    return msgs

HELP = "Commands: /memory  /remember TEXT  /forget N  /forget all  /reset  /system [text]  /stats  /quit"


def show_memory(lt: LongTermMemory):
    if not lt.items:
        print("(long-term memory is empty)")
        return
    print(f"long-term memory ({lt.design}, saved in {lt.path}):")
    for i in lt.items:
        print(f"  {i['id']}. [{i['date']}] {i['text']}")


_QUESTION_START = ("what", "which", "who", "whom", "whose", "how", "why", "when", "where", "can", "could",
                   "would", "will", "is", "are", "was", "were", "do", "does", "did", "should", "tell",
                   "explain", "give", "list", "show", "describe", "summarise", "summarize", "and")


def looks_like_question(message: str) -> bool:
    m = message.strip().lower()
    return m.endswith("?") or m.startswith(_QUESTION_START)


def report(r: dict, message: str = ""):
    """Tell the user whenever memory changes: nothing is stored or deleted silently."""
    if r["action"] == "blocked":
        print(f"   [memory: NOT saved: {r['why']}]")
    elif r["action"] in ("stored", "corrected"):
        print(f"   [memory: saved: {r['added']}]")
        if r.get("replaced"):
            print(f"   [memory: replaced: {r['replaced']}]")
    elif r["action"] == "ignored":
        # Not saving a STATEMENT is an outcome the user must see (7.2: a fact written with loose
        # grammar was silently not saved). Not saving a QUESTION is normal: no notice, or the
        # notice becomes noise that teaches you to ignore it (7.3).
        if message and not looks_like_question(message):
            print("   [memory: nothing saved from this message; use /remember TEXT to save it yourself]")
    elif r["action"] == "forgot":
        if r["removed"]:
            print("   [memory: removed:]")
            for t in r["removed"]:
                print(f"      - {t}")
            print("   [whole entries are removed; if one also held something you want kept, say it again]")
        else:
            print("   [memory: found nothing matching that to forget; see /memory]")


def main():
    if not llm.is_up():
        raise SystemExit(f"Ollama is not running at {config.OLLAMA_HOST}. Start it with: ollama serve")

    docs_mode = "--docs" in sys.argv
    system = config.SYSTEM_PROMPT
    short = Memory(strategy="window")
    lt = LongTermMemory()
    previous_user = []                                  # for follow-up search queries
    print(f"Chatting with {config.LLM_MODEL}{' + your documents' if docs_mode else ''}. "
          f"Long-term memory: {len(lt.items)} entr{'y' if len(lt.items) == 1 else 'ies'} loaded. {HELP}")

    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "bye"):     # 7.3: a bare "quit" went to the model,
            break                                       # and memory treated it as a forget request

        # Anything starting with "/" is a command and NEVER reaches the model.
        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            arg = arg.strip()
            if cmd == "/quit":
                break
            elif cmd == "/memory":
                show_memory(lt)
            elif cmd == "/remember" and arg:
                report(lt.remember(arg))
            elif cmd in ("/remember", "/forget") and not arg:
                print("usage: /remember I work on the DMP&PMS project at Mercedes-Benz\n"
                      "       /forget 3   or   /forget 2 5   or   /forget all")
            elif cmd == "/forget" and arg == "all":
                lt.forget_all()
                print("(long-term memory erased)")
            elif cmd == "/forget" and arg:
                try:
                    ids = [int(x) for x in arg.replace(",", " ").split()]
                except ValueError:
                    print("usage: /forget 3   or   /forget 2 5   or   /forget all")
                    continue
                removed = lt.forget_ids(ids)
                print("removed:\n" + "\n".join(f"  - {t}" for t in removed) if removed
                      else "(no entry with that number; see /memory)")
            elif cmd == "/reset":
                short, previous_user = Memory(strategy="window"), []
                print("(conversation cleared; long-term memory kept, use /forget all to erase it)")
            elif cmd == "/system" and arg:
                system, short = arg, Memory(strategy="window")
                print("(system message replaced, conversation cleared)")
            elif cmd == "/system":
                print(f"current system message: {system!r}")
            elif cmd == "/stats":
                msgs = short.messages(system + lt.as_prompt(), "")
                print(f"recent exchanges: {len(short.turns)}   long-term entries: {len(lt.items)}   "
                      f"context: {tokens.count_messages(msgs)} / budget {short.budget} tokens   "
                      f"memory model calls so far: {lt.model_calls}")
            else:
                print(f"unknown command. {HELP}")
            continue

        hits = []
        if docs_mode:
            hits = search(previous_user, user)
            about = "\n".join(f"- [{i['date']}] {i['text']}" for i in lt.items)
            msgs = docs_messages(system, short.turns, user, hits, about_user=about)
        else:
            msgs = short.messages(system + lt.as_prompt(), user)
        expected = tokens.count_messages(msgs)          # counted BEFORE sending
        print("bot > ", end="", flush=True)
        try:
            text, s = llm.stream_chat(msgs, on_token=lambda t: print(t, end="", flush=True))
        except (requests.RequestException, RuntimeError) as e:
            print(f"\n(error: {e})")
            continue
        short.add(user, text)
        previous_user.append(user)
        print(f"\n   [prompt {s['prompt_tokens']} tokens | out {s['output_tokens']} | {s['total_s']:.1f}s]")
        if hits:
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
            for i, h in enumerate(hits, 1):
                print(f"   [{i}] {h.source}, page {h.page}{'  (cited)' if i in cited else ''}")
        if s["prompt_tokens"] != expected:
            print(f"   ! token count mismatch: predicted {expected}, Ollama read {s['prompt_tokens']}")
        for w in s["warnings"]:
            print(f"   ! {w}")
        # Only YOUR message goes to memory, never retrieved document text: a poisoned PDF
        # cannot write itself into what the assistant "knows" about you (6.3).
        report(lt.process(user), user)                  # AFTER the reply: you never wait for it


if __name__ == "__main__":
    main()
