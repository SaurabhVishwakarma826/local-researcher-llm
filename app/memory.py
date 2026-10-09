"""
app/memory.py — short-term (conversation) memory strategies (Phase 7.1).

The model has no memory: the conversation is the message list we resend every turn.
When it grows past the budget, something has to go. Three strategies:

  trim     keep all exchanges; drop the OLDEST until it fits (Phase 1's chat.py).
           Early facts ("I'm Ravi") are the first thing lost.
  window   keep only the last N exchanges.
  summary  fold exchanges older than the window into a running summary that the
           model rewrites (one extra model call per folded exchange).
           7.1 finding: the 3B model did NOT summarise. It copied the transcript until
           its output limit cut it off, so only the BEGINNING survived.
  facts    when an exchange leaves the window, the model EXTRACTS facts the user stated
           (JSON list, forced by Ollama's `format`), from the user's message only.
           Append-only: an existing fact is never rewritten, so it cannot drift.

Token counts use app.tokens.count_messages: exact here, because no tools are involved.
"""

import json
from dataclasses import dataclass, field

from app import config, llm, tokens

SUMMARY_SYSTEM = (
    "You maintain a running summary of a conversation between a user and an assistant. "
    "Keep EVERY fact about the user: name, role, place, numbers, preferences, decisions, and "
    "any unresolved question. Drop small talk and general explanations. "
    "Write at most {words} words, as short factual sentences.")


def update_summary(summary: str, user: str, assistant: str) -> str:
    text, _ = llm.chat(
        [{"role": "system", "content": SUMMARY_SYSTEM.format(words=config.MEMORY_SUMMARY_MAX_WORDS)},
         {"role": "user", "content": f"Current summary:\n{summary or '(empty)'}\n\n"
                                     f"New exchange to fold in:\nUser: {user}\nAssistant: {assistant}\n\n"
                                     "Write the updated summary."}],
        sampling=config.SAMPLING_DETERMINISTIC, num_predict=220)
    return text.strip()


FACTS_SYSTEM = (
    "Extract durable facts that the USER states about themselves or their situation: name, role, "
    "places, organisations, people, numbers, preferences, decisions. Only facts stated in the "
    "message, each as one short sentence. Questions and requests are NOT facts. "
    "If there are none, return an empty list.")
FACTS_SCHEMA = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
                "required": ["facts"]}


def extract_facts(user: str) -> list:
    text, _ = llm.chat([{"role": "system", "content": FACTS_SYSTEM},
                        {"role": "user", "content": f"Message from the user:\n{user}"}],
                       sampling=config.SAMPLING_DETERMINISTIC, fmt=FACTS_SCHEMA, num_predict=200)
    try:
        facts = json.loads(text).get("facts", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


@dataclass
class Memory:
    strategy: str = "trim"                  # "trim" | "window" | "summary" | "facts"
    budget: int = None                      # max tokens for system + history + new question
    window_turns: int = None
    turns: list = field(default_factory=list)          # [(user, assistant), ...] kept word for word
    summary: str = ""
    facts: list = field(default_factory=list)
    summary_calls: int = 0                  # extra model calls for summary OR fact extraction
    dropped: int = 0                        # exchanges forgotten entirely (trim / window)

    def __post_init__(self):
        self.budget = self.budget or config.MEMORY_BUDGET_TOKENS
        self.window_turns = self.window_turns or config.MEMORY_WINDOW_TURNS
        if self.strategy not in ("trim", "window", "summary", "facts"):
            raise ValueError(f"unknown memory strategy {self.strategy!r}")

    def add(self, user: str, assistant: str):
        self.turns.append((user, assistant))
        if self.strategy in ("window", "summary", "facts"):
            while len(self.turns) > self.window_turns:
                u, a = self.turns.pop(0)
                if self.strategy == "summary":
                    self.summary = update_summary(self.summary, u, a)
                    self.summary_calls += 1
                elif self.strategy == "facts":
                    seen = {f.lower() for f in self.facts}
                    self.facts += [f for f in extract_facts(u) if f.lower() not in seen]
                    self.facts = self.facts[-config.MEMORY_MAX_FACTS:]
                    self.summary_calls += 1
                else:
                    self.dropped += 1

    def messages(self, system: str, question: str) -> list:
        sys = system + (f"\n\nSummary of the earlier conversation:\n{self.summary}" if self.summary else "")
        if self.facts:
            sys += "\n\nFacts the user told you earlier:\n" + "\n".join(f"- {f}" for f in self.facts)

        def build(turns):
            msgs = [{"role": "system", "content": sys}]
            for u, a in turns:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            return msgs + [{"role": "user", "content": question}]

        msgs = build(self.turns)
        # Every strategy respects the budget in the end; trim relies on it entirely.
        while self.turns and tokens.count_messages(msgs) > self.budget:
            self.turns.pop(0)
            self.dropped += 1
            msgs = build(self.turns)
        return msgs
