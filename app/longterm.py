"""
app/longterm.py — long-term memory across sessions (Phase 7.2).

Two designs, compared in 19_longterm.py:
  atomic    the model extracts short facts; a correction REPLACES the outdated fact;
            forgetting deletes single facts. Risk: rewriting loses detail (7.1: "Role:
            Manager" lost "cement plant"; one session lost "Hyderabad").
  verbatim  the user's exact message is stored. A correction is ADDED and the recency
            rule (later entries win) resolves it. Forgetting deletes whole messages,
            which may also hold facts the user wanted kept.

7.2 result (7B, one scripted scenario): verbatim 4/5, atomic 3/5. Atomic's failures were
unpredictable rewrites: it merged "from Mumbai" and "works in Hyderabad" into
"places: Mumbai, Hyderabad", then a correction about moving to Pune deleted that AND an
unrelated NHAI fact. Verbatim's failure was predictable: forgetting Mercedes deleted the
whole message, NHAI included. Predictable failures can be SHOWN to the user and fixed.

Every message goes through small jobs:
  1. never-store check  CODE (regex), not the model: IDs, account/card numbers, passwords
  2. classify           model, JSON yes/no: durable fact? correction? forget request?
  3. store / correct / forget
"""

import json
import re
from datetime import date
from pathlib import Path

from app import config, llm
from app.memory import extract_facts

# 1 — never store. Deliberately strict: a false block costs little, a stored secret costs a lot.
# Order matters: the FIRST match names the reason, so the most specific patterns come first
# (a 16-digit card number also contains a 12-digit "Aadhaar-like" run).
NEVER_STORE = [
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "a PAN number"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "a card-like number"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "an Aadhaar-like 12-digit number"),
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "a phone number"),
    (re.compile(r"\b\d{9,18}\b"), "a long account-like number"),
    (re.compile(r"(?i)\b(password|passcode|otp|pin|cvv)\b"), "a password, PIN or code"),
]


def never_store_reason(text: str):
    for pattern, what in NEVER_STORE:
        if pattern.search(text):
            return what
    return None


def _json(system: str, user: str, schema: dict) -> dict:
    text, _ = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                       sampling=config.SAMPLING_DETERMINISTIC, fmt=schema, num_predict=200)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# 2 — classify
CLASSIFY_SYSTEM = (
    "Classify one message from a user, for a personal memory.\n"
    "durable_fact: the user states a LASTING fact about themselves: name, home town, city, job, "
    "employer, projects, skills, long-term preferences. NOT durable: questions, requests, greetings, "
    "how they feel today, opinions about the moment.\n"
    "correction: the message changes or contradicts something about them that may have been said "
    "before (e.g. 'actually', 'not anymore', 'I moved', 'I changed').\n"
    "forget_request: the user asks you to forget or stop remembering something.")
CLASSIFY_SCHEMA = {"type": "object", "properties": {
    "durable_fact": {"type": "boolean"}, "correction": {"type": "boolean"},
    "forget_request": {"type": "boolean"}},
    "required": ["durable_fact", "correction", "forget_request"]}

SELECT_SYSTEM = (
    "Below are stored memories about the user, numbered. The user's new message {what}. "
    "Return the numbers of ONLY the memories it {verb}. Return an empty list if none.")
SELECT_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
                 "required": ["ids"]}


class LongTermMemory:
    def __init__(self, design: str = None, path=None):
        self.design = design or config.LONGTERM_DESIGN
        if self.design not in ("atomic", "verbatim"):
            raise ValueError(f"unknown design {self.design!r}")
        self.path = Path(path or config.LONGTERM_MEMORY_PATH)
        self.items, self.next_id, self.model_calls = [], 1, 0
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items, self.next_id = data["items"], data["next_id"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"design": self.design, "items": self.items,
                                         "next_id": self.next_id}, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    def _add(self, text: str):
        if text.lower() not in {i["text"].lower() for i in self.items}:
            self.items.append({"id": self.next_id, "text": text, "date": date.today().isoformat()})
            self.next_id += 1

    def _select(self, message: str, what: str, verb: str) -> list:
        if not self.items:
            return []
        listing = "\n".join(f"{i['id']}. {i['text']}" for i in self.items)
        out = _json(SELECT_SYSTEM.format(what=what, verb=verb),
                    f"Stored memories:\n{listing}\n\nNew message: {message}", SELECT_SCHEMA)
        self.model_calls += 1
        valid = {i["id"] for i in self.items}
        return [x for x in out.get("ids", []) if x in valid]          # code checks the model's ids

    def process(self, message: str) -> dict:
        """Handle one user message. Returns what happened, so it can be SHOWN to the user."""
        reason = never_store_reason(message)
        if reason:
            return {"action": "blocked", "why": f"contains {reason}; never stored"}

        c = _json(CLASSIFY_SYSTEM, message, CLASSIFY_SCHEMA)
        self.model_calls += 1

        if c.get("forget_request"):
            ids = self._select(message, "asks to forget something", "asks to forget")
            removed = [i["text"] for i in self.items if i["id"] in ids]
            self.items = [i for i in self.items if i["id"] not in ids]
            self.save()
            return {"action": "forgot", "removed": removed}

        if not (c.get("durable_fact") or c.get("correction")):
            return {"action": "ignored"}

        replaced = []
        if c.get("correction") and self.design == "atomic":
            ids = self._select(message, "corrects or replaces something said before",
                               "makes outdated")
            replaced = [i["text"] for i in self.items if i["id"] in ids]
            self.items = [i for i in self.items if i["id"] not in ids]

        if self.design == "atomic":
            new = extract_facts(message)
            self.model_calls += 1
            for f in new:
                if not never_store_reason(f):
                    self._add(f)
        else:
            new = [message]
            self._add(message)
        self.save()
        return {"action": "corrected" if c.get("correction") else "stored", "added": new,
                "replaced": replaced}

    def remember(self, text: str) -> dict:
        """Explicit save (/remember): the user overrules the classifier, which missed a fact
        written with loose grammar ("I working for mercdeez benz..."). The never-store
        check still applies. Always stored verbatim: the user's own words."""
        reason = never_store_reason(text)
        if reason:
            return {"action": "blocked", "why": f"contains {reason}; never stored"}
        self._add(text.strip())
        self.save()
        return {"action": "stored", "added": [text.strip()], "replaced": []}

    def forget_ids(self, ids: list) -> list:
        """Delete exact entries by number: code only, no model (/forget 3)."""
        removed = [i["text"] for i in self.items if i["id"] in ids]
        self.items = [i for i in self.items if i["id"] not in ids]
        self.save()
        return removed

    def forget_all(self):
        self.items = []
        self.save()

    def as_prompt(self) -> str:
        if not self.items:
            return ""
        lines = "\n".join(f"- [{i['date']}] {i['text']}" for i in self.items)
        return ("\n\nWhat you know about the user from earlier conversations (oldest first; if two "
                "entries conflict, the LATER one is true):\n" + lines)
