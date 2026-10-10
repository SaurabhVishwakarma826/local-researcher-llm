"""
app/rewrite.py — turn a follow-up into a standalone search query (Phase 7.3).

"and what must hotels do there?" means nothing to a retriever. The model rewrites it
using the user's EARLIER MESSAGES ONLY, not the assistant's answers: cheaper, and the
assistant's own guesses never leak into the query (7.2: the model treated its invented
"Dealer Management System" as something the user had said).

RULE: a rewritten query is for SEARCHING only. The answer must still respond to the
user's original words (7.3: "What can SWM 2026 charge tourists" retrieved the right chunk,
but the rules don't charge anyone; local authorities do).
"""

import json

from app import config, llm

REWRITE_SYSTEM = (
    "Rewrite the user's LATEST message as ONE standalone search query for their documents. "
    "Replace words like 'it', 'they', 'there', 'that', 'the second one' with what they refer to "
    "in the earlier messages. Keep the user's own key terms. If the latest message already "
    "makes sense on its own, return it unchanged. Reply only with the JSON.")
REWRITE_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


def rewrite(previous: list, message: str) -> str:
    """previous: the user's earlier messages in this conversation, oldest first."""
    if not previous:
        return message
    history = "\n".join(f"- {m}" for m in previous[-config.REWRITE_HISTORY_MESSAGES:])
    text, _ = llm.chat([{"role": "system", "content": REWRITE_SYSTEM},
                        {"role": "user", "content": f"Earlier messages from the user:\n{history}\n\n"
                                                    f"Latest message: {message}"}],
                       sampling=config.SAMPLING_DETERMINISTIC, fmt=REWRITE_SCHEMA, num_predict=80)
    try:
        q = json.loads(text).get("query", "").strip()
    except (json.JSONDecodeError, AttributeError):
        q = ""
    return q or message          # never search with nothing


def concat(previous: list, message: str, n: int = None) -> str:
    """The free alternative: the last n earlier questions + the follow-up, joined.
    7.3: tied the model rewrite (10/10) at zero cost, on conversations WITHOUT topic changes."""
    n = n or config.REWRITE_HISTORY_MESSAGES
    return " ".join(previous[-n:] + [message])


def union_search(previous: list, message: str, k: int) -> list:
    """Hedge between two interpretations (7.3): search with the latest message ALONE (raw wins
    when the user switches topic) and with the joined messages (concat wins on vague
    follow-ups), then take the best of each in turn until k chunks. Live test: concat alone
    dragged 'what does it say about hilly areas?' back to the old RDF topic."""
    from app import retrieve
    raw = retrieve.search(message, k=k)
    joined = retrieve.search(concat(previous, message), k=k) if previous else []
    out, seen = [], set()
    for pair in zip(raw + [None] * k, joined + [None] * k):
        for h in pair:
            if h is not None and h.id not in seen:
                out.append(h)
                seen.add(h.id)
    return out[:k]
