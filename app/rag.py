"""
app/rag.py — question -> retrieve -> assemble prompt -> answer with citations.

Prompt versions (3.4). Changed ONE thing at a time, measured with 13_rag.py:
  v1  original: <doc id="n"> tags, rules described in words.
      Result: answerable 3/4, refusals 4/4. Invented citations like [D3] with wrong numbers,
      and claimed News Aggregator used Django (multi-item chunk).
  v2  v1 + each document labelled VISIBLY as [n] inside its tags.
  v3  v2 + ONE worked example exchange before the real question, which SHOWS the
      citation format and "only the items the question asks about" (1.4: examples beat
      instructions). The example is about an unrelated parking notice so it can't leak facts.
  (A version that DESCRIBED stricter rules scored answerable 0/4: the model refused
   answerable questions and copied the format sample 'Some fact from a document [2].'
   verbatim. Removed.)

Other choices: rules AFTER the documents (no measured difference vs system message, 3.4),
temperature 0 (1.2), token count checked before sending, chunks dropped if over budget (2.1).

Usage from the project root:
    python -m app.rag "your question"
    python -m app.rag "your question" --source file.pdf --show-prompt --style v3
"""

import re
import sys
from dataclasses import dataclass

from app import config, llm, store, tokens
from app.prompt import sanitize

REFUSAL = "I don't know based on the provided documents."
STYLES = ("v1", "v2", "v3")

SYSTEM = ("You are a research assistant. You answer questions using only the documents "
          "you are given, and you cite them.")

RULES = ("Answer the question using ONLY the documents above.\n"
         "- After every fact, cite the document number in square brackets, like [1] or [2][3].\n"
         "- Never use your own knowledge.\n"
         f"- If the documents do not contain the answer, reply with exactly: {REFUSAL}")

# v3's worked example: made-up and unrelated to any real document, so nothing can leak.
# Doc 1 mentions residents AND visitors; the question is only about visitors. Doc 2 is irrelevant.
EXAMPLE_DOCS = [("city_notice.pdf", 2, "PARKING\nResidents may park on Elm Street overnight. "
                 "Visitors must use the Oak Street garage, which closes at 10 pm."),
                ("city_notice.pdf", 5, "LIBRARY\nThe central library opens at 9 am on weekdays.")]
EXAMPLE_QUESTION = "Where can visitors park?"
EXAMPLE_ANSWER = "Visitors must use the Oak Street garage [1], which closes at 10 pm [1]."
EXAMPLE_LEAK_WORDS = ("oak street", "elm street")

# The model invented formats like [D3], [B2], [M1] (3.4).
_MALFORMED = re.compile(r"\[[A-Za-z]+\s?\d+\]")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    cited: list             # Hits actually cited, in citation order
    bad_citations: list     # numbers cited that match no document
    malformed: list         # invented citation formats like [D3]
    hits: list              # Hits that were put in the prompt
    prompt_tokens: int
    seconds: float
    messages: list          # the exact messages sent, for inspection
    style: str


def _docs_block(items: list, style: str) -> str:
    """items: (source, page, text)."""
    if style == "v1":
        docs = [f'<doc id="{i}" source="{s}" page="{p}">\n{sanitize(t)}\n</doc>'
                for i, (s, p, t) in enumerate(items, 1)]
    else:
        docs = [f"<doc>\n[{i}] {s}, page {p}\n{sanitize(t)}\n</doc>"
                for i, (s, p, t) in enumerate(items, 1)]
    return "<documents>\n" + ("\n\n".join(docs) or "(no documents)") + "\n</documents>"


def _user_turn(items: list, question: str, style: str) -> dict:
    return {"role": "user", "content": f"{_docs_block(items, style)}\n\nQuestion: {question}\n\n{RULES}"}


def build_messages(question: str, hits: list, style: str = None) -> list:
    style = style or config.RAG_PROMPT_STYLE
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, not {style!r}")
    msgs = [{"role": "system", "content": SYSTEM}]
    if style == "v3":
        msgs += [_user_turn(EXAMPLE_DOCS, EXAMPLE_QUESTION, style),
                 {"role": "assistant", "content": EXAMPLE_ANSWER}]
    msgs.append(_user_turn([(h.source, h.page, h.text) for h in hits], question, style))
    return msgs


def _fit(question: str, hits: list, style: str) -> tuple:
    """Drop the lowest-ranked chunks until the prompt fits the budget."""
    hits = list(hits)
    while True:
        msgs = build_messages(question, hits, style)
        n = tokens.count_messages(msgs)
        if n <= config.RAG_MAX_PROMPT_TOKENS or not hits:
            return msgs, hits, n
        hits.pop()


def _is_refusal(text: str) -> bool:
    return "i don't know" in text.lower().replace("\u2019", "'")


def answer_from_hits(question: str, hits: list, style: str = None) -> Answer:
    style = style or config.RAG_PROMPT_STYLE
    msgs, hits, _ = _fit(question, hits, style)
    text, s = llm.chat(msgs, sampling=config.SAMPLING_DETERMINISTIC,
                       num_predict=config.RAG_ANSWER_MAX_TOKENS)
    text = text.strip()
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    cited = [hits[i - 1] for i in dict.fromkeys(nums) if 1 <= i <= len(hits)]
    bad = sorted({i for i in nums if not 1 <= i <= len(hits)})
    return Answer(question=question, text=text, refused=_is_refusal(text), cited=cited,
                  bad_citations=bad, malformed=_MALFORMED.findall(text), hits=hits,
                  prompt_tokens=s["prompt_tokens"], seconds=s["total_s"], messages=msgs, style=style)


def answer(question: str, k: int = None, source: str = None, style: str = None) -> Answer:
    return answer_from_hits(question, store.search(question, k=k, source=source), style)


def print_answer(a: Answer, show_prompt: bool = False):
    if show_prompt:
        print("=" * 30 + f" EXACT PROMPT SENT (style {a.style}) " + "=" * 20)
        for m in a.messages:
            print(f"--- [{m['role']}] ---\n{m['content']}\n")
        print("=" * 79)
    print(f"\nQ: {a.question}\n\nA: {a.text}\n")
    if a.refused:
        print("(refused: the model says the documents do not contain the answer)")
    for i, h in enumerate(a.hits, 1):
        mark = "cited" if h in a.cited else "not cited"
        print(f"  [{i}] {h.source}  page {h.page}   score {h.score:.3f}   ({mark})")
    if a.bad_citations:
        print(f"  ! cited documents that do not exist: {a.bad_citations}")
    if a.malformed:
        print(f"  ! invented citation format (not counted): {a.malformed}")
    print(f"\n  {a.prompt_tokens} prompt tokens, {a.seconds:.1f}s, style {a.style}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    src = args[args.index("--source") + 1] if "--source" in args else None
    sty = args[args.index("--style") + 1] if "--style" in args else None
    print_answer(answer(args[0], source=src, style=sty), show_prompt="--show-prompt" in args)
