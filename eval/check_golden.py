"""
eval/check_golden.py — validate eval/golden.jsonl before trusting any score.

The most important check: every `evidence` phrase must really exist on the
pages you named. One typo there would make retrieval look broken when it isn't.

    python eval/check_golden.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.ingest import ingest_pdf  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
TARGET_TEST = 30          # aim for at least this many test questions
REFUSE_SHARE = (0.25, 0.35)
SUGGESTED_TAGS = ["numbers", "table", "multi-item", "informal", "list", "world-knowledge",
                  "near-miss", "premise", "cross-page"]


def norm(s: str) -> str:
    """Same normalisation the harness uses: lowercase, hyphens as spaces, one space."""
    return " ".join(s.lower().replace("-", " ").split())


def load():
    items, errors = [], []
    for n, line in enumerate(GOLDEN.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append((n, json.loads(line)))
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg}). Check quotes and commas.")
    return items, errors


_page_cache = {}


def page_texts(source: str) -> dict:
    if source not in _page_cache:
        path = ROOT / config.DOCS_DIR / source
        _page_cache[source] = {p.page: norm(p.text) for p in ingest_pdf(path)[0]} if path.exists() else None
    return _page_cache[source]


def check_item(n: int, q: dict) -> tuple:
    err, warn = [], []
    where = f"line {n} ({q.get('id', '?')})"
    for k in ("id", "split", "type", "question"):
        if not q.get(k):
            err.append(f"{where}: missing '{k}'")
    if q.get("split") not in ("dev", "test"):
        err.append(f"{where}: split must be 'dev' or 'test'")
    if q.get("type") not in ("answer", "refuse"):
        err.append(f"{where}: type must be 'answer' or 'refuse'")
        return err, warn

    if q["type"] == "answer":
        for k in ("answer", "source", "pages", "evidence", "must_include"):
            if not q.get(k):
                err.append(f"{where}: answerable question needs '{k}'")
        if err:
            return err, warn
        mi = q["must_include"]
        if not all(isinstance(alts, list) and alts for alts in mi):
            err.append(f"{where}: must_include must be a list of lists, e.g. [[\"15%\", \"fifteen\"]]")
        texts = page_texts(q["source"])
        if texts is None:
            err.append(f"{where}: file not found in {config.DOCS_DIR}: {q['source']}")
            return err, warn
        ev = norm(q["evidence"])
        if len(ev.split()) < 2:
            warn.append(f"{where}: evidence is one word; it may match unrelated chunks")
        on_named = [p for p in q["pages"] if ev in texts.get(p, "")]
        on_any = [p for p, t in texts.items() if ev in t]
        if not on_named:
            if on_any:
                err.append(f"{where}: evidence not on page(s) {q['pages']} but found on {on_any}")
            else:
                err.append(f"{where}: evidence NOT FOUND anywhere in {q['source']}: {q['evidence']!r}. "
                           "Copy it exactly from the PDF.")
        elif set(on_any) - set(q["pages"]):
            warn.append(f"{where}: evidence also appears on page(s) {sorted(set(on_any) - set(q['pages']))}; "
                        "add them to 'pages'")
        for alts in mi:
            if isinstance(alts, list) and not any(norm(a) in " ".join(texts.values()) for a in alts):
                warn.append(f"{where}: none of {alts} appears in the document. Fine if the answer "
                            "is a paraphrase, but check for typos.")
    else:
        if q.get("evidence"):
            warn.append(f"{where}: a 'refuse' question should not have evidence")
    if q.get("only_source") and page_texts(q["only_source"]) is None:
        err.append(f"{where}: only_source file not found: {q['only_source']}")
    return err, warn


def main():
    if not GOLDEN.exists():
        sys.exit(f"Not found: {GOLDEN}")
    items, errors = load()
    warnings = []
    ids = Counter(q.get("id") for _, q in items)
    errors += [f"duplicate id: {i}" for i, c in ids.items() if c > 1]
    for n, q in items:
        e, w = check_item(n, q)
        errors += e
        warnings += w

    qs = [q for _, q in items]
    test = [q for q in qs if q.get("split") == "test"]
    refuse = [q for q in test if q.get("type") == "refuse"]
    tags = Counter(t for q in test for t in q.get("tags", []))

    print(f"{len(qs)} questions: {len(qs) - len(test)} dev, {len(test)} test")
    print(f"test: {len(test) - len(refuse)} answerable, {len(refuse)} refuse")
    print(f"test tags: {dict(tags) if tags else '(none)'}\n")

    for e in errors:
        print(f"ERROR   {e}")
    for w in warnings:
        print(f"warning {w}")

    todo = []
    if len(test) < TARGET_TEST:
        todo.append(f"write {TARGET_TEST - len(test)} more test questions (target {TARGET_TEST}+)")
    share = len(refuse) / len(test) if test else 0
    if test and not REFUSE_SHARE[0] <= share <= REFUSE_SHARE[1]:
        todo.append(f"refuse questions are {share:.0%} of the test set; aim for 25-35%")
    missing = [t for t in SUGGESTED_TAGS if t not in tags]
    if missing:
        todo.append(f"no test questions tagged: {', '.join(missing)}")
    print("\n" + ("Still to do:\n  - " + "\n  - ".join(todo) if todo else "Coverage looks good."))
    print(f"\n{'FIX THE ERRORS ABOVE before running any evaluation.' if errors else 'No errors.'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
