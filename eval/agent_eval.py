"""
eval/agent_eval.py — multi-step agent test set (Phase 6.2).

    python eval/agent_eval.py                      # plain loop AND plan-then-execute, compared
    python eval/agent_eval.py --mode loop --label x
    python eval/agent_eval.py --mode plan --label x
    python eval/agent_eval.py --mode workflow      # the LangGraph workflow (app/workflow.py)
    python eval/agent_eval.py --mode loop,workflow # any two modes, compared

Each case is checked on: the right answer, the required tools used, the right ORDER
when one step needs another's result, and flagged (unverified) numbers.
Date answers are computed on the day you run it.
"""

import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import agent, llm  # noqa: E402

RESULTS = ROOT / "eval" / "results"
TODAY = date.today()
IN_FORCE = date(2026, 4, 1)          # SWM 2026 in force from 1 April 2026 (paper, page 1)

# expect: list of alternatives; the answer must contain at least one (numbers compared as numbers)
# also:   extra required phrases, each a list of alternatives (e.g. the CONCLUSION of a comparison)
# never:  phrases that make the answer wrong
# order:  tools that must appear in this order (first use of each)
# (also/never added after a05 said "15% ... meets the target of below 10-12%" and still passed)
CASES = [
    dict(id="a01", kind="search+calc",
         q="What RDF substitution percentage do the SWM 2026 rules finally reach? If a cement plant "
           "burns 2,400 tonnes of fuel a year, how many tonnes would be RDF at that level?",
         expect=[360], tools=["search_documents", "calculator"], order=["search_documents", "calculator"]),
    dict(id="a02", kind="search+calc",
         q="At the STARTING RDF substitution percentage in the SWM 2026 rules, how many tonnes of RDF "
           "would a plant burning 2,400 tonnes of fuel a year use?",
         expect=[120], tools=["search_documents", "calculator"], order=["search_documents", "calculator"]),
    dict(id="a03", kind="search+calc",
         q="A hostel generates 40 kg of waste a day. How many more kg per day could it generate before "
           "it counts as a bulk waste generator under SWM 2026?",
         expect=[60], tools=["search_documents", "calculator"], order=["search_documents", "calculator"]),
    dict(id="a04", kind="search+calc",
         q="An MRF designed for 200 TPD actually processes 130 TPD. What is its utilisation, and is it "
           "below the level the paper says indicates upstream problems?",
         expect=[65, 0.65], also=[["below", "under", "less than"]], tools=["search_documents"], order=[]),
    dict(id="a05", kind="search+calc",
         q="A 500 kg sample of dry recyclables contains 75 kg of impurities. What is its contamination "
           "rate, and does it meet the paper's best-practice target?",
         expect=[15, 0.15],
         also=[["not meet", "doesn't meet", "does not meet", "exceeds", "above", "fails", "higher than"]],
         never=["meets the best", "meets the target", "within the target"],
         tools=["search_documents"], order=[]),
    dict(id="a06", kind="search+calc",
         q="What was Saurabh's CGPA in engineering? Express it as a percentage of 10.",
         expect=[85.3], tools=["search_documents", "calculator"], order=["search_documents", "calculator"]),
    dict(id="a07", kind="dates",
         q="How many days are there between today and 1 April 2027?",
         expect=[(date(2027, 4, 1) - TODAY).days], tools=["current_date", "days_between"],
         order=["current_date", "days_between"]),
    dict(id="a08", kind="search+dates",
         q="As of today, for how many days have the SWM 2026 rules been in force?",
         expect=[(TODAY - IN_FORCE).days], tools=["search_documents", "current_date", "days_between"],
         order=["search_documents", "days_between"]),
    dict(id="a09", kind="one tool",
         q="What is 23.7% of 9,845?", expect=[2333.265], tools=["calculator"], order=[]),
    dict(id="a10", kind="one tool",
         q="According to the SWM 2026 paper, what may MRFs also act as, besides sorting facilities?",
         expect=["deposition point"], tools=["search_documents"], order=[]),
    dict(id="a11", kind="no tool",
         q="Hi! What can you help me with?", expect=[], tools=[], order=[], no_tools=True),
]

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _has_value(answer: str, want) -> bool:
    if isinstance(want, str):
        return want.lower() in answer.lower().replace("-", " ")
    nums = [float(m.replace(",", "")) for m in _NUM.findall(answer)]
    return any(abs(n - want) <= max(0.01, 1e-4 * abs(want)) for n in nums)


def check(case: dict, r: agent.AgentResult) -> dict:
    # A FAILED call does not count as using the tool (a01: the calculator errored, the answer
    # step computed 360 itself, and the case still passed).
    used = [s.name for s in r.steps if s.kind == "tool" and not s.result.startswith(("SKIPPED", "ERROR"))]
    text = r.answer.lower().replace("-", " ")
    answer_ok = (not case["expect"] or any(_has_value(r.answer, w) for w in case["expect"]))
    answer_ok = answer_ok and all(any(a.replace("-", " ") in text for a in alts) for alts in case.get("also", []))
    answer_ok = answer_ok and not any(w.replace("-", " ") in text for w in case.get("never", []))
    # Right by luck is not right: if the expected number is itself FLAGGED as unverified,
    # it came from no tool or document (6.2: a01 said 360, flagged, and still passed).
    unverified_hit = any(isinstance(w, (int, float)) and
                         any(abs(f - w) <= max(0.01, 1e-4 * abs(w)) for f in r.unverified)
                         for w in case["expect"])
    tools_ok = all(t in used for t in case["tools"])
    if case.get("no_tools"):
        tools_ok = not used
    firsts = [used.index(t) for t in case["order"] if t in used]
    order_ok = len(firsts) == len(case["order"]) and firsts == sorted(firsts)
    ok = answer_ok and tools_ok and order_ok and not unverified_hit
    tool_problem = "used a tool needlessly" if case.get("no_tools") else "missing tool"
    why = [] if ok else [w for w, bad in (("wrong answer", not answer_ok), (tool_problem, not tools_ok),
                                          ("wrong order", not order_ok),
                                          ("answer unverified", unverified_hit)) if bad]
    return {"ok": ok, "why": why, "used": used, "flagged": r.unverified}


def run_mode(mode: str) -> list:
    rows = []
    for c in CASES:
        if mode == "workflow":
            from app import workflow
            r = workflow.run(c["q"])
        else:
            r = agent.run(c["q"], plan=(mode == "plan"))
        chk = check(c, r)
        rows.append({"id": c["id"], "kind": c["kind"], "question": c["q"], "expect": c["expect"],
                     "answer": r.answer, "plan": r.plan, "model_calls": r.model_calls,
                     "seconds": round(r.seconds, 1), "max_prompt_tokens": r.max_prompt_tokens,
                     "stopped": r.stopped,
                     # every step, so a failure can be READ instead of guessed (6.2: a01's
                     # calculator probably failed, but the file had no way to show it)
                     "trace": [{"kind": st.kind, "name": st.name, "args": st.args,
                                "result": st.result[:300]} for st in r.steps],
                     **chk})
        mark = "OK  " if chk["ok"] else "FAIL"
        print(f"  {mode:<4} {c['id']} {mark} {', '.join(chk['why']):<26} tools={chk['used']}  "
              f"{r.model_calls} calls {r.seconds:5.1f}s"
              + (f"  flagged={chk['flagged']}" if chk["flagged"] else ""), flush=True)
    return rows


def summary(rows: list) -> dict:
    n = len(rows)
    return {"correct": sum(r["ok"] for r in rows), "n": n,
            "wrong_order": sum("wrong order" in r["why"] for r in rows),
            "wrong_answer": sum("wrong answer" in r["why"] for r in rows),
            "flagged_answers": sum(bool(r["flagged"]) for r in rows),
            "mean_calls": round(sum(r["model_calls"] for r in rows) / n, 1),
            "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1)}


def save(mode: str, label: str, rows: list, s: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS / f"{stamp}_agent_{mode}{'_' + label if label else ''}.json"
    path.write_text(json.dumps({"timestamp": stamp, "kind": "agent", "mode": mode, "label": label,
                                "today": TODAY.isoformat(), "summary": s, "per_case": rows},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    args = sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "both"
    label = args[args.index("--label") + 1] if "--label" in args else ""
    if not llm.is_up():
        sys.exit("Ollama is not running. Start it with: ollama serve")
    modes = ["loop", "plan"] if mode == "both" else mode.split(",")
    print(f"{len(CASES)} cases, today = {TODAY}. Roughly {len(CASES) * 25 * len(modes) // 60} min.\n")
    results = {}
    for m in modes:
        rows = run_mode(m)
        results[m] = (rows, summary(rows))
        print(f"  saved {save(m, label, rows, results[m][1]).relative_to(ROOT)}\n")

    print(f"{'':<22}" + "".join(f"{m:>12}" for m in modes))
    for key in ("correct", "wrong_order", "wrong_answer", "flagged_answers", "mean_calls", "mean_seconds"):
        vals = "".join(f"{(str(results[m][1][key]) + ('/' + str(results[m][1]['n']) if key == 'correct' else '')):>12}"
                       for m in modes)
        print(f"  {key:<20}{vals}")
    if len(modes) == 2:
        a, b = ({r["id"]: r for r in results[m][0]} for m in modes)
        flips = [(i, a[i]["ok"], b[i]["ok"]) for i in a if a[i]["ok"] != b[i]["ok"]]
        print(f"\nCases that differ ({len(flips)}):")
        for i, x, y in flips:
            print(f"  {i}: {modes[0]} {'OK' if x else 'FAIL'} -> {modes[1]} {'OK' if y else 'FAIL'}   "
                  f"{modes[1]} plan/route: {b[i]['plan'][:110]!r}")
    print("\nRead the failures in the saved JSON: 'answer' and 'plan' for each case.")


if __name__ == "__main__":
    main()
