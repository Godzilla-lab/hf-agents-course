"""
Local harness: you, standing next to the agent with a clipboard.

Usage (run from the agents-course folder):

    ./.venv/bin/python unit4/run_local.py                  list the 20 questions, no LLM calls
    ./.venv/bin/python unit4/run_local.py --only 3,4,8     run those (1-based) and cache answers
    ./.venv/bin/python unit4/run_local.py --only 3 --force ignore the cache for those
    ./.venv/bin/python unit4/run_local.py --all            run everything not yet cached
    ./.venv/bin/python unit4/run_local.py --table          show the cache
    ./.venv/bin/python unit4/run_local.py --check          compare cache to expected.json
    ./.venv/bin/python unit4/run_local.py --submit         guarded: asks you to type SUBMIT

Why a cache? Every LLM call costs free-tier budget. Once a question has an
answer we keep it in answers.json and never re-run it unless you say --force.
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

API_BASE = "https://agents-course-unit4-scoring.hf.space"
ANSWERS_PATH = os.path.join(HERE, "answers.json")
EXPECTED_PATH = os.path.join(HERE, "expected.json")
LOG_PATH = os.path.join(HERE, "submissions.log")
PAUSE_BETWEEN_QUESTIONS = 8  # seconds, keeps us under the free-tier requests-per-minute cap


# ---------------------------------------------------------------- API helpers

def fetch_questions() -> list[dict]:
    """GET /questions. An API is just a URL that returns JSON instead of a web page."""
    resp = requests.get(f"{API_BASE}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- cache helpers

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- display

def short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def print_questions(questions: list[dict]) -> None:
    print(f"{'#':>2}  {'task_id':36}  {'file':5}  question")
    for i, q in enumerate(questions, start=1):
        ext = os.path.splitext(q.get("file_name") or "")[1] or "-"
        print(f"{i:>2}  {q['task_id']}  {ext:5}  {short(q['question'])}")
    with_files = sum(1 for q in questions if q.get("file_name"))
    print(f"\n{len(questions)} questions, {with_files} with attachments")


def print_table(questions: list[dict], cache: dict) -> None:
    print(f"{'#':>2}  {'status':6}  {'steps':>5}  {'secs':>5}  answer")
    for i, q in enumerate(questions, start=1):
        row = cache.get(q["task_id"])
        if not row:
            print(f"{i:>2}  {'-':6}  {'':>5}  {'':>5}  (not run)")
            continue
        ans = row["answer"] if row["status"] == "ok" else f"ERROR: {short(row.get('error', ''), 60)}"
        print(f"{i:>2}  {row['status']:6}  {row.get('steps', ''):>5}  {row.get('seconds', 0):>5.0f}  {short(ans, 70)}")


# ---------------------------------------------------------------- running

def run_questions(questions: list[dict], indexes: list[int], force: bool, stop_on_error: bool) -> None:
    # Imported here so the list/table modes never load the agent (or need a key).
    from agent import build_agent, answer_question

    cache = load_json(ANSWERS_PATH)
    agent = build_agent()
    todo = [i for i in indexes if force or questions[i - 1]["task_id"] not in cache]
    skipped = sorted(set(indexes) - set(todo))
    if skipped:
        print(f"cached, skipping: {skipped} (use --force to re-run)")

    for n, i in enumerate(todo):
        q = questions[i - 1]
        print("\n" + "=" * 70)
        print(f"Q{i}  {q['task_id']}  file={q.get('file_name') or '-'}")
        print(short(q["question"], 200))
        print("=" * 70)
        row = {
            "index": i,
            "question": q["question"],
            "file_name": q.get("file_name") or "",
            "model": os.environ.get("AGENT_MODEL", "xai/grok-4.6"),
            "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        t0 = time.time()
        try:
            raw, answer, steps = answer_question(agent, q["task_id"], q["question"], q.get("file_name") or "")
            row.update(status="ok", raw=str(raw), answer=answer, steps=steps)
            print(f"\n>>> Q{i} answer: {answer!r}")
        except Exception as e:  # keep going, one bad question must not kill the run
            row.update(status="error", error=f"{type(e).__name__}: {e}")
            print(f"\n>>> Q{i} ERROR: {row['error']}")
            traceback.print_exc()
            if stop_on_error:
                row["seconds"] = time.time() - t0
                cache = load_json(ANSWERS_PATH)
                cache[q["task_id"]] = row
                save_json(ANSWERS_PATH, cache)
                sys.exit(1)
        row["seconds"] = time.time() - t0
        cache = load_json(ANSWERS_PATH)  # re-read: another run may have added rows meanwhile
        cache[q["task_id"]] = row
        save_json(ANSWERS_PATH, cache)  # written after EVERY question, a crash loses nothing

        if n < len(todo) - 1:
            time.sleep(PAUSE_BETWEEN_QUESTIONS)

    print()
    print_table(questions, cache)


def check(questions: list[dict], cache: dict) -> None:
    """Compare cached answers to expected.json, which holds ONLY answers you verified by hand."""
    expected = load_json(EXPECTED_PATH)
    if not expected:
        print("expected.json is empty or missing. Fill it with {\"<index or task_id>\": \"answer\"}.")
        return
    by_index = {str(i): q["task_id"] for i, q in enumerate(questions, start=1)}
    ok = 0
    for key, want in expected.items():
        task_id = by_index.get(key, key)
        idx = next((i for i, q in enumerate(questions, 1) if q["task_id"] == task_id), "?")
        got = cache.get(task_id, {}).get("answer")
        mark = "PASS" if got == want else "FAIL"
        ok += mark == "PASS"
        print(f"Q{idx:<3} {mark}  want={want!r}  got={got!r}")
    print(f"\n{ok}/{len(expected)} hand-verified answers match")


# ---------------------------------------------------------------- submit (guarded)

def submit(questions: list[dict], cache: dict, yes: bool = False) -> None:
    """The ONLY place in this project that talks to /submit."""
    username = os.environ.get("HF_USERNAME", "").strip()
    space_id = os.environ.get("SPACE_ID", "").strip()
    if not username or not space_id:
        sys.exit("Set HF_USERNAME and SPACE_ID in unit4/.env before submitting.")
    if not yes and not sys.stdin.isatty():
        sys.exit("Refusing to submit from a non-interactive shell (pass --yes to confirm).")

    answers = []
    for q in questions:
        row = cache.get(q["task_id"], {})
        ans = row.get("answer") if row.get("status") == "ok" else None
        answers.append({"task_id": q["task_id"], "submitted_answer": ans or "unknown"})

    payload = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers,
    }
    print(f"username   : {payload['username']}")
    print(f"agent_code : {payload['agent_code']}")
    for i, a in enumerate(answers, start=1):
        print(f"  Q{i:<3} {a['submitted_answer']!r}")
    missing = sum(1 for a in answers if a["submitted_answer"] == "unknown")
    if missing:
        print(f"\nWARNING: {missing} answers are 'unknown'")

    if yes:
        print("\n--yes given, sending")
    elif input("\nType SUBMIT to send, anything else to abort: ").strip() != "SUBMIT":
        print("aborted")
        return

    resp = requests.post(f"{API_BASE}/submit", json=payload, timeout=60)
    print(f"\nHTTP {resp.status_code}")
    try:
        result = resp.json()
    except ValueError:
        result = {"raw": resp.text}
    print(json.dumps(result, indent=2))
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"sent_at": datetime.now(timezone.utc).isoformat(), "result": result}) + "\n")


# ---------------------------------------------------------------- main

def parse_indexes(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    bad = [i for i in out if not 1 <= i <= n]
    if bad:
        sys.exit(f"index out of range: {bad} (valid 1..{n})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma list of 1-based indexes, ranges ok: 3,4,8-11")
    p.add_argument("--all", action="store_true", help="run every question not yet cached")
    p.add_argument("--force", action="store_true", help="re-run even if cached")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--table", action="store_true", help="print the cache")
    p.add_argument("--check", action="store_true", help="compare cache with expected.json")
    p.add_argument("--submit", action="store_true", help="send cached answers to the scorer (asks first)")
    p.add_argument("--yes", action="store_true", help="with --submit: skip the SUBMIT prompt")
    args = p.parse_args()

    questions = fetch_questions()

    if args.table:
        print_table(questions, load_json(ANSWERS_PATH))
    elif args.check:
        check(questions, load_json(ANSWERS_PATH))
    elif args.submit:
        submit(questions, load_json(ANSWERS_PATH), yes=args.yes)
    elif args.only or args.all:
        indexes = parse_indexes(args.only, len(questions)) if args.only else list(range(1, len(questions) + 1))
        run_questions(questions, indexes, args.force, args.stop_on_error)
    else:
        print_questions(questions)


if __name__ == "__main__":
    # Make "from agent import ..." work no matter where you run this from.
    sys.path.insert(0, HERE)
    main()
