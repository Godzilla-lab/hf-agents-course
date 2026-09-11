"""
The worker: a CodeAgent that answers one GAIA question at a time.

Three jobs live here and nowhere else, so run_local.py and app.py behave
identically:

  build_agent()       assemble brain + toolbox + rules
  answer_question()   wrap one question (and its attachment) and run the agent
  normalize_answer()  scrub the reply into exact-match shape

GAIA scoring is exact string match. "3" is right, "3 albums" is wrong,
"The answer is 3." is wrong. So the rules below are all about the SHAPE of
the final answer, not just its content.
"""

import os
import re

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from smolagents.utils import Retrying

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

# The brain. LiteLLM routes on the prefix: "xai/", "gemini/", "anthropic/", "openai/"...
# Swap the string in .env and nothing else in this file changes.
MODEL_ID = os.environ.get("AGENT_MODEL", "xai/grok-4.6")

# Which environment variable holds the key for each provider prefix.
KEY_VARS = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def api_key_for(model_id: str) -> str:
    provider = model_id.split("/", 1)[0]
    var = KEY_VARS.get(provider, f"{provider.upper()}_API_KEY")
    key = os.environ.get(var)
    if not key:
        raise SystemExit(f"{var} missing in unit4/.env (needed for model {model_id})")
    return key

# Goes into the system prompt. This is the "job description" the model reads
# before every single step, so keep it short and unambiguous.
INSTRUCTIONS = """
You are answering questions from the GAIA benchmark. Scoring is EXACT string match,
so the value you pass to final_answer() must be ONLY the answer, nothing else:

- a number: digits only. No commas as thousands separators, no units, no $ or %
  unless the question explicitly asks for them. Keep decimals if the question asks.
- a short string: no articles (a, an, the), no abbreviations, no trailing period,
  no quotes, no explanation.
- a comma separated list: apply the rules above to every element, join with ", ".
  If the question asks for alphabetical or ascending order, sort before answering.

Never pass a sentence to final_answer(). Never include the words "FINAL ANSWER".
Think step by step in your code comments, verify facts by reading the actual page
or file, then call final_answer() with the bare value.
If after several tries you are still unsure, call final_answer() with your single
best guess. A guess scores more than an apology.

How to research:
- web_search gives snippets only. Never answer from a snippet; open the page.
- For Wikipedia facts use wikipedia_page(title, as_of_date) which returns the URL,
  then find_in_page(url, "keyword") to jump to the relevant section. If the question
  names a year or version, pass as_of_date so you read that version.
- For other sites use visit_webpage(url) then find_in_page(url, "keyword").
- If a page is long and the part you need is not in the first chunk, do NOT search
  the web again: call find_in_page(url, "<name or number you expect>") on that page.
- If a site returns 403 or an error, try a different site once, then fall back to
  Wikipedia. Do not retry the same blocked site.
- Read tables carefully and count with code, not by eye.
- Attachments: read_text_file / run_python_file / read_excel_file / transcribe_audio /
  analyze_image. YouTube links: watch_youtube_video(url, question).
""".strip()


def _is_transient_error(exc: BaseException) -> bool:
    """True for errors that go away if you simply wait: rate limits and overload."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "503", "rate limit", "rate_limit", "too many requests",
                       "high demand", "overloaded", "unavailable", "resource_exhausted",
                       "disconnected", "connection", "timeout", "timed out")
    )


def build_agent(tools: list | None = None) -> CodeAgent:
    """Brain + toolbox + rules."""
    if tools is None:
        from tools import ALL_TOOLS
        tools = ALL_TOOLS
    model = LiteLLMModel(model_id=MODEL_ID, api_key=api_key_for(MODEL_ID))
    # smolagents retries only on 429 (too many requests). Providers also throw
    # 503 (overloaded) in bursts, so swap in a retryer that treats both as
    # "wait, then try again": 6 attempts, waiting 10, 20, 40, 80, 160 seconds.
    model.retryer = Retrying(
        max_attempts=6,
        wait_seconds=10,
        exponential_base=2,
        jitter=True,
        retry_predicate=_is_transient_error,
        reraise=True,
    )
    return CodeAgent(
        tools=tools,
        model=model,
        instructions=INSTRUCTIONS,
        # The sandbox only allows a short list of imports by default. Add the
        # data-wrangling ones the questions need. Never os/subprocess/pathlib:
        # anything touching disk goes through a tool, not agent-written code.
        additional_authorized_imports=["pandas", "numpy", "json", "csv", "io"],
        max_steps=15,
        verbosity_level=1,
    )


def answer_question(agent: CodeAgent, task_id: str, question: str, file_name: str = "") -> tuple:
    """
    Run the agent on one question. Returns (raw_output, clean_answer, steps_used).

    file_name is non-empty for 5 of the 20 questions. When it is, the file is
    downloaded BEFORE the agent starts and its path is put in the prompt, so the
    agent does not waste a step deciding to fetch it.
    """
    task = f"Question: {question}"
    if file_name:
        from tools import fetch_task_file  # local import: keeps tools optional until step 3
        path = fetch_task_file(task_id, file_name)
        task += (
            f"\n\nAn attachment for this question is already downloaded at: {path}"
            "\nUse the file tools (read_excel_file, read_text_file, run_python_file, "
            "transcribe_audio, analyze_image) to look inside it. Do not try to open it "
            "with plain Python, the sandbox will refuse."
        )

    result = agent.run(task, return_full_result=True)
    raw = result.output
    answer = normalize_answer(raw)
    if looks_like_prose(answer):
        # Safety net: the agent (or the out-of-steps fallback) wrote a paragraph.
        # One cheap extra call pulls the bare value out of it.
        answer = normalize_answer(extract_bare_answer(agent, question, str(raw)))
    return raw, answer, result.steps and len(result.steps)


def looks_like_prose(text: str) -> bool:
    """A bare GAIA answer is short and has no sentence furniture."""
    return len(text) > 80 or "\n" in text or "**" in text or text.count(". ") >= 1


def extract_bare_answer(agent: CodeAgent, question: str, prose: str) -> str:
    """Ask the brain to reduce a verbose reply to the exact-match value."""
    prompt = (
        "A research agent produced the text below while answering a GAIA question. "
        "Extract the final answer ONLY, formatted for exact string match:\n"
        "- number: digits only, no commas, no units, no $ or % unless the question asks\n"
        "- string: no articles, no abbreviations, no trailing period, no quotes\n"
        "- list: comma separated with a single space after each comma, sorted if the question asks\n"
        "If the text gives no answer, reply with the single most likely value anyway. "
        "Reply with the value and nothing else.\n\n"
        f"QUESTION:\n{question}\n\nAGENT OUTPUT:\n{prose[:6000]}"
    )
    reply = agent.model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    return reply.content or ""


def normalize_answer(raw) -> str:
    """
    Scrub the model's reply into exact-match shape. Pure function, no LLM.

    Deliberately conservative: we strip wrappers (quotes, "FINAL ANSWER:",
    trailing period) and tidy list spacing, but never lowercase or drop words.
    Those could break a correct answer, and the prompt already asks for the
    right shape.
    """
    text = str(raw if raw is not None else "").strip()

    # "FINAL ANSWER: 42"  ->  "42"
    text = re.sub(r"^\s*final\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)

    # Wrapping quotes or backticks: "Right" -> Right
    text = text.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()

    # One trailing period, but not a decimal like "89706.00"
    if text.endswith(".") and not re.fullmatch(r"-?\d+\.", text):
        text = text[:-1].rstrip()

    # Collapse runs of whitespace
    text = " ".join(text.split())

    # Lists: "a,b , c" -> "a, b, c" (skip numbers like "1,000" which we also do not want,
    # but the prompt handles that; here we only touch obvious lists)
    if "," in text and not re.fullmatch(r"-?[\d,]+(\.\d+)?", text):
        parts = [p.strip() for p in text.split(",")]
        text = ", ".join(p for p in parts if p)

    return text or "unknown"


if __name__ == "__main__":
    # Tiny self-test of the scrubber. Costs zero LLM calls.
    cases = {
        'FINAL ANSWER: "Right."': "Right",
        "  3  ": "3",
        "89706.00": "89706.00",
        "b,e": "b, e",
        "`CUB`": "CUB",
        "": "unknown",
        "final answer: broccoli, celery ,lettuce.": "broccoli, celery, lettuce",
    }
    for given, want in cases.items():
        got = normalize_answer(given)
        print(f"{'ok ' if got == want else 'BAD'}  {given!r:45} -> {got!r}")
