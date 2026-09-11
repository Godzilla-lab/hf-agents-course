"""
The same agent, in a shop window on Hugging Face.

Derived from the course template (agents-course/Final_Assignment_Template).
Two changes that matter:
  1. the loop hands task_id and file_name to the agent, so attachments work
  2. a second button runs a few questions WITHOUT submitting, for safe testing

Secrets (XAI_API_KEY, GEMINI_API_KEY, HF_TOKEN) come from the Space settings,
not from a .env file. SPACE_ID is set automatically by Hugging Face.
"""

import os
import time

import gradio as gr
import pandas as pd
import requests

from agent import answer_question, build_agent

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
PAUSE_BETWEEN_QUESTIONS = 3

# Built once when the Space starts, reused for every question.
AGENT = build_agent()


def fetch_questions() -> list[dict]:
    resp = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


def run_agent(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run the agent on each question. Returns (answers_payload, log_rows)."""
    answers, log = [], []
    for i, item in enumerate(questions, start=1):
        task_id, question, file_name = item.get("task_id"), item.get("question"), item.get("file_name") or ""
        if not task_id or question is None:
            continue
        t0 = time.time()
        try:
            raw, answer, steps = answer_question(AGENT, task_id, question, file_name)
        except Exception as e:
            answer, steps = f"AGENT ERROR: {e}", None
        answers.append({"task_id": task_id, "submitted_answer": answer})
        log.append({"#": i, "task_id": task_id, "question": question[:120], "answer": answer,
                    "steps": steps, "seconds": round(time.time() - t0)})
        if i < len(questions):
            time.sleep(PAUSE_BETWEEN_QUESTIONS)
    return answers, log


def run_without_submitting(n: float) -> tuple[str, pd.DataFrame]:
    """Safe smoke test: answer the first N questions, never touch /submit."""
    try:
        questions = fetch_questions()[: int(n)]
    except Exception as e:
        return f"Could not fetch questions: {e}", pd.DataFrame()
    answers, log = run_agent(questions)
    return f"Ran {len(answers)} question(s). Nothing was submitted.", pd.DataFrame(log)


def run_and_submit_all(profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame]:
    """Template flow: log in, run all 20, submit, show the score."""
    if profile is None:
        return "Please log in to Hugging Face with the button.", pd.DataFrame()
    username = profile.username
    space_id = os.getenv("SPACE_ID", "")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        questions = fetch_questions()
    except Exception as e:
        return f"Could not fetch questions: {e}", pd.DataFrame()

    answers, log = run_agent(questions)
    if not answers:
        return "The agent produced no answers.", pd.DataFrame(log)

    payload = {"username": username.strip(), "agent_code": agent_code, "answers": answers}
    try:
        resp = requests.post(f"{DEFAULT_API_URL}/submit", json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        status = (f"Submission successful. User: {result.get('username')}  "
                  f"Score: {result.get('score', 'N/A')}% "
                  f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} correct)  "
                  f"Message: {result.get('message', '')}")
    except requests.exceptions.HTTPError as e:
        detail = e.response.text[:500] if e.response is not None else str(e)
        status = f"Submission failed: {detail}"
    except Exception as e:
        status = f"Submission failed: {e}"
    return status, pd.DataFrame(log)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA agent (Unit 4 final assignment)")
    gr.Markdown(
        "smolagents CodeAgent with Grok as the brain and Gemini as the senses "
        "(audio, images, YouTube), plus Stockfish for chess. Log in, then run and submit. "
        "Or use the test button to answer a few questions without submitting."
    )
    gr.LoginButton()
    with gr.Row():
        n_box = gr.Number(value=1, label="Questions to test", precision=0, minimum=1, maximum=20)
        test_button = gr.Button("Run N questions without submitting")
    run_button = gr.Button("Run all 20 and submit", variant="primary")
    status_output = gr.Textbox(label="Status", lines=3, interactive=False)
    results_table = gr.DataFrame(label="Answers", wrap=True)

    test_button.click(fn=run_without_submitting, inputs=[n_box], outputs=[status_output, results_table])
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
