---
title: GAIA Agent
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# GAIA agent, Hugging Face Agents Course Unit 4

A smolagents `CodeAgent` that answers the 20 GAIA Level 1 questions of the course's final
assignment.

| part | what it does |
| --- | --- |
| `agent.py` | builds the agent, wraps each question, scrubs the answer into exact-match shape |
| `tools.py` | web search, page reading with in-page find, Wikipedia as of a date, file readers, audio transcription (Gemini), image analysis (Grok vision), YouTube viewing (yt-dlp captions plus sampled frames read by Grok), chess board reading plus Stockfish |
| `app.py` | Gradio app: log in, run all 20 and submit, or run N without submitting |
| `run_local.py` | local harness with a cache, for running questions one at a time |

## Models

- Brain: `AGENT_MODEL` (default `xai/grok-4.6`) through LiteLLM. Any provider prefix works.
- Hearing: `MEDIA_MODELS`, a comma list of Gemini models tried in order for audio (free tier allows about 20 requests per day per model). Images and video frames go to the brain model.

## Secrets needed on the Space

`XAI_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN` (read access to the gated
`gaia-benchmark/GAIA` dataset, used only to download question attachments when the
course file endpoint is unavailable).

## Running locally

    cp .env.example .env   # fill in the keys
    python run_local.py             # list questions
    python run_local.py --only 3,6  # run some, cached in answers.json
    python run_local.py --submit    # asks you to type SUBMIT
