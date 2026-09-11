"""
Step 0: prove the Gemini key works, with ONE tiny request.

Nothing agent-y happens here. We only check that:
  1. the key in .env is picked up,
  2. LiteLLM can reach Gemini,
  3. the model answers.

LiteLLM is an adapter: it speaks one Python interface on our side and translates
to whichever provider the model_id names. The "gemini/" prefix is the routing
label. Swap it for "anthropic/..." or "openai/..." and nothing else changes.
"""

import os

from dotenv import load_dotenv
from smolagents import LiteLLMModel

# Reads unit4/.env and puts its lines into os.environ. Nothing is printed.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit(
        "GEMINI_API_KEY is empty. Create a free key at https://aistudio.google.com/apikey "
        "and put it in unit4/.env (see .env.example)."
    )

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini/gemini-3.8-flash")

model = LiteLLMModel(model_id=MODEL_ID, api_key=os.environ["GEMINI_API_KEY"])

# A message is a list of "turns". Each turn has a role and a list of content parts.
# smolagents uses this shape everywhere, so learn it once here.
messages = [
    {"role": "user", "content": [{"type": "text", "text": "Reply with the single word: pong"}]}
]

reply = model(messages)
print(f"model : {MODEL_ID}")
print(f"reply : {reply.content!r}")
