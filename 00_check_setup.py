"""Run this FIRST. It checks your setup and tells you exactly what's missing."""

import os

print("1. Is smolagents installed?")
import smolagents
print(f"   YES - version {smolagents.__version__}\n")

print("2. Can I find a Hugging Face token?")
token = os.environ.get("HF_TOKEN")
source = "the HF_TOKEN environment variable"

if not token:
    # huggingface-cli login saves the token to a file instead of an env var.
    from huggingface_hub import get_token
    token = get_token()
    source = "the file saved by 'huggingface-cli login'"

if not token:
    print("   NO - no token found.")
    print("   Fix it: run this in your terminal, then paste your token when asked:")
    print("       /Users/godzilla/CLI/HuggingFace/agents-course/.venv/bin/hf auth login")
    print("   Get a token at: https://huggingface.co/settings/tokens (New token -> Read)")
    raise SystemExit(1)

print(f"   YES - found in {source}")
print(f"   It starts with: {token[:7]}...\n")

print("3. Does the token actually work?")
from huggingface_hub import whoami
me = whoami(token=token)
print(f"   YES - logged in as: {me['name']}\n")

print("Setup is good. Next: run 01_first_agent.py")
