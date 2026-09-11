"""
The simplest possible CodeAgent: NO tools at all.

Why no tools? Because I want you to see the LOOP working first, with nothing
else that can break. An agent with zero tools can still do a lot, because it
can write and run Python - and Python is itself a tool.
"""

from smolagents import CodeAgent, InferenceClientModel

# The "brain". InferenceClientModel talks to Hugging Face's servers.
# It finds your token automatically - that's why you ran 00_check_setup.py.
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# The agent. tools=[] means: no tools. Just the brain and a Python interpreter.
agent = CodeAgent(tools=[], model=model)

# Ask it something that needs real calculation, not recall.
# A plain chatbot would guess at this. An agent will compute it.
result = agent.run(
    "What is 4573 * 8821, and is that number divisible by 7? "
    "Show me the remainder."
)

print("\n" + "=" * 60)
print("FINAL ANSWER:", result)
print("=" * 60)
