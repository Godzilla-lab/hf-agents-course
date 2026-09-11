"""
WHY input tokens grow every step: the LLM has no memory.

Everyone assumes the model "remembers" the conversation. It does not.
Each call is a brand new stranger who has never heard of you.
The only reason chat feels continuous is that the ENTIRE transcript is
re-sent, from scratch, every single time.
"""

from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

def ask(messages):
    return model.generate(messages).content.strip()

def user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}

def assistant(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}

print("=" * 70)
print("EXPERIMENT A: two separate calls")
print("=" * 70)

a1 = ask([user("My favourite number is 8821. Just say OK.")])
print("Call 1 -> ", a1)

a2 = ask([user("What is my favourite number?")])
print("Call 2 -> ", a2)
print("\nThe model has NO IDEA. Call 2 knows nothing about call 1.\n")

print("=" * 70)
print("EXPERIMENT B: same thing, but we re-send the history")
print("=" * 70)

history = [
    user("My favourite number is 8821. Just say OK."),
    assistant(a1),
    user("What is my favourite number?"),
]
b2 = ask(history)
print("Call 2 -> ", b2)
print("\nNow it knows. Not because it remembered.")
print("Because we PAID to send the whole conversation again.\n")

print("=" * 70)
print("THE COST OF THAT, over 6 steps")
print("=" * 70)

STEP = 1000   # roughly 1000 new tokens of thinking + result per step
print(f"{'step':>5} | {'sent this step':>15} | {'paid so far':>13}")
print("-" * 42)
total = 0
for step in range(1, 7):
    sent = STEP * step      # every step re-sends all previous steps
    total += sent
    print(f"{step:>5} | {sent:>15,} | {total:>13,}")
print("-" * 42)
print("Doubling the steps did NOT double the cost. Look at the last column.")
print("3 steps cost 6,000. 6 steps cost 21,000. That is 3.5x, not 2x.")
