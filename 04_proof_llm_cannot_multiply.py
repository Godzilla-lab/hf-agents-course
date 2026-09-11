"""
PROOF that the LLM did not do the multiplication in script 01.

The claim to test: "LLMs are good at coding, so they can multiply without tools."

The experiment: ask the EXACT SAME model the EXACT SAME sums, but forbid it
from writing code. If the model can really multiply, it will get them right.
If it was Python doing the work all along, it will get them wrong.
"""

from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

SUMS = [
    (4573, 8821),    # the one from script 01
    (6829, 4417),
    (91237, 5849),
    (37, 42),        # small, as a control
]

def ask_the_raw_model(a, b):
    prompt = (
        f"What is {a} * {b}?\n"
        "Reply with ONLY the digits of the answer. "
        "Do not write code. Do not explain. Do not show working."
    )
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return model.generate(msgs).content.strip()

print(f"{'PROBLEM':>18} | {'TRUTH (python)':>16} | {'LLM GUESS':>16} | VERDICT")
print("-" * 78)

wrong = 0
for a, b in SUMS:
    truth = a * b                     # Python. Actually correct, always.
    guess = ask_the_raw_model(a, b)   # The LLM. Pattern-matching from memory.

    digits = "".join(c for c in guess if c.isdigit())
    ok = digits == str(truth)
    if not ok:
        wrong += 1

    print(f"{a:>7} * {b:<8} | {truth:>16,} | {guess[:16]:>16} | {'correct' if ok else 'WRONG'}")

print("-" * 78)
print(f"The raw LLM got {wrong} of {len(SUMS)} wrong.")
print()
print("In script 01 the agent got 4573 * 8821 right every time, because it")
print("never tried to multiply. It wrote 'a * b' and Python did the work.")
print("The LLM's skill is WRITING the code. The CPU's job is RUNNING it.")
