"""
Same agent, now with a tool: web search.

This is the course's playlist example. The ONLY difference from 01 is the
tools=[...] list. That is the whole idea - tools are just functions you hand
the agent, and it decides when to call them.
"""

from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# DuckDuckGoSearchTool needs no API key. Inside the agent's code it appears
# as a function named web_search(query="...").
search = DuckDuckGoSearchTool()

agent = CodeAgent(tools=[search], model=model)

result = agent.run(
    "Search for music recommendations for a Batman-themed party at a mansion. "
    "Give me a playlist of 5 specific songs with artist names."
)

print("\n" + "=" * 60)
print("FINAL ANSWER:", result)
print("=" * 60)
