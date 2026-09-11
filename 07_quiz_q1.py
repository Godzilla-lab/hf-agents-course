"""Unit 2.1 quiz Q1: a CodeAgent with web search."""

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
)

if __name__ == "__main__":
    print("tools :", sorted(agent.tools))
    print("model :", type(agent.model).__name__)
    print("id    :", agent.model.model_id)
    print("\nConstructed successfully.")
