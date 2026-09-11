"""
Unit 2.1 challenge: a manager agent that delegates to a web search agent.

The shape of the idea: one agent is a SPECIALIST that only knows how to browse.
Another is a MANAGER with no tools of its own, whose only power is handing work
to the specialist and reasoning about what comes back.
"""

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
)

model = InferenceClientModel()

# THE SPECIALIST. Searching the web is two jobs, not one:
#   DuckDuckGoSearchTool -> find candidate URLs
#   VisitWebpageTool     -> actually open one and read it
# Search alone gives you snippets. Without VisitWebpage the agent can only
# ever skim, which is how you get confident, wrong answers.
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model,
    max_steps=10,
    name="web_agent",
    description=(
        "Browses the web to answer questions. "
        "Give it a specific query as an argument."
    ),
)

# THE MANAGER. tools=[] on purpose: its capability comes from managed_agents.
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
    max_steps=10,
)

if __name__ == "__main__":
    print("web_agent name        :", web_agent.name)
    print("web_agent tools       :", sorted(web_agent.tools))
    print("web_agent max_steps   :", web_agent.max_steps)
    print("manager managed_agents:", sorted(manager_agent.managed_agents))
    print("manager extra imports :", manager_agent.additional_authorized_imports)
    print("\nConstructed successfully.")
