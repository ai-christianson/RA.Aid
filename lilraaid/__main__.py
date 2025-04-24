import sys

from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, tool

def main():
    model = LiteLLMModel(
        model_id="openrouter/thudm/glm-4-32b:free",
        temperature=0.2
    )
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
    agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

if __name__ == "__main__":
    main()
