import os
import uuid
from dotenv import load_dotenv
from ra_aid.agent_utils import run_agent_with_retry
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from ra_aid.tools.list_directory import list_directory_tree
from ra_aid.tool_configs import get_read_only_tools
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console
from ra_aid.agents.ciayn_agent import CiaynAgent

console = Console()

# Load environment variables
load_dotenv()

@tool
def check_weather(location: str) -> str:
    """Gets the weather at the given location."""
    return f"The weather in {location} is sunny!"

@tool
def output_message(message: str, prompt_user_input: bool = False) -> str:
    """Outputs a message to the user, optionally prompting for input."""
    console.print(Panel(Markdown(message.strip())))
    if prompt_user_input:
        user_input = input("\n> ").strip()
        print()
        return user_input
    return ""

if __name__ == "__main__":
    # Initialize the chat model
    chat = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen-2.5-coder-32b-instruct"
    )
    
    # Get tools
    tools = get_read_only_tools(True, True)
    tools.append(output_message)
    
    # Initialize agent
    agent = CiaynAgent(chat, tools)
    
    # Test chat prompt
    test_prompt = "Find the tests in this codebase."

    # Run the agent using run_agent_with_retry
    result = run_agent_with_retry(agent, test_prompt, {"configurable": {"thread_id": str(uuid.uuid4())}})
    
    # Initial greeting
    print("Welcome to the Chat Interface! (Type 'quit' to exit)")
