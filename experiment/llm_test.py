import os
import uuid
from dotenv import load_dotenv
from ra_aid.agent_utils import run_agent_with_retry
from typing import Dict, Any, Generator, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from ra_aid.tools.list_directory import list_directory_tree
from ra_aid.tool_configs import get_read_only_tools
import inspect
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console

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

class CiaynAgent:
    def get_function_info(self, func):
        """
        Returns a well-formatted string containing the function signature and docstring,
        designed to be easily readable by both humans and LLMs.
        """
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func)
        if docstring is None:
            docstring = "No docstring provided"
        full_signature = f"{func.__name__}{signature}"
        info = f"""{full_signature}
\"\"\"
{docstring}
\"\"\"  """
        return info

    def __init__(self, model, tools: list):
        """Initialize the agent with a model and list of tools."""
        self.model = model
        self.tools = tools
        self.available_functions = []
        for t in tools:
            self.available_functions.append(self.get_function_info(t.func))

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        base_prompt = ""
        if last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"
            
        base_prompt += f"""
<available functions>
{"\n\n".join(self.available_functions)}
</available functions>

<agent instructions>
You are a ReAct agent. You run in a loop and use ONE of the available functions per iteration.
If the current query does not require a function call, just use output_message to say what you would normally say.
The result of that function call will be given to you in the next message.
Call one function at a time. Function arguments can be complex objects, long strings, etc. if needed.
The user cannot see the results of function calls, so you have to explicitly call output_message if you want them to see something.
You must always respond with a single line of python that calls one of the available tools.
Use as many steps as you need to in order to fully complete the task.
Start by asking the user what they want.
</agent instructions>

<example response>
check_weather("London")
</example response>
    
<example response>
output_message(\"\"\"How can I help you today?\"\"\", True)
</example response>

Output **ONLY THE CODE** and **NO MARKDOWN BACKTICKS**"""
        return base_prompt

    def _execute_tool(self, code: str) -> str:
        """Execute a tool call and return its result."""
        globals_dict = {
            tool.func.__name__: tool.func
            for tool in self.tools
        }
        
        try:
            result = eval(code.strip(), globals_dict)
            return result
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            console.print(f"[red]Error:[/red] {error_msg}")
            return error_msg


    def _create_agent_chunk(self, content: str) -> Dict[str, Any]:
        """Create an agent chunk in the format expected by print_agent_output."""
        return {
            "agent": {
                "messages": [AIMessage(content=content)]
            }
        }

    def _create_error_chunk(self, content: str) -> Dict[str, Any]:
        """Create an error chunk in the format expected by print_agent_output."""
        return {
            "tools": {
                "messages": [{"status": "error", "content": content}]
            }
        }

    def stream(self, messages_dict: Dict[str, List[Any]], config: Dict[str, Any] = None) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        chat_history = []
        last_result = None
        first_iteration = True
        
        while True:
            base_prompt = self._build_prompt(None if first_iteration else last_result)
            chat_history.append(HumanMessage(content=base_prompt))
            
            try:
                full_history = initial_messages + chat_history
                response = self.model.invoke(full_history)
                
                last_result = self._execute_tool(response.content)
                chat_history.append(response)
                first_iteration = False
                yield {}
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield self._create_error_chunk(error_msg)
                break

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
