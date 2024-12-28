import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from ra_aid.tools.list_directory import list_directory_tree
from ra_aid.tool_configs import get_read_only_tools
import inspect
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console

console = Console()

# Load environment variables
load_dotenv()


def get_function_info(func):
    """
    Returns a well-formatted string containing the function signature and docstring,
    designed to be easily readable by both humans and LLMs.
    """
    # Get signature
    signature = inspect.signature(func)
    
    # Get docstring - use getdoc to clean up indentation
    docstring = inspect.getdoc(func)
    if docstring is None:
        docstring = "No docstring provided"
        
    # Format full signature including return type
    full_signature = f"{func.__name__}{signature}"
    
    # Build the complete string
    info = f"""{full_signature}
\"\"\"
{docstring}
\"\"\"  """
    
    return info

@tool
def check_weather(location: str) -> str:
    """
    Gets the weather at the given location.
    """
    return f"The weather in {location} is sunny!"

@tool
def output_message(message: str, prompt_user_input: bool = False) -> str:
    """
    Outputs a message to the user, optionally prompting for input.
    """
    print()
    console.print(Panel(Markdown(message.strip())))
    if prompt_user_input:
        user_input = input("\n> ").strip()
        print()
        return user_input
    return ""

def evaluate_response(code: str, tools: list) -> any:
    """
    Evaluates a single function call and returns its result
    
    Args:
        code (str): The code to evaluate
        tools (list): List of tool objects that have a .func property
        
    Returns:
        any: Result of the code evaluation
    """
    # Create globals dictionary from tool functions
    globals_dict = {
        tool.func.__name__: tool.func
        for tool in tools
    }
    
    try:
        # Using eval() instead of exec() since we're evaluating a single expression
        result = eval(code, globals_dict)
        return result
    except Exception as e:
        print(f"Code:\n\n{code}\n\n")
        print(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"

def create_chat_interface():
    # Initialize the chat model
    chat = ChatOpenAI(
        # api_key=os.getenv("OPENROUTER_API_KEY"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7 ,
        # base_url="https://openrouter.ai/api/v1",
        base_url="https://api.deepseek.com/v1",
        # model="deepseek/deepseek-chat"
        model="deepseek-chat"
        # model="openai/gpt-4o-mini"
        # model="qwen/qwen-2.5-coder-32b-instruct"
        # model="qwen/qwen-2.5-72b-instruct"
    )
    
    # Chat loop
    print("Welcome to the Chat Interface! (Type 'quit' to exit)")
    
    chat_history = []
    last_result = None
    first_iteration = True
    
    tools = get_read_only_tools(True, True)
    
    tools.extend([output_message])
    
    available_functions = []
    
    for t in tools:
        available_functions.append(get_function_info(t.func))
    
    while True:
        base_prompt = ""
        
        # Add the last result to the prompt if it's not the first iteration
        if not first_iteration and last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"
            
        # Construct the tool documentation and context
        base_prompt += f"""
        <available functions>
        {"\n\n".join(available_functions)}
        </available functions>
        """
        
        base_prompt += """
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
            output_message(\"\"\"
            How can I help you today?
            \"\"\", True)
        </example response>
        """
        
        base_prompt += "\nOutput **ONLY THE CODE** and **NO MARKDOWN BACKTICKS**"
            
        # Add user message to history
        # Remove the previous messages if they exist
        # if len(chat_history) > 1:
        #     chat_history.pop()  # Remove the last assistant message
        #     chat_history.pop()  # Remove the last human message
            
        chat_history.append(HumanMessage(content=base_prompt))
        
        try:
            # Get response from model
            # print("PRECHAT")
            response = chat.invoke(chat_history)
            # print("POSTCHAT")
            
            # # Print the code response
            # print("\nAssistant generated code:")
            # print(response.content)
            
            # Evaluate the code
            # print("\nExecuting code:")
            # print("PREEVAL")
            last_result = evaluate_response(response.content.strip(), tools)
            # print("POSTEVAL")
            # if last_result is not None:
            #     print(f"Result: {last_result}")
            
            # Add assistant response to history
            chat_history.append(response)
            
            # Set first_iteration to False after the first loop
            first_iteration = False
            # print("LOOP")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    create_chat_interface()