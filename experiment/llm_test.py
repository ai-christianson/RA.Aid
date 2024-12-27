import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def check_weather(location: str) -> str:
    return f"The weather in {location} is sunny!"

def ask_human(query: str) -> str:
    print()
    print(f"Assistant: {query}")
    user_input = input("\nYou: ").strip()
    return user_input

def output_message(message: str, prompt_user_input: bool = False) -> str:
    print()
    print(f"Assistant: {message.strip()}")
    if prompt_user_input:
        user_input = input("\nYou: ").strip()
        return user_input
    return ""

def evaluate_response(code: str) -> any:
    """
    Evaluates a single function call and returns its result
    """
    globals_dict = {
        'check_weather': check_weather,
        'ask_human': ask_human,
        'output_message': output_message
    }
    
    try:
        # Using eval() instead of exec() since we're evaluating a single expression
        result = eval(code, globals_dict)
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"

def create_chat_interface():
    # Initialize the chat model
    chat = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3,
        base_url="https://openrouter.ai/api/v1",
        # model="deepseek/deepseek-chat"
        model="qwen/qwen-2.5-coder-32b-instruct"
    )
    
    # Chat loop
    print("Welcome to the Chat Interface! (Type 'quit' to exit)")
    
    chat_history = []
    last_result = None
    first_iteration = True
    
    while True:
        base_prompt = ""
        
        # Add the last result to the prompt if it's not the first iteration
        if not first_iteration and last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"
            
        # Construct the tool documentation and context
        base_prompt += """
        <available functions>
        # Get the weather at a location:
        check_weather(location: str) -> str
        
        # Output a message and optionally get their response:
        output_message(message: str, prompt_user_input: bool = False) -> str
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
            response = chat.invoke(chat_history)
            
            # # Print the code response
            # print("\nAssistant generated code:")
            # print(response.content)
            
            # Evaluate the code
            # print("\nExecuting code:")
            last_result = evaluate_response(response.content.strip())
            # if last_result is not None:
                # print(f"Result: {last_result}")
            
            # Add assistant response to history
            chat_history.append(response)
            
            # Set first_iteration to False after the first loop
            first_iteration = False
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    create_chat_interface()