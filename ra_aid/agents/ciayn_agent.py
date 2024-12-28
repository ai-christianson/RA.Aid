import inspect
from typing import Dict, Any, Generator, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from ra_aid.exceptions import ToolExecutionError

class CiaynAgent:
    def _get_function_info(self, func):
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
\"\"\""""
        return info

    def __init__(self, model, tools: list, max_history_messages: int = 50):
        """Initialize the agent with a model and list of tools.
        
        Args:
            model: The language model to use
            tools: List of tools available to the agent
            max_history_messages: Maximum number of messages to keep in chat history
        """
        self.model = model
        self.tools = tools
        self.max_history_messages = max_history_messages
        self.available_functions = []
        for t in tools:
            self.available_functions.append(self._get_function_info(t.func))

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
            raise ToolExecutionError(error_msg)

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

    def _trim_chat_history(self, initial_messages: List[Any], chat_history: List[Any]) -> List[Any]:
        """Trim chat history to maximum length while preserving initial messages.
        
        Only trims the chat_history portion while preserving all initial messages.
        Returns the concatenated list of initial_messages + trimmed chat_history.
        
        Args:
            initial_messages: List of initial messages to preserve
            chat_history: List of chat messages that may be trimmed
            
        Returns:
            List[Any]: Concatenated initial_messages + trimmed chat_history
        """
        if len(chat_history) <= self.max_history_messages:
            return initial_messages + chat_history
            
        # Keep last max_history_messages from chat_history
        return initial_messages + chat_history[-self.max_history_messages:]

    def stream(self, messages_dict: Dict[str, List[Any]], config: Dict[str, Any] = None) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        chat_history = []
        last_result = None
        first_iteration = True
        
        while True:
            base_prompt = self._build_prompt(None if first_iteration else last_result)
            chat_history.append(HumanMessage(content=base_prompt))
            
            full_history = self._trim_chat_history(initial_messages, chat_history)
            response = self.model.invoke(full_history)
                
            try:
                last_result = self._execute_tool(response.content)
                chat_history.append(response)
                first_iteration = False
                yield {}

            except ToolExecutionError as e:
                yield self._create_error_chunk(str(e))
                break
