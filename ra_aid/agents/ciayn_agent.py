import inspect
from dataclasses import dataclass
from typing import Dict, Any, Generator, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from ra_aid.exceptions import ToolExecutionError
from ra_aid.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ChunkMessage:
    content: str
    status: str

class CiaynAgent:
    """Code Is All You Need (CIAYN) agent that uses generated Python code for tool interaction.

    The CIAYN philosophy emphasizes direct code generation and execution over structured APIs:
    - Language model generates executable Python code snippets
    - Tools are invoked through natural Python code rather than fixed schemas
    - Flexible and adaptable approach to tool usage through dynamic code
    - Complex workflows emerge from composing code segments

    Code Generation & Function Calling:
    - Dynamic generation of Python code for tool invocation
    - Handles complex nested function calls and argument structures
    - Natural integration of tool outputs into Python data flow
    - Runtime code composition for multi-step operations

    ReAct Pattern Implementation:
    - Observation: Captures tool execution results
    - Reasoning: Analyzes outputs to determine next steps
    - Action: Generates and executes appropriate code
    - Reflection: Updates state and plans next iteration
    - Maintains conversation context across iterations

    Core Capabilities:
    - Dynamic tool registration with automatic documentation
    - Sandboxed code execution environment
    - Token-aware chat history management
    - Comprehensive error handling and recovery
    - Streaming interface for real-time interaction
    - Memory management with configurable limits
    """

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

    def __init__(self, model, tools: list, max_history_messages: int = 50, max_tokens: Optional[int] = 100000):
        """Initialize the agent with a model and list of tools.
        
        Args:
            model: The language model to use
            tools: List of tools available to the agent
            max_history_messages: Maximum number of messages to keep in chat history
            max_tokens: Maximum number of tokens allowed in message history (None for no limit)
        """
        self.model = model
        self.tools = tools
        self.max_history_messages = max_history_messages
        self.max_tokens = max_tokens
        self.available_functions = []
        for t in tools:
            self.available_functions.append(self._get_function_info(t.func))

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        base_prompt = ""
        if last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"
            
        base_prompt += f"""

<agent instructions>
You are a ReAct agent. You run in a loop and use ONE of the available functions per iteration.
The result of that function call will be given to you in the next message.
Call one function at a time. Function arguments can be complex objects, long strings, etc. if needed.
The user cannot see the results of function calls, so you have to explicitly use a tool like ask_human if you want them to see something.
You must always respond with a single line of python that calls one of the available tools.
Use as many steps as you need to in order to fully complete the task.
Start by asking the user what they want.

You must carefully review the conversation history, which functions were called so far, returned results, etc., and make sure the very next function call you make makes sense in order to achieve the original goal.
You must achieve the goal in as few steps possible, but no fewer.
You typically don't want to keep calling the same function over and over with the same parameters.
</agent instructions>

You must ONLY use ONE of the following functions (these are the ONLY functions that exist):

<available functions>
{"\n\n".join(self.available_functions)}
</available functions>

You may use ANY of the above functions to complete your job. Use the best one for the current step you are on. Be efficient, avoid getting stuck in repetitive loops, and do not hesitate to call functions which delegate your work to make your life easier.
But you MUST NOT assume tools exist that are not in the above list, e.g. write_file_tool.
Consider your task done only once you have taken *ALL* the steps required to complete it.

<example bad output>
write_file_tool(...)
</example bad output>

<example good output>
request_research_and_implementation(\"\"\"
Example query.
\"\"\")

run_programming_task(\"\"\"
# Example Programming Task

Implement a widget factory satisfying the following requirements:

- Requirement A
- Requirement B

...
\"\"\")
</example good output>
DO NOT CLAIM YOU ARE FINISHED UNTIL YOU ACTUALLY ARE!
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
        message = ChunkMessage(content=content, status="error")
        return {
            "tools": {
                "messages": [message]
            }
        }

    @staticmethod
    def _estimate_tokens(content: Optional[Union[str, BaseMessage]]) -> int:
        """Estimate number of tokens in content using simple byte length heuristic.
        
        Estimates 1 token per 4 bytes of content. For messages, uses the content field.
        
        Args:
            content: String content or Message object to estimate tokens for
            
        Returns:
            int: Estimated number of tokens, 0 if content is None/empty
        """
        if content is None:
            return 0
            
        if isinstance(content, BaseMessage):
            text = content.content
        else:
            text = content
            
        if not text:
            return 0
            
        return len(text.encode('utf-8')) // 4

    def _trim_chat_history(self, initial_messages: List[Any], chat_history: List[Any]) -> List[Any]:
        """Trim chat history based on message count and token limits while preserving initial messages.
        
        Applies both message count and token limits (if configured) to chat_history,
        while preserving all initial_messages. Returns concatenated result.
        
        Args:
            initial_messages: List of initial messages to preserve
            chat_history: List of chat messages that may be trimmed
            
        Returns:
            List[Any]: Concatenated initial_messages + trimmed chat_history
        """
        # First apply message count limit
        if len(chat_history) > self.max_history_messages:
            chat_history = chat_history[-self.max_history_messages:]
            
        # Skip token limiting if max_tokens is None
        if self.max_tokens is None:
            return initial_messages + chat_history
            
        # Calculate initial messages token count
        initial_tokens = sum(self._estimate_tokens(msg) for msg in initial_messages)
        
        # Remove messages from start of chat_history until under token limit
        while chat_history:
            total_tokens = initial_tokens + sum(self._estimate_tokens(msg) for msg in chat_history)
            if total_tokens <= self.max_tokens:
                break
            chat_history.pop(0)
            
        return initial_messages + chat_history

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
            response = self.model.invoke([SystemMessage("Execute efficiently yet completely as a fully autonomous agent.")] + full_history)
                
            try:
                logger.debug(f"Code generated by agent: {response.content}")
                last_result = self._execute_tool(response.content)
                chat_history.append(response)
                first_iteration = False
                yield {}

            except ToolExecutionError as e:
                chat_history.append(HumanMessage(content=f"Your tool call caused an error: {e}\n\nPlease correct your tool call and try again."))
                yield self._create_error_chunk(str(e))
