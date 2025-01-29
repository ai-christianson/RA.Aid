import re
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ra_aid.exceptions import ToolExecutionError
from ra_aid.logging_config import get_logger
from ra_aid.models_tokens import DEFAULT_TOKEN_LIMIT
from ra_aid.tools.reflection import get_function_info

logger = get_logger(__name__)


@dataclass
class ChunkMessage:
    content: str
    status: str


def validate_function_call_pattern(s: str) -> bool:
    """Check if a string matches the expected function call pattern.

    Validates that the string represents a valid function call with:
    - Function name consisting of word characters, underscores or hyphens
    - Opening/closing parentheses with balanced nesting
    - Arbitrary arguments inside parentheses
    - Optional whitespace

    Args:
        s: String to validate

    Returns:
        bool: False if pattern matches (valid), True if invalid
    """
    pattern = r"^\s*[\w_\-]+\s*\([^)(]*(?:\([^)(]*\)[^)(]*)*\)\s*$"
    return not re.match(pattern, s, re.DOTALL)


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

    def __init__(
        self,
        model,
        tools: list,
        max_history_messages: int = 50,
        max_tokens: Optional[int] = DEFAULT_TOKEN_LIMIT,
    ):
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
            self.available_functions.append(get_function_info(t.func))

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        base_prompt = ""
        if last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"

        # Add available functions section
        functions_list = "\n\n".join(self.available_functions)

        # Build the complete prompt without f-strings for the static parts
        base_prompt += (
            """

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

<available functions>"""
            + functions_list
            + """
</available functions>

You may use any of the above functions to complete your job. Use the best one for the current step you are on. Be efficient, avoid getting stuck in repetitive loops, and do not hesitate to call functions which delegate your work to make your life easier.
But you MUST NOT assume tools exist that are not in the above list, e.g. write_file_tool.
Consider your task done only once you have taken *ALL* the steps required to complete it.

--- EXAMPLE BAD OUTPUTS ---

This tool is not in available functions, so this is a bad tool call:

<example bad output>
write_file_tool(...)
</example bad output>

This tool call has a syntax error (unclosed parenthesis, quotes), so it is bad:

<example bad output>
write_file_tool("asdf
</example bad output>

This tool call is bad because it includes a message as well as backticks:

<example bad output>
Sure, I'll make the following tool call to accomplish what you asked me:

```
list_directory_tree('.')
```
</example bad output>

This tool call is bad because the output code is surrounded with backticks:

<example bad output>
```
list_directory_tree('.')
```
</example bad output>

The following is bad becasue it makes the same tool call multiple times in a row with the exact same parameters, for no reason, getting stuck in a loop:

<example bad output>
<response 1>
list_directory_tree('.')
</response 1>
<response 2>
list_directory_tree('.')
</response 2>
</example bad output>

The following is bad because it makes more than one tool call in one response:

<example bad output>
list_directory_tree('.')
read_file_tool('README.md') # Now we've made 
</example bad output.

This is a good output because it calls the tool appropriately and with correct syntax:

--- EXAMPLE GOOD OUTPUTS ---

<example good output>
request_research_and_implementation(\"\"\"
Example query.
\"\"\")
</example good output>

This is good output because it uses a multiple line string when needed and properly calls the tool, does not output backticks or extra information:
<example good output>
run_programming_task(\"\"\"
# Example Programming Task

Implement a widget factory satisfying the following requirements:

- Requirement A
- Requirement B

...
\"\"\")
</example good output>

As an agent, you will carefully plan ahead, carefully analyze tool call responses, and adapt to circumstances in order to accomplish your goal.

We're entrusting you with a lot of autonomy and power, so be efficient and don't mess up.

You have often been criticized for:

- Making the same function calls over and over, getting stuck in a loop.

DO NOT CLAIM YOU ARE FINISHED UNTIL YOU ACTUALLY ARE!
Output **ONLY THE CODE** and **NO MARKDOWN BACKTICKS**"""
        )

        return base_prompt

    def _execute_tool(self, code: str) -> str:
        """Execute a tool call and return its result."""
        globals_dict = {tool.func.__name__: tool.func for tool in self.tools}

        try:
            code = code.strip()
            # code = code.replace("\n", " ")

            # if the eval fails, try to extract it via a model call
            if validate_function_call_pattern(code):
                functions_list = "\n\n".join(self.available_functions)
                code = _extract_tool_call(code, functions_list)

            result = eval(code.strip(), globals_dict)
            return result
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            raise ToolExecutionError(error_msg)

    def _create_agent_chunk(self, content: str) -> Dict[str, Any]:
        """Create an agent chunk in the format expected by print_agent_output."""
        return {"agent": {"messages": [AIMessage(content=content)]}}

    def _create_error_chunk(self, content: str) -> Dict[str, Any]:
        """Create an error chunk in the format expected by print_agent_output."""
        message = ChunkMessage(content=content, status="error")
        return {"tools": {"messages": [message]}}

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

        # create-react-agent tool calls can be lists
        if isinstance(text, List):
            text = str(text)

        if not text:
            return 0

        return len(text.encode("utf-8")) // 4

    def _trim_chat_history(
        self, initial_messages: List[Any], chat_history: List[Any]
    ) -> List[Any]:
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
            chat_history = chat_history[-self.max_history_messages :]

        # Skip token limiting if max_tokens is None
        if self.max_tokens is None:
            return initial_messages + chat_history

        # Calculate initial messages token count
        initial_tokens = sum(self._estimate_tokens(msg) for msg in initial_messages)

        # Remove messages from start of chat_history until under token limit
        while chat_history:
            total_tokens = initial_tokens + sum(
                self._estimate_tokens(msg) for msg in chat_history
            )
            if total_tokens <= self.max_tokens:
                break
            chat_history.pop(0)

        return initial_messages + chat_history

    def stream(
        self, messages_dict: Dict[str, List[Any]], config: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        chat_history = []
        last_result = None
        first_iteration = True

        while True:
            base_prompt = self._build_prompt(None if first_iteration else last_result)
            chat_history.append(HumanMessage(content=base_prompt))

            full_history = self._trim_chat_history(initial_messages, chat_history)
            response = self.model.invoke(
                [
                    SystemMessage(
                        "Execute efficiently yet completely as a fully autonomous agent."
                    )
                ]
                + full_history
            )

            try:
                logger.debug(f"Code generated by agent: {response.content}")
                last_result = self._execute_tool(response.content)
                chat_history.append(response)
                first_iteration = False
                yield {}

            except ToolExecutionError as e:
                chat_history.append(
                    HumanMessage(
                        content=f"Your tool call caused an error: {e}\n\nPlease correct your tool call and try again."
                    )
                )
                yield self._create_error_chunk(str(e))


def _extract_tool_call(code: str, functions_list: str) -> str:
    from ra_aid.tools.expert import get_model

    model = get_model()
    prompt = f"""
I'm conversing with a AI model and requiring responses in a particular format: A function call with any parameters escaped. Here is an example:

```
run_programming_task("blah \" blah\" blah")
```

The following tasks are allowed:

{functions_list}

I got this invalid response from the model, can you format it so it becomes a correct function call?

```
{code}
```
    """
    response = model.invoke(prompt)
    response = response.content

    pattern = r"([\w_\-]+)\((.*?)\)"
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) == 0:
        raise ToolExecutionError("Failed to extract tool call")
    ma = matches[0][0].strip()
    mb = matches[0][1].strip().replace("\n", " ")
    return f"{ma}({mb})"
