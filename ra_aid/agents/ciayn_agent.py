import re
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ra_aid.config import DEFAULT_MAX_TOOL_FAILURES
from ra_aid.console.output import cpm
from ra_aid.exceptions import ToolExecutionError
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.logging_config import get_logger
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT
from ra_aid.prompts import CIAYN_AGENT_BASE_PROMPT, EXTRACT_TOOL_CALL_PROMPT
from ra_aid.tools.expert import get_model
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
        model: BaseChatModel,
        tools: list[BaseTool],
        max_history_messages: int = 50,
        max_tokens: Optional[int] = DEFAULT_TOKEN_LIMIT,
        config: Optional[dict] = None,
    ):
        """Initialize the agent with a model and list of tools.

        Args:
            model: The language model to use
            tools: List of tools available to the agent
            max_history_messages: Maximum number of messages to keep in chat history
            max_tokens: Maximum number of tokens allowed in message history (None for no limit)
            config: Optional configuration dictionary
        """
        if config is None:
            config = {}
        self.config = config
        self.provider = config.get("provider", "openai")

        self.model = model
        self.tools = tools
        self.max_history_messages = max_history_messages
        self.max_tokens = max_tokens
        self.chat_history = []
        self.available_functions = []
        for t in tools:
            self.available_functions.append(get_function_info(t.func))

        self.tool_failure_current_provider = None
        self.tool_failure_current_model = None
        self.fallback_handler = FallbackHandler(config, tools)
        self.sys_message = SystemMessage(
            "Execute efficiently yet completely as a fully autonomous agent."
        )
        self.error_message_template = "Your tool call caused an error: {e}\n\nPlease correct your tool call and try again."

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        base_prompt = ""

        if last_result is not None:
            base_prompt += f"\n<last result>{last_result}</last result>"

        # Add available functions section
        functions_list = "\n\n".join(self.available_functions)

        # Build the complete prompt without f-strings for the static parts
        base_prompt += CIAYN_AGENT_BASE_PROMPT.format(functions_list=functions_list)

        # base_prompt += "\n\nYou must reply with ONLY ONE of the functions given in available functions."

        return base_prompt

    def _execute_tool(self, msg: BaseMessage) -> str:
        """Execute a tool call and return its result."""

        code = msg.content
        globals_dict = {tool.func.__name__: tool.func for tool in self.tools}

        try:
            code = code.strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            # logger.debug(f"_execute_tool: stripped code: {code}")

            # if the eval fails, try to extract it via a model call
            if validate_function_call_pattern(code):
                functions_list = "\n\n".join(self.available_functions)
                code = self._extract_tool_call(code, functions_list)
                pass

            result = eval(code.strip(), globals_dict)
            return result
        except Exception as e:
            error_msg = f"Error: {str(e)} \n Could not excute code: {code}"
            tool_name = self.extract_tool_name(code)
            raise ToolExecutionError(error_msg, base_message=msg, tool_name=tool_name)

    def extract_tool_name(self, code: str) -> str:
        match = re.match(r"\s*([\w_\-]+)\s*\(", code)
        if match:
            return match.group(1)
        return ""

    def handle_fallback_response(
        self, fallback_response: list[Any], e: ToolExecutionError
    ) -> str:
        err_msg = HumanMessage(content=self.error_message_template.format(e=e))
        self.chat_history.append(err_msg)

        if not fallback_response:
            return ""

        msg = f"Fallback tool handler has triggered after consecutive failed tool calls reached {DEFAULT_MAX_TOOL_FAILURES} failures.\n"
        # Passing the fallback raw invocation may confuse our llm, as invocation methods may differ.
        # msg += f"<fallback llm raw invocation>{fallback_response[0]}</fallback llm raw invocation>\n"
        msg += f"<fallback tool name>{e.tool_name}</fallback tool name>"
        msg += f"<fallback tool call result>{fallback_response[1]}</fallback tool call result>"
        return msg

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
        Estimates 1 token per 2.0 bytes of content. For messages, uses the content field.

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

        return len(text.encode("utf-8")) // 2.0

    def _extract_tool_call(self, code: str, functions_list: str) -> str:
        model = get_model()
        prompt = EXTRACT_TOOL_CALL_PROMPT.format(
            functions_list=functions_list, code=code
        )
        response = model.invoke(prompt)
        response = response.content

        pattern = r"([\w_\-]+)\((.*?)\)"
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) == 0:
            raise ToolExecutionError("Failed to extract tool call")
        ma = matches[0][0].strip()
        mb = matches[0][1].strip().replace("\n", " ")
        logger.debug(f"Extracted tool call: {ma}({mb})")
        return f"{ma}({mb})"

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
        self, messages_dict: Dict[str, List[Any]], _config: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        self.chat_history = []
        last_result = None

        while True:
            base_prompt = self._build_prompt(last_result)
            self.chat_history.append(HumanMessage(content=base_prompt))
            full_history = self._trim_chat_history(initial_messages, self.chat_history)
            response = self.model.invoke([self.sys_message] + full_history)

            try:
                last_result = self._execute_tool(response)
                self.chat_history.append(response)
                self.fallback_handler.reset_fallback_handler()
                yield {}

            except ToolExecutionError as e:
                fallback_response = self.fallback_handler.handle_failure(
                    e, self, self.chat_history
                )
                last_result = self.handle_fallback_response(fallback_response, e)
                yield {}
