import re
import ast
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ra_aid.config import DEFAULT_MAX_TOOL_FAILURES
from ra_aid.exceptions import ToolExecutionError
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.logging_config import get_logger
from ra_aid.models_params import DEFAULT_TOKEN_LIMIT
from ra_aid.prompts.ciayn_prompts import CIAYN_AGENT_SYSTEM_PROMPT, CIAYN_AGENT_HUMAN_PROMPT, EXTRACT_TOOL_CALL_PROMPT, NO_TOOL_CALL_PROMPT
from ra_aid.tools.expert import get_model
from ra_aid.tools.reflection import get_function_info
from ra_aid.console.output import cpm

logger = get_logger(__name__)


@dataclass
class ChunkMessage:
    content: str
    status: str


def validate_function_call_pattern(s: str) -> bool:
    """Check if a string matches the expected function call pattern.

    Validates that the string represents a valid function call using AST parsing.
    Valid function calls must be syntactically valid Python code.

    Args:
        s: String to validate

    Returns:
        bool: False if pattern matches (valid), True if invalid
    """
    # Clean up the code before parsing
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = s[3:-3].strip()
    elif s.startswith("```"):
        s = s[3:].strip()
    elif s.endswith("```"):
        s = s[:-3].strip()
    
    # Check for multiple function calls - this can't be handled by AST parsing alone
    if re.search(r'\)\s*[\w\-]+\s*\(', s):
        return True  # Invalid - contains multiple function calls
    
    # Use AST parsing as the single validation method
    try:
        tree = ast.parse(s)
        
        # Valid pattern is a single expression that's a function call
        if (len(tree.body) == 1 and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Call)):
            
            return False  # Valid function call
        
        return True  # Invalid pattern
    
    except Exception:
        # Any exception during parsing means it's not valid
        return True


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

    # List of tools that can be bundled (called multiple times in one response)
    BUNDLEABLE_TOOLS = [
        "emit_expert_context", 
        "ask_expert", 
        "emit_key_facts", 
        "emit_key_snippet",
        "request_implementation",
        "read_file_tool",
        "emit_research_notes",
        "ripgrep_search"
    ]

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

        self.fallback_handler = FallbackHandler(config, tools)
        
        # Include the functions list in the system prompt
        functions_list = "\n\n".join(self.available_functions)
        self.sys_message = SystemMessage(
            CIAYN_AGENT_SYSTEM_PROMPT.format(functions_list=functions_list)
        )
        
        self.error_message_template = "Your tool call caused an error: {e}\n\nPlease correct your tool call and try again."
        self.fallback_fixed_msg = HumanMessage(
            "Fallback tool handler has fixed the tool call see: <fallback tool call result> for the output."
        )

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        # Add last result section if provided
        last_result_section = ""
        if last_result is not None:
            last_result_section = f"\n<last result>{last_result}</last result>"
        
        # Build the human prompt without the function list
        return CIAYN_AGENT_HUMAN_PROMPT.format(
            last_result_section=last_result_section
        )

    def _detect_multiple_tool_calls(self, code: str) -> List[str]:
        """Detect if there are multiple tool calls in the code using AST parsing.
        
        Args:
            code: The code string to analyze
            
        Returns:
            List of individual tool call strings if bundleable, or just the original code as a single element
        """
        try:
            # Clean up the code for parsing
            code = code.strip()
            if code.startswith("```"):
                code = code[3:].strip()
            if code.endswith("```"):
                code = code[:-3].strip()
                
            # Try to parse the code as a sequence of expressions
            parsed = ast.parse(code)
            
            # Check if we have multiple expressions and they are all valid function calls
            if isinstance(parsed.body, list) and len(parsed.body) > 1:
                calls = []
                for node in parsed.body:
                    # Only process expressions that are function calls
                    if (isinstance(node, ast.Expr) and 
                        isinstance(node.value, ast.Call) and 
                        isinstance(node.value.func, ast.Name)):
                        
                        func_name = node.value.func.id
                        
                        # Only consider this a bundleable call if the function is in our allowed list
                        if func_name in self.BUNDLEABLE_TOOLS:
                            # Extract the exact call text from the original code
                            call_str = ast.unparse(node)
                            calls.append(call_str)
                        else:
                            # If any function is not bundleable, return just the original code
                            cpm(
                                f"Found multiple tool calls, but {func_name} is not bundleable.",
                                title="⚠ Non-bundleable Tools",
                                border_style="yellow"
                            )
                            return [code]
                
                if calls:
                    cpm(
                        f"Detected {len(calls)} bundleable tool calls.",
                        title="✓ Bundling Tools",
                        border_style="green"
                    )
                    return calls
            
            # Default case: just return the original code as a single element
            return [code]
            
        except SyntaxError:
            # If we can't parse the code with AST, just return the original
            return [code]

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

            # Check for multiple tool calls that can be bundled
            tool_calls = self._detect_multiple_tool_calls(code)
            
            # If we have multiple valid bundleable calls, execute them in sequence
            if len(tool_calls) > 1:
                results = []
                for call in tool_calls:
                    # Validate and fix each call if needed
                    if validate_function_call_pattern(call):
                        functions_list = "\n\n".join(self.available_functions)
                        call = self._extract_tool_call(call, functions_list)
                    
                    # Execute the call and collect the result
                    result = eval(call.strip(), globals_dict)
                    results.append(result)
                
                # Return the result of the last tool call
                return results[-1]
            
            # Regular single tool call case
            if validate_function_call_pattern(code):
                cpm(
                    f"Tool call validation failed. Attempting to extract function call using LLM.",
                    title="⚠ Validation Warning",
                    border_style="yellow"
                )
                functions_list = "\n\n".join(self.available_functions)
                code = self._extract_tool_call(code, functions_list)
                pass

            result = eval(code.strip(), globals_dict)
            return result
        except Exception as e:
            error_msg = f"Error: {str(e)} \n Could not execute code: {code}"
            tool_name = self.extract_tool_name(code)
            cpm(
                f"Tool execution failed for `{tool_name}`: {str(e)}",
                title="❗ Tool Error",
                border_style="red"
            )
            raise ToolExecutionError(
                error_msg, base_message=msg, tool_name=tool_name
            ) from e

    def extract_tool_name(self, code: str) -> str:
        """Extract the tool name from the code."""
        match = re.match(r"\s*([\w_\-]+)\s*\(", code)
        if match:
            return match.group(1)
        return ""

    def handle_fallback_response(
        self, fallback_response: list[Any], e: ToolExecutionError
    ) -> str:
        """Handle a fallback response from the fallback handler."""
        err_msg = HumanMessage(content=self.error_message_template.format(e=e))

        if not fallback_response:
            self.chat_history.append(err_msg)
            cpm(
                f"Tool fallback was attempted but did not succeed. Original error: {str(e)}",
                title="❗ Fallback Failed",
                border_style="red bold"
            )
            return ""

        self.chat_history.append(self.fallback_fixed_msg)
        msg = f"Fallback tool handler has triggered after consecutive failed tool calls reached {DEFAULT_MAX_TOOL_FAILURES} failures.\n"
        # Passing the fallback raw invocation may confuse our llm, as invocation methods may differ.
        # msg += f"<fallback llm raw invocation>{fallback_response[0]}</fallback llm raw invocation>\n"
        msg += f"<fallback tool name>{e.tool_name}</fallback tool name>\n"
        msg += f"<fallback tool call result>\n{fallback_response[1]}\n</fallback tool call result>\n"
        
        cpm(
            f"Fallback successful for tool `{e.tool_name}` after {DEFAULT_MAX_TOOL_FAILURES} consecutive failures.",
            title="✓ Fallback Success",
            border_style="green"
        )
        
        return msg

    def _create_agent_chunk(self, content: str) -> Dict[str, Any]:
        """Create an agent chunk in the format expected by print_agent_output."""
        return {"agent": {"messages": [AIMessage(content=content)]}}

    def _create_error_chunk(self, error_message: str) -> Dict[str, Any]:
        """Create an error chunk for the agent output stream."""
        return {
            "type": "error",
            "message": error_message,
            "tool_call": {
                "name": "report_error",
                "args": {"error": error_message},
            },
        }

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

    @staticmethod
    def _estimate_tokens(content: Optional[Union[str, BaseMessage]]) -> int:
        """Estimate token count for a message or string."""
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
        """Extract a tool call from the code using a language model."""
        model = get_model()
        cpm(
            f"Attempting to fix malformed tool call using LLM. Original code:\n```\n{code}\n```",
            title="🔧 Tool Call Extraction",
            border_style="blue"
        )
        prompt = EXTRACT_TOOL_CALL_PROMPT.format(
            functions_list=functions_list, code=code
        )
        response = model.invoke(prompt)
        response = response.content

        pattern = r"([\w_\-]+)\((.*?)\)"
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) == 0:
            cpm(
                "Failed to extract a valid tool call from the model's response.",
                title="❗ Extraction Failed", 
                border_style="red"
            )
            raise ToolExecutionError("Failed to extract tool call")
        ma = matches[0][0].strip()
        mb = matches[0][1].strip().replace("\n", " ")
        fixed_code = f"{ma}({mb})"
        cpm(
            f"Successfully extracted tool call: `{fixed_code}`",
            title="✓ Extraction Success",
            border_style="green"
        )
        return fixed_code

    def stream(
        self, messages_dict: Dict[str, List[Any]], _config: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        self.chat_history = []
        last_result = None
        empty_response_count = 0
        max_empty_responses = 3  # Maximum number of consecutive empty responses before giving up

        while True:
            base_prompt = self._build_prompt(last_result)
            self.chat_history.append(HumanMessage(content=base_prompt))
            full_history = self._trim_chat_history(initial_messages, self.chat_history)
            response = self.model.invoke([self.sys_message] + full_history)

            # Check if the response is empty or doesn't contain a valid tool call
            if not response.content or not response.content.strip():
                empty_response_count += 1
                logger.warning(f"Model returned empty response (count: {empty_response_count})")
                
                cpm(
                    f"The model returned an empty response (attempt {empty_response_count} of {max_empty_responses}). Requesting the model to make a valid tool call.",
                    title="⚠ Empty Response",
                    border_style="yellow bold"
                )
                
                if empty_response_count >= max_empty_responses:
                    # If we've had too many empty responses, raise an error to break the loop
                    from ra_aid.agent_context import mark_agent_crashed
                    crash_message = "Agent failed to make any tool calls after multiple attempts"
                    mark_agent_crashed(crash_message)
                    logger.error(crash_message)
                    
                    cpm(
                        "The agent has crashed after multiple failed attempts to generate a valid tool call.",
                        title="❗ Agent Crashed",
                        border_style="red bold"
                    )
                    
                    yield self._create_error_chunk(crash_message)
                    return
                
                # Send a message to the model explicitly telling it to make a tool call
                self.chat_history.append(AIMessage(content=""))  # Add the empty response
                self.chat_history.append(HumanMessage(content=NO_TOOL_CALL_PROMPT))
                continue
            
            # Reset empty response counter on successful response
            empty_response_count = 0

            try:
                last_result = self._execute_tool(response)
                self.chat_history.append(response)
                self.fallback_handler.reset_fallback_handler()
                yield {}

            except ToolExecutionError as e:
                cpm(
                    f"Tool execution error: {str(e)}. Attempting fallback...",
                    title="↻ Fallback Attempt",
                    border_style="yellow"
                )
                fallback_response = self.fallback_handler.handle_failure(
                    e, self, self.chat_history
                )
                last_result = self.handle_fallback_response(fallback_response, e)
                yield {}