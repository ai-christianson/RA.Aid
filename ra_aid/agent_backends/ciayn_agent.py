import re
import ast
import string
import random
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from ra_aid.callbacks.default_callback_handler import (
    initialize_callback_handler,
)
from ra_aid.config import DEFAULT_MAX_TOOL_FAILURES
from ra_aid.exceptions import ToolExecutionError
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.logging_config import get_logger

# ADDED IMPORT
from ra_aid.models_params import (
    models_params,
    DEFAULT_TOKEN_LIMIT,
)  # Need DEFAULT_TOKEN_LIMIT too
from ra_aid.prompts.ciayn_prompts import (
    CIAYN_AGENT_SYSTEM_PROMPT,
)
from ra_aid.tools.reflection import get_function_info
from ra_aid.tool_configs import CUSTOM_TOOLS
import ra_aid.console.formatting
from ra_aid.agent_context import should_exit
from ra_aid.text.processing import process_thinking_content
from ra_aid.text import fix_triple_quote_contents

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

    # Handle markdown code blocks more comprehensively
    if s.startswith("```"):
        # Extract the content between the backticks
        lines = s.split("\n")
        # Remove first line (which may contain ```python or just ```)
        lines = lines[1:] if len(lines) > 1 else []
        # Remove last line if it contains closing backticks
        if lines and "```" in lines[-1]:
            lines = lines[:-1]
        # Rejoin the content
        s = "\n".join(lines).strip()

    # Use AST parsing as the single validation method
    try:
        tree = ast.parse(s)

        # Valid pattern is a single expression that's a function call
        if (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Call)
        ):

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

    # List of tools that can be bundled together in a single response
    BUNDLEABLE_TOOLS = [
        "emit_expert_context",
        "ask_expert",
        "emit_key_facts",
        "emit_key_snippet",
        "request_implementation",
        "read_file_tool",
        "emit_research_notes",
        "ripgrep_search",
        "plan_implementation_completed",
        "request_research_and_implementation",
        "run_shell_command",
    ]

    # List of tools that should not be called repeatedly with the same parameters
    # This prevents the agent from getting stuck in a loop calling the same tool
    # with the same arguments multiple times
    NO_REPEAT_TOOLS = [
        "emit_expert_context",
        "ask_expert",
        "emit_key_facts",
        "emit_key_snippet",
        "request_implementation",
        "read_file_tool",
        "emit_research_notes",
        "ripgrep_search",
        "plan_implementation_completed",
        "request_research_and_implementation",
        "run_shell_command",
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

        self.callback_handler, self.stream_config = initialize_callback_handler(
            model=self.model,
            track_cost=self.config.get("track_cost", True),
        )

        # Include the functions list in the system prompt
        functions_list = "\\n\\n".join(self.available_functions)
        # Use  HumanMessage because not all models support SystemMessage
        self.sys_message = HumanMessage(
            CIAYN_AGENT_SYSTEM_PROMPT.format(functions_list=functions_list)
        )

        self.error_message_template = "Your tool call caused an error: {e}\\n\\nPlease correct your tool call and try again."
        self.fallback_fixed_msg = HumanMessage(
            "Fallback tool handler has fixed the tool call see: <fallback tool call result> for the output."
        )

        # Track the most recent tool call and parameters to prevent repeats
        # This is used to detect and prevent identical tool calls with the same parameters
        # to avoid redundant operations and encourage the agent to try different approaches
        self.last_tool_call = None
        self.last_tool_params = None

    def _build_prompt(self, last_result: Optional[str] = None) -> str:
        """Build the prompt for the agent including available tools and context."""
        # Add last result section if provided
        last_result_section = ""
        if last_result is not None:
            last_result_section = f"\\n<last result>{last_result}</last result>"

        return last_result_section

    def strip_code_markup(self, code: str) -> str:
        """
        Strips markdown code block markup from a string.

        Handles cases for:
        - Code blocks with language specifiers (```python)
        - Code blocks without language specifiers (```)
        - Code surrounded with single backticks (`)

        Args:
            code: The string potentially containing code markup

        Returns:
            The code with markup removed
        """
        code = code.strip()

        # Check for code blocks with any language specifier
        if code.startswith("```") and not code.startswith("``` "):
            # Extract everything after the first newline to skip the language specifier
            first_newline = code.find("\n")
            if first_newline != -1:
                code = code[first_newline + 1 :].strip()
            else:
                # If there's no newline, just remove the backticks and handle special cases
                # like language specifiers with spaces
                code = code[3:].strip()
                if code.endswith("```"):
                    code = code[:-3].strip()

                # Try to detect language specifier followed by space
                import re

                match = re.match(r"^([a-zA-Z0-9_\-+]+)(\s+)(.*)", code)
                if match:
                    # Only remove language specifier if there's a clear space delimiter
                    # This preserves behavior for cases like "pythonprint" where there's no clear
                    # way to separate the language from the code
                    lang, space, remaining = match.groups()
                    if remaining:  # Make sure there's content after the space
                        code = remaining
        # Additional check for simple code blocks without language
        elif code.startswith("```"):
            code = code[3:].strip()
        # Check for code surrounded with single backticks (`)
        elif code.startswith("`") and not code.startswith("``"):
            code = code[1:].strip()

        if code.endswith("```"):
            code = code[:-3].strip()
        # Check for code ending with single backtick
        elif code.endswith("`") and not code.endswith("``"):
            code = code[:-1].strip()

        return code

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
                    if (
                        isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                    ):

                        func_name = node.value.func.id

                        # Only consider this a bundleable call if the function is in our allowed list
                        if func_name in self.BUNDLEABLE_TOOLS:
                            # Extract the exact call text from the original code
                            call_str = ast.unparse(node)
                            calls.append(call_str)
                        else:
                            # If any function is not bundleable, return just the original code
                            logger.debug(
                                f"Found multiple tool calls, but {func_name} is not bundleable."
                            )
                            return [code]

                if calls:
                    logger.debug(f"Detected {len(calls)} bundleable tool calls.")
                    return calls

            # Default case: just return the original code as a single element
            return [code]

        except SyntaxError:
            # If we can't parse the code with AST, just return the original
            return [code]

    def _execute_tool(self, msg: BaseMessage) -> str:
        """Execute a tool call and return its result."""

        # Check for should_exit before executing tool calls
        if should_exit():
            logger.debug("Agent should exit flag detected in _execute_tool")
            return "Tool execution aborted - agent should exit flag is set"

        code = msg.content
        globals_dict = {tool.func.__name__: tool.func for tool in self.tools}

        try:
            code = self.strip_code_markup(code)
            
            # Only call fix_triple_quote_contents if:
            # 1. The code is not valid Python AND
            # 2. The first line includes "put_complete_file_contents"
            is_valid_python = True
            try:
                ast.parse(code)
            except SyntaxError:
                is_valid_python = False
            
            # Check if first line includes "put_complete_file_contents"
            first_line = code.splitlines()[0] if code.splitlines() else ""
            contains_put_complete = "put_complete_file_contents" in first_line
            
            if not is_valid_python and contains_put_complete:
                code = fix_triple_quote_contents(code)

            # Check for multiple tool calls that can be bundled
            tool_calls = self._detect_multiple_tool_calls(code)

            # If we have multiple valid bundleable calls, execute them in sequence
            if len(tool_calls) > 1:
                # Check for should_exit before executing bundled tool calls
                if should_exit():
                    logger.debug(
                        "Agent should exit flag detected before executing bundled tool calls"
                    )
                    return (
                        "Bundled tool execution aborted - agent should exit flag is set"
                    )

                results = []
                result_strings = []

                for call in tool_calls:
                    # Check if agent should exit
                    if should_exit():
                        logger.debug(
                            "Agent should exit flag detected during bundled tool execution"
                        )
                        return (
                            "Tool execution interrupted: agent_should_exit flag is set."
                        )

                    # Validate and fix each call if needed (using conditional extraction)
                    if validate_function_call_pattern(call):
                        provider = self.config.get("provider", "")
                        model_name = self.config.get("model", "")
                        model_config = models_params.get(provider, {}).get(
                            model_name, {}
                        )
                        attempt_extraction = model_config.get(
                            "attempt_llm_tool_extraction", False
                        )

                        if attempt_extraction:
                            logger.info(
                                f"Bundled call validation failed. Attempting extraction for: {call}"
                            )
                            ra_aid.console.formatting.print_warning(
                                "Bundled call validation failed. Attempting LLM-based tool call extraction.",
                                title="Bundled Call Validation",
                            )
                            functions_list = "\\n\\n".join(self.available_functions)
                            try:
                                call = self._extract_tool_call(call, functions_list)
                            except ToolExecutionError as extraction_error:
                                raise extraction_error  # Propagate extraction errors
                        else:
                            logger.info(
                                f"Invalid bundled tool call format detected and LLM extraction is disabled. Call: {call}"
                            )
                            warning_message = f"Invalid bundled tool call format detected:\\n```\\n{call}\\n```\\nLLM extraction is disabled. Skipping this call."
                            # Add an error message to results instead of raising, to allow other bundled calls to proceed
                            error_msg = f"Invalid tool call format and LLM extraction is disabled. Call: {call}"
                            results.append(f"Error: {error_msg}")
                            result_id = self._generate_random_id()
                            result_strings.append(
                                f"<result-{result_id}>\\nError: tool call was not structured correctly. Re-read the instructions, carefully consider what went wrong, and try again with a *CORRECT AND COMPLETE* tool call.\\n</result-{result_id}>"
                            )
                            continue  # Skip executing this invalid call

                    # Check for repeated tool calls with the same parameters
                    tool_name = self.extract_tool_name(call)

                    if tool_name in self.NO_REPEAT_TOOLS:
                        # Use AST to extract parameters
                        try:
                            tree = ast.parse(call)
                            if isinstance(tree.body[0], ast.Expr) and isinstance(
                                tree.body[0].value, ast.Call
                            ):

                                # Debug - print full AST structure
                                logger.debug(
                                    f"AST structure for bundled call: {ast.dump(tree.body[0].value)}"
                                )

                                # Extract and normalize parameter values
                                param_pairs = []

                                # Handle positional arguments
                                if tree.body[0].value.args:
                                    logger.debug(
                                        f"Found positional args in bundled call: {[ast.unparse(arg) for arg in tree.body[0].value.args]}"
                                    )

                                    for i, arg in enumerate(tree.body[0].value.args):
                                        arg_value = ast.unparse(arg)

                                        # Normalize string literals by removing outer quotes
                                        if (
                                            arg_value.startswith("'")
                                            and arg_value.endswith("'")
                                        ) or (
                                            arg_value.startswith('"')
                                            and arg_value.endswith('"')
                                        ):
                                            arg_value = arg_value[1:-1]

                                        param_pairs.append((f"arg{i}", arg_value))

                                # Handle keyword arguments
                                for k in tree.body[0].value.keywords:
                                    param_name = k.arg
                                    param_value = ast.unparse(k.value)

                                    # Debug - print each parameter
                                    logger.debug(
                                        f"Processing parameter: {param_name} = {param_value}"
                                    )

                                    # Normalize string literals by removing outer quotes
                                    if (
                                        param_value.startswith("'")
                                        and param_value.endswith("'")
                                    ) or (
                                        param_value.startswith('"')
                                        and param_value.endswith('"')
                                    ):
                                        param_value = param_value[1:-1]

                                    param_pairs.append((param_name, param_value))

                                # Debug - print extracted parameters
                                logger.debug(f"Extracted parameters: {param_pairs}")

                                # Create a fingerprint of the call
                                current_call = (tool_name, str(sorted(param_pairs)))

                                # Debug information to help diagnose false positives
                                logger.debug(
                                    f"Tool call: {tool_name}\\nCurrent call fingerprint: {current_call}\\nLast call fingerprint: {self.last_tool_call}"
                                )

                                # If this fingerprint matches the last tool call, reject it
                                if current_call == self.last_tool_call:
                                    logger.info(
                                        f"Detected repeat call of {tool_name} with the same parameters."
                                    )
                                    result = f"Repeat calls of {tool_name} with the same parameters are not allowed. You must try something different!"
                                    results.append(result)

                                    # Generate a random ID for this result
                                    result_id = self._generate_random_id()
                                    result_strings.append(
                                        f"<result-{result_id}>\n{result}\n</result-{result_id}>"
                                    )
                                    continue

                                # Update last tool call fingerprint for next comparison
                                self.last_tool_call = current_call
                        except Exception as e:
                            # If we can't parse parameters, just continue
                            # This ensures robustness when dealing with complex or malformed tool calls
                            logger.debug(
                                f"Failed to parse parameters for duplicate detection: {str(e)}"
                            )
                            pass

                    # Execute the call and collect the result
                    result = eval(call.strip(), globals_dict)
                    results.append(result)

                    # Generate a random ID for this result
                    result_id = self._generate_random_id()
                    result_strings.append(
                        f"<result-{result_id}>\n{result}\n</result-{result_id}>"
                    )

                # Return all results as one big string with tagged sections
                return "\n\n".join(result_strings)

            # Regular single tool call case
            if validate_function_call_pattern(code):
                # Retrieve the configuration flag
                provider = self.config.get("provider", "")
                model_name = self.config.get("model", "")
                model_config = models_params.get(provider, {}).get(model_name, {})
                attempt_extraction = model_config.get(
                    "attempt_llm_tool_extraction", False
                )

                if attempt_extraction:
                    logger.warning(
                        "Tool call validation failed. Attempting to extract function call using LLM."
                    )
                    ra_aid.console.formatting.print_warning(
                        "Tool call validation failed. Attempting to extract function call using LLM.",
                        title="Tool Validation Error",
                    )
                    functions_list = "\\n\\n".join(self.available_functions)
                    # Handle potential errors during extraction itself
                    try:
                        code = self._extract_tool_call(code, functions_list)
                    except ToolExecutionError as extraction_error:
                        # If extraction fails, re-raise the error to be caught by the main loop
                        raise extraction_error
                else:
                    logger.info(
                        f"Invalid tool call format detected and LLM extraction is disabled for this model. Code: {code}"
                    )

                    error_msg = (
                        "Invalid tool call format and LLM extraction is disabled."
                    )
                    # Try to get tool name for better error reporting, default if fails
                    tool_name = self.extract_tool_name(code) or "unknown_tool_format"
                    # Use the original message `msg` available in the scope
                    raise ToolExecutionError(
                        error_msg, base_message=msg, tool_name=tool_name
                    )

            # Check for repeated tool call with the same parameters (single tool case)
            tool_name = self.extract_tool_name(code)

            # If the tool is in the NO_REPEAT_TOOLS list, check for repeat calls
            if tool_name in self.NO_REPEAT_TOOLS:
                # Use AST to extract parameters
                try:
                    tree = ast.parse(code)
                    if isinstance(tree.body[0], ast.Expr) and isinstance(
                        tree.body[0].value, ast.Call
                    ):

                        # Debug - print full AST structure
                        logger.debug(
                            f"AST structure for single call: {ast.dump(tree.body[0].value)}"
                        )

                        # Extract and normalize parameter values
                        param_pairs = []

                        # Handle positional arguments
                        if tree.body[0].value.args:
                            logger.debug(
                                f"Found positional args in single call: {[ast.unparse(arg) for arg in tree.body[0].value.args]}"
                            )

                            for i, arg in enumerate(tree.body[0].value.args):
                                arg_value = ast.unparse(arg)

                                # Normalize string literals by removing outer quotes
                                if (
                                    arg_value.startswith("'")
                                    and arg_value.endswith("'")
                                ) or (
                                    arg_value.startswith('"')
                                    and arg_value.endswith('"')
                                ):
                                    arg_value = arg_value[1:-1]

                                param_pairs.append((f"arg{i}", arg_value))

                        # Handle keyword arguments
                        for k in tree.body[0].value.keywords:
                            param_name = k.arg
                            param_value = ast.unparse(k.value)

                            # Debug - print each parameter
                            logger.debug(
                                f"Processing parameter: {param_name} = {param_value}"
                            )

                            # Normalize string literals by removing outer quotes
                            if (
                                param_value.startswith("'")
                                and param_value.endswith("'")
                            ) or (
                                param_value.startswith('"')
                                and param_value.endswith('"')
                            ):
                                param_value = param_value[1:-1]

                            param_pairs.append((param_name, param_value))

                        # Also check for positional arguments
                        if tree.body[0].value.args:
                            logger.debug(
                                f"Found positional args: {[ast.unparse(arg) for arg in tree.body[0].value.args]}"
                            )

                        # Create a fingerprint of the call
                        current_call = (tool_name, str(sorted(param_pairs)))

                        # Debug information to help diagnose false positives
                        logger.debug(
                            f"Tool call: {tool_name}\\nCurrent call fingerprint: {current_call}\\nLast call fingerprint: {self.last_tool_call}"
                        )

                        # If this fingerprint matches the last tool call, reject it
                        if current_call == self.last_tool_call:
                            logger.info(
                                f"Detected repeat call of {tool_name} with the same parameters."
                            )
                            return f"Repeat calls of {tool_name} with the same parameters are not allowed. You must try something different!"

                        # Update last tool call fingerprint for next comparison
                        self.last_tool_call = current_call
                except Exception as e:
                    # If we can't parse parameters, just continue with the tool execution
                    # This ensures robustness when dealing with complex or malformed tool calls
                    logger.debug(
                        f"Failed to parse parameters for duplicate detection: {str(e)}"
                    )
                    pass

            # Before executing the call
            if should_exit():
                logger.debug("Agent should exit flag detected before tool execution")
                return "Tool execution interrupted: agent_should_exit flag is set."

            # Retrieve tool name
            tool_name = self.extract_tool_name(code)

            # Check if this is a custom tool and print output
            is_custom_tool = tool_name in [tool.name for tool in CUSTOM_TOOLS]

            # Execute tool
            result = eval(code.strip(), globals_dict)

            # Only display console output for custom tools
            if is_custom_tool:
                custom_tool_output = f"Executing custom tool: {tool_name}\\n"
                custom_tool_output += f"\\n\tResult: {result}"
                ra_aid.console.formatting.console.print(
                    ra_aid.console.formatting.Panel(
                        ra_aid.console.formatting.Markdown(custom_tool_output.strip()),
                        title=" Custom Tool",
                        border_style="magenta",
                    )
                )

            return result
        except Exception as e:
            error_msg = f"Error: {str(e)} \\n Could not execute code: {code}"
            tool_name = self.extract_tool_name(code)
            logger.info(f"Tool execution failed for `{tool_name}`: {str(e)}")

            # Record error in trajectory
            try:
                # Import here to avoid circular imports
                from ra_aid.database.repositories.trajectory_repository import (
                    TrajectoryRepository,
                )
                from ra_aid.database.repositories.human_input_repository import (
                    HumanInputRepository,
                )
                from ra_aid.database.connection import get_db

                # Create repositories directly
                trajectory_repo = TrajectoryRepository(get_db())
                human_input_repo = HumanInputRepository(get_db())
                human_input_id = human_input_repo.get_most_recent_id()

                trajectory_repo.create(
                    step_data={
                        "error_message": f"Tool execution failed for `{tool_name}`:\\nError: {str(e)}",
                        "display_title": "Tool Error",
                        "code": code,
                        "tool_name": tool_name,
                    },
                    record_type="tool_execution",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=str(e),
                    error_type="ToolExecutionError",
                    tool_name=tool_name,
                    tool_parameters={"code": code},
                )
            except Exception as trajectory_error:
                # Just log and continue if there's an error in trajectory recording
                logger.error(
                    f"Error recording trajectory for tool error display: {trajectory_error}"
                )

            ra_aid.console.formatting.print_warning(
                f"Tool execution failed for `{tool_name if tool_name else 'unknown'}`:\nError: {str(e)}\n\nCode:\n\n````\n{code}\n````",
                title="Tool Error",
            )
            # Re-raise the original error if it's already a ToolExecutionError and we didn't modify it
            if isinstance(e, ToolExecutionError) and not (
                "error_msg" in locals()
                and e.base_message == error_msg  # Check if we created a new error msg
            ):
                raise e
            # Otherwise, raise a new ToolExecutionError
            else:
                raise ToolExecutionError(
                    error_msg, base_message=msg, tool_name=tool_name
                ) from e

    def _generate_random_id(self, length: int = 6) -> str:
        """Generate a random ID string for result tagging.

        Args:
            length: Length of the random ID to generate

        Returns:
            String of random alphanumeric characters
        """
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choice(chars) for _ in range(length))

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
            logger.info(
                f"Tool fallback was attempted but did not succeed. Original error: {str(e)}"
            )

            # Record error in trajectory
            try:
                # Import here to avoid circular imports
                from ra_aid.database.repositories.trajectory_repository import (
                    TrajectoryRepository,
                )
                from ra_aid.database.repositories.human_input_repository import (
                    HumanInputRepository,
                )
                from ra_aid.database.connection import get_db

                # Create repositories directly
                trajectory_repo = TrajectoryRepository(get_db())
                human_input_repo = HumanInputRepository(get_db())
                human_input_id = human_input_repo.get_most_recent_id()

                trajectory_repo.create(
                    step_data={
                        "error_message": f"Tool fallback was attempted but did not succeed. Original error: {str(e)}",
                        "display_title": "Fallback Failed",
                        "tool_name": (
                            e.tool_name if hasattr(e, "tool_name") else "unknown_tool"
                        ),
                    },
                    record_type="error",
                    human_input_id=human_input_id,
                    is_error=True,
                    error_message=str(e),
                    error_type="FallbackFailedError",
                    tool_name=(
                        e.tool_name if hasattr(e, "tool_name") else "unknown_tool"
                    ),
                )
            except Exception as trajectory_error:
                # Just log and continue if there's an error in trajectory recording
                logger.error(
                    f"Error recording trajectory for fallback failed warning: {trajectory_error}"
                )

            ra_aid.console.formatting.print_warning(
                f"Tool fallback was attempted but did not succeed. Original error: {str(e)}",
                title="Fallback Failed",
            )
            return ""

        self.chat_history.append(self.fallback_fixed_msg)
        msg = f"Fallback tool handler has triggered after consecutive failed tool calls reached {DEFAULT_MAX_TOOL_FAILURES} failures.\\n"
        # Passing the fallback raw invocation may confuse our llm, as invocation methods may differ.
        # msg += f"<fallback llm raw invocation>{fallback_response[0]}</fallback llm raw invocation>\\n"
        msg += f"<fallback tool name>{e.tool_name}</fallback tool name>\\n"
        msg += f"<fallback tool call result>\\n{fallback_response[1]}\\n</fallback tool call result>\\n"

        logger.info(
            f"Fallback successful for tool `{e.tool_name}` after {DEFAULT_MAX_TOOL_FAILURES} consecutive failures."
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

    def stream(
        self, messages_dict: Dict[str, List[Any]], _config: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream agent responses in a format compatible with print_agent_output."""
        initial_messages = messages_dict.get("messages", [])
        self.chat_history = []
        last_result = None
        empty_response_count = 0
        max_empty_responses = (
            3  # Maximum number of consecutive empty responses before giving up
        )

        while True:
            # Check for should_exit
            if should_exit():
                logger.debug("Agent should exit flag detected in stream loop")
                break

            base_prompt = self._build_prompt(last_result)
            if base_prompt:  # Only add if non-empty
                self.chat_history.append(HumanMessage(content=base_prompt))
            full_history = self._trim_chat_history(initial_messages, self.chat_history)

            response = self.model.invoke(
                [self.sys_message] + full_history, self.stream_config
            )
            # print(f"response={response}")

            # Get settings from config and models_params
            provider = self.config.get("provider", "")
            model_name = self.config.get("model", "")
            model_config_from_params = models_params.get(provider, {}).get(model_name, {})

            # Determine supports_think_tag: prioritize self.config, then models_params
            if "supports_think_tag" in self.config:
                supports_think_tag = self.config.get("supports_think_tag")
            else:
                supports_think_tag = model_config_from_params.get("supports_think_tag") # Defaults to None if not in either

            supports_thinking = model_config_from_params.get("supports_thinking", False)
            # show_thoughts defaults to True if not specified (matching process_thinking_content default)
            show_thoughts = self.config.get("show_thoughts", None)

            # Process thinking content if supported
            response.content, _ = process_thinking_content(
                content=response.content,
                supports_think_tag=supports_think_tag,
                supports_thinking=supports_thinking,
                panel_title=" Thoughts",
                show_thoughts=show_thoughts,
            )

            # Check if the response is empty or doesn't contain a valid tool call
            if not response.content or not response.content.strip():
                empty_response_count += 1
                logger.info(
                    f"Model returned empty response (count: {empty_response_count})"
                )

                warning_message = f"The model returned an empty response (attempt {empty_response_count} of {max_empty_responses}). Requesting the model to make a valid tool call."
                logger.info(warning_message)

                # Record warning in trajectory
                try:
                    # Import here to avoid circular imports
                    from ra_aid.database.repositories.trajectory_repository import (
                        TrajectoryRepository,
                    )
                    from ra_aid.database.repositories.human_input_repository import (
                        HumanInputRepository,
                    )
                    from ra_aid.database.connection import get_db_connection

                    # Create repositories directly
                    trajectory_repo = TrajectoryRepository(get_db_connection())
                    human_input_repo = HumanInputRepository(get_db_connection())
                    human_input_id = human_input_repo.get_most_recent_id()

                    trajectory_repo.create(
                        step_data={
                            "warning_message": warning_message,
                            "display_title": "Empty Response",
                            "attempt": empty_response_count,
                            "max_attempts": max_empty_responses,
                        },
                        record_type="error",
                        human_input_id=human_input_id,
                        is_error=True,
                        error_message=warning_message,
                        error_type="EmptyResponseWarning",
                    )
                except Exception as trajectory_error:
                    # Just log and continue if there's an error in trajectory recording
                    logger.error(
                        f"Error recording trajectory for empty response warning: {trajectory_error}"
                    )

                ra_aid.console.formatting.print_warning(
                    warning_message, title="Empty Response"
                )

                if empty_response_count >= max_empty_responses:
                    # If we've had too many empty responses, raise an error to break the loop
                    from ra_aid.agent_context import mark_agent_crashed

                    crash_message = (
                        "Agent failed to make any tool calls after multiple attempts"
                    )
                    mark_agent_crashed(crash_message)
                    logger.error(crash_message)

                    error_message = "The agent has crashed after multiple failed attempts to generate a valid tool call."
                    logger.error(error_message)

                    # Record error in trajectory
                    try:
                        # Import here to avoid circular imports
                        from ra_aid.database.repositories.trajectory_repository import (
                            TrajectoryRepository,
                        )
                        from ra_aid.database.repositories.human_input_repository import (
                            HumanInputRepository,
                        )
                        from ra_aid.database.connection import get_db_connection

                        # Create repositories directly
                        trajectory_repo = TrajectoryRepository(get_db_connection())
                        human_input_repo = HumanInputRepository(get_db_connection())
                        human_input_id = human_input_repo.get_most_recent_id()

                        trajectory_repo.create(
                            step_data={
                                "error_message": error_message,
                                "display_title": "Agent Crashed",
                                "crash_reason": crash_message,
                                "attempts": empty_response_count,
                            },
                            record_type="error",
                            human_input_id=human_input_id,
                            is_error=True,
                            error_message=error_message,
                            error_type="AgentCrashError",
                            tool_name="unknown_tool",
                        )
                    except Exception as trajectory_error:
                        # Just log and continue if there's an error in trajectory recording
                        logger.error(
                            f"Error recording trajectory for agent crash: {trajectory_error}"
                        )

                    ra_aid.console.formatting.print_error(error_message)

                    yield self._create_error_chunk(crash_message)
                    return

                # If not max empty, continue the loop to retry
                last_result = "Model returned an empty response. Please provide a valid tool call."  # Provide feedback
                continue  # Go to the next iteration immediately

            # Reset empty response counter on successful response
            empty_response_count = 0

            try:
                last_result = self._execute_tool(response)
                self.chat_history.append(response)
                if hasattr(self.fallback_handler, "reset_fallback_handler"):
                    self.fallback_handler.reset_fallback_handler()
                yield {}

            except ToolExecutionError as e:
                logger.info(f"Tool execution error: {str(e)}. Attempting fallback...")
                fallback_response = self.fallback_handler.handle_failure(
                    e, self, self.chat_history
                )
                last_result = self.handle_fallback_response(fallback_response, e)
                # If fallback failed (last_result is empty string), don't yield empty dict
                if last_result:
                    yield {}
                else:
                    # Add the error message to the chat history so the model sees it
                    err_msg_content = self.error_message_template.format(e=e)
                    self.chat_history.append(HumanMessage(content=err_msg_content))
                    last_result = (
                        err_msg_content  # Set last result for the next loop iteration
                    )
                    yield {}  # Yield empty dict to allow stream to continue
