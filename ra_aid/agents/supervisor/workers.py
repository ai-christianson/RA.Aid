"""Workers for the supervisor architecture.

Workers are simple, stateless executors that:
- Execute ONE task
- Return result
- Cannot spawn other agents
- Cannot create new tasks
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from ra_aid.agents.supervisor.models import Task
from ra_aid.console.formatting import cpm
from ra_aid.tools.read_file import read_file_tool
from ra_aid.tools.ripgrep import ripgrep_search
from ra_aid.tools.list_directory import list_directory_tree
from ra_aid.tools.shell import run_shell_command
from ra_aid.tools.write_file import put_complete_file_contents
from ra_aid.tools.file_str_replace import file_str_replace

logger = logging.getLogger(__name__)

# Maximum iterations for worker tool loop
MAX_WORKER_ITERATIONS = 20


class Worker(ABC):
    """Base class for workers. Execute task, return result, done."""

    def __init__(self, model, tools: List[BaseTool]):
        self.model = model
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this worker type."""
        pass

    def _extract_text_content(self, content) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle list of content blocks (Claude format)
            texts = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    texts.append(block['text'])
                elif hasattr(block, 'text'):
                    texts.append(block.text)
            return '\n'.join(texts)
        return str(content)

    def _get_tool_info(self, tool) -> str:
        """Get formatted tool info with argument schema."""
        name = tool.name
        # Get required args from schema
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
            required = schema.get('required', [])
            props = schema.get('properties', {})
            args_info = []
            for arg_name in required:
                arg_type = props.get(arg_name, {}).get('type', 'string')
                args_info.append(f"{arg_name}: {arg_type}")
            if args_info:
                return f"- {name}({', '.join(args_info)})"
        return f"- {name}()"

    def execute(self, task: Task, context: str) -> str:
        """
        Execute a task with given context.

        This is a bounded tool-calling loop:
        - Runs until model says DONE or max iterations
        - Cannot spawn agents
        - Returns result string
        """
        system_prompt = self.get_system_prompt()

        # Build tool descriptions with arg info
        tool_descriptions = "\n".join([self._get_tool_info(t) for t in self.tools])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Task: {task.description}

Context:
{context}

Available tools:
{tool_descriptions}

To use a tool, respond EXACTLY like this:
TOOL: tool_name
ARGS: {{"arg_name": "value"}}

Examples:
TOOL: run_shell_command
ARGS: {{"command": "git diff"}}

TOOL: read_file_tool
ARGS: {{"filepath": "src/main.py"}}

TOOL: ripgrep_search
ARGS: {{"pattern": "function_name"}}

TOOL: list_directory_tree
ARGS: {{"path": "."}}

When you have completed the task, respond with:
DONE: <your findings/results>

Do NOT announce what you're doing. Just use the tools.
""")
        ]

        result = ""

        for iteration in range(MAX_WORKER_ITERATIONS):
            response = self.model.invoke(messages)
            content = self._extract_text_content(response.content)

            # Check if done
            if "DONE:" in content:
                # Extract result after DONE:
                result = content.split("DONE:", 1)[1].strip()
                break

            # Check for tool call
            if "TOOL:" in content:
                tool_result = self._execute_tool_call(content)
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content=f"Tool result:\n{tool_result}"))
            else:
                # Model didn't use tool or say DONE - prompt it
                messages.append(AIMessage(content=content))
                messages.append(HumanMessage(content="Please either use a TOOL or say DONE with your results."))

        if not result:
            result = f"Worker reached max iterations ({MAX_WORKER_ITERATIONS}) without completing."

        return result

    def _execute_tool_call(self, content: str) -> str:
        """Parse and execute a tool call from model response."""
        try:
            # Parse TOOL: line
            tool_line = ""
            args_line = ""

            for line in content.split("\n"):
                line_stripped = line.strip()
                if line_stripped.startswith("TOOL:"):
                    tool_line = line_stripped.replace("TOOL:", "").strip()
                elif line_stripped.startswith("ARGS:"):
                    args_line = line_stripped.replace("ARGS:", "").strip()

            if not tool_line:
                return "Error: Could not parse tool name"

            tool_name = tool_line
            if tool_name not in self.tools_by_name:
                return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools_by_name.keys())}"

            tool = self.tools_by_name[tool_name]

            # Parse args
            import json
            args = {}
            if args_line:
                try:
                    args = json.loads(args_line)
                except json.JSONDecodeError:
                    # If not valid JSON, try to infer the primary argument
                    primary_arg = self._get_primary_arg(tool)
                    if primary_arg:
                        args = {primary_arg: args_line}
                    else:
                        return f"Error: Invalid JSON in ARGS. Expected format: {{\"arg\": \"value\"}}"

            # Execute tool
            result = tool.invoke(args)
            return str(result)

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing tool: {str(e)}"

    def _get_primary_arg(self, tool) -> Optional[str]:
        """Get the primary (first required) argument name for a tool."""
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
                required = schema.get('required', [])
                if required:
                    return required[0]
            except Exception:
                pass
        return None


class Researcher(Worker):
    """Worker that gathers information. Has read tools + shell for git/info commands."""

    def __init__(self, model):
        tools = [
            read_file_tool,
            ripgrep_search,
            list_directory_tree,
            run_shell_command,  # For git diff, git status, git log, etc.
        ]
        super().__init__(model, tools)

    def get_system_prompt(self) -> str:
        return """You are a research worker. Your job is to gather information.

Rules:
- Use tools to find the requested information
- Return ONLY facts you discovered
- Do NOT suggest improvements or next steps
- Do NOT make changes to any files
- For git info, use run_shell_command with: git diff, git status, git log, etc.
- Be concise and factual

When done, respond with DONE: followed by your findings."""


class Implementor(Worker):
    """Worker that makes changes. Has read and write tools."""

    def __init__(self, model):
        tools = [
            read_file_tool,
            ripgrep_search,
            list_directory_tree,
            put_complete_file_contents,
            file_str_replace,
            run_shell_command,
        ]
        super().__init__(model, tools)

    def get_system_prompt(self) -> str:
        return """You are an implementation worker. Your job is to make specific changes.

Rules:
- Make ONLY the changes specified in the task
- Do NOT add extra features or improvements
- Do NOT refactor unrelated code
- Work with the context provided - do not request more research
- Be precise and minimal

When done, respond with DONE: followed by a summary of what you changed."""


class Verifier(Worker):
    """Worker that verifies changes. Runs tests, checks syntax, etc."""

    def __init__(self, model):
        tools = [
            read_file_tool,
            run_shell_command,
            list_directory_tree,
        ]
        super().__init__(model, tools)

    def get_system_prompt(self) -> str:
        return """You are a verification worker. Your job is to verify that changes work correctly.

Rules:
- Run tests if they exist
- Check for syntax errors
- Verify the expected behavior
- Report pass/fail status clearly

When done, respond with DONE: followed by verification results (PASS or FAIL with details)."""
