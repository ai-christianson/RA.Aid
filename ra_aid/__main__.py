import argparse
import logging
import os
import sys
import json
import time # <-- ADDED IMPORT
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ra_aid.utils.mcp_use_client import MCPUseClientSync
import uuid
from datetime import datetime

import litellm
import uvicorn

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm

from ra_aid import print_error, print_stage_header
from ra_aid.__version__ import __version__
from ra_aid.version_check import check_for_newer_version
from ra_aid.agent_utils import (
    create_agent,
    run_agent_with_retry,
)
from ra_aid.agents.research_agent import run_research_agent
from ra_aid.config import (
    DEFAULT_MAX_TEST_CMD_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_TEST_CMD_TIMEOUT,
    VALID_PROVIDERS,
)
from ra_aid.database.repositories.key_fact_repository import (
    KeyFactRepositoryManager,
    get_key_fact_repository,
)
from ra_aid.database.repositories.key_snippet_repository import (
    KeySnippetRepositoryManager,
    get_key_snippet_repository,
)
from ra_aid.database.repositories.human_input_repository import (
    HumanInputRepositoryManager,
    get_human_input_repository,
)
from ra_aid.database.repositories.session_repository import get_session_repository
from ra_aid.database.repositories.research_note_repository import (
    ResearchNoteRepositoryManager,
    get_research_note_repository,
)
from ra_aid.database.repositories.trajectory_repository import (
    TrajectoryRepositoryManager,
    get_trajectory_repository,
)
from ra_aid.database.repositories.session_repository import SessionRepositoryManager
from ra_aid.database.repositories.related_files_repository import (
    RelatedFilesRepositoryManager,
)

from ra_aid.database.repositories.work_log_repository import WorkLogRepositoryManager
from ra_aid.database.repositories.config_repository import (
    ConfigRepositoryManager,
    get_config_repository,
)
from ra_aid.env_inv import EnvDiscovery
from ra_aid.env_inv_context import EnvInvManager, get_env_inv
from ra_aid.model_formatters import format_key_facts_dict
from ra_aid.model_formatters.key_snippets_formatter import format_key_snippets_dict
from ra_aid.console.formatting import cpm
from ra_aid.database import (
    DatabaseManager,
    ensure_migrations_applied,
)
from ra_aid.dependencies import check_dependencies
from ra_aid.env import validate_environment
from ra_aid.exceptions import AgentInterrupt
from ra_aid.fallback_handler import FallbackHandler
from ra_aid.llm import initialize_llm, get_model_default_temperature
from ra_aid.logging_config import get_logger, setup_logging
from ra_aid.models_params import models_params
from ra_aid.project_info import format_project_info, get_project_info
from ra_aid.prompts.chat_prompts import CHAT_PROMPT
from ra_aid.prompts.web_research_prompts import WEB_RESEARCH_PROMPT_SECTION_CHAT
from ra_aid.prompts.custom_tools_prompts import DEFAULT_CUSTOM_TOOLS_PROMPT
from ra_aid.server.server import app as fastapi_app
from ra_aid.tool_configs import (
    get_chat_tools,
    set_modification_tools,
    get_custom_tools,
    get_planning_tools,
    get_implementation_tools,
)
from ra_aid.tools.human import ask_human

import importlib.resources as pkg_resources

import atexit

logger = get_logger(__name__)

# Configure litellm to suppress debug logs
os.environ["LITELLM_LOG"] = "ERROR"
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Explicitly configure LiteLLM's loggers
for logger_name in ["litellm", "LiteLLM"]:
    litellm_logger = logging.getLogger(logger_name)
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = True

# Use litellm's internal method to disable debugging
if hasattr(litellm, "_logging") and hasattr(litellm._logging, "_disable_debugging"):
    litellm._logging._disable_debugging()


def launch_server(host: str, port: int, args):
    """Launch the RA.Aid web interface."""
    from ra_aid.database.connection import DatabaseManager
    from ra_aid.database.repositories.session_repository import SessionRepositoryManager
    from ra_aid.database.repositories.key_fact_repository import (
        KeyFactRepositoryManager,
    )
    from ra_aid.database.repositories.key_snippet_repository import (
        KeySnippetRepositoryManager,
    )
    from ra_aid.database.repositories.human_input_repository import (
        HumanInputRepositoryManager,
    )
    from ra_aid.database.repositories.research_note_repository import (
        ResearchNoteRepositoryManager,
    )
    from ra_aid.database.repositories.related_files_repository import (
        RelatedFilesRepositoryManager,
    )
    from ra_aid.database.repositories.trajectory_repository import (
        TrajectoryRepositoryManager,
    )
    from ra_aid.database.repositories.work_log_repository import (
        WorkLogRepositoryManager,
    )
    from ra_aid.database.repositories.config_repository import ConfigRepositoryManager
    from ra_aid.env_inv_context import EnvInvManager
    from ra_aid.env_inv import EnvDiscovery

    # Set the console handler level to INFO for server mode
    # Get the root logger and modify the console handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        # Check if this is a console handler (outputs to stdout/stderr)
        if isinstance(handler, logging.StreamHandler) and handler.stream in [
            sys.stdout,
            sys.stderr,
        ]:
            # Set console handler to INFO level for better visibility in server mode
            handler.setLevel(logging.INFO)
            logger.debug("Modified console logging level to INFO for server mode")

    # Apply any pending database migrations
    from ra_aid.database import ensure_migrations_applied

    try:
        migration_result = ensure_migrations_applied()
        if not migration_result:
            logger.warning("Database migrations failed but execution will continue")
    except Exception as e:
        logger.error(f"Database migration error: {str(e)}")

    # Check dependencies before proceeding
    check_dependencies()

    # Validate environment (expert_enabled, web_research_enabled)
    (
        expert_enabled,
        expert_missing,
        web_research_enabled,
        web_research_missing,
    ) = validate_environment(
        args
    )  # Will exit if main env vars missing
    logger.debug("Environment validation successful")

    # Validate model configuration early
    model_config = models_params.get(args.provider, {}).get(args.model or "", {})
    supports_temperature = model_config.get(
        "supports_temperature",
        args.provider
        in [
            "anthropic",
            "openai",
            "openrouter",
            "openai-compatible",
            "deepseek",
        ],
    )

    if supports_temperature and args.temperature is None:
        args.temperature = model_config.get("default_temperature")
        if args.temperature is None:
            args.temperature = get_model_default_temperature(args.provider, args.model)
            cpm(
                f"This model supports temperature argument but none was given. Using model default temperature: {args.temperature}."
            )
        logger.debug(
            f"Using default temperature {args.temperature} for model {args.model}"
        )

    # Initialize environment discovery
    env_discovery = EnvDiscovery()
    env_discovery.discover()
    env_data = env_discovery.format_markdown()

    print(f"Starting RA.Aid web interface on http://{host}:{port}")

    # Initialize database connection and repositories
    with (
        DatabaseManager(base_dir=args.project_state_dir) as db,
        SessionRepositoryManager(db) as session_repo,
        KeyFactRepositoryManager(db) as key_fact_repo,
        KeySnippetRepositoryManager(db) as key_snippet_repo,
        HumanInputRepositoryManager(db) as human_input_repo,
        ResearchNoteRepositoryManager(db) as research_note_repo,
        RelatedFilesRepositoryManager() as related_files_repo,
        TrajectoryRepositoryManager(db) as trajectory_repo,
        WorkLogRepositoryManager() as work_log_repo,
        ConfigRepositoryManager() as config_repo,
        EnvInvManager(env_data) as env_inv,
    ):
        # This initializes all repositories and makes them available via their respective get methods
        logger.debug("Initialized SessionRepository")
        logger.debug("Initialized KeyFactRepository")
        logger.debug("Initialized KeySnippetRepository")
        logger.debug("Initialized HumanInputRepository")
        logger.debug("Initialized ResearchNoteRepository")
        logger.debug("Initialized RelatedFilesRepository")
        logger.debug("Initialized TrajectoryRepository")
        logger.debug("Initialized WorkLogRepository")
        logger.debug("Initialized ConfigRepository")
        logger.debug("Initialized Environment Inventory")

        # Update config repo with values from args and environment validation
        config_repo.update(
            {
                "provider": args.provider,
                "model": args.model,
                "num_ctx": args.num_ctx,
                "expert_provider": args.expert_provider,
                "expert_model": args.expert_model,
                "expert_num_ctx": args.expert_num_ctx,
                "temperature": args.temperature,
                "experimental_fallback_handler": args.experimental_fallback_handler,
                "expert_enabled": expert_enabled,
                "web_research_enabled": web_research_enabled,
                "show_thoughts": args.show_thoughts,
                "show_cost": args.show_cost,
                "force_reasoning_assistance": args.reasoning_assistance,
                "disable_reasoning_assistance": args.no_reasoning_assistance,
                "cowboy_mode": args.cowboy_mode,
            }
        )

        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


def parse_arguments(args=None):
    ANTHROPIC_DEFAULT_MODEL = DEFAULT_MODEL
    OPENAI_DEFAULT_MODEL = "o4-mini"

    # Case-insensitive log level argument type
    def log_level_type(value):
        value = value.lower()
        if value not in ["debug", "info", "warning", "error", "critical"]:
            raise argparse.ArgumentTypeError(
                f"Invalid log level: {value}. Choose from debug, info, warning, error, critical."
            )
        return value

    parser = argparse.ArgumentParser(
        description="RA.Aid - AI Agent for executing programming and research tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ra-aid -m "Add error handling to the database module"
    ra-aid -m "Explain the authentication flow" --research-only
    ra-aid --msg-file task_description.txt
    ra-aid -m "Use context7 to find latest pandas API for reading CSVs" # Context7 enabled by default
    ra-aid -m "Implement feature X" --no-context7 # Disable default Context7
    ra-aid -m "Use my custom browser tool" --mcp-use-config ./my_browser_mcp.json --no-context7 # Use specific MCP config
        """,
    )
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        help="The task or query to be executed by the agent (cannot be used with --msg-file)",
    )
    parser.add_argument(
        "--msg-file",
        type=str,
        help="Path to a text file containing the task/message (cannot be used with --message)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version number and exit",
    )
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Only perform research without implementation",
    )
    parser.add_argument(
        "--web-research",
        action="store_true",
        help="Enable web research capabilities",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable search capabilities",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=(
            "gemini"
            if os.getenv("GEMINI_API_KEY")
            else ("openai" if os.getenv("OPENAI_API_KEY") else "anthropic")
        ),
        choices=VALID_PROVIDERS,
        help="The LLM provider to use",
    )
    parser.add_argument("--model", type=str, help="The model name to use")
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=262144,
        help="Context window size for Ollama models",
    )
    parser.add_argument(
        "--research-provider",
        type=str,
        choices=VALID_PROVIDERS,
        help="Provider to use specifically for research tasks",
    )
    parser.add_argument(
        "--research-model",
        type=str,
        help="Model to use specifically for research tasks",
    )
    parser.add_argument(
        "--planner-provider",
        type=str,
        choices=VALID_PROVIDERS,
        help="Provider to use specifically for planning tasks",
    )
    parser.add_argument(
        "--planner-model", type=str, help="Model to use specifically for planning tasks"
    )
    parser.add_argument(
        "--cowboy-mode",
        action="store_true",
        help="Skip interactive approval for shell commands",
    )
    parser.add_argument(
        "--expert-provider",
        type=str,
        default=None,
        choices=VALID_PROVIDERS,
        help="The LLM provider to use for expert knowledge queries",

    )
    parser.add_argument(
        "--expert-model",
        type=str,
        help="The model name to use for expert knowledge queries (required for non-OpenAI providers)",
    )
    parser.add_argument(
        "--expert-num-ctx",
        type=int,
        default=262144,
        help="Context window size for expert Ollama models",
    )
    parser.add_argument(
        "--hil",
        "-H",
        action="store_true",
        help="Enable human-in-the-loop mode, where the agent can prompt the user for additional information.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable chat mode with direct human interaction (implies --hil)",
    )
    parser.add_argument(
        "--log-mode",
        choices=["console", "file"],
        default="file",
        help="Logging mode: 'console' shows all logs in console, 'file' logs to file with only warnings+ in console",
    )
    parser.add_argument(
        "--pretty-logger", action="store_true", help="Enable pretty logging output"
    )
    parser.add_argument(
        "--log-level",
        type=log_level_type,
        default="debug",
        help="Set specific logging level (case-insensitive, affects file and console logging based on --log-mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="LLM temperature (0.0-2.0). Controls randomness in responses",
        default=None,
    )
    parser.add_argument(
        "--disable-limit-tokens",
        action="store_false",
        help="Whether to disable token limiting for Anthropic Claude react agents. Token limiter removes older messages to prevent maximum token limit API errors.",
    )
    parser.add_argument(
        "--experimental-fallback-handler",
        action="store_true",
        help="Enable experimental fallback handler.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=DEFAULT_RECURSION_LIMIT,
        help="Maximum recursion depth for agent operations (default: 100)",
    )
    parser.add_argument(
        "--aider-config", type=str, help="Specify the aider config file path"
    )
    parser.add_argument(
        "--use-aider",
        action="store_true",
        help="Use aider for code modifications instead of default file tools (file_str_replace, put_complete_file_contents)",
    )
    parser.add_argument(
        "--test-cmd",
        type=str,
        help="Test command to run before completing tasks (e.g. 'pytest tests/')",
    )
    parser.add_argument(
        "--auto-test",
        action="store_true",
        help="Automatically run tests before completing tasks",
    )
    parser.add_argument(
        "--max-test-cmd-retries",
        type=int,
        default=DEFAULT_MAX_TEST_CMD_RETRIES,
        help="Maximum number of retries for the test command (default: 3)",
    )
    parser.add_argument(
        "--test-cmd-timeout",
        type=int,
        default=DEFAULT_TEST_CMD_TIMEOUT,
        help=f"Timeout in seconds for test command execution (default: {DEFAULT_TEST_CMD_TIMEOUT})",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch the web interface",
    )
    parser.add_argument(
        "--server-host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on for web interface (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=1818,
        help="Port to listen on for web interface (default: 1818)",
    )
    parser.add_argument(
        "--wipe-project-memory",
        action="store_true",
        help="Delete the project database file (.ra-aid/pk.db) before starting, effectively wiping all stored memory",
    )
    parser.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )
    parser.add_argument(
        "--show-thoughts",
        action="store_true",
        help="Display model thinking content extracted from think tags when supported by the model",
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Display cost information as the agent works",
    )
    parser.add_argument(
        "--track-cost",
        action="store_true",
        default=False,
        help="Track token usage and costs (default: False)",
    )
    parser.add_argument(
        "--no-track-cost",
        action="store_false",
        dest="track_cost",
        help="Disable tracking of token usage and costs",
    )
    parser.add_argument(
        "--reasoning-assistance",
        action="store_true",
        help="Force enable reasoning assistance regardless of model defaults",
    )
    parser.add_argument(
        "--no-reasoning-assistance",
        action="store_true",
        help="Force disable reasoning assistance regardless of model defaults",
    )
    parser.add_argument(
        "--custom-tools",
        type=str,
        help="File path of Python module containing custom tools (e.g. ./path/to_custom_tools.py)",
    )
    parser.add_argument(
        "--mcp-use-config",
        type=str,
        help="Path to MCP-Use JSON config file (enables MCP-Use tools)",
    )
    parser.add_argument(
        "--disable-default-mcp",
        nargs='*',
        metavar='SERVER_NAME',
        help="Disable specific default MCP servers (e.g., context7 taskmaster-ai tree_sitter) or all if no names are given.",
        default=None # Default is None, meaning don't disable any unless flag is present
    )
    parser.add_argument(
        "--disable-task-master-planning",
        action="store_true",
        help="Force RA.Aid to use its internal planner instead of Task Master tools, even if Task Master MCP is available.",
    )


    if args is None:
        args = sys.argv[1:]
    parsed_args = parser.parse_args(args)

    # Eagerly validate file paths provided via arguments
    if parsed_args.msg_file and not os.path.isfile(parsed_args.msg_file):
        parser.error(f"Message file not found: {parsed_args.msg_file}")
    if parsed_args.custom_tools and not os.path.isfile(parsed_args.custom_tools):
        parser.error(f"Custom tools file not found: {parsed_args.custom_tools}")
    if parsed_args.mcp_use_config and not os.path.isfile(parsed_args.mcp_use_config):
        parser.error(f"MCP-Use config file not found: {parsed_args.mcp_use_config}")
    if parsed_args.aider_config and not os.path.isfile(parsed_args.aider_config):
        parser.error(f"Aider config file not found: {parsed_args.aider_config}")

    # Validate message vs msg-file usage
    if parsed_args.message and parsed_args.msg_file:
        parser.error("Cannot use both --message and --msg-file")
    if parsed_args.msg_file:
        try:
            with open(parsed_args.msg_file, "r") as f:
                parsed_args.message = f.read()
        except IOError as e:
            parser.error(f"Failed to read message file: {str(e)}")

    # Set hil=True when chat mode is enabled
    if parsed_args.chat:
        parsed_args.hil = True

    # Validate provider
    if parsed_args.provider not in VALID_PROVIDERS:
        parser.error(f"Invalid provider: {parsed_args.provider}")
    # Handle model defaults and requirements

    if parsed_args.provider == "openai":
        parsed_args.model = parsed_args.model or OPENAI_DEFAULT_MODEL
    elif parsed_args.provider == "anthropic":
        # Use default model for Anthropic only if not specified
        parsed_args.model = parsed_args.model or ANTHROPIC_DEFAULT_MODEL
    elif parsed_args.provider == "gemini":
        parsed_args.model = parsed_args.model or "gemini-2.5-pro-preview-03-25"
    elif not parsed_args.model and not parsed_args.research_only:
        # Require model for other providers unless in research mode
        parser.error(
            f"--model is required when using provider '{parsed_args.provider}'"
        )

    # Handle expert provider/model defaults
    if not parsed_args.expert_provider:
        # Priority: Explicit EXPERT_* -> Both Gemini/OpenAI -> Gemini -> Anthropic Expert -> DeepSeek -> Main Provider Fallback
        if os.environ.get("EXPERT_OPENAI_API_KEY"):
            parsed_args.expert_provider = "openai"
            parsed_args.expert_model = None  # Will be auto-selected later
        elif os.environ.get("EXPERT_ANTHROPIC_API_KEY"):
             parsed_args.expert_provider = "anthropic"
             # Use main anthropic model if expert model not specified
             parsed_args.expert_model = parsed_args.expert_model or ANTHROPIC_DEFAULT_MODEL
        # Add other explicit EXPERT_* checks here if needed in the future...

        # NEW: Check if both base Gemini and OpenAI keys are present (and no specific EXPERT_* key was found)
        elif os.environ.get("GEMINI_API_KEY") and os.environ.get("OPENAI_API_KEY"):
            # Both keys present, default main to Gemini (already done) and expert to OpenAI
            parsed_args.expert_provider = "openai"
            # Let llm.py auto-select 'o3' unless user specified --expert-model
            parsed_args.expert_model = parsed_args.expert_model or None

        # Fallback checks for individual base keys (if the combined check didn't match or only one key exists)
        elif os.environ.get("GEMINI_API_KEY"): # Check main Gemini key as fallback
            parsed_args.expert_provider = "gemini"
            # Use gemini-2.5-pro if not specified, matching main model default
            parsed_args.expert_model = parsed_args.expert_model or "gemini-2.5-pro-preview-03-25"
        elif os.environ.get("DEEPSEEK_API_KEY"): # Check main Deepseek key as fallback
            parsed_args.expert_provider = "deepseek"
            parsed_args.expert_model = "deepseek-reasoner" # Specific default for Deepseek expert
        else:
            # Final Fallback: Use main provider settings if none of the above conditions met
            # Special-case OpenAI main provider: we want to use the provider but let later logic choose the best expert model.
            if parsed_args.provider == "openai":
                parsed_args.expert_provider = "openai"
                parsed_args.expert_model = None  # trigger auto-selection later (prefer o3)
            else:
                # For other main providers, use their settings as the expert fallback
                parsed_args.expert_provider = parsed_args.provider
                parsed_args.expert_model = parsed_args.model

    # Validate temperature range if provided
    if parsed_args.temperature is not None and not (
        0.0 <= parsed_args.temperature <= 2.0
    ):
        parser.error("Temperature must be between 0.0 and 2.0")

    # Validate recursion limit is positive
    if parsed_args.recursion_limit <= 0:
        parser.error("Recursion limit must be positive")

    # if auto-test command is provided, validate test-cmd is also provided
    if parsed_args.auto_test and not parsed_args.test_cmd:
        parser.error("Test command is required when using --auto-test")

    # If show_cost is true, we must also enable track_cost
    if parsed_args.show_cost:
        parsed_args.track_cost = True

    return parsed_args


# Create console instance
console = Console()

# Create individual memory objects for each agent
research_memory = MemorySaver()
planning_memory = MemorySaver()
implementation_memory = MemorySaver()


def is_informational_query() -> bool:
    """Determine if the current query is informational based on config settings."""
    return get_config_repository().get("research_only", False)


def is_stage_requested(stage: str) -> bool:
    """Check if a stage has been requested to proceed."""
    # This is kept for backward compatibility but no longer does anything
    return False


def display_welcome_message():
    """Display a welcome message when starting in interactive mode."""
    welcome_text = """
Welcome to RA.Aid Interactive Mode!

I can help you with:
• Code analysis and understanding
• Implementing new features
• Fixing bugs
• Refactoring code
• Researching technical topics

Just type your request and press Enter.
Press Ctrl+C to exit.
    """
    console.print(Panel(welcome_text, title="RA.Aid Interactive", border_style="green"))


def wipe_project_memory(custom_dir=None):
    """Delete the project database file to wipe all stored memory.

    Args:
        custom_dir: Optional custom directory to use instead of .ra-aid in current directory

    Returns:
        str: A message indicating the result of the operation
    """
    import os
    from pathlib import Path

    if custom_dir:
        ra_aid_dir = Path(custom_dir)
        db_path = os.path.join(custom_dir, "pk.db")
    else:
        cwd = os.getcwd()
        ra_aid_dir = Path(os.path.join(cwd, ".ra-aid"))
        db_path = os.path.join(ra_aid_dir, "pk.db")

    if not os.path.exists(db_path):
        return "No project memory found to wipe."

    try:
        os.remove(db_path)
        return "Project memory wiped successfully."
    except PermissionError:
        return "Error: Could not wipe project memory due to permission issues."
    except Exception as e:
        return f"Error: Failed to wipe project memory: {str(e)}"


def build_status():
    """Build status panel with model and feature information.

    Includes memory statistics at the bottom with counts of key facts, snippets, and research notes.
    """
    status = Text()

    # Get the config repository to get model/provider information
    config_repo = get_config_repository()
    provider = config_repo.get("provider", "")
    model = config_repo.get("model", "")
    temperature = config_repo.get("temperature")
    expert_provider = config_repo.get("expert_provider", "")
    expert_model = config_repo.get("expert_model", "")
    experimental_fallback_handler = config_repo.get(
        "experimental_fallback_handler", False
    )
    web_research_enabled = config_repo.get("web_research_enabled", False)
    custom_tools_enabled = config_repo.get("custom_tools_enabled", False)
    active_mcp_servers = config_repo.get("active_mcp_servers", [])
    logger.debug("MCP Servers read by build_status: %s", active_mcp_servers)

    # Get the expert enabled status
    expert_enabled = bool(expert_provider and expert_model)

    # Basic model information
    status.append("🤖 ")
    status.append(f"{provider}/{model}")
    if temperature is not None:
        status.append(f" @ T{temperature}")
    status.append("\n")

    # Expert model information
    status.append("🤔 ")
    if expert_enabled:
        status.append(f"{expert_provider}/{expert_model}")
    else:
        status.append("Expert: ")
        status.append("Disabled", style="italic")
    status.append("\n")

    # Web research status
    status.append("🔍 Search: ")
    status.append(
        "Enabled" if web_research_enabled else "Disabled",
        style=None if web_research_enabled else "italic",
    )
    status.append("\n")

    # Custom tools status
    if custom_tools_enabled:
        status.append("🛠️ Custom Tools: ")
        status.append(
            "Enabled" if custom_tools_enabled else "Disabled",
            style=None if custom_tools_enabled else "italic",
        )
        status.append("\n")
        
    # MCP servers status
    if active_mcp_servers:
        status.append("🔌 MCP Servers: ")
        status.append(", ".join(active_mcp_servers))
        status.append("\n")

    # Fallback handler status
    if experimental_fallback_handler:
        fb_handler = FallbackHandler({}, [])
        status.append("\n🔧 FallbackHandler Enabled: ")
        msg = ", ".join(
            [fb_handler._format_model(m) for m in fb_handler.fallback_tool_models]
        )
        status.append(msg)
        status.append("\n")

    # Add memory statistics
    # Get counts of key facts, snippets, and research notes with error handling
    fact_count = 0
    snippet_count = 0
    note_count = 0

    try:
        fact_count = len(get_key_fact_repository().get_all())
    except RuntimeError as e:
        logger.debug(f"Failed to get key facts count: {e}")

    try:
        snippet_count = len(get_key_snippet_repository().get_all())
    except RuntimeError as e:
        logger.debug(f"Failed to get key snippets count: {e}")

    try:
        note_count = len(get_research_note_repository().get_all())
    except RuntimeError as e:
        logger.debug(f"Failed to get research notes count: {e}")

    # Add memory statistics line with reset option note
    status.append(
        f"💾 Memory: {fact_count} facts, {snippet_count} snippets, {note_count} notes"
    )
    if fact_count > 0 or snippet_count > 0 or note_count > 0:
        status.append(" (use --wipe-project-memory to reset)")

    # Check for newer version
    version_message = check_for_newer_version()
    if version_message:
        status.append("\n\n")
        status.append(version_message, style="yellow")

    return status


def process_task(args, mcp_use_client_instance=None):
    """Process a single task with the given arguments."""
    # This function will contain the core task processing logic
    # that was previously in main()
    
    # Get or create config repository for storing runtime configuration
    try:
        # Try to get existing config repository
        config_repo = get_config_repository()
    except Exception as e:
        # If it fails, initialize a new one
        logger.debug(f"Config repository not found, initializing: {e}")
        from ra_aid.database.repositories.config_repository import ConfigRepositoryManager
        ConfigRepositoryManager.initialize()
        config_repo = get_config_repository()
    
    # Store key arguments in config repository for access by tools
    config_repo.set("provider", args.provider)
    config_repo.set("model", args.model)
    config_repo.set("temperature", args.temperature)
    config_repo.set("expert_provider", args.expert_provider)
    config_repo.set("expert_model", args.expert_model)
    config_repo.set("research_only", args.research_only)
    
    # Safely set web_research and search flags
    web_research_enabled = getattr(args, 'web_research', False)
    search_enabled = getattr(args, 'search', False)
    config_repo.set("web_research_enabled", web_research_enabled)
    config_repo.set("search_enabled", search_enabled)
    
    config_repo.set("hil", args.hil)
    config_repo.set("test_cmd_timeout", args.test_cmd_timeout)
    config_repo.set("max_test_cmd_retries", args.max_test_cmd_retries)
    
    
    # Validate message is provided
    if not args.message:
        error_message = "No task provided"
        print_error(error_message)
        return
    
    # Store the human input in the database
    human_input_repo = get_human_input_repository()
    session_id = get_session_repository().get_current_session_id()
    human_input_repo.create(content=args.message, source="cli", session_id=session_id)
    
    # Run the appropriate agent based on the arguments
    if args.research_only:
        # Run research agent
        research_model = initialize_llm(
            args.provider, args.model or DEFAULT_MODEL, temperature=args.temperature
        )
        run_research_agent(
            args.message,
            research_model,
            expert_enabled=False,
            research_only=True,
            hil=args.hil,
            web_research_enabled=config_repo.get("web_research_enabled", False),
            memory=research_memory,
        )
    else:
        # Initialize the LLM
        llm = initialize_llm(
            args.provider, args.model, temperature=args.temperature
        )
        
        # Initialize the expert LLM if needed
        expert_llm = None
        if args.expert_provider and args.expert_model:
            expert_llm = initialize_llm(
                args.expert_provider, args.expert_model, temperature=args.temperature
            )
        
        # Get/Load custom tools (call might have side effects)
        get_custom_tools(mcp_use_client=mcp_use_client_instance)
        
        # Create the planning agent
        planning_agent = create_agent(
            llm,
            get_planning_tools(
                expert_enabled=bool(expert_llm),
                web_research_enabled=web_research_enabled,
            ),
            checkpointer=planning_memory,
            agent_type="planning",
        )
        
        # Run the planning agent
        run_agent_with_retry(
            planning_agent,
            args.message,
        )
        
        # Create and run the implementation agent if not in research-only mode
        implementation_agent = create_agent(
            llm,
            get_implementation_tools(
                expert_enabled=bool(expert_llm),
                web_research_enabled=web_research_enabled,
            ),
            checkpointer=implementation_memory,
            agent_type="implementation",
        )
        
        run_agent_with_retry(
            implementation_agent,
            args.message,
        )
    
    # Clean up MCP-Use client if it was initialized
    if mcp_use_client_instance:
        try:
            mcp_use_client_instance.close()
        except Exception as e:
            logger.error(f"Error closing MCP-Use client: {e}")

def run_interactive_mode(args, mcp_use_client_instance=None):
    """Run RA.Aid in interactive mode with a command prompt."""
    display_welcome_message()
    
    # Config repository is initialized within the main() context manager
        
    # --- Display initial status ---    
    try:
        console.print(Panel(build_status(), title="Status", border_style="blue"))
    except Exception as e:
        logger.debug(f"Initial status display failed: {e}")
    # --- END NEW ---
    
    try:
        while True:
            try:
                user_input = input("RA.Aid> ").strip()
                if not user_input:
                    continue
                args.message = user_input
                process_task(args, mcp_use_client_instance)
                
                # Show status after each task
                console.print(Panel(build_status(), title="Status", border_style="blue"))
                
                # Clear message for next iteration
                args.message = None
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                sys.exit(0)
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        print_error(f"Error: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the ra-aid command line tool."""
    args = parse_arguments()
    setup_logging(
        args.log_mode,
        args.pretty_logger,
        args.log_level,
        base_dir=args.project_state_dir,
    )
    logger.debug("Starting RA.Aid with arguments: %s", args)

    # Initialize MCP client instance to None
    mcp_use_client_instance = None

    # Check if we need to wipe project memory before starting
    if args.wipe_project_memory:
        result = wipe_project_memory(custom_dir=args.project_state_dir)
        logger.info(result)
        print(f"📋 {result}")
        
    # Let run_interactive_mode handle the welcome message

    # Launch web interface if requested
    if args.server:
        if args.cowboy_mode:
            if not Confirm.ask(
                "WARNING: Running in server mode with cowboy mode enabled allows the Web UI " \
                "to execute shell commands without confirmation. Continue?", default=False
            ):
                print("Exiting due to user cancellation.")
                sys.exit(0)

        launch_server(args.server_host, args.server_port, args)
        return

    try:
        with DatabaseManager(base_dir=args.project_state_dir) as db:
            # Apply any pending database migrations
            try:
                migration_result = ensure_migrations_applied()
                if not migration_result:
                    logger.warning(
                        "Database migrations failed but execution will continue"
                    )
            except Exception as e:
                logger.error(f"Database migration error: {str(e)}")

            # Initialize empty config dictionary to be populated later
            config = {}

            # Initialize repositories with database connection
            # Create environment inventory data
            env_discovery = EnvDiscovery()
            env_discovery.discover()
            env_data = env_discovery.format_markdown()

            with (
                SessionRepositoryManager(db) as session_repo,
                KeyFactRepositoryManager(db) as key_fact_repo,
                KeySnippetRepositoryManager(db) as key_snippet_repo,
                HumanInputRepositoryManager(db) as human_input_repo,
                ResearchNoteRepositoryManager(db) as research_note_repo,
                RelatedFilesRepositoryManager() as related_files_repo,
                TrajectoryRepositoryManager(db) as trajectory_repo,
                WorkLogRepositoryManager() as work_log_repo,
                ConfigRepositoryManager() as config_repo,
                EnvInvManager(env_data) as env_inv,
            ):
                # This initializes all repositories and makes them available via their respective get methods

                # Initialize and register MCP-Use client for cleanup if enabled
                # Note: mcp_use_client_instance was initialized to None earlier
                active_mcp_servers = [] # Default to empty list
                logger.debug("Attempting MCP initialization...")
                if config_repo.get("mcp_use_enabled", False):
                    try:
                        mcp_use_config = config_repo.get("mcp_use_config")
                        from ra_aid.utils.mcp_use_client import MCPUseClientSync 
                        logger.info("Attempting to initialize MCP-Use client...")
                        mcp_use_client_instance = MCPUseClientSync(mcp_use_config)
                        atexit.register(mcp_use_client_instance.close)
                        
                        # --- NEW: Wait for servers and show message ---
                        console.print("[yellow]Initializing MCP servers (waiting 20s)...[/yellow]")
                        time.sleep(20)
                        # --- END NEW ---
                        
                        logger.debug("MCPUseClientSync instance created. Getting active servers...")
                        active_mcp_servers = mcp_use_client_instance.get_active_server_names()
                        logger.info(f"MCP-Use client initialized. Active servers: {active_mcp_servers}")
                    except Exception as e:
                        logger.error(f"MCP-Use client initialization failed: {e}", exc_info=True)
                        logger.warning("MCP-Use integration will be disabled for this run.")
                        # Ensure flags reflect the failure
                        config_repo.set("mcp_use_enabled", False)
                        mcp_enabled = False # Update local var too
                        active_mcp_servers = []
                        mcp_use_client_instance = None # Reset instance on failure
                else:
                    logger.debug("MCP-Use is disabled in config.")
                
                config_repo.set("active_mcp_servers", active_mcp_servers) # Store active servers
                logger.debug("MCP Servers list stored in config_repo: %s", active_mcp_servers)

                # Determine if we should start in interactive mode
                start_interactive = not args.message and not args.msg_file and not args.server and not args.chat

                # Process initial task if provided
                if not start_interactive:
                    process_task(args, mcp_use_client_instance)
                    # Show status after initial task
                    console.print(Panel(build_status(), title="Status", border_style="blue"))

                # Enter interactive mode if requested or after initial task
                if start_interactive or (not args.server and not args.chat):
                    run_interactive_mode(args, mcp_use_client_instance)
                logger.debug("Initialized SessionRepository")
                logger.debug("Initialized KeyFactRepository")
                logger.debug("Initialized KeySnippetRepository")
                logger.debug("Initialized HumanInputRepository")
                logger.debug("Initialized ResearchNoteRepository")
                logger.debug("Initialized RelatedFilesRepository")
                logger.debug("Initialized TrajectoryRepository")
                logger.debug("Initialized WorkLogRepository")
                logger.debug("Initialized ConfigRepository")
                logger.debug("Initialized Environment Inventory")

                logger.debug("Initializing new session")
                session_repo.create_session()

                check_dependencies()

                (
                    expert_enabled,
                    expert_missing,
                    web_research_enabled,
                    web_research_missing,
                ) = validate_environment(
                    args
                )  # Will exit if main env vars missing
                logger.debug("Environment validation successful")

                # Validate model configuration early
                model_config = models_params.get(args.provider, {}).get(
                    args.model or "", {}
                )
                supports_temperature = model_config.get(
                    "supports_temperature",
                    args.provider
                    in [
                        "anthropic",
                        "openai",
                        "openrouter",
                        "openai-compatible",
                        "deepseek",
                    ],
                )

                if supports_temperature and args.temperature is None:
                    args.temperature = model_config.get("default_temperature")
                    if args.temperature is None:
                        args.temperature = get_model_default_temperature(
                            args.provider, args.model
                        )
                        cpm(
                            f"This model supports temperature argument but none was given. Using model default temperature: {args.temperature}."
                        )
                    logger.debug(
                        f"Using default temperature {args.temperature} for model {args.model}"
                    )

                # Update config repo with values from CLI arguments
                config_repo.update(config)
                config_repo.set("provider", args.provider)
                config_repo.set("model", args.model)
                config_repo.set("num_ctx", args.num_ctx)
                config_repo.set("expert_provider", args.expert_provider)
                config_repo.set("expert_model", args.expert_model)
                config_repo.set("expert_num_ctx", args.expert_num_ctx)
                config_repo.set("temperature", args.temperature)
                config_repo.set(
                    "experimental_fallback_handler", args.experimental_fallback_handler
                )
                config_repo.set("web_research_enabled", web_research_enabled)
                config_repo.set("show_thoughts", args.show_thoughts)
                config_repo.set("show_cost", args.show_cost)
                config_repo.set("track_cost", args.track_cost)
                config_repo.set("force_reasoning_assistance", args.reasoning_assistance)
                config_repo.set(
                    "disable_reasoning_assistance", args.no_reasoning_assistance
                )
                config_repo.set("custom_tools", args.custom_tools)
                config_repo.set(
                    "custom_tools_enabled", True if args.custom_tools else False
                )
                # Handle MCP-Use config loading and filtering
                logger.debug("Starting MCP config determination...") # <-- ADDED
                user_mcp_config_path = args.mcp_use_config # User-specified path
                disabled_defaults = args.disable_default_mcp # List of defaults to disable, or None
                final_mcp_config_source = None # Will be dict or path string
                mcp_enabled = False
                logger.debug(f"User MCP config path: {user_mcp_config_path}") # <-- ADDED
                logger.debug(f"Disabled defaults: {disabled_defaults}") # <-- ADDED

                # Determine if we should attempt to load the default config
                should_load_defaults = True
                if user_mcp_config_path and disabled_defaults is None:
                    # Case 1: User specified a config, didn't touch --disable-default-mcp.
                    # Load ONLY the user's config.
                    logger.info(
                        f"--mcp-use-config ('{user_mcp_config_path}') provided without --disable-default-mcp. "
                        "Loading only the specified config."
                    )
                    should_load_defaults = False
                    final_mcp_config_source = user_mcp_config_path
                # Else (Case 2: No user config OR --disable-default-mcp was used): 
                # We will potentially load defaults (and maybe merge user config later).
                logger.debug(f"Should load default MCP config? {should_load_defaults}") # <-- ADDED
                
                default_config_dict = {}
                if should_load_defaults:
                    logger.debug("Attempting to load default MCP config...")
                    try:
                        # Corrected path relative to project root
                        default_config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../examples/default_mcp_servers.json')) 
                        logger.debug(f"Looking for default config at: {default_config_file_path}")
                        if os.path.isfile(default_config_file_path):
                            logger.info(f"Loading default MCP server config: {default_config_file_path}")
                            with open(default_config_file_path, 'r') as f:
                                default_config_dict = json.load(f)
                            logger.debug(f"Default config loaded: {list(default_config_dict.get('mcpServers', {}).keys())}")

                            # Filter based on --disable-default-mcp list
                            if isinstance(disabled_defaults, list):
                                logger.debug(f"Filtering default servers based on: {disabled_defaults}") # <-- ADDED
                                if not disabled_defaults: # Empty list means disable all
                                    logger.info("--disable-default-mcp used with no arguments. Disabling all default MCP servers.")
                                    default_config_dict["mcpServers"] = {}
                                else:
                                    servers_to_keep = {}
                                    disabled_set = set(disabled_defaults)
                                    logger.info(f"Disabling default MCP servers specified by flag: {disabled_set}")
                                    for name, config in default_config_dict.get("mcpServers", {}).items():
                                        if name not in disabled_set:
                                            servers_to_keep[name] = config
                                    default_config_dict["mcpServers"] = servers_to_keep
                                    logger.debug(f"Filtered default config: {list(default_config_dict.get('mcpServers', {}).keys())}") # <-- ADDED
                        else:
                            logger.warning("Default MCP config file not found.") # <-- ADDED
                    except (ImportError, FileNotFoundError, NotADirectoryError, json.JSONDecodeError) as e:
                        logger.error(f"Error loading or parsing default MCP config: {e}", exc_info=True) # <-- ADDED exc_info
                        default_config_dict = {} # Ensure it's empty on error
                        logger.error(f"Error loading or processing default MCP config: {e}. Proceeding without defaults.")
                        default_config_dict = {} # Ensure it's empty on error

                # Now, determine the final config source (path or merged dict)
                if user_mcp_config_path and should_load_defaults:
                    # Case 2a: Merge user config with (potentially filtered) defaults
                    logger.debug("Merging user MCP config with filtered defaults...") # <-- ADDED
                    try:
                        with open(user_mcp_config_path, 'r') as f:
                            user_config_dict = json.load(f)
                        merged_servers = default_config_dict.get("mcpServers", {})
                        merged_servers.update(user_config_dict.get("mcpServers", {}))
                        final_mcp_config_source = {"mcpServers": merged_servers}
                        logger.info(f"Merged user MCP config '{user_mcp_config_path}' with loaded defaults.")
                        logger.debug(f"Merged config: {list(final_mcp_config_source.get('mcpServers', {}).keys())}") # <-- ADDED
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.error(f"Error loading user-specified MCP config '{user_mcp_config_path}': {e}. Using only defaults (if any).", exc_info=True)
                        final_mcp_config_source = default_config_dict # Fallback to defaults
                elif user_mcp_config_path: # Case 1: Only user config (defaults were skipped)
                    final_mcp_config_source = user_mcp_config_path
                elif should_load_defaults: # Case 2b: Only defaults (filtered or not)
                    final_mcp_config_source = default_config_dict
                # Else: No user config, no defaults loaded -> final_mcp_config_source remains None
                logger.debug(f"Final MCP config source type: {type(final_mcp_config_source)}") # <-- ADDED
                logger.debug(f"Final MCP config source value (partial): {str(final_mcp_config_source)[:200]}...") # <-- ADDED

                # Set config repo based on the final source
                if final_mcp_config_source and (isinstance(final_mcp_config_source, str) or final_mcp_config_source.get("mcpServers")):
                    config_repo.set("mcp_use_config", final_mcp_config_source) # Store dict or path
                    mcp_enabled = True
                    logger.debug("MCP config source found, setting mcp_use_enabled=True") # <-- ADDED
                else:
                    logger.info("No MCP servers configured or enabled.")
                    mcp_enabled = False
                    logger.debug("No valid MCP config source found, mcp_use_enabled=False") # <-- ADDED
                
                config_repo.set("mcp_use_enabled", mcp_enabled)

                config_repo.set("cowboy_mode", args.cowboy_mode) # Also add here for non-server mode


                # Determine if Task Master planning integration should be enabled
                task_master_active = "taskmaster-ai" in active_mcp_servers
                task_master_planning_enabled = task_master_active and not args.disable_task_master_planning
                config_repo.set("task_master_planning_enabled", task_master_planning_enabled)
                logger.info(f"Task Master planning integration enabled: {task_master_planning_enabled}")

                # Check for required API keys if Task Master planning is enabled
                if task_master_planning_enabled:
                    if not os.getenv("ANTHROPIC_API_KEY"):
                        logger.warning("Task Master planning is enabled, but ANTHROPIC_API_KEY environment variable is not set. Task Master tools requiring it may fail.")


                # Check for GitHub token if GitHub MCP server might be active
                if "github" in active_mcp_servers:
                     if not os.getenv("GITHUB_TOKEN"):
                          logger.warning("GitHub MCP server is active, but GITHUB_TOKEN environment variable is not set. GitHub tools will likely fail.")

                # Validate custom tools function signatures (this will now load MCP tools via the instance if available)
                get_custom_tools(mcp_use_client=mcp_use_client_instance)
                custom_tools_enabled = config_repo.get("custom_tools_enabled", False)

                # Build status panel with memory statistics
                status = build_status()

                console.print(
                    Panel(
                        status,
                        title=f"RA.Aid v{__version__}",
                        border_style="bright_blue",
                        padding=(0, 1),
                    )
                )

                # Handle chat mode
                if args.chat:
                    # Initialize chat model with default provider/model
                    chat_model = initialize_llm(
                        args.provider, args.model, temperature=args.temperature
                    )

                    if args.research_only:
                        try:
                            trajectory_repo = get_trajectory_repository()
                            human_input_id = (
                                get_human_input_repository().get_most_recent_id()
                            )
                            error_message = (
                                "Chat mode cannot be used with --research-only"
                            )
                            trajectory_repo.create(
                                step_data={
                                    "display_title": "Error",
                                    "error_message": error_message,
                                },
                                record_type="error",
                                human_input_id=human_input_id,
                                is_error=True,
                                error_message=error_message,
                            )
                        except Exception as traj_error:
                            # Swallow exception to avoid recursion
                            logger.debug(f"Error recording trajectory: {traj_error}")
                            pass
                        print_error("Chat mode cannot be used with --research-only")
                        sys.exit(1)

                    print_stage_header("Chat Mode")

                    # Record stage transition in trajectory
                    trajectory_repo = get_trajectory_repository()
                    human_input_id = get_human_input_repository().get_most_recent_id()
                    trajectory_repo.create(
                        step_data={
                            "stage": "chat_mode",
                            "display_title": "Chat Mode",
                        },
                        record_type="stage_transition",
                        human_input_id=human_input_id,
                    )

                    # Get project info
                    try:
                        project_info = get_project_info(".", file_limit=2000)
                        formatted_project_info = format_project_info(project_info)
                    except Exception as e:
                        logger.warning(f"Failed to get project info: {e}")
                        formatted_project_info = ""

                    # Get initial request from user
                    initial_request = ask_human.invoke(
                        {"question": "What would you like help with?"}
                    )

                    # Record chat input in database (redundant as ask_human already records it,
                    # but needed in case the ask_human implementation changes)
                    try:
                        # Using get_human_input_repository() to access the repository from context
                        human_input_repository = get_human_input_repository()
                        # Get current session ID
                        session_id = session_repo.get_current_session_id()
                        human_input_repository.create(
                            content=initial_request,
                            source="chat",
                            session_id=session_id,
                        )
                        human_input_repository.garbage_collect()
                    except Exception as e:
                        logger.error(f"Failed to record initial chat input: {str(e)}")

                    # Get working directory and current date
                    working_directory = os.getcwd()
                    current_date = datetime.now().strftime("%Y-%m-%d")

                    # Run chat agent with CHAT_PROMPT
                    config = {
                        "configurable": {"thread_id": str(uuid.uuid4())},
                        "recursion_limit": args.recursion_limit,
                        "chat_mode": True,
                        "cowboy_mode": args.cowboy_mode,
                        "web_research_enabled": web_research_enabled,
                        "initial_request": initial_request,
                        "limit_tokens": args.disable_limit_tokens,
                    }

                    # Store config in repository
                    config_repo.update(config)
                    config_repo.set("provider", args.provider)
                    config_repo.set("model", args.model)
                    config_repo.set("num_ctx", args.num_ctx)
                    config_repo.set("expert_provider", args.expert_provider)
                    config_repo.set("expert_model", args.expert_model)
                    config_repo.set("expert_num_ctx", args.expert_num_ctx)
                    config_repo.set("temperature", args.temperature)
                    config_repo.set("show_thoughts", args.show_thoughts)
                    config_repo.set("show_cost", args.show_cost)
                    config_repo.set("track_cost", args.track_cost)
                    config_repo.set(
                        "force_reasoning_assistance", args.reasoning_assistance
                    )
                    config_repo.set(
                        "disable_reasoning_assistance", args.no_reasoning_assistance
                    )
                    config_repo.set("cowboy_mode", args.cowboy_mode) # Chat mode also needs cowboy mode

                    # Set modification tools based on use_aider flag
                    set_modification_tools(args.use_aider)

                    # Create chat agent with appropriate tools
                    chat_agent = create_agent(
                        chat_model,
                        get_chat_tools(
                            expert_enabled=expert_enabled,
                            web_research_enabled=web_research_enabled,
                        ),
                        checkpointer=MemorySaver(),
                    )

                    # Run chat agent and exit
                    run_agent_with_retry(
                        chat_agent,
                        CHAT_PROMPT.format(
                            initial_request=initial_request,
                            web_research_section=(
                                WEB_RESEARCH_PROMPT_SECTION_CHAT
                                if web_research_enabled
                                else ""
                            ),
                            custom_tools_section=(
                                DEFAULT_CUSTOM_TOOLS_PROMPT
                                if custom_tools_enabled
                                else ""
                            ),
                            working_directory=working_directory,
                            current_date=current_date,
                            key_facts=format_key_facts_dict(
                                get_key_fact_repository().get_facts_dict()
                            ),
                            key_snippets=format_key_snippets_dict(
                                get_key_snippet_repository().get_snippets_dict()
                            ),
                            project_info=formatted_project_info,
                            env_inv=get_env_inv(),
                        ),
                        config,
                    )
                    return

                # Validate message is provided
                # (This check is no longer needed as interactive mode is handled within the main with block)
                # if (
                #     not args.message and not args.wipe_project_memory
                # ):  # Add check for wipe_project_memory flag
                #     # Instead of showing an error, set a flag to enter interactive mode later
                #     args.enter_interactive_mode = True

                # Initialize base_task with a default value
                base_task = ""
                if args.message:  # Set base_task if message exists
                    base_task = args.message

                # Record CLI input in database
                try:
                    # Using get_human_input_repository() to access the repository from context
                    human_input_repository = get_human_input_repository()
                    # Get current session ID
                    session_id = session_repo.get_current_session_id()
                    human_input_repository.create(
                        content=base_task, source="cli", session_id=session_id
                    )
                    # Run garbage collection to ensure we don't exceed 100 inputs
                    human_input_repository.garbage_collect()
                    logger.debug(f"Recorded CLI input: {base_task}")
                except Exception as e:
                    logger.error(f"Failed to record CLI input: {str(e)}")
                config = {
                    "configurable": {"thread_id": str(uuid.uuid4())},
                    "recursion_limit": args.recursion_limit,
                    "research_only": args.research_only,
                    "cowboy_mode": args.cowboy_mode,
                    "web_research_enabled": web_research_enabled,
                    "aider_config": args.aider_config,
                    "use_aider": args.use_aider,
                    "limit_tokens": args.disable_limit_tokens,
                    "auto_test": args.auto_test,
                    "test_cmd": args.test_cmd,
                    "max_test_cmd_retries": args.max_test_cmd_retries,
                    "experimental_fallback_handler": args.experimental_fallback_handler,
                    "test_cmd_timeout": args.test_cmd_timeout,
                }

                # Store config in repository
                config_repo.update(config)

                # Store base provider/model configuration
                config_repo.set("provider", args.provider)
                config_repo.set("model", args.model)
                config_repo.set("num_ctx", args.num_ctx)

                # Store expert provider/model (no fallback)
                config_repo.set("expert_provider", args.expert_provider)
                config_repo.set("expert_model", args.expert_model)
                config_repo.set("expert_num_ctx", args.expert_num_ctx)

                # Store planner config with fallback to base values
                config_repo.set(
                    "planner_provider", args.planner_provider or args.provider
                )
                config_repo.set("planner_model", args.planner_model or args.model)

                # Store research config with fallback to base values
                config_repo.set(
                    "research_provider", args.research_provider or args.provider
                )
                config_repo.set("research_model", args.research_model or args.model)

                # Store temperature in config
                config_repo.set("temperature", args.temperature)

                # Store reasoning assistance flags
                config_repo.set("force_reasoning_assistance", args.reasoning_assistance)
                config_repo.set(
                    "disable_reasoning_assistance", args.no_reasoning_assistance
                )
                # Store cowboy_mode for the main agent run
                config_repo.set("cowboy_mode", args.cowboy_mode)

                # Set modification tools based on use_aider flag
                set_modification_tools(args.use_aider)

                # Run research stage
                print_stage_header("Research Stage")

                # Record stage transition in trajectory
                trajectory_repo = get_trajectory_repository()
                human_input_id = get_human_input_repository().get_most_recent_id()
                trajectory_repo.create(
                    step_data={
                        "stage": "research_stage",
                        "display_title": "Research Stage",
                    },
                    record_type="stage_transition",
                    human_input_id=human_input_id,
                )

                # Initialize research model with potential overrides
                research_provider = args.research_provider or args.provider
                research_model_name = args.research_model or args.model
                research_model = initialize_llm(
                    research_provider, research_model_name, temperature=args.temperature
                )

                run_research_agent(
                    base_task,
                    research_model,
                    expert_enabled=expert_enabled,
                    research_only=args.research_only,
                    hil=args.hil,
                    memory=research_memory,
                )

                # for how long have we had a second planning agent triggered here?

                # After task completion in normal mode, set flag to enter interactive mode
                # (This logic is now handled within the main with block)
                # if not args.server and not args.chat:
                #     # Clear message args to prompt for next task
                #     args.message = None
                #     args.msg_file = None
                #     
                #     # Show status after task completion
                #     console.print(Panel(build_status(), title="Status", border_style="blue"))
                #     
                #     # Set flag to enter interactive mode after completing the current task
                #     args.enter_interactive_mode = True

    except (KeyboardInterrupt, AgentInterrupt):
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
