import argparse
import logging
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv

# Load .env file early, before any other imports that might read environment variables
load_dotenv()

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
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_TEST_CMD_TIMEOUT,
    VALID_PROVIDERS,
    DEFAULT_EXPERT_ANTHROPIC_MODEL,
    DEFAULT_EXPERT_GEMINI_MODEL,
    DEFAULT_EXPERT_OPENAI_MODEL,
    DEFAULT_EXPERT_DEEPSEEK_MODEL
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
from ra_aid.tool_configs import get_chat_tools, set_modification_tools, get_custom_tools
from ra_aid.tools.human import ask_human

logger = get_logger(__name__)


def store_limit_config(config_repo, args):
    """Store limit-related configuration values in the repository.
    
    Args:
        config_repo: ConfigRepository instance
        args: Parsed command line arguments
    """
    config_repo.set("max_cost", args.max_cost)
    config_repo.set("max_tokens", args.max_tokens)
    config_repo.set("exit_at_limit", args.exit_at_limit)

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
            "makehub",
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
                "max_cost": args.max_cost,
                "max_tokens": args.max_tokens,
                "exit_at_limit": args.exit_at_limit,
            }
        )

        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


def parse_arguments(args=None):

    # Case-insensitive log level argument type
    def log_level_type(value):
        value = value.lower()
        if value not in ["debug", "info", "warning", "error", "critical"]:
            raise argparse.ArgumentTypeError(
                f"Invalid log level: {value}. Choose from debug, info, warning, error, critical."
            )
        return value

    parser = argparse.ArgumentParser(
        description="RA.Aid - AI Agent for executing programming and research tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Epilog will be added after subparsers are defined
    )

    # Global arguments (relevant for agent/server modes and some script commands)
    parser.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )
    parser.add_argument(
        "--log-mode",
        choices=["console", "file"],
        default="file",
        help="Logging mode: 'console' shows all logs in console, 'file' logs to file with only warnings+ in console (default: file)",
    )
    parser.add_argument(
        "--pretty-logger", action="store_true", help="Enable pretty logging output"
    )
    parser.add_argument(
        "--log-level",
        type=log_level_type,
        default="debug",
        help="Set specific logging level (case-insensitive, affects file and console logging based on --log-mode, default: debug)",
    )
    # Version argument needs to be accessible early
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version number and exit",
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command", help="Available commands. Use <command> --help for specific command help.")
    subparsers.required = False # Make command optional for default agent behavior

    # Agent/Server related arguments (will be grouped or handled if no script command is given)
    # These are defined here so they appear in the main help, but primarily used when no script subcommand is chosen.
    # Alternatively, they could be added to a "run" or "agent" subcommand. For now, keep them global.

    agent_parser = subparsers.add_parser("agent", help="Run the AI agent (default if no command is specified).", add_help=False) # add_help=False to avoid conflict with main parser's args

    # Arguments for the main agent functionality (original parser arguments)
    # These are added to the main parser so they can be used if no subcommand is given,
    # or potentially passed to an "agent" subcommand in the future.
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
    # Removed redundant --version argument here, it's already defined globally.
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Only perform research without implementation",
    )
    parser.add_argument(
        "--research-and-plan-only", "-rap",
        action="store_true",
        help="Run research and planning, then exit.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=(
            "gemini"
            if os.getenv("GEMINI_API_KEY")
            else ("openai" if os.getenv("OPENAI_API_KEY") else DEFAULT_PROVIDER)
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
        "--supervisor",
        action="store_true",
        default=True,
        help="Enable supervisor mode: expert plans tasks, workers execute (default: on)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy agent architecture instead of supervisor mode",
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
        "--skip-version-check",
        action="store_true",
        help="Skip checking for newer versions on startup (faster startup)",
    )
    parser.add_argument(
        "--skip-env-discovery",
        action="store_true",
        help="Skip environment discovery on startup (faster startup, but environment info won't be available to agents)",
    )
    parser.add_argument(
        "--fast-startup",
        action="store_true",
        default=True,
        help="Enable fast startup mode (implies --skip-version-check and --skip-env-discovery). Enabled by default.",
    )
    parser.add_argument(
        "--no-fast-startup",
        action="store_false",
        dest="fast_startup",
        help="Disable fast startup mode",
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
    # Removed redundant --project-state-dir argument here, it's already defined globally.
    parser.add_argument(
        "--show-thoughts",
        action="store_true",
        default=True,
        help="Display model thinking content extracted from think tags when supported by the model. Enabled by default.",
    )
    parser.add_argument(
        "--no-show-thoughts",
        action="store_false",
        dest="show_thoughts",
        help="Hide model thinking content",
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
        "--max-cost",
        type=float,
        default=None,
        help="Maximum cost threshold in USD (positive float)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token threshold (positive integer)",
    )
    parser.add_argument(
        "--exit-at-limit",
        action="store_true",
        help="Exit immediately without prompt when limits are reached",
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
        "--set-default-provider",
        type=str,
        choices=VALID_PROVIDERS,
        help="Set the default provider to use for future runs",
    )
    parser.add_argument(
        "--set-default-model",
        type=str,
        help="Set the default model to use for future runs",
    )
    parser.add_argument(
        "--price-performance-ratio",
        type=float,
        help="Price-performance ratio for Makehub API (0.0-1.0, where 0.0 prioritizes speed and 1.0 prioritizes cost efficiency)",
    )

    # --- Script Subcommands ---

    # last-cost
    parser_last_cost = subparsers.add_parser("last-cost", help="Display cost and token usage for the latest session.")
    parser_last_cost.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # all-costs
    parser_all_costs = subparsers.add_parser("all-costs", help="Display cost and token usage for all sessions.")
    parser_all_costs.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # extract-plan
    parser_extract_plan = subparsers.add_parser("extract-plan", help="Extract the plan for a session. Defaults to the latest session.")
    parser_extract_plan.add_argument("session_id", type=int, nargs='?', default=None, help="The ID of the session. If omitted, the latest session is used.")
    parser_extract_plan.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # extract-last-plan
    parser_extract_last_plan = subparsers.add_parser("extract-last-plan", help="Extract the plan for the most recent session.")
    parser_extract_last_plan.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # extract-last-research-notes
    parser_extract_last_research_notes = subparsers.add_parser("extract-last-research-notes", help="Extract research notes for the most recent session.")
    parser_extract_last_research_notes.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )
    
    # generate-openapi
    parser_generate_openapi = subparsers.add_parser("generate-openapi", help="Generate and print the OpenAPI specification for the server.")

    # create-migration
    parser_create_migration = subparsers.add_parser("create-migration", help="Create a new database migration.")
    parser_create_migration.add_argument("name", type=str, help="Name for the new migration (e.g., add_user_table).")
    parser_create_migration.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # migrate
    parser_migrate = subparsers.add_parser("migrate", help="Run all pending database migrations.")
    parser_migrate.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # migration-status
    parser_migration_status = subparsers.add_parser("migration-status", help="Show the current status of database migrations.")
    parser_migration_status.add_argument(
        "--project-state-dir",
        help="Directory to store project state (database and logs). By default, a .ra-aid directory is created in the current working directory.",
    )

    # extract-changelog
    parser_extract_changelog = subparsers.add_parser("extract-changelog", help="Extract changelog entries for a specific version from CHANGELOG.md.")
    parser_extract_changelog.add_argument("version", type=str, help="The version string to extract (e.g., 0.30.0).")

    # Update epilog with examples including new subcommands
    parser.epilog = """
Examples:
  ra-aid -m "Add error handling to the database module"
  ra-aid --server
  ra-aid last-cost
  ra-aid extract-plan 123
  ra-aid create-migration add_new_feature
  ra-aid extract-changelog 0.25.0
    """

    if args is None:
        args = sys.argv[1:]
    
    # If only 'ra-aid' is run with no subcommand or message, it should show help or default to agent.
    # For now, parse_args will handle if a command is required or not.
    # If sys.argv is just the program name, args will be empty.
    # If first arg is not a recognized command or option, it might be treated as agent message if -m is not used.
    # This needs careful handling if we want `ra-aid "my task"` to work without -m.
    # Current setup: `ra-aid -m "task"` or `ra-aid <subcommand>`.

    parsed_args = parser.parse_args(args)

    # If no command is specified, and no message/server/wipe flag, show help.
    # However, the default behavior is to run the agent.
    # We'll check `parsed_args.command` later to dispatch.

    # Validate message vs msg-file usage (only if not a script command that ignores messages)
    if not parsed_args.command or parsed_args.command not in [
        "last-cost", "all-costs", "extract-plan", "extract-last-plan", 
        "extract-last-research-notes", "generate-openapi", "create-migration", 
        "migrate", "migration-status", "extract-changelog"
    ]:
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
            parsed_args.model = parsed_args.model or DEFAULT_OPENAI_MODEL
        elif parsed_args.provider == "anthropic":
            # Use default model for Anthropic only if not specified
            parsed_args.model = parsed_args.model or DEFAULT_ANTHROPIC_MODEL
        elif parsed_args.provider == "gemini":
            parsed_args.model = parsed_args.model or DEFAULT_GEMINI_MODEL
        elif parsed_args.provider == "openrouter":
            parsed_args.model = parsed_args.model or DEFAULT_OPENROUTER_MODEL
        elif not parsed_args.model and not parsed_args.research_only:
            # Require model for other providers unless in research mode
            if not hasattr(parsed_args, 'command') or parsed_args.command is None or parsed_args.command == "agent":
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
                parsed_args.expert_model = parsed_args.expert_model or DEFAULT_EXPERT_ANTHROPIC_MODEL
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
                # Use default Gemini model if not specified
                parsed_args.expert_model = parsed_args.expert_model or DEFAULT_EXPERT_GEMINI_MODEL
            elif os.environ.get("DEEPSEEK_API_KEY"): # Check main Deepseek key as fallback
                parsed_args.expert_provider = "deepseek"
                parsed_args.expert_model = DEFAULT_EXPERT_DEEPSEEK_MODEL # Specific default for Deepseek expert
            elif os.environ.get("ANTHROPIC_API_KEY"): # Use Anthropic for expert if available
                parsed_args.expert_provider = "anthropic"
                parsed_args.expert_model = parsed_args.expert_model or DEFAULT_EXPERT_ANTHROPIC_MODEL
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

    # Validate max_cost is positive if provided
    if parsed_args.max_cost is not None and parsed_args.max_cost <= 0:
        parser.error("--max-cost must be a positive number")

    # Validate max_tokens is positive if provided
    if parsed_args.max_tokens is not None and parsed_args.max_tokens <= 0:
        parser.error("--max-tokens must be a positive integer")

    # Validate price-performance-ratio range only for MakeHub provider
    if parsed_args.provider == "makehub" and parsed_args.price_performance_ratio is not None:
        if not (0.0 <= parsed_args.price_performance_ratio <= 1.0):
            parser.error("--price-performance-ratio must be between 0.0 and 1.0")

    return parsed_args


# Create console instance (global for the module)
console = Console()

# --- Handler functions for script subcommands ---
# Note: These will be defined before `main()` uses them.

def handle_last_cost(args):
    import json
    from ra_aid.scripts.last_session_usage import get_latest_session_usage
    result, status_code = get_latest_session_usage(
        project_dir=args.project_state_dir, 
        db_path=None # db_path is not exposed as a direct CLI arg for subcommands yet
    )
    console.print(json.dumps(result, indent=2))
    sys.exit(status_code)

def handle_all_costs(args):
    import json
    from ra_aid.scripts.all_sessions_usage import get_all_sessions_usage
    results, status_code = get_all_sessions_usage(
        project_dir=args.project_state_dir,
        db_path=None 
    )
    console.print(json.dumps(results, indent=2))
    sys.exit(status_code)

def handle_extract_plan(args):
    from ra_aid.scripts.extract_plan import get_plan_for_session
    from ra_aid.scripts.extract_last_plan import main as extract_last_plan_main
    
    if args.session_id is not None:
        plan = get_plan_for_session(args.session_id, project_state_dir=args.project_state_dir)
        if plan:
            console.print(f"Plan for session {args.session_id}:")
            console.print(plan)
        else:
            console.print(f"No plan found for session {args.session_id}.")
    else:
        # No session_id provided, use extract_last_plan logic
        extract_last_plan_main(project_state_dir=args.project_state_dir)
    sys.exit(0)

def handle_extract_last_plan(args):
    from ra_aid.scripts.extract_last_plan import main as extract_last_plan_main
    extract_last_plan_main(project_state_dir=args.project_state_dir)
    sys.exit(0)

def handle_extract_last_research_notes(args):
    from ra_aid.scripts.extract_last_research_notes import main as extract_last_research_notes_main
    extract_last_research_notes_main(project_state_dir=args.project_state_dir)
    sys.exit(0)

def handle_generate_openapi(args):
    from ra_aid.scripts.generate_openapi import main as generate_openapi_main
    generate_openapi_main() # This script prints to stdout
    sys.exit(0)

def handle_create_migration(args):
    from ra_aid.database.migrations import create_new_migration
    # Ensure DB context for migration path resolution if needed, though create_new_migration might handle paths independently
    # For safety, operations that might implicitly init DBManager should be aware of project_state_dir
    # However, create_new_migration primarily deals with file system paths based on project structure.
    # It might be fine without a full DBManager context if it resolves paths correctly.
    # Let's assume it works correctly with current project path detection or add chdir if problems arise.
    # The `get_migrations_dir` used by `create_new_migration` should ideally respect `project_state_dir`.
    # This is a deeper refactor. For now, we call it directly.
    # If `project_state_dir` is set, `create_new_migration` needs to know where `.ra-aid/migrations` should be.
    # This might require `create_new_migration` to accept `base_dir`.
    # For now, we assume it uses cwd correctly or that `project_state_dir` is handled by underlying path functions.
    
    console.print(f"Creating migration: [cyan]{args.name}[/cyan]")
    # `create_new_migration` needs to be aware of `project_state_dir` if it's not the CWD.
    # This is a limitation if `create_new_migration` doesn't take `base_dir`.
    # A temporary workaround could be `os.chdir` if `args.project_state_dir` is set.
    original_cwd = None
    if args.project_state_dir and os.getcwd() != os.path.abspath(args.project_state_dir):
        original_cwd = os.getcwd()
        try:
            os.chdir(args.project_state_dir)
            logger.info(f"Changed CWD to {args.project_state_dir} for migration creation.")
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Project state directory not found: {args.project_state_dir}")
            sys.exit(1)
            
    result = create_new_migration(args.name, auto=True)

    if original_cwd:
        os.chdir(original_cwd)
        logger.info(f"Restored CWD to {original_cwd}.")

    if result:
        console.print(f"[bold green]Successfully created migration:[/bold green] {result}")
    else:
        console.print("[bold red]Failed to create migration.[/bold red]")
        sys.exit(1)
    sys.exit(0)

def handle_migrate(args):
    from ra_aid.database.migrations import ensure_migrations_applied
    console.print("Applying pending migrations...")
    # ensure_migrations_applied should be called within a DatabaseManager context
    # that is initialized with args.project_state_dir
    with DatabaseManager(base_dir=args.project_state_dir):
        success, error_message = ensure_migrations_applied()
    if success:
        console.print("[bold green]Migrations applied successfully (or no pending migrations).[/bold green]")
    else:
        if error_message:
            console.print(f"[bold red]Failed to apply migrations.[/bold red]\nDetails: {error_message}")
        else:
            console.print("[bold red]Failed to apply migrations. No specific error detail provided.[/bold red]")
        sys.exit(1)
    sys.exit(0)

def handle_migration_status(args):
    from ra_aid.database.migrations import get_migration_status
    from rich.table import Table as RichTable # Alias to avoid conflict with other Table types

    console.print("Checking migration status...")
    with DatabaseManager(base_dir=args.project_state_dir):
        status = get_migration_status()
    
    if "error" in status:
        console.print(f"[bold red]Error getting status:[/bold red] {status['error']}")
        sys.exit(1)

    table = RichTable(title="Migration Status")
    table.add_column("Status", justify="right", style="cyan", no_wrap=True)
    table.add_column("Count", justify="left", style="magenta")
    table.add_row("Applied", str(status.get("applied_count", 0)))
    table.add_row("Pending", str(status.get("pending_count", 0)))
    console.print(table)

    if status.get("pending"):
        console.print("\n[bold yellow]Pending Migrations:[/bold yellow]")
        for p in status["pending"]:
            console.print(f"- {p}")
    sys.exit(0)

def handle_extract_changelog(args):
    from ra_aid.scripts.extract_changelog import extract_version_content # Use the core function
    from pathlib import Path

    version = args.version
    # CHANGELOG.md is assumed to be in project root, 3 levels up from __main__.py's script dir
    # This path logic might need adjustment if ra-aid is installed as a package.
    # For development: Path(__file__).parent.parent.parent / "CHANGELOG.md"
    # For installed package: This needs a robust way to find CHANGELOG.md relative to package or CWD.
    # Assuming CWD for now if run from project root.
    changelog_path = Path("CHANGELOG.md") # Simpler assumption, user runs from project root
    if not changelog_path.exists():
        # Try relative to __file__ as a fallback (might work in some editable installs)
        script_dir_changelog_path = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
        if script_dir_changelog_path.exists():
            changelog_path = script_dir_changelog_path
        else:
            console.print(f"Error: Could not find {changelog_path} or {script_dir_changelog_path}", file=sys.stderr)
            sys.exit(1)
            
    try:
        content = changelog_path.read_text()
        version_content = extract_version_content(content, version)
        console.print(version_content)
    except FileNotFoundError:
        console.print(f"Error: Could not find {changelog_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # Raised by extract_version_content if version not found
        console.print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        console.print(f"Error reading changelog or extracting content: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


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


def build_status(skip_version_check=False):
    """Build status panel with model and feature information.

    Includes memory statistics at the bottom with counts of key facts, snippets, and research notes.

    Args:
        skip_version_check: If True, skip checking for newer versions
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

    # Check for newer version (unless skipped)
    if not skip_version_check:
        version_message = check_for_newer_version()
        if version_message:
            status.append("\n\n")
            status.append(version_message, style="yellow")

    return status


def main():
    """Main entry point for the ra-aid command line tool."""
    args = parse_arguments() # This now parses global args and subcommands

    # Handle --fast-startup flag by enabling skip flags
    if args.fast_startup:
        args.skip_version_check = True
        args.skip_env_discovery = True

    # Setup logging early. project_state_dir is a global arg.
    setup_logging(
        args.log_mode,
        args.pretty_logger,
        args.log_level,
        base_dir=args.project_state_dir, # Pass explicitly
    )
    logger.debug("Starting RA.Aid with arguments: %s", args)

    # Dispatch to subcommand handlers if a command is given
    if args.command:
        if args.command == "last-cost":
            handle_last_cost(args)
        elif args.command == "all-costs":
            handle_all_costs(args)
        elif args.command == "extract-plan":
            handle_extract_plan(args)
        elif args.command == "extract-last-plan":
            handle_extract_last_plan(args)
        elif args.command == "extract-last-research-notes":
            handle_extract_last_research_notes(args)
        elif args.command == "generate-openapi":
            handle_generate_openapi(args)
        elif args.command == "create-migration":
            handle_create_migration(args)
        elif args.command == "migrate":
            handle_migrate(args)
        elif args.command == "migration-status":
            handle_migration_status(args)
        elif args.command == "extract-changelog":
            handle_extract_changelog(args)
        # Add other command dispatches here
        # If a command was handled, the handler function should sys.exit()
        # If we reach here after a command, it means it wasn't a script command or didn't exit.
        # This part might need refinement based on how subcommands are structured.
        # For now, assuming script handlers exit. Agent/server logic follows.
        # If args.command was "agent", it will fall through.

    # If no specific script-like command was run and exited, proceed with agent/server logic.
    # Check if we need to set default provider or model (this is a global option, not a subcommand)
    from ra_aid.config import save_default_values

    if args.set_default_provider or args.set_default_model:
        values_to_save = {}
        if args.set_default_provider:
            values_to_save["provider"] = args.set_default_provider
            logger.info(f"Setting default provider to: {args.set_default_provider}")
            print(f"✅ Default provider set to: {args.set_default_provider}")

        if args.set_default_model:
            values_to_save["model"] = args.set_default_model
            logger.info(f"Setting default model to: {args.set_default_model}")
            print(f"✅ Default model set to: {args.set_default_model}")

        # Save the values to the configuration file
        save_default_values(values_to_save, args.project_state_dir)

        # If only setting defaults and no other operation, exit
        if not args.message and not args.msg_file and not args.server and not args.wipe_project_memory:
            return

    # Check if we need to wipe project memory before starting
    if args.wipe_project_memory:
        result = wipe_project_memory(custom_dir=args.project_state_dir)
        logger.info(result)
        print(f"📋 {result}")

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
                migration_success, migration_error_msg = ensure_migrations_applied()
                if not migration_success:
                    logger.warning(
                        f"Database migrations failed but execution will continue. Error: {migration_error_msg or 'No specific error detail provided.'}"
                    )
            except Exception as e:
                logger.error(f"Unexpected database migration error: {str(e)}")

            # Initialize empty config dictionary to be populated later
            config = {}

            # Initialize repositories with database connection
            # Create environment inventory data (skip if requested for faster startup)
            if args.skip_env_discovery:
                logger.debug("Skipping environment discovery (--skip-env-discovery)")
                env_data = "Environment discovery skipped for faster startup."
            else:
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
                        "makehub",
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
                store_limit_config(config_repo, args)
                config_repo.set("research_and_plan_only", args.research_and_plan_only)
                config_repo.update(config)
                config_repo.set("provider", args.provider)
                config_repo.set("model", args.model)
                config_repo.set("num_ctx", args.num_ctx)
                config_repo.set("expert_provider", args.expert_provider)
                config_repo.set("expert_model", args.expert_model)
                config_repo.set("expert_num_ctx", args.expert_num_ctx)
                config_repo.set("temperature", args.temperature)
                config_repo.set("price_performance_ratio", args.price_performance_ratio)
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
                config_repo.set("cowboy_mode", args.cowboy_mode) # Also add here for non-server mode

                # Validate custom tools function signatures
                get_custom_tools()
                custom_tools_enabled = config_repo.get("custom_tools_enabled", False)

                # Build status panel with memory statistics
                status = build_status(skip_version_check=args.skip_version_check)

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

                    # Get project info (skip deep scan if fast startup is enabled)
                    # Always run pwd and ls -la for basic context
                    import subprocess
                    try:
                        pwd_result = subprocess.run(['pwd'], capture_output=True, text=True, timeout=5)
                        ls_result = subprocess.run(['ls', '-la'], capture_output=True, text=True, timeout=5)
                        quick_context = f"Current Directory:\n{pwd_result.stdout}\nDirectory Listing:\n{ls_result.stdout}"
                    except Exception as e:
                        logger.warning(f"Failed to get directory context: {e}")
                        quick_context = ""

                    if args.fast_startup or args.skip_env_discovery:
                        logger.debug("Skipping deep project info scan for faster startup")
                        formatted_project_info = f"{quick_context}\n\n(Deep project scan skipped for faster startup. Use fuzzy_find_project_files or list_dir to explore further.)"
                    else:
                        try:
                            project_info = get_project_info(".", file_limit=2000)
                            formatted_project_info = f"{quick_context}\n\n{format_project_info(project_info)}"
                        except Exception as e:
                            logger.warning(f"Failed to get project info: {e}")
                            formatted_project_info = quick_context

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
                    config_repo.set("price_performance_ratio", args.price_performance_ratio)
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

                # Handle supervisor mode (default) unless --legacy is specified
                if args.supervisor and not getattr(args, 'legacy', False):
                    print_stage_header("Supervisor Mode")

                    # Initialize models
                    # Expert model = main model (or expert model if specified)
                    expert_provider = args.expert_provider or args.provider
                    expert_model_name = args.expert_model or args.model

                    expert_model = initialize_llm(
                        expert_provider,
                        expert_model_name,
                        temperature=args.temperature,
                    )

                    # Worker model = can be cheaper/faster
                    # For now, use the same model, but could be configured separately
                    worker_model = initialize_llm(
                        args.provider,
                        args.model,
                        temperature=args.temperature,
                    )

                    # Import and run supervisor
                    from ra_aid.agents.supervisor import run_supervisor

                    result = run_supervisor(
                        user_request=args.message or "",
                        expert_model=expert_model,
                        worker_model=worker_model,
                    )

                    # Print final result
                    print()
                    console.print(Panel(
                        Markdown(result),
                        title="Supervisor Complete",
                        border_style="green bold",
                    ))
                    return

                # Validate message is provided
                if (
                    not args.message and not args.wipe_project_memory
                ):  # Add check for wipe_project_memory flag
                    error_message = "--message or --msg-file is required. Use --help for available commands."
                    try:
                        trajectory_repo = get_trajectory_repository()
                        human_input_id = (
                            get_human_input_repository().get_most_recent_id()
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
                    print_error(error_message)
                    sys.exit(1)

                base_task = "" # Initialize base_task
                if args.message:  # Only set base_task if message exists
                    base_task = args.message

                # Record CLI input in database
                human_input_id = None # Initialize before try
                session_id = None # Initialize before try
                try:
                    human_input_repository = get_human_input_repository()
                    session_id = session_repo.get_current_session_id() # Capture session_id here
                    human_input_record = human_input_repository.create( # Capture record
                        content=base_task, source="cli", session_id=session_id
                    )
                    human_input_id = human_input_record.id # Get ID
                    # Run garbage collection to ensure we don't exceed 100 inputs
                    human_input_repository.garbage_collect()
                    logger.debug(f"Recorded CLI input: {base_task}")
                except Exception as e:
                    logger.error(f"Failed to record CLI input: {str(e)}")
                    human_input_id = None # Ensure None on failure

                if human_input_id:
                    try:
                        trajectory_repo = get_trajectory_repository() # Get the repository instance
                        logger.debug(f"Creating user_query trajectory record for session {session_id} (CLI), human_input_id {human_input_id}.")
                        trajectory_repo.create(
                            session_id=session_id,
                            human_input_id=human_input_id,
                            record_type="user_query",
                            step_data={
                                "display_title": "User Query",
                                "query": base_task,
                            },
                        )
                        logger.info(f"Created user_query trajectory for session {session_id} (CLI).")
                    except Exception as e:
                        logger.exception(f"Error creating user_query trajectory for session {session_id} (CLI): {e}")
                else:
                    logger.warning(f"Skipping user_query trajectory creation for session {session_id} (CLI) due to missing human_input_id.")

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

                # Display message if research and plan only mode is enabled
                if args.research_and_plan_only:
                    cpm("Research and plan only mode enabled. The agent will exit after planning.")

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
                config_repo.set("price_performance_ratio", args.price_performance_ratio)

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
                # Use the human_input_id captured earlier for stage transition
                trajectory_repo.create(
                    step_data={
                        "stage": "research_stage",
                        "display_title": "Research Stage",
                    },
                    record_type="stage_transition",
                    human_input_id=human_input_id, # Pass the potentially None ID
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

                if args.research_and_plan_only:
                    # If this flag is active, the research agent is expected to handle
                    # the plan creation and emission. We exit after research.
                    print_stage_header(
                        "Research phase complete. Plan and notes saved. Exiting as per --research-and-plan-only.\n"
                        "You can extract the plan using 'ra-aid extract-last-plan' "
                        "and research notes using 'ra-aid extract-last-research-notes'."
                    )
                    sys.exit(0)

                # for how long have we had a second planning agent triggered here?

    except (KeyboardInterrupt, AgentInterrupt):
        print()
        print(" 👋 Bye!")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
