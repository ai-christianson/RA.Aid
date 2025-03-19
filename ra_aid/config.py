"""Configuration utilities."""

DEFAULT_RECURSION_LIMIT = 100
DEFAULT_MAX_TEST_CMD_RETRIES = 3
DEFAULT_MAX_TOOL_FAILURES = 3
FALLBACK_TOOL_MODEL_LIMIT = 5
RETRY_FALLBACK_COUNT = 3
DEFAULT_TEST_CMD_TIMEOUT = 60 * 5  # 5 minutes in seconds
DEFAULT_MODEL="claude-3-7-sonnet-20250219"
DEFAULT_SHOW_COST = False


VALID_PROVIDERS = [
    "anthropic",
    "openai",
    "openrouter",
    "openai-compatible",
    "deepseek",
    "gemini",
    "ollama",
]
