"""Configuration utilities."""

DEFAULT_RECURSION_LIMIT = 100
DEFAULT_MAX_TEST_CMD_RETRIES = 3
DEFAULT_MAX_TOOL_FAILURES = 3
FALLBACK_TOOL_MODEL_LIMIT = 5
RETRY_FALLBACK_COUNT = 3
RETRY_FALLBACK_DELAY = 2

VALID_PROVIDERS = [
    "anthropic",
    "openai",
    "openrouter",
    "openai-compatible",
    "deepseek",
    "gemini",
]
