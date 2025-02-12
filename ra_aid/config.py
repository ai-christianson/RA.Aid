"""Configuration utilities."""

DEFAULT_RECURSION_LIMIT = 100
DEFAULT_MAX_TEST_CMD_RETRIES = 3
DEFAULT_MAX_TOOL_FAILURES = 2
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

from ra_aid.agents.ciayn_agent import CiaynAgent
from langgraph.graph.graph import CompiledGraph

RAgents = CompiledGraph | CiaynAgent
