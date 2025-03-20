"""Custom callback handlers for tracking token usage and costs."""

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from playhouse.shortcuts import model_to_dict

from ra_aid.config import DEFAULT_MODEL
from ra_aid.database.repositories.session_repository import SessionRepository
from ra_aid.database.repositories.trajectory_repository import TrajectoryRepository
from ra_aid.logging_config import get_logger
from ra_aid.utils.singleton import Singleton

logger = get_logger(__name__)

# Define cost per 1K tokens for various models
ANTHROPIC_MODEL_COSTS = {
    # Claude 3.7 Sonnet input
    "claude-3-7-sonnet-20250219": 0.003,
    "anthropic/claude-3.7-sonnet": 0.003,
    "claude-3.7-sonnet": 0.003,
    # Claude 3.7 Sonnet output
    "claude-3-7-sonnet-20250219-completion": 0.015,
    "anthropic/claude-3.7-sonnet-completion": 0.015,
    "claude-3.7-sonnet-completion": 0.015,
    # Claude 3 Opus input
    "claude-3-opus-20240229": 0.015,
    "anthropic/claude-3-opus": 0.015,
    "claude-3-opus": 0.015,
    # Claude 3 Opus output
    "claude-3-opus-20240229-completion": 0.075,
    "anthropic/claude-3-opus-completion": 0.075,
    "claude-3-opus-completion": 0.075,
    # Claude 3 Sonnet input
    "claude-3-sonnet-20240229": 0.003,
    "anthropic/claude-3-sonnet": 0.003,
    "claude-3-sonnet": 0.003,
    # Claude 3 Sonnet output
    "claude-3-sonnet-20240229-completion": 0.015,
    "anthropic/claude-3-sonnet-completion": 0.015,
    "claude-3-sonnet-completion": 0.015,
    # Claude 3 Haiku input
    "claude-3-haiku-20240307": 0.00025,
    "anthropic/claude-3-haiku": 0.00025,
    "claude-3-haiku": 0.00025,
    # Claude 3 Haiku output
    "claude-3-haiku-20240307-completion": 0.00125,
    "anthropic/claude-3-haiku-completion": 0.00125,
    "claude-3-haiku-completion": 0.00125,
    # Claude 2 input
    "claude-2": 0.008,
    "claude-2.0": 0.008,
    "claude-2.1": 0.008,
    # Claude 2 output
    "claude-2-completion": 0.024,
    "claude-2.0-completion": 0.024,
    "claude-2.1-completion": 0.024,
    # Claude Instant input
    "claude-instant-1": 0.0016,
    "claude-instant-1.2": 0.0016,
    # Claude Instant output
    "claude-instant-1-completion": 0.0055,
    "claude-instant-1.2-completion": 0.0055,
}


def standardize_model_name(model_name: str, is_completion: bool = False) -> str:
    """
    Standardize the model name to a format that can be used for cost calculation.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.

    Returns:
        Standardized model name.
    """
    if not model_name:
        model_name = "claude-3-sonnet"

    model_name = model_name.lower()

    # Handle OpenRouter prefixes
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/") :]

    # Add completion suffix if needed
    if is_completion and not model_name.endswith("-completion"):
        model_name = model_name + "-completion"

    return model_name


def get_anthropic_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.

    Returns:
        Cost in USD.
    """
    model_name = standardize_model_name(model_name, is_completion)

    if model_name not in ANTHROPIC_MODEL_COSTS:
        logger.warning(
            "Could not find model_name in ANTHROPIC_MODEL_COSTS dictionary, defaulting to claude-3-sonnet."
        )
        # Default to Claude 3 Sonnet pricing if model not found
        model_name = (
            "claude-3-sonnet" if not is_completion else "claude-3-sonnet-completion"
        )

    cost_per_1k = ANTHROPIC_MODEL_COSTS[model_name]
    total_cost = cost_per_1k * (num_tokens / 1000)

    return total_cost


def calculate_token_cost(
    model_name: str, input_tokens: int = 0, output_tokens: int = 0
) -> float:
    """
    Calculate the total cost for input and output tokens.

    Args:
        model_name: Name of the model
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        float: Total cost in USD
    """
    input_cost = get_anthropic_token_cost_for_model(
        model_name, input_tokens, is_completion=False
    )
    output_cost = get_anthropic_token_cost_for_model(
        model_name, output_tokens, is_completion=True
    )
    return input_cost + output_cost


class AnthropicCallbackHandler(BaseCallbackHandler, metaclass=Singleton):
    """Callback Handler that tracks Anthropic token usage and costs.

    This class uses the Singleton metaclass to ensure only one instance exists.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    model_name: str = DEFAULT_MODEL

    # Track cumulative totals separately from last message
    cumulative_total_tokens: int = 0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    # Repositories for callback handling
    trajectory_repo = None
    session_repo = None

    # Session totals to maintain consistency across agent switches
    session_totals = {
        "cost": 0.0,
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": None,
    }

    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        if model_name:
            self.model_name = model_name

        from ra_aid.database.repositories.trajectory_repository import (
            get_trajectory_repository,
        )
        from ra_aid.database.repositories.session_repository import (
            get_session_repository,
        )

        self.trajectory_repo = get_trajectory_repository()

        try:
            session_repo = get_session_repository()
            current_session = session_repo.get_current_session_record()
            if current_session:
                self.session_totals["session_id"] = current_session.get_id()

        except Exception as e:
            logger.warning(f"Failed to get current session: {e}")

        # Default costs for Claude 3.7 Sonnet
        self.input_cost_per_token = 0.003 / 1000  # $3/M input tokens
        self.output_cost_per_token = 0.015 / 1000  # $15/M output tokens

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.prompt_tokens + self.completion_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost:.6f}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Record the model name if available."""
        if "name" in serialized:
            self.model_name = serialized["name"]

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Collect token usage from response."""
        token_usage = {}

        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            if "token_usage" in llm_output:
                token_usage = llm_output["token_usage"]
            elif "usage" in llm_output:
                usage = llm_output["usage"]

                if "input_tokens" in usage:
                    token_usage["prompt_tokens"] = usage["input_tokens"]
                if "output_tokens" in usage:
                    token_usage["completion_tokens"] = usage["output_tokens"]

            if "model_name" in llm_output:
                self.model_name = llm_output["model_name"]

        # Try to get usage from response.usage
        elif hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                token_usage["prompt_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                token_usage["completion_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                token_usage["total_tokens"] = usage.total_tokens

        # Extract usage from generations if available
        elif hasattr(response, "generations") and response.generations:
            for gen in response.generations:
                if gen and hasattr(gen[0], "generation_info"):
                    gen_info = gen[0].generation_info or {}
                    if "usage" in gen_info:
                        token_usage = gen_info["usage"]
                        break

        # Update counts with lock to prevent race conditions
        with self._lock:
            # Store the current message's tokens directly (non-cumulative)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            # Update cumulative totals
            self.cumulative_prompt_tokens += prompt_tokens
            self.cumulative_completion_tokens += completion_tokens
            self.cumulative_total_tokens += prompt_tokens + completion_tokens

            # Set the current message tokens (non-cumulative)
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens

            # Calculate costs based on the current message
            if prompt_tokens > 0:
                prompt_cost = get_anthropic_token_cost_for_model(
                    self.model_name, prompt_tokens, is_completion=False
                )
                self.total_cost += prompt_cost

            if completion_tokens > 0:
                completion_cost = get_anthropic_token_cost_for_model(
                    self.model_name, completion_tokens, is_completion=True
                )
                self.total_cost += completion_cost

            self.successful_requests += 1

            self._handle_callback_update()

    def _handle_callback_update(self) -> None:
        """
        Handle callback updates and record trajectory data.
        """
        if not self.trajectory_repo:
            return

        try:
            if not self.session_totals["session_id"]:
                logger.warning("session_id not initialized")
                return

            last_cost = calculate_token_cost(
                self.model_name, self.prompt_tokens, self.completion_tokens
            )
            
            self.session_totals["cost"] += last_cost
            self.session_totals["tokens"] += self.prompt_tokens + self.completion_tokens
            self.session_totals["input_tokens"] += self.prompt_tokens
            self.session_totals["output_tokens"] += self.completion_tokens

            self.trajectory_repo.create(
                record_type="model_usage",
                current_cost=last_cost,
                input_tokens=self.prompt_tokens,
                output_tokens=self.completion_tokens,
                session_id=self.session_totals["session_id"],
            )

        except Exception as e:
            logger.error(f"Failed to store token usage data: {e}")

    def __copy__(self) -> "AnthropicCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "AnthropicCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self


# Create a context variable for our custom callback
anthropic_callback_var: ContextVar[Optional[AnthropicCallbackHandler]] = ContextVar(
    "anthropic_callback", default=None
)


@contextmanager
def get_anthropic_callback(
    model_name: Optional[str] = None,
) -> AnthropicCallbackHandler:
    """Get the Anthropic callback handler in a context manager.
    which conveniently exposes token and cost information.

    Args:
        model_name: Optional model name to use for cost calculation.

    Returns:
        AnthropicCallbackHandler: The Anthropic callback handler.

    Example:
        >>> with get_anthropic_callback("claude-3-sonnet") as cb:
        ...     # Use the callback handler
        ...     # cb.total_tokens, cb.total_cost will be available after
    """
    cb = AnthropicCallbackHandler(model_name)
    anthropic_callback_var.set(cb)
    yield cb
    anthropic_callback_var.set(None)
