"""Custom callback handlers for tracking token usage and costs."""

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

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
        # Default to Claude 3 Sonnet pricing if model not found
        model_name = (
            "claude-3-sonnet" if not is_completion else "claude-3-sonnet-completion"
        )

    cost_per_1k = ANTHROPIC_MODEL_COSTS[model_name]
    total_cost = cost_per_1k * (num_tokens / 1000)

    return total_cost


class AnthropicCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks Anthropic token usage and costs."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0
    model_name: str = "claude-3-sonnet"  # Default model

    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        if model_name:
            self.model_name = model_name

        # Default costs for Claude 3.7 Sonnet
        self.input_cost_per_token = 0.003 / 1000  # $3/M input tokens
        self.output_cost_per_token = 0.015 / 1000  # $15/M output tokens

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
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
        """Count tokens as they're generated."""
        with self._lock:
            self.completion_tokens += 1
            self.total_tokens += 1
            token_cost = get_anthropic_token_cost_for_model(
                self.model_name, 1, is_completion=True
            )
            self.total_cost += token_cost

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Collect token usage from response."""
        token_usage = {}

        # Try to extract token usage from response
        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            if "token_usage" in llm_output:
                token_usage = llm_output["token_usage"]
            elif "usage" in llm_output:
                usage = llm_output["usage"]

                # Handle Anthropic's specific usage format
                if "input_tokens" in usage:
                    token_usage["prompt_tokens"] = usage["input_tokens"]
                if "output_tokens" in usage:
                    token_usage["completion_tokens"] = usage["output_tokens"]

            # Extract model name if available
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
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            # Only update prompt tokens if we have them
            if prompt_tokens > 0:
                self.prompt_tokens += prompt_tokens
                self.total_tokens += prompt_tokens
                prompt_cost = get_anthropic_token_cost_for_model(
                    self.model_name, prompt_tokens, is_completion=False
                )
                self.total_cost += prompt_cost

            # Only update completion tokens if not already counted by on_llm_new_token
            if completion_tokens > 0 and completion_tokens > self.completion_tokens:
                additional_tokens = completion_tokens - self.completion_tokens
                self.completion_tokens = completion_tokens
                self.total_tokens += additional_tokens
                completion_cost = get_anthropic_token_cost_for_model(
                    self.model_name, additional_tokens, is_completion=True
                )
                self.total_cost += completion_cost

            self.successful_requests += 1

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
