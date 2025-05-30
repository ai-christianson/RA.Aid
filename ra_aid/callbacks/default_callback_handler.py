import threading
import time
import sys
from langchain.chat_models.base import BaseChatModel
import litellm
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional, Union, Any, List
from decimal import Decimal, getcontext

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich.prompt import Confirm

from ra_aid.model_detection import (
    get_model_name_from_chat_model,
    get_provider_from_chat_model,
)
from ra_aid.utils.singleton import Singleton
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.session_repository import get_session_repository
from ra_aid.logging_config import get_logger

# Added imports
from ra_aid.config import DEFAULT_SHOW_COST
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.console.formatting import cpm

logger = get_logger(__name__)

getcontext().prec = 16

MODEL_COSTS = {
    "claude-3-7-sonnet-20250219": {
        "input": Decimal("0.000003"),
        "output": Decimal("0.000015"),
    },
    "claude-3-opus-20240229": {
        "input": Decimal("0.000015"),
        "output": Decimal("0.000075"),
    },
    "claude-3-sonnet-20240229": {
        "input": Decimal("0.000003"),
        "output": Decimal("0.000015"),
    },
    "claude-3-haiku-20240307": {
        "input": Decimal("0.00000025"),
        "output": Decimal("0.00000125"),
    },
    "claude-2": {
        "input": Decimal("0.00001102"),
        "output": Decimal("0.00003268"),
    },
    "claude-instant-1": {
        "input": Decimal("0.00000163"),
        "output": Decimal("0.00000551"),
    },
    "google/gemini-2.5-pro-exp-03-25:free": {
        "input": Decimal("0"),
        "output": Decimal("0"),
    },
    # Tiered pricing based on prompt tokens (input)
    "google/gemini-2.5-pro-exp-03-25": {
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "gemini-2.5-pro-exp-03-25": {
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "google/gemini-2.5-pro-preview-03-25": { # Added google/ prefix
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "gemini-2.5-pro-preview-03-25": { # Kept original entry without prefix just in case
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "google/gemini-2.5-pro-preview-05-06": { # Created May 7, 2025
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "gemini-2.5-pro-preview-05-06": { # Created May 7, 2025
        "input_under_200k": Decimal("0.00000125"),  # $1.25/M tokens
        "input_over_200k": Decimal("0.00000250"),   # $2.50/M tokens
        "output_under_200k": Decimal("0.00001000"), # $10.00/M tokens
        "output_over_200k": Decimal("0.00001500"),  # $15.00/M tokens
        "tier_threshold": 200000,
    },
    "deepseek/deepseek-chat-v3-0324": {
        "input": Decimal("0.00000027"),
        "output": Decimal("0.0000011"),
    },
    "weaver-ai": {
        "input": Decimal("0.001875"),
        "output": Decimal("0.00225"),
    },
    "airoboros-v1": {
        "input": Decimal("0.0005"),
        "output": Decimal("0.0005"),
    },
    "mistral-nemo": {
        "input": Decimal("0.00015"),
        "output": Decimal("0.00015"),
    },
    "pixtral-12b": {
        "input": Decimal("0.00015"),
        "output": Decimal("0.00015"),
    },
    "mistral-large-24b11": {
        "input": Decimal("0.002"),
        "output": Decimal("0.006"),
    },
    "mistralai/mistral-small-3.1-24b-instruct": {
        "input": Decimal("0.0001"),  # $0.1/M input tokens
        "output": Decimal("0.0003"),  # $0.3/M output tokens
    },
    "anthropic/claude-sonnet-4": { # Created May 22, 2025
        "input": Decimal("0.000003"),  # $3/M input tokens
        "output": Decimal("0.000015"), # $15/M output tokens
        # Image costs ($4.80/K input imgs) are not currently supported by this structure
    },
}


class DefaultCallbackHandler(BaseCallbackHandler, metaclass=Singleton):
    def __init__(self, model_name: str, provider: Optional[str] = None):
        super().__init__()
        self._lock = threading.Lock()
        self._initialize(model_name, provider)

    def _initialize(self, model_name: str, provider: Optional[str] = None):
        with self._lock:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.successful_requests = 0
            self.total_cost = Decimal("0.0")
            self.model_name = model_name
            self.provider = provider
            self._last_request_time = None
            self.__post_init__()

    cumulative_total_tokens: int = 0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    trajectory_repo = None
    session_repo = None
    config_repo = None

    input_cost_per_token: Decimal = Decimal("0.0")  # Default/non-tiered input cost
    output_cost_per_token: Decimal = Decimal("0.0") # Default/non-tiered output cost
    tiered_costs: Optional[Dict[str, Union[Decimal, int]]] = None # Store tiered costs if applicable

    session_totals = {
        "cost": Decimal("0.0"),
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": None,
        "duration": 0.0,
    }

    def __post_init__(self):
        try:
            if not hasattr(self, "trajectory_repo") or self.trajectory_repo is None:
                self.trajectory_repo = get_trajectory_repository()

            if not hasattr(self, "session_repo") or self.session_repo is None:
                self.session_repo = get_session_repository()

            if not hasattr(self, "config_repo") or self.config_repo is None:
                self.config_repo = get_config_repository()

            if self.session_repo:
                current_session = self.session_repo.get_current_session_record()
                if current_session:
                    self.session_totals["session_id"] = current_session.get_id()

            self._initialize_model_costs()
        except Exception as e:
            logger.error(f"Failed to initialize callback handler: {e}", exc_info=True)

    def _initialize_model_costs(self) -> None:
        try:
            model_info = litellm.get_model_info(
                model=self.model_name, custom_llm_provider=self.provider
            )
            if model_info:
                input_cost = model_info.get("input_cost_per_token", 0.0)
                output_cost = model_info.get("output_cost_per_token", 0.0)
                self.input_cost_per_token = Decimal(str(input_cost))
                self.output_cost_per_token = Decimal(str(output_cost))
                if self.input_cost_per_token and self.output_cost_per_token:
                    return
        except Exception as e:
            logger.debug(f"Could not get model info from litellm: {e}")
            # Fallback logic will proceed

        # Fallback logic: Check MODEL_COSTS dictionary
        model_cost_info = MODEL_COSTS.get(self.model_name)
        if model_cost_info:
            if "tier_threshold" in model_cost_info:
                # Store the tiered structure and set default rates (under threshold)
                self.tiered_costs = model_cost_info
                self.input_cost_per_token = model_cost_info.get("input_under_200k", Decimal("0"))
                self.output_cost_per_token = model_cost_info.get("output_under_200k", Decimal("0"))
            elif "input" in model_cost_info and "output" in model_cost_info:
                # Standard non-tiered model
                self.input_cost_per_token = model_cost_info["input"]
                self.output_cost_per_token = model_cost_info["output"]
                self.tiered_costs = None
            else:
                # Unknown structure, default to zero
                logger.warning(f"Unknown cost structure for model {self.model_name} in MODEL_COSTS. Defaulting to 0.")
                self.input_cost_per_token = Decimal("0")
                self.output_cost_per_token = Decimal("0")
                self.tiered_costs = None
        else:
            # Model not found, default to zero
            logger.warning(f"Model {self.model_name} not found in litellm or MODEL_COSTS. Defaulting to 0 cost.")
            # --- START MODIFICATION ---
            config_repo = get_config_repository()
            show_cost = config_repo.get("show_cost", DEFAULT_SHOW_COST)
            if show_cost:
                cpm(
                    f"Could not find costs for model '{self.model_name}'. Defaulting to 0.0.",
                    border_style="yellow",
                )
            # --- END MODIFICATION ---
            self.input_cost_per_token = Decimal("0")
            self.output_cost_per_token = Decimal("0")
            self.tiered_costs = None

    def _get_tiered_cost_rates(self, prompt_tokens: int) -> tuple[Decimal, Decimal]:
        """
        Calculates the applicable input and output cost per token based on
        the current call's prompt tokens and the model's tiered pricing structure.

        Args:
            prompt_tokens: The number of prompt tokens in the current LLM call.

        Returns:
            A tuple containing (input_cost_per_token, output_cost_per_token).
        """
        current_input_cost_per_token = self.input_cost_per_token
        current_output_cost_per_token = self.output_cost_per_token

        if self.tiered_costs and isinstance(self.tiered_costs, dict):
            threshold = self.tiered_costs.get("tier_threshold", 0)
            if prompt_tokens > threshold:
                current_input_cost_per_token = self.tiered_costs.get("input_over_200k", current_input_cost_per_token)
                current_output_cost_per_token = self.tiered_costs.get("output_over_200k", current_output_cost_per_token)
            else:
                current_input_cost_per_token = self.tiered_costs.get("input_under_200k", current_input_cost_per_token)
                current_output_cost_per_token = self.tiered_costs.get("output_under_200k", current_output_cost_per_token)

        return current_input_cost_per_token, current_output_cost_per_token

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
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        try:
            self._last_request_time = time.time()
            if "name" in serialized:
                self.model_name = serialized["name"]
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)

    def _extract_token_usage(self, response: LLMResult) -> dict:
        """Extract token usage information from various response formats."""
        token_usage = {}

        # Check in llm_output
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

        # Check in response.usage
        elif hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                token_usage["prompt_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                token_usage["completion_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                token_usage["total_tokens"] = usage.total_tokens

        # Check in generations
        if (
            not token_usage
            and hasattr(response, "generations")
            and response.generations
        ):
            for gen in response.generations:
                # Check in generation_info
                if gen and hasattr(gen[0], "generation_info"):
                    gen_info = gen[0].generation_info or {}
                    if "usage" in gen_info:
                        token_usage = gen_info["usage"]
                        break

                # Check in message.usage_metadata (for Gemini models)
                if (
                    gen
                    and hasattr(gen[0], "message")
                    and hasattr(gen[0].message, "usage_metadata")
                ):
                    usage_metadata = gen[0].message.usage_metadata
                    if usage_metadata:
                        if "input_tokens" in usage_metadata:
                            token_usage["prompt_tokens"] = usage_metadata[
                                "input_tokens"
                            ]
                        if "output_tokens" in usage_metadata:
                            token_usage["completion_tokens"] = usage_metadata[
                                "output_tokens"
                            ]
                        if (
                            "total_tokens" in usage_metadata
                            and not token_usage.get("prompt_tokens")
                            and not token_usage.get("completion_tokens")
                        ):
                            # If we only have total but not input/output breakdown
                            token_usage["total_tokens"] = usage_metadata["total_tokens"]
                        break

        return token_usage

    def _update_token_counts(self, token_usage: dict, duration: float) -> None:
        """Update token counts and costs with thread safety."""
        with self._lock:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )

            # If we only have total_tokens but not the breakdown
            if total_tokens > 0 and prompt_tokens == 0 and completion_tokens == 0:
                # Make a reasonable guess about the split (e.g., 90% prompt, 10% completion)
                prompt_tokens = int(total_tokens * 0.9)
                completion_tokens = total_tokens - prompt_tokens

            self.cumulative_prompt_tokens += prompt_tokens
            self.cumulative_completion_tokens += completion_tokens
            self.cumulative_total_tokens += total_tokens

            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

            # Determine cost per token for this specific call, considering tiers
            current_input_cost_per_token, current_output_cost_per_token = self._get_tiered_cost_rates(prompt_tokens)

            # Calculate costs using Decimal arithmetic with potentially tiered rates
            input_cost = Decimal(prompt_tokens) * current_input_cost_per_token
            output_cost = Decimal(completion_tokens) * current_output_cost_per_token
            cost = input_cost + output_cost
            self.total_cost += cost

            self.successful_requests += 1

            # Update session totals
            self.session_totals["cost"] += cost if "cost" in locals() else 0
            self.session_totals["tokens"] += total_tokens
            self.session_totals["input_tokens"] += prompt_tokens
            self.session_totals["output_tokens"] += completion_tokens
            self.session_totals["duration"] += duration

            # Check limits after updating session totals
            limit_info = self._check_limits()
            if limit_info:
                limit_type, current_val, limit_val, exit_at_limit = limit_info
                self._record_limit_reached(limit_type, current_val, limit_val, exit_at_limit)
                should_prompt = not exit_at_limit
            else:
                should_prompt = False

        # Handle limit exceeded outside of lock
        if limit_info:
            if should_prompt:
                message = self._format_limit_message(limit_type, current_val, limit_val)
                if not Confirm.ask(message, default=False):
                    sys.exit(0)
            else:
                # Auto-exit mode
                print(f"Exiting due to {limit_type} limit reached: {current_val} >= {limit_val}")
                sys.exit(0)

        # Continue with normal callback handling
        with self._lock:
            self._handle_callback_update(
                total_tokens, prompt_tokens, completion_tokens, duration
            )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            if self._last_request_time is None:
                logger.debug("No request start time found, using default duration")
                duration = 0.1  # Default duration in seconds
            else:
                duration = time.time() - self._last_request_time
                self._last_request_time = None

            token_usage = self._extract_token_usage(response)

            self._update_token_counts(token_usage, duration)

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)

    def _handle_callback_update(
        self,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
    ) -> None:
        try:
            if not self.trajectory_repo:
                return

            if not self.session_totals["session_id"]:
                logger.warning("session_id not initialized")
                return

            # Recalculate cost for the trajectory record using potentially tiered rates for this call
            current_input_cost_per_token, current_output_cost_per_token = self._get_tiered_cost_rates(prompt_tokens)

            input_cost = Decimal(prompt_tokens) * current_input_cost_per_token
            output_cost = Decimal(completion_tokens) * current_output_cost_per_token
            cost = input_cost + output_cost

            # Must Convert Decimal to float compatible JSON serialization in repository
            cost_float = float(cost)

            trajectory_record = self.trajectory_repo.create(
                record_type="model_usage",
                current_cost=cost_float,
                input_tokens=self.prompt_tokens,
                output_tokens=self.completion_tokens,
                session_id=self.session_totals["session_id"],
                step_data={
                    "duration": duration,
                    "model": self.model_name,
                },
            )
        except Exception as e:
            logger.error(f"Failed to store token usage data: {e}", exc_info=True)

    def reset_session_totals(self) -> None:
        try:
            current_session_id = self.session_totals.get("session_id")
            self.session_totals = {
                "cost": Decimal("0.0"),
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "session_id": current_session_id,
                "duration": 0.0,
            }
        except Exception as e:
            logger.error(f"Error resetting session totals: {e}", exc_info=True)

    def reset_all_totals(self) -> None:
        with self._lock:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.successful_requests = 0
            self.total_cost = Decimal("0.0")
            self._last_request_time = None

            self.session_totals = {
                "cost": Decimal("0.0"),
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "session_id": None,
                "duration": 0.0,
            }

            self.cumulative_total_tokens = 0
            self.cumulative_prompt_tokens = 0
            self.cumulative_completion_tokens = 0

            # Reset cost-related attributes before re-initializing
            self.input_cost_per_token = Decimal("0.0")
            self.output_cost_per_token = Decimal("0.0")
            self.tiered_costs = None

            self._initialize_model_costs() # Re-fetch costs based on current model_name
            if self.session_repo:
                current_session = self.session_repo.get_current_session_record()
                if current_session:
                    self.session_totals["session_id"] = current_session.get_id()

    def _check_limits(self) -> Optional[tuple]:
        """Check if cost or token limits have been exceeded.
        
        Returns:
            tuple: (limit_type, current_value, limit_value, exit_at_limit) if limit exceeded, None otherwise
        """
        try:
            if not self.config_repo:
                return None
                
            max_cost = self.config_repo.get("max_cost")
            max_tokens = self.config_repo.get("max_tokens")
            exit_at_limit = self.config_repo.get("exit_at_limit", False)
            
            current_cost = self.session_totals["cost"]
            current_tokens = self.session_totals["tokens"]
            
            # Check cost limit (convert to float for comparison)
            if max_cost is not None and max_cost > 0 and float(current_cost) >= max_cost:
                return ("cost", float(current_cost), max_cost, exit_at_limit)
                
            # Check token limit
            if max_tokens is not None and max_tokens > 0 and current_tokens >= max_tokens:
                return ("tokens", current_tokens, max_tokens, exit_at_limit)
                
            return None
        except Exception as e:
            logger.error(f"Error checking limits: {e}", exc_info=True)
            return None
    
    def _record_limit_reached(self, limit_type: str, current_value: Union[int, float], 
                             limit_value: Union[int, float], exit_at_limit: bool) -> None:
        """Record a limit reached event in the trajectory."""
        try:
            if not self.trajectory_repo or not self.session_totals["session_id"]:
                return
                
            self.trajectory_repo.create(
                record_type="limit_reached",
                session_id=self.session_totals["session_id"],
                step_data={
                    "limit_type": limit_type,
                    "current_value": current_value,
                    "limit_value": limit_value,
                    "exit_at_limit": exit_at_limit
                }
            )
        except Exception as e:
            logger.error(f"Failed to record limit reached event: {e}", exc_info=True)
    
    def _format_limit_message(self, limit_type: str, current_value: Union[int, float], 
                             limit_value: Union[int, float]) -> str:
        """Format a user-friendly message for limit exceeded scenarios."""
        if limit_type == "cost":
            return f"Cost limit exceeded: ${current_value:.6f} >= ${limit_value:.6f}. Continue anyway?"
        else:
            return f"Token limit exceeded: {current_value:,} >= {limit_value:,}. Continue anyway?"

    def get_stats(self) -> Dict[str, Union[int, float]]:
        try:
            return {
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_cost": self.total_cost,
                "successful_requests": self.successful_requests,
                "model_name": self.model_name,
                "session_totals": dict(self.session_totals),
                "cumulative_tokens": {
                    "total": self.cumulative_total_tokens,
                    "prompt": self.cumulative_prompt_tokens,
                    "completion": self.cumulative_completion_tokens,
                },
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {}


default_callback_var: ContextVar[Optional[DefaultCallbackHandler]] = ContextVar(
    "default_callback", default=None
)


@contextmanager
def get_default_callback(
    model_name: str,
    provider: Optional[str] = None,
) -> DefaultCallbackHandler:
    cb = DefaultCallbackHandler(model_name=model_name, provider=provider)
    default_callback_var.set(cb)
    yield cb
    default_callback_var.set(None)


def _initialize_callback_handler_internal(
    model_name: str, provider: Optional[str] = None, track_cost: bool = True
) -> tuple[Optional[DefaultCallbackHandler], dict]:
    cb = None
    stream_config = {"callbacks": []}

    if not track_cost:
        logger.debug("Cost tracking is disabled, skipping callback handler")
        return cb, stream_config

    logger.debug(f"Using callback handler for model {model_name}")
    cb = DefaultCallbackHandler(model_name, provider)
    stream_config["callbacks"].append(cb)

    return cb, stream_config


def initialize_callback_handler(
    model: BaseChatModel, track_cost: bool = True
) -> tuple[Optional[DefaultCallbackHandler], dict]:
    model_name = get_model_name_from_chat_model(model)
    provider = get_provider_from_chat_model(model)
    return _initialize_callback_handler_internal(model_name, provider, track_cost)
