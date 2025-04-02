import threading
import time
from langchain.chat_models.base import BaseChatModel
import litellm
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional, Union, Any, List
from decimal import Decimal, getcontext

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

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
    # Newly added models
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

    input_cost_per_token: Decimal = Decimal("0.0")
    output_cost_per_token: Decimal = Decimal("0.0")

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
            # --- START MODIFICATION ---
            config_repo = get_config_repository()
            show_cost = config_repo.get("show_cost", DEFAULT_SHOW_COST)
            if show_cost:
                cpm(
                    "Could not find model costs from litellm defaulting to MODEL_COSTS table or 0",
                    border_style="yellow",
                )
            # --- END MODIFICATION ---

        # Fallback logic remains the same
        model_cost = MODEL_COSTS.get(
            self.model_name, {"input": Decimal("0"), "output": Decimal("0")}
        )
        self.input_cost_per_token = model_cost["input"]
        self.output_cost_per_token = model_cost["output"]

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

            # Calculate costs using Decimal arithmetic
            input_cost = Decimal(prompt_tokens) * self.input_cost_per_token
            output_cost = Decimal(completion_tokens) * self.output_cost_per_token
            cost = input_cost + output_cost
            self.total_cost += cost

            self.successful_requests += 1

            # Update session totals
            self.session_totals["cost"] += cost if "cost" in locals() else 0
            self.session_totals["tokens"] += total_tokens
            self.session_totals["input_tokens"] += prompt_tokens
            self.session_totals["output_tokens"] += completion_tokens
            self.session_totals["duration"] += duration

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

            input_cost = Decimal(prompt_tokens) * self.input_cost_per_token
            output_cost = Decimal(completion_tokens) * self.output_cost_per_token
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

            self._initialize_model_costs()
            if self.session_repo:
                current_session = self.session_repo.get_current_session_record()
                if current_session:
                    self.session_totals["session_id"] = current_session.get_id()

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
