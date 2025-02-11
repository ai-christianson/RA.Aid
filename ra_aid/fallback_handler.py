from ra_aid.config import DEFAULT_MAX_TOOL_FAILURES, FALLBACK_TOOL_MODEL_LIMIT
from ra_aid.tool_leaderboard import supported_top_tool_models
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from ra_aid.llm import initialize_llm, merge_chat_history, validate_provider_env


class FallbackHandler:
    def __init__(self, config):
        self.config = config
        self.fallback_enabled = config.get("fallback_tool_enabled", True)
        self.fallback_tool_models = self._load_fallback_tool_models(config)
        self.tool_failure_consecutive_failures = 0
        self.tool_failure_used_fallbacks = set()

    def _load_fallback_tool_models(self, config):
        fallback_tool_models_config = config.get("fallback_tool_models")
        if fallback_tool_models_config:
            return [
                m.strip() for m in fallback_tool_models_config.split(",") if m.strip()
            ]
        else:
            console = Console()
            supported = []
            skipped = []
            for item in supported_top_tool_models:
                provider = item.get("provider")
                model_name = item.get("model")
                if validate_provider_env(provider):
                    supported.append(model_name)
                    if len(supported) == FALLBACK_TOOL_MODEL_LIMIT:
                        break
                else:
                    skipped.append(model_name)
            final_models = supported[:FALLBACK_TOOL_MODEL_LIMIT]
            message = "Fallback models selected: " + ", ".join(final_models)
            if skipped:
                message += (
                    "\nSkipped top tool calling models due to missing provider ENV API keys: "
                    + ", ".join(skipped)
                )
            console.print(Panel(Markdown(message), title="Fallback Models"))
            return final_models

    def handle_failure(self, code: str, error: Exception, logger, agent):
        logger.debug(
            f"_handle_tool_failure: tool failure encountered for code '{code}' with error: {error}"
        )
        self.tool_failure_consecutive_failures += 1
        max_failures = self.config.get("max_tool_failures", DEFAULT_MAX_TOOL_FAILURES)
        logger.debug(
            f"_handle_tool_failure: failure count {self.tool_failure_consecutive_failures}, max_failures {max_failures}"
        )
        if (
            self.fallback_enabled
            and self.tool_failure_consecutive_failures >= max_failures
            and self.fallback_tool_models
        ):
            logger.debug(
                "_handle_tool_failure: threshold reached, invoking fallback mechanism."
            )
            self.attempt_fallback(code, logger, agent)

    def attempt_fallback(self, code: str, logger, agent):
        logger.debug(f"_attempt_fallback: initiating fallback for code: {code}")
        new_model = self.fallback_tool_models[0]
        failed_tool_call_name = code.split("(")[0].strip()
        logger.error(
            f"Tool call failed {self.tool_failure_consecutive_failures} times. Attempting fallback to model: {new_model} for tool: {failed_tool_call_name}"
        )
        try:
            logger.debug(f"_attempt_fallback: validating provider {agent.provider}")
            if not validate_provider_env(agent.provider):
                logger.error(
                    f"Missing environment configuration for provider {agent.provider}. Cannot fallback."
                )
            else:
                logger.debug(
                    f"_attempt_fallback: initializing fallback model {new_model}"
                )
                agent.model = initialize_llm(agent.provider, new_model)
                logger.debug(
                    f"_attempt_fallback: binding tools to new model using tool: {failed_tool_call_name}"
                )
                agent.model.bind_tools(agent.tools, tool_choice=failed_tool_call_name)
                self.tool_failure_used_fallbacks.add(new_model)
                logger.debug("_attempt_fallback: merging chat history for fallback")
                merge_chat_history()
                self.tool_failure_consecutive_failures = 0
                logger.debug(
                    "_attempt_fallback: fallback successful and tool failure counter reset"
                )
        except Exception as switch_e:
            logger.error(f"Fallback model switching failed: {switch_e}")

    def reset_fallback_handler(self):
        self.tool_failure_consecutive_failures = 0
        self.tool_failure_used_fallbacks.clear()
