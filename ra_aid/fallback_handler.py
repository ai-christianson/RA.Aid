from ra_aid.config import DEFAULT_MAX_TOOL_FAILURES, FALLBACK_TOOL_MODEL_LIMIT, RETRY_FALLBACK_COUNT, RETRY_FALLBACK_DELAY
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
            # Assume comma-separated model names; wrap each in a dict with default type "prompt"
            models = []
            for m in [x.strip() for x in fallback_tool_models_config.split(",") if x.strip()]:
                models.append({"model": m, "type": "prompt"})
            return models
        else:
            console = Console()
            supported = []
            skipped = []
            for item in supported_top_tool_models:
                provider = item.get("provider")
                model_name = item.get("model")
                if validate_provider_env(provider):
                    supported.append(item)
                    if len(supported) == FALLBACK_TOOL_MODEL_LIMIT:
                        break
                else:
                    skipped.append(model_name)
            final_models = supported  # list of dicts
            message = "Fallback models selected: " + ", ".join([m["model"] for m in final_models])
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
        fallback_model = self.fallback_tool_models[0]
        failed_tool_call_name = code.split("(")[0].strip()
        logger.error(
            f"Tool call failed {self.tool_failure_consecutive_failures} times. Attempting fallback to model: {fallback_model['model']} for tool: {failed_tool_call_name}"
        )
        Console().print(Panel(Markdown(f"**Tool fallback activated**: Switching to fallback model {fallback_model['model']} for tool {failed_tool_call_name}."), title="Fallback Notification"))
        if fallback_model.get("type", "prompt").lower() == "fc":
            self.attempt_fallback_function(code, logger, agent)
        else:
            self.attempt_fallback_prompt(code, logger, agent)

    def reset_fallback_handler(self):
        self.tool_failure_consecutive_failures = 0
        self.tool_failure_used_fallbacks.clear()
    def attempt_fallback_prompt(self, code: str, logger, agent):
        logger.debug("Attempting prompt-based fallback using fallback models")
        failed_tool_call_name = code.split("(")[0].strip()
        for fallback_model in self.fallback_tool_models:
            try:
                logger.debug(f"Trying fallback model: {fallback_model['model']}")
                model = initialize_llm(agent.provider, fallback_model['model']).with_retry(retries=RETRY_FALLBACK_COUNT, delay=RETRY_FALLBACK_DELAY)
                model.bind_tools(agent.tools, tool_choice=failed_tool_call_name)
                response = model.invoke(code)
                self.tool_failure_used_fallbacks.add(fallback_model['model'])
                agent.model = model
                self.reset_fallback_handler()
                logger.debug("Prompt-based fallback executed successfully with model: " + fallback_model['model'])
                return response
            except Exception as e:
                logger.error(f"Prompt-based fallback with model {fallback_model['model']} failed: {e}")
        raise Exception("All prompt-based fallback models failed")

    def attempt_fallback_function(self, code: str, logger, agent):
        logger.debug("Attempting function-calling fallback using fallback models")
        failed_tool_call_name = code.split("(")[0].strip()
        for fallback_model in self.fallback_tool_models:
            try:
                logger.debug(f"Trying fallback model: {fallback_model['model']}")
                model = initialize_llm(agent.provider, fallback_model['model']).with_retry(retries=RETRY_FALLBACK_COUNT, delay=RETRY_FALLBACK_DELAY)
                model.bind_tools(agent.tools, tool_choice=failed_tool_call_name)
                response = model.invoke(code)
                self.tool_failure_used_fallbacks.add(fallback_model['model'])
                agent.model = model
                self.reset_fallback_handler()
                logger.debug("Function-calling fallback executed successfully with model: " + fallback_model['model'])
                return response
            except Exception as e:
                logger.error(f"Function-calling fallback with model {fallback_model['model']} failed: {e}")
        raise Exception("All function-calling fallback models failed")
