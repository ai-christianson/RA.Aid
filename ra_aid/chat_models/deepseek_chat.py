from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from typing import Any, List, Optional, Dict


# Docs: https://api-docs.deepseek.com/guides/reasoning_model
class ChatDeepseekReasoner(ChatOpenAI):
    """ChatDeepseekReasoner with custom overrides for R1/reasoner models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invocation_params(
        self, options: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        params = super().invocation_params(options, **kwargs)

        # Remove unsupported params for R1 models
        params.pop("temperature", None)
        params.pop("top_p", None)
        params.pop("presence_penalty", None)
        params.pop("frequency_penalty", None)

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _generate to ensure message alternation in accordance to Deepseek API."""

        processed = []
        prev_role = None

        for msg in messages:
            current_role = "user" if msg.type == "human" else "assistant"

            if prev_role == current_role:
                if processed:
                    processed[-1].content += f"\n\n{msg.content}"
            else:
                processed.append(msg)
                prev_role = current_role

        return super()._generate(
            processed, stop=stop, run_manager=run_manager, **kwargs
        )
