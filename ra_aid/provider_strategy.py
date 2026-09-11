"""Provider validation strategies."""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    missing_vars: List[str]


class ProviderStrategy(ABC):
    """Abstract base class for provider validation strategies."""

    @abstractmethod
    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate provider environment variables."""
        pass


class OpenAIStrategy(ProviderStrategy):
    """OpenAI provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate OpenAI environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "openai"
        ):
            key = os.environ.get("EXPERT_OPENAI_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("OPENAI_API_KEY")
                if base_key:
                    os.environ["EXPERT_OPENAI_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append("EXPERT_OPENAI_API_KEY environment variable is not set")

            # Handle expert model selection if none specified
            if hasattr(args, "expert_model") and not args.expert_model:
                from ra_aid.llm import select_expert_model

                model = select_expert_model("openai")
                if model:
                    args.expert_model = model
                elif hasattr(args, "research_only") and args.research_only:
                    missing.append("No suitable expert model available")

        else:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                missing.append("OPENAI_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class OpenAICompatibleStrategy(ProviderStrategy):
    """OpenAI-compatible provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate OpenAI-compatible environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "openai-compatible"
        ):
            key = os.environ.get("EXPERT_OPENAI_API_KEY")
            base = os.environ.get("EXPERT_OPENAI_API_BASE")

            # Try to copy from base if not set
            if not key or key == "":
                base_key = os.environ.get("OPENAI_API_KEY")
                if base_key:
                    os.environ["EXPERT_OPENAI_API_KEY"] = base_key
                    key = base_key
            if not base or base == "":
                base_base = os.environ.get("OPENAI_API_BASE")
                if base_base:
                    os.environ["EXPERT_OPENAI_API_BASE"] = base_base
                    base = base_base

            if not key:
                missing.append("EXPERT_OPENAI_API_KEY environment variable is not set")
            if not base:
                missing.append("EXPERT_OPENAI_API_BASE environment variable is not set")

            # Check expert model only for research-only mode
            if hasattr(args, "research_only") and args.research_only:
                model = args.expert_model if hasattr(args, "expert_model") else None
                if not model:
                    model = os.environ.get("EXPERT_OPENAI_MODEL")
                    if not model:
                        model = os.environ.get("OPENAI_MODEL")
                if not model:
                    missing.append(
                        "Model is required for OpenAI-compatible provider in research-only mode"
                    )
        else:
            key = os.environ.get("OPENAI_API_KEY")
            base = os.environ.get("OPENAI_API_BASE")

            if not key:
                missing.append("OPENAI_API_KEY environment variable is not set")
            if not base:
                missing.append("OPENAI_API_BASE environment variable is not set")

            # Check model only for research-only mode
            if hasattr(args, "research_only") and args.research_only:
                model = args.model if hasattr(args, "model") else None
                if not model:
                    model = os.environ.get("OPENAI_MODEL")
                if not model:
                    missing.append(
                        "Model is required for OpenAI-compatible provider in research-only mode"
                    )

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class AnthropicStrategy(ProviderStrategy):
    """Anthropic provider validation strategy."""

    VALID_MODELS = ["claude-"]

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Anthropic environment variables and model."""
        missing = []

        # Check if we're validating expert config
        is_expert = (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "anthropic"
        )

        # Check API key
        if is_expert:
            key = os.environ.get("EXPERT_ANTHROPIC_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("ANTHROPIC_API_KEY")
                if base_key:
                    os.environ["EXPERT_ANTHROPIC_API_KEY"] = base_key
                    key = base_key
            if not key:
                # If neither expert nor base key is set, show base key message
                base_key = os.environ.get("ANTHROPIC_API_KEY")
                if not base_key:
                    missing.append("ANTHROPIC_API_KEY environment variable is not set")
                else:
                    missing.append(
                        "EXPERT_ANTHROPIC_API_KEY environment variable is not set"
                    )
        else:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                missing.append("ANTHROPIC_API_KEY environment variable is not set")

        # Check model
        model_matched = False
        model_to_check = None

        # First check command line argument
        if is_expert:
            if hasattr(args, "expert_model") and args.expert_model:
                model_to_check = args.expert_model
            else:
                # If no expert model, check environment variable
                model_to_check = os.environ.get("EXPERT_ANTHROPIC_MODEL")
                if not model_to_check or model_to_check == "":
                    # Try to copy from base if not set
                    base_model = os.environ.get("ANTHROPIC_MODEL")
                    if base_model:
                        os.environ["EXPERT_ANTHROPIC_MODEL"] = base_model
                        model_to_check = base_model
        else:
            if hasattr(args, "model") and args.model:
                model_to_check = args.model
            else:
                model_to_check = os.environ.get("ANTHROPIC_MODEL")

        if not model_to_check:
            missing.append("ANTHROPIC_MODEL environment variable is not set")
            return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

        # Validate model format
        for pattern in self.VALID_MODELS:
            if re.match(pattern, model_to_check):
                model_matched = True
                break

        if not model_matched:
            missing.append(
                f'Invalid Anthropic model: {model_to_check}. Must match one of these patterns: {", ".join(self.VALID_MODELS)}'
            )

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class OpenRouterStrategy(ProviderStrategy):
    """OpenRouter provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate OpenRouter environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "openrouter"
        ):
            key = os.environ.get("EXPERT_OPENROUTER_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("OPENROUTER_API_KEY")
                if base_key:
                    os.environ["EXPERT_OPENROUTER_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "EXPERT_OPENROUTER_API_KEY environment variable is not set"
                )
        else:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                missing.append("OPENROUTER_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class RequestyStrategy(ProviderStrategy):
    """Requesty provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Requesty environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "requesty"
        ):
            key = os.environ.get("EXPERT_REQUESTY_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("REQUESTY_API_KEY")
                if base_key:
                    os.environ["EXPERT_REQUESTY_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "EXPERT_REQUESTY_API_KEY environment variable is not set"
                )
        else:
            key = os.environ.get("REQUESTY_API_KEY")
            if not key:
                missing.append("REQUESTY_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class GeminiStrategy(ProviderStrategy):
    """Gemini provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Gemini environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "gemini"
        ):
            key = os.environ.get("EXPERT_GEMINI_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("GEMINI_API_KEY")
                if base_key:
                    os.environ["EXPERT_GEMINI_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append("EXPERT_GEMINI_API_KEY environment variable is not set")
        else:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                missing.append("GEMINI_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class DeepSeekStrategy(ProviderStrategy):
    """DeepSeek provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate DeepSeek environment variables."""
        missing = []

        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "deepseek"
        ):
            key = os.environ.get("EXPERT_DEEPSEEK_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("DEEPSEEK_API_KEY")
                if base_key:
                    os.environ["EXPERT_DEEPSEEK_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "EXPERT_DEEPSEEK_API_KEY environment variable is not set"
                )
        else:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                missing.append("DEEPSEEK_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class OllamaStrategy(ProviderStrategy):
    """Ollama provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Ollama environment variables."""
        # No required environment variables for Ollama
        # We use a default URL of "http://localhost:11434" if OLLAMA_BASE_URL is not set
        return ValidationResult(valid=True, missing_vars=[])


class FireworksStrategy(ProviderStrategy):
    """Fireworks provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Fireworks environment variables.

        Args:
            args: Optional command line arguments

        Returns:
            Result of validation
        """
        missing = []

        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "fireworks"
        ):
            key = os.environ.get("EXPERT_FIREWORKS_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("FIREWORKS_API_KEY")
                if base_key:
                    os.environ["EXPERT_FIREWORKS_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "EXPERT_FIREWORKS_API_KEY environment variable is not set"
                )
        else:
            key = os.environ.get("FIREWORKS_API_KEY")
            if not key:
                missing.append("FIREWORKS_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class GroqStrategy(ProviderStrategy):
    """Groq provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Groq environment variables.

        Args:
            args: Optional command line arguments

        Returns:
            Result of validation
        """
        missing = []

        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "groq"
        ):
            key = os.environ.get("EXPERT_GROQ_API_KEY")
            if not key or key == "":
                # Try to copy from base if not set
                base_key = os.environ.get("GROQ_API_KEY")
                if base_key:
                    os.environ["EXPERT_GROQ_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "EXPERT_GROQ_API_KEY environment variable is not set"
                )
        else:
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                missing.append("GROQ_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

class BedrockStrategy(ProviderStrategy):
    """Bedrock provider validation strategy."""

    DEFAULT_REGION = "us-east-1"

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Bedrock environment variables.

        Args:
            args: Optional command line arguments

        Returns:
            Result of validation
        """
        missing = []

        # Check if we're validating expert config
        is_expert = (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "bedrock"
        )

        # Check for AWS credentials
        if is_expert:
            # Check for AWS profile
            aws_profile = os.environ.get("EXPERT_AWS_PROFILE")
            if not aws_profile or aws_profile == "default":
                # Try to copy from base if not set
                base_profile = os.environ.get("AWS_PROFILE")
                if base_profile:
                    os.environ["EXPERT_AWS_PROFILE"] = base_profile
                    aws_profile = base_profile

            # Check for AWS direct credentials
            aws_access_key = os.environ.get("EXPERT_AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("EXPERT_AWS_SECRET_ACCESS_KEY")

            if not aws_access_key or aws_access_key == "":
                # Try to copy from base if not set
                base_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
                if base_access_key:
                    os.environ["EXPERT_AWS_ACCESS_KEY_ID"] = base_access_key
                    aws_access_key = base_access_key

            if not aws_secret_key or aws_secret_key == "":
                # Try to copy from base if not set
                base_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
                if base_secret_key:
                    os.environ["EXPERT_AWS_SECRET_ACCESS_KEY"] = base_secret_key
                    aws_secret_key = base_secret_key

            # Optional: AWS Session Token
            aws_session_token = os.environ.get("EXPERT_AWS_SESSION_TOKEN")
            if not aws_session_token or aws_session_token == "":
                base_session_token = os.environ.get("AWS_SESSION_TOKEN")
                if base_session_token:
                    os.environ["EXPERT_AWS_SESSION_TOKEN"] = base_session_token
                    aws_session_token = base_session_token

            # Region
            aws_region = os.environ.get("EXPERT_AWS_REGION")
            if not aws_region or aws_region == "":
                base_region = os.environ.get("AWS_REGION")
                if base_region:
                    os.environ["EXPERT_AWS_REGION"] = base_region
                    aws_region = base_region
                else:
                    os.environ["EXPERT_AWS_REGION"] = self.DEFAULT_REGION

            # Validate credentials
            if not aws_profile and not (aws_access_key and aws_secret_key):
                missing.append(
                    "Either EXPERT_AWS_PROFILE or both EXPERT_AWS_ACCESS_KEY_ID and EXPERT_AWS_SECRET_ACCESS_KEY must be set"
                )

        else:
            # Check for AWS profile
            aws_profile = os.environ.get("AWS_PROFILE")

            # Check for AWS direct credentials
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

            # Region
            aws_region = os.environ.get("AWS_REGION")
            if not aws_region or aws_region == "":
                os.environ["AWS_REGION"] = self.DEFAULT_REGION

            # Validate credentials
            if not aws_profile and not (aws_access_key and aws_secret_key):
                missing.append(
                    "Either AWS_PROFILE or both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
                )

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class MakehubStrategy(ProviderStrategy):
    """Makehub provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Makehub environment variables."""
        missing = []

        # Check if we're validating expert config
        if (
            args
            and hasattr(args, "expert_provider")
            and args.expert_provider == "makehub"
        ):
            key = os.environ.get("EXPERT_MAKEHUB_API_KEY")
            if not key or key == "":
                # Automatically copy from base if not set (since Makehub uses one API key)
                base_key = os.environ.get("MAKEHUB_API_KEY")
                if base_key:
                    os.environ["EXPERT_MAKEHUB_API_KEY"] = base_key
                    key = base_key
            if not key:
                missing.append(
                    "MAKEHUB_API_KEY environment variable is not set (needed for expert mode)"
                )
        else:
            key = os.environ.get("MAKEHUB_API_KEY")
            if not key:
                missing.append("MAKEHUB_API_KEY environment variable is not set")

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class ProviderFactory:
    """Factory for creating provider validation strategies."""

    @staticmethod
    def create(provider: str, args: Optional[Any] = None) -> Optional[ProviderStrategy]:
        """Create a provider validation strategy.

        Args:
            provider: Provider name
            args: Optional command line arguments

        Returns:
            Provider validation strategy or None if provider not found
        """
        strategies = {
            "openai": OpenAIStrategy(),
            "openai-compatible": OpenAICompatibleStrategy(),
            "anthropic": AnthropicStrategy(),
            "openrouter": OpenRouterStrategy(),
            "requesty": RequestyStrategy(),
            "gemini": GeminiStrategy(),
            "ollama": OllamaStrategy(),
            "deepseek": DeepSeekStrategy(),
            "fireworks": FireworksStrategy(),
            "groq": GroqStrategy(),
            "bedrock": BedrockStrategy(),
            "makehub": MakehubStrategy(),
        }
        strategy = strategies.get(provider)
        return strategy
