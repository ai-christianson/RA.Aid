"""Provider validation strategies."""

from abc import ABC, abstractmethod
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Any

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
        if args and hasattr(args, 'expert_provider') and args.expert_provider == 'openai':
            key = os.environ.get('EXPERT_OPENAI_API_KEY')
            if not key or key == '':
                # Try to copy from base if not set
                base_key = os.environ.get('OPENAI_API_KEY')
                if base_key:
                    os.environ['EXPERT_OPENAI_API_KEY'] = base_key
                    key = base_key
            if not key:
                missing.append('EXPERT_OPENAI_API_KEY environment variable is not set')

            # Check expert model only for research-only mode
            if hasattr(args, 'research_only') and args.research_only:
                model = args.expert_model if hasattr(args, 'expert_model') else None
                if not model:
                    model = os.environ.get('EXPERT_OPENAI_MODEL')
                    if not model:
                        model = os.environ.get('OPENAI_MODEL')
                if not model:
                    missing.append('Model is required for OpenAI provider in research-only mode')
        else:
            key = os.environ.get('OPENAI_API_KEY')
            if not key:
                missing.append('OPENAI_API_KEY environment variable is not set')

            # Check model only for research-only mode
            if hasattr(args, 'research_only') and args.research_only:
                model = args.model if hasattr(args, 'model') else None
                if not model:
                    model = os.environ.get('OPENAI_MODEL')
                if not model:
                    missing.append('Model is required for OpenAI provider in research-only mode')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

class OpenAICompatibleStrategy(ProviderStrategy):
    """OpenAI-compatible provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate OpenAI-compatible environment variables."""
        missing = []

        # Check if we're validating expert config
        if args and hasattr(args, 'expert_provider') and args.expert_provider == 'openai-compatible':
            key = os.environ.get('EXPERT_OPENAI_API_KEY')
            base = os.environ.get('EXPERT_OPENAI_API_BASE')

            # Try to copy from base if not set
            if not key or key == '':
                base_key = os.environ.get('OPENAI_API_KEY')
                if base_key:
                    os.environ['EXPERT_OPENAI_API_KEY'] = base_key
                    key = base_key
            if not base or base == '':
                base_base = os.environ.get('OPENAI_API_BASE')
                if base_base:
                    os.environ['EXPERT_OPENAI_API_BASE'] = base_base
                    base = base_base

            if not key:
                missing.append('EXPERT_OPENAI_API_KEY environment variable is not set')
            if not base:
                missing.append('EXPERT_OPENAI_API_BASE environment variable is not set')

            # Check expert model only for research-only mode
            if hasattr(args, 'research_only') and args.research_only:
                model = args.expert_model if hasattr(args, 'expert_model') else None
                if not model:
                    model = os.environ.get('EXPERT_OPENAI_MODEL')
                    if not model:
                        model = os.environ.get('OPENAI_MODEL')
                if not model:
                    missing.append('Model is required for OpenAI-compatible provider in research-only mode')
        else:
            key = os.environ.get('OPENAI_API_KEY')
            base = os.environ.get('OPENAI_API_BASE')

            if not key:
                missing.append('OPENAI_API_KEY environment variable is not set')
            if not base:
                missing.append('OPENAI_API_BASE environment variable is not set')

            # Check model only for research-only mode
            if hasattr(args, 'research_only') and args.research_only:
                model = args.model if hasattr(args, 'model') else None
                if not model:
                    model = os.environ.get('OPENAI_MODEL')
                if not model:
                    missing.append('Model is required for OpenAI-compatible provider in research-only mode')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

class AnthropicStrategy(ProviderStrategy):
    """Anthropic provider validation strategy."""

    VALID_MODELS = [
        "claude-"
    ]

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Anthropic environment variables and model."""
        missing = []

        # Check if we're validating expert config
        is_expert = args and hasattr(args, 'expert_provider') and args.expert_provider == 'anthropic'

        # Check API key
        if is_expert:
            key = os.environ.get('EXPERT_ANTHROPIC_API_KEY')
            if not key or key == '':
                # Try to copy from base if not set
                base_key = os.environ.get('ANTHROPIC_API_KEY')
                if base_key:
                    os.environ['EXPERT_ANTHROPIC_API_KEY'] = base_key
                    key = base_key
            if not key:
                missing.append('EXPERT_ANTHROPIC_API_KEY environment variable is not set')
        else:
            key = os.environ.get('ANTHROPIC_API_KEY')
            if not key:
                missing.append('ANTHROPIC_API_KEY environment variable is not set')

        # Check model
        model_matched = False
        model_to_check = None

        # First check command line argument
        if is_expert:
            if hasattr(args, 'expert_model') and args.expert_model:
                model_to_check = args.expert_model
            else:
                # If no expert model, check environment variable
                model_to_check = os.environ.get('EXPERT_ANTHROPIC_MODEL')
                if not model_to_check or model_to_check == '':
                    # Try to copy from base if not set
                    base_model = os.environ.get('ANTHROPIC_MODEL')
                    if base_model:
                        os.environ['EXPERT_ANTHROPIC_MODEL'] = base_model
                        model_to_check = base_model
        else:
            if hasattr(args, 'model') and args.model:
                model_to_check = args.model
            else:
                model_to_check = os.environ.get('ANTHROPIC_MODEL')

        if not model_to_check:
            missing.append('ANTHROPIC_MODEL environment variable is not set')
            return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

        # Validate model format
        for pattern in self.VALID_MODELS:
            if re.match(pattern, model_to_check):
                model_matched = True
                break

        if not model_matched:
            missing.append(f'Invalid Anthropic model: {model_to_check}. Must match one of these patterns: {", ".join(self.VALID_MODELS)}')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

class OpenRouterStrategy(ProviderStrategy):
    """OpenRouter provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate OpenRouter environment variables."""
        missing = []

        # Check if we're validating expert config
        if args and hasattr(args, 'expert_provider') and args.expert_provider == 'openrouter':
            key = os.environ.get('EXPERT_OPENROUTER_API_KEY')
            if not key or key == '':
                # Try to copy from base if not set
                base_key = os.environ.get('OPENROUTER_API_KEY')
                if base_key:
                    os.environ['EXPERT_OPENROUTER_API_KEY'] = base_key
                    key = base_key
            if not key:
                missing.append('EXPERT_OPENROUTER_API_KEY environment variable is not set')
        else:
            key = os.environ.get('OPENROUTER_API_KEY')
            if not key:
                missing.append('OPENROUTER_API_KEY environment variable is not set')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)

class GeminiStrategy(ProviderStrategy):
    """Gemini provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Gemini environment variables."""
        missing = []

        # Check if we're validating expert config
        if args and hasattr(args, 'expert_provider') and args.expert_provider == 'gemini':
            key = os.environ.get('EXPERT_GEMINI_API_KEY')
            if not key or key == '':
                # Try to copy from base if not set
                base_key = os.environ.get('GEMINI_API_KEY')
                if base_key:
                    os.environ['EXPERT_GEMINI_API_KEY'] = base_key
                    key = base_key
            if not key:
                missing.append('EXPERT_GEMINI_API_KEY environment variable is not set')
        else:
            key = os.environ.get('GEMINI_API_KEY')
            if not key:
                missing.append('GEMINI_API_KEY environment variable is not set')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class DeepSeekStrategy(ProviderStrategy):
    """DeepSeek provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate DeepSeek environment variables."""
        missing = []

        if args and hasattr(args, 'expert_provider') and args.expert_provider == 'deepseek':
            key = os.environ.get('EXPERT_DEEPSEEK_API_KEY')
            if not key or key == '':
                # Try to copy from base if not set
                base_key = os.environ.get('DEEPSEEK_API_KEY')
                if base_key:
                    os.environ['EXPERT_DEEPSEEK_API_KEY'] = base_key
                    key = base_key
            if not key:
                missing.append('EXPERT_DEEPSEEK_API_KEY environment variable is not set')
        else:
            key = os.environ.get('DEEPSEEK_API_KEY')
            if not key:
                missing.append('DEEPSEEK_API_KEY environment variable is not set')

        return ValidationResult(valid=len(missing) == 0, missing_vars=missing)


class OllamaStrategy(ProviderStrategy):
    """Ollama provider validation strategy."""

    def validate(self, args: Optional[Any] = None) -> ValidationResult:
        """Validate Ollama environment variables."""
        missing = []
        
        base_url = os.environ.get('OLLAMA_BASE_URL')
        if not base_url:
            missing.append('OLLAMA_BASE_URL environment variable is not set')

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
            'openai': OpenAIStrategy(),
            'openai-compatible': OpenAICompatibleStrategy(),
            'anthropic': AnthropicStrategy(),
            'openrouter': OpenRouterStrategy(),
            'gemini': GeminiStrategy(),
            'ollama': OllamaStrategy(),
            'deepseek': DeepSeekStrategy()
        }
        strategy = strategies.get(provider)
        return strategy
