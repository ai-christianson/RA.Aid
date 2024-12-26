import pytest
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class WebUIConfig:
    provider: str
    model: str
    max_tokens: int = 1000
    max_context_length: int = 2000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "max_context_length": self.max_context_length
        }

def test_webui_config_defaults():
    config = WebUIConfig(provider="test-provider", model="test-model")
    assert config.provider == "test-provider"
    assert config.model == "test-model"
    assert config.max_tokens == 1000
    assert config.max_context_length == 2000

def test_webui_config_custom_values():
    config = WebUIConfig(
        provider="test-provider",
        model="test-model",
        max_tokens=2000,
        max_context_length=4000
    )
    assert config.provider == "test-provider"
    assert config.model == "test-model"
    assert config.max_tokens == 2000
    assert config.max_context_length == 4000

def test_webui_config_to_dict():
    config = WebUIConfig(
        provider="test-provider",
        model="test-model",
        max_tokens=2000,
        max_context_length=4000
    )
    config_dict = config.to_dict()
    assert config_dict["provider"] == "test-provider"
    assert config_dict["model"] == "test-model"
    assert config_dict["max_tokens"] == 2000
    assert config_dict["max_context_length"] == 4000 