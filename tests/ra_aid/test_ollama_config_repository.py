"""Test that the Ollama provider correctly uses the config repository."""

import pytest
from unittest.mock import patch, MagicMock

from ra_aid.database.repositories.config_repository import ConfigRepositoryManager
from ra_aid.llm import create_llm_client


def test_ollama_uses_get_config_repository():
    """Test that the Ollama provider correctly uses get_config_repository()."""
    # Create a mock for the ChatOllama class
    with patch("langchain_ollama.ChatOllama") as mock_chat_ollama, \
         patch("ra_aid.llm.get_provider_config") as mock_get_provider_config:
        
        # Set up the mock provider config
        mock_get_provider_config.return_value = {
            "base_url": "http://localhost:11434"
        }
        
        # Set up the mock ChatOllama instance
        mock_chat_ollama.return_value = MagicMock()
        
        # Use the ConfigRepositoryManager to initialize the repository
        with ConfigRepositoryManager() as config_repo:
            # Set a custom num_ctx value in the config repository
            config_repo.set("num_ctx", 32768)
            
            # Create the LLM client
            create_llm_client("ollama", "llama3", temperature=0.7)
            
            # Verify that ChatOllama was called with the correct parameters
            mock_chat_ollama.assert_called_once()
            call_args = mock_chat_ollama.call_args[1]
            
            # Check that the num_ctx value from the config repository was used
            assert call_args["num_ctx"] == 32768
            assert call_args["model"] == "llama3"
            assert call_args["base_url"] == "http://localhost:11434"
            assert call_args["temperature"] == 0.7