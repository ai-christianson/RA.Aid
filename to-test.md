# Common Provider and Model Combinations for Testing

## OpenAI Models
```bash
# GPT-4o
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "openai" --model "gpt-4o" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost

# GPT-4o-mini
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "openai" --model "gpt-4o-mini" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost
```

## Google Models
```bash
# Gemini 1.5 Flash
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "google" --model "gemini-1.5-flash" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost

# Gemini 1.5 Pro
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "google" --model "gemini-1.5-pro" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost
```

## Anthropic Models
```bash
# Claude 3.5 Sonnet
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "anthropic" --model "claude-3-5-sonnet" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost

# Claude 3 Opus
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "anthropic" --model "claude-3-opus" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost
```

## Mistral Models
```bash
# Mistral Large
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "mistral" --model "mistral-large" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost

# Mistral Medium
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "mistral" --model "mistral-medium" --expert-provider "openai" --expert-model "gpt-4o" --log-level debug --show-cost
```

## Deepseek Models (as in your example)
```bash
# Deepseek Chat
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "deepseek" --model "deepseek-chat" --expert-provider "openrouter" --expert-model "deepseek/deepseek-r1" --log-level debug --show-cost
```

## Openrouter flash gemeni 2.5 pro
ra-aid --cowboy-mode -m "Go look through ra_aid/agent_utils.py and look at corresponding test and add tests that are missing." --provider "openrouter" --model "google/gemini-2.5-pro-exp-03-25:free" --log-level debug --show-cost
