---
sidebar_position: 3
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Open Models Configuration

RA.Aid supports a variety of open source and compatible model providers. This guide covers configuration options and best practices for using different models with RA.Aid.

## Overview

<Tabs groupId="provider-overview">
  <TabItem value="providers" label="Supported Providers" default>

RA.Aid supports these model providers:

| Provider | Description | Key Features |
|----------|-------------|--------------|
| DeepSeek | Chinese hedge fund who creates sophisticated LLMs | Strong, open models like R1 |
| OpenRouter | Multi-model gateway service | Access to 100+ models, unified API interface, pay-per-token |
| OpenAI-compatible | Self-hosted model endpoints | Compatible with Llama, Mistral and other open models |
| Anthropic | Claude model series | 200k token context, strong tool use, JSON/XML parsing |
| Gemini | Google's multimodal models | Code generation in 20+ languages, parallel request support |
| Ollama | Local LLM hosting framework | Run models locally, no API keys required, offline usage |

  </TabItem>
  <TabItem value="setup" label="Quick Setup">

### Basic Configuration

1. Set your provider's API key:
```bash
# Choose the appropriate provider
export DEEPSEEK_API_KEY=your_key
export OPENROUTER_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GEMINI_API_KEY=your_key
# Ollama doesn't require an API key
```

2. Run RA.Aid with your chosen provider:
```bash
ra-aid -m "Your task" --provider <provider> --model <model>
```

  </TabItem>
</Tabs>

## Provider Configuration

<Tabs groupId="model-provider">
  <TabItem value="deepseek" label="DeepSeek" default>

### DeepSeek Models

DeepSeek offers powerful reasoning models optimized for complex tasks.

```bash
# Environment setup
export DEEPSEEK_API_KEY=your_api_key_here

# Basic usage
ra-aid -m "Your task" --provider deepseek --model deepseek-reasoner

# With temperature control
ra-aid -m "Your task" --provider deepseek --model deepseek-reasoner --temperature 0.7
```

**Available Models:**
- `deepseek-reasoner`: Optimized for reasoning tasks
- Access via OpenRouter: `deepseek/deepseek-r1`
</TabItem>
<TabItem value="openrouter" label="OpenRouter">

### OpenRouter Integration

OpenRouter provides access to multiple open source models through a single API.

```bash
# Environment setup
export OPENROUTER_API_KEY=your_api_key_here

# Example commands
ra-aid -m "Your task" --provider openrouter --model mistralai/mistral-large-2411
ra-aid -m "Your task" --provider openrouter --model deepseek/deepseek-r1
```

**Popular Models:**
- `mistralai/mistral-large-2411`
- `anthropic/claude-3`
- `deepseek/deepseek-r1`
</TabItem>
<TabItem value="openai-compatible" label="OpenAI-compatible">

### OpenAI-compatible Endpoints

Use OpenAI-compatible API endpoints with custom hosting solutions.

```bash
# Environment setup
export OPENAI_API_KEY=your_api_key_here
export OPENAI_API_BASE=https://your-api-endpoint

# Usage
ra-aid -m "Your task" --provider openai-compatible --model your-model-name
```

**Configuration Options:**
- Set custom base URL with `OPENAI_API_BASE`
- Supports temperature control
- Compatible with most OpenAI-style APIs
</TabItem>
<TabItem value="gemini" label="Google Gemini">

### Google Gemini Models

Google's Gemini models offer powerful multimodal capabilities with extensive code generation support.

```bash
# Environment setup
export GEMINI_API_KEY=your_api_key_here

# Basic usage
ra-aid -m "Your task" --provider gemini --model gemini-1.5-pro-latest

# With temperature control
ra-aid -m "Your task" --provider gemini --model gemini-1.5-flash-latest --temperature 0.5
```

**Available Models:**
- `gemini-pro`: Original Gemini Pro model
- `gemini-1.5-flash-latest`: Latest Gemini 1.5 Flash model (fast responses)
- `gemini-1.5-pro-latest`: Latest Gemini 1.5 Pro model (strong reasoning)
- `gemini-1.5-flash`: Gemini 1.5 Flash release
- `gemini-1.5-pro`: Gemini 1.5 Pro release 
- `gemini-1.0-pro`: Original Gemini 1.0 Pro model

**Configuration Notes:**
- All Gemini models support a 128,000 token context window
- Temperature control is supported for creative vs. deterministic responses
- Obtain your API key from [AI Studio](https://aistudio.google.com/app/apikey)
</TabItem>
<TabItem value="ollama" label="Ollama">

### Ollama Integration

Ollama provides a framework for running large language models locally on your machine.

```bash
# No environment variables required by default
# Ollama uses http://localhost:11434 by default

# Basic usage
ra-aid -m "Your task" --provider ollama --model justinledwards/mistral-small-3.1-Q6_K

# With context window control
ra-aid -m "Your task" --provider ollama --model qwq:32b --num-ctx 8192

# With temperature control
ra-aid -m "Your task" --provider ollama --model MHKetbi/Qwen2.5-Coder-32B-Instruct --temperature 0.3
```

**Popular Models:**
- `justinledwards/mistral-small-3.1-Q6_K`: Optimized Mistral small model
- `qwq:32b`: High-performing yet compact reasoning model
- `MHKetbi/Qwen2.5-Coder-32B-Instruct`: Qwen 2.5 optimized for code tasks

**Configuration Notes:**
- Requires [Ollama](https://ollama.com/download) to be installed and running
- No API keys needed - all processing happens locally
- Adjust context window with `--num-ctx` (default: 262144)
- Works completely offline after model download

For detailed setup instructions and advanced configuration options, see our [Ollama Configuration Guide](../configuration/ollama.md).
</TabItem>
</Tabs>

## Advanced Configuration

<Tabs groupId="advanced-config">
<TabItem value="expert" label="Expert Model">

### Expert Tool Configuration

Configure the expert model for specialized tasks; this usually benefits from a more powerful, slower, reasoning model:

```bash
# DeepSeek expert
export EXPERT_DEEPSEEK_API_KEY=your_key
ra-aid -m "Your task" --expert-provider deepseek --expert-model deepseek-reasoner

# OpenRouter expert
export EXPERT_OPENROUTER_API_KEY=your_key
ra-aid -m "Your task" --expert-provider openrouter --expert-model mistralai/mistral-large-2411

# Gemini expert
export EXPERT_GEMINI_API_KEY=your_key
ra-aid -m "Your task" --expert-provider gemini --expert-model gemini-2.0-flash-thinking-exp-1219

# Ollama expert
ra-aid -m "Your task" --expert-provider ollama --expert-model qwq:32b
```

</TabItem>
<TabItem value="temperature" label="Temperature Control">

### Temperature Settings

Control model creativity vs determinism:

```bash
# More deterministic (good for coding)
ra-aid -m "Your task" --temperature 0.2

# More creative (good for brainstorming)
ra-aid -m "Your task" --temperature 0.8
```

**Note:** Not all models support temperature control. Check provider documentation.
</TabItem>
</Tabs>

## Best Practices

- Set environment variables in your shell configuration file
- Use lower temperatures (0.1-0.3) for coding tasks
- Test different models to find the best fit for your use case
- Consider using expert mode for complex programming tasks

## Environment Variables

Complete list of supported environment variables:

| Variable | Provider | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter | Main API access |
| `DEEPSEEK_API_KEY` | DeepSeek | Main API access |
| `OPENAI_API_KEY` | OpenAI-compatible | API access |
| `OPENAI_API_BASE` | OpenAI-compatible | Custom endpoint |
| `ANTHROPIC_API_KEY` | Anthropic | API access |
| `GEMINI_API_KEY` | Gemini | API access |
| `OLLAMA_BASE_URL` | Ollama | Custom endpoint (default: http://localhost:11434) |
| `EXPERT_OPENROUTER_API_KEY` | OpenRouter | Expert tool |
| `EXPERT_DEEPSEEK_API_KEY` | DeepSeek | Expert tool |
| `EXPERT_GEMINI_API_KEY` | Gemini | Expert tool |
| `EXPERT_OLLAMA_BASE_URL` | Ollama | Expert tool endpoint |

## Troubleshooting

- Verify API keys are set correctly
- Check endpoint URLs for OpenAI-compatible setups
- Monitor API rate limits and quotas
- For Ollama, ensure the service is running (`ollama list`)

## See Also

- [Expert Model Configuration](/configuration/expert-model) - Detailed information about expert model setup and configuration
- [Thinking Models](/configuration/thinking-models) - Learn about models that can show their reasoning process