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
| DeepSeek | Specialized reasoning models | High performance on complex tasks |
| OpenRouter | Gateway to multiple open models | Wide model selection |
| OpenAI-compatible | API-compatible endpoints | Use with compatible hosting |
| Anthropic | Claude model family | Strong reasoning capabilities |
| Gemini | Google AI models | Competitive performance |

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
</Tabs>

## Advanced Configuration

<Tabs groupId="advanced-config">
<TabItem value="expert" label="Expert Mode">

### Expert Tool Configuration

Configure the expert tool for specialized tasks:

```bash
# DeepSeek expert
export EXPERT_DEEPSEEK_API_KEY=your_key
ra-aid -m "Your task" --expert-provider deepseek --expert-model deepseek-reasoner

# OpenRouter expert
export EXPERT_OPENROUTER_API_KEY=your_key
ra-aid -m "Your task" --expert-provider openrouter --expert-model mistralai/mistral-large-2411
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
|----------|----------|----------|
| `OPENROUTER_API_KEY` | OpenRouter | Main API access |
| `DEEPSEEK_API_KEY` | DeepSeek | Main API access |
| `OPENAI_API_KEY` | OpenAI-compatible | API access |
| `OPENAI_API_BASE` | OpenAI-compatible | Custom endpoint |
| `ANTHROPIC_API_KEY` | Anthropic | API access |
| `GEMINI_API_KEY` | Gemini | API access |
| `EXPERT_OPENROUTER_API_KEY` | OpenRouter | Expert tool |
| `EXPERT_DEEPSEEK_API_KEY` | DeepSeek | Expert tool |

## Troubleshooting

- Verify API keys are set correctly
- Check endpoint URLs for OpenAI-compatible setups
- Monitor API rate limits and quotas
