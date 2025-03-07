---
sidebar_position: 2
---

# Recommended Config

This configuration combines the strengths of multiple AI models to provide the best experience:

- Anthropic Sonnet excels at driving the agent's core reasoning and planning
- OpenAI's models provide robust debugging and logical analysis capabilities  
- Tavily web search integration allows the agent to find relevant information online

:::info
RA.Aid must be installed before using these configurations. If you haven't installed it yet, please see the [Installation Guide](installation).
:::

## Getting API Keys

To use RA.Aid with the recommended configuration, you'll need to obtain API keys from the following services:

1. **OpenAI API Key**: Create an account at [OpenAI's platform](https://platform.openai.com) and generate an API key from your dashboard.

2. **Anthropic API Key**: Sign up at [Anthropic's Console](https://console.anthropic.com), then generate an API key from the API Keys section.

3. **Tavily API Key** (optional): Create an account at [Tavily](https://app.tavily.com/sign-in) and get your API key from the dashboard.

Please keep your API keys secure and never share them publicly. Each service has its own pricing and usage terms.

## Configuration

Configure your API keys:

```bash
# For OpenAI (required)
export OPENAI_API_KEY=your_api_key_here

# For Anthropic (required)
export ANTHROPIC_API_KEY=your_api_key_here

# For web search capability (optional)
export TAVILY_API_KEY=your_api_key_here
```

## Basic Usage

Start RA.Aid in interactive chat mode:

```bash
ra-aid --chat
```

Or run with a single command:

```bash
ra-aid -m "Help me understand this code"
```

If you prefer to use aider's specialized code editing capabilities instead of RA.Aid's built-in file modification tools:

```bash
ra-aid -m "Implement this feature" --use-aider
```

You can control logging verbosity and location using the `--log-mode` and `--log-level` options:

```bash
# Log to file (with only warnings to console)
ra-aid -m "Your task" --log-mode file --log-level debug

# Log everything to console
ra-aid -m "Your task" --log-mode console --log-level info
```

For more detailed logging configuration, see the [Logging documentation](../configuration/logging.md).

For information on RA.Aid's memory management and how to reset memory when needed, see the [Memory Management documentation](../configuration/memory-management.md).
