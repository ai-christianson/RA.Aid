# Recommended Config

:::info
RA.Aid must be installed before using these configurations. If you haven't installed it yet, please see the [Installation Guide](installation).
:::

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

Start RA.Aid in interactive mode:

```bash
ra-aid
```

Or run with a specific command:

```bash
ra-aid -m "Help me understand this code"
```
