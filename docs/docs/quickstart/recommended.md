---
sidebar_position: 2
---

# Recommended Config

The simplest and recommended configuration uses Google's Gemini 2.5 Pro model, which provides a strong balance of reasoning, planning, and analysis capabilities for most tasks.

- **Primary Model:** Google Gemini 2.5 Pro (`gemini-2.5-pro-preview-03-25`) handles core agent functions and expert consultations by default when only the `GEMINI_API_KEY` is set.
- **Web Search:** Tavily web search integration (optional) allows the agent to find relevant information online.

:::info
RA.Aid must be installed before using these configurations. If you haven't installed it yet, please see the [Installation Guide](installation).
:::

## Getting API Keys

To use RA.Aid with the recommended configuration, you'll need to obtain the following API key:

1.  **Gemini API Key**: Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

You might also consider keys for other services if you want to explicitly configure different models or providers:

2.  **OpenAI API Key** (Optional): Create an account at [OpenAI's platform](https://platform.openai.com) and generate an API key. Useful if you want to use OpenAI models, perhaps for the expert model specifically (see [Expert Model Configuration](/configuration/expert-model.md)).
3.  **Anthropic API Key** (Optional): Sign up at [Anthropic's Console](https://console.anthropic.com) and generate an API key. Useful if you prefer Anthropic models.
4.  **Tavily API Key** (Optional): Create an account at [Tavily](https://app.tavily.com/sign-in) and get your API key for web search capabilities.

Please keep your API keys secure and never share them publicly. Each service has its own pricing and usage terms.

## Configuration

Configure your API keys as environment variables. For the recommended setup, you only need:

```bash
# Recommended: For Google Gemini
export GEMINI_API_KEY='your_gemini_api_key'

# Optional: For web search capability
export TAVILY_API_KEY='your_tavily_api_key'
```

If you want to use other providers, you can set their keys as well:

```bash
# Optional: For OpenAI
export OPENAI_API_KEY='your_openai_api_key'

# Optional: For Anthropic
export ANTHROPIC_API_KEY='your_anthropic_api_key'

# Optional: To use a specific expert model (see Expert Model Config)
# export EXPERT_OPENAI_API_KEY='your_openai_api_key_for_expert'
# export EXPERT_GEMINI_API_KEY='your_gemini_api_key_for_expert'
# etc.
```

RA.Aid will automatically detect the available keys and select the default provider. If only `GEMINI_API_KEY` is set, it will use the `gemini` provider and the `gemini-2.5-pro-preview-03-25` model.

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

Note: aider must be installed separately. See [aider-chat](https://pypi.org/project/aider-chat/) for more information.

You can control logging verbosity and location using the `--log-mode` and `--log-level` options:

```bash
# Log to file (with only warnings to console)
ra-aid -m "Your task" --log-mode file --log-level debug

# Log everything to console
ra-aid -m "Your task" --log-mode console --log-level info
```

For more detailed logging configuration, see the [Logging documentation](../configuration/logging.md).

For information on RA.Aid's memory management and how to reset memory when needed, see the [Memory Management documentation](../configuration/memory-management.md).

For advanced reasoning capabilities, see the [Expert Model Configuration](../configuration/expert-model.md) to configure a specialized expert model for complex tasks.
