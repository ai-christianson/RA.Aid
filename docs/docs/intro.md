---
sidebar_position: 1
slug: /
---

# Welcome to RA.Aid

RA.Aid (pronounced "raid") is your AI-powered development companion that helps you build software autonomously. As a standalone coding agent built on LangChain's agent-based task execution framework, RA.Aid can handle research, planning, and implementation of your development tasks. Whether you're working on new features, refactoring code, or researching solutions, RA.Aid makes development faster and more efficient.

## Why RA.Aid?

- 🤖 **Autonomous Development**: Let RA.Aid handle complex programming tasks while you focus on the big picture
- 🔍 **Smart Research**: Automatically researches solutions and best practices
- 📋 **Intelligent Planning**: Breaks down complex tasks into manageable steps
- 💬 **Interactive Mode**: Get help when you need it through natural conversation

## Quick Start

Ready to get started? Jump right to:

- [Installation Guide](/quickstart/installation)
- [Basic Usage Examples](/usage/modern-web-app)

### Basic Example

Here's how simple it is to use RA.Aid with the recommended Gemini configuration:

```bash
# Install RA.Aid
pip install ra-aid

# Set up API keys (Gemini recommended, Tavily optional for web search)
export GEMINI_API_KEY='your_gemini_api_key'
export TAVILY_API_KEY='your_tavily_api_key'

# Start using it
ra-aid -m "Add input validation to the login form"
```

RA.Aid will automatically use Gemini 2.5 Pro if only the `GEMINI_API_KEY` is set. You can optionally add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` if you prefer to use those models. See the [Recommended Config](/quickstart/recommended) for more details.

## Key Features

- **Three-Stage Workflow**: Research → Planning → Implementation
- **Web Research**: Automatically searches for best practices and solutions
- **Interactive Mode**: Get help when you need it through natural conversation
- **Multiple AI Providers**: Support for various AI models (Gemini, OpenAI, Anthropic, etc.)
- **Git Integration**: Works seamlessly with your version control
- **Standalone Code Agent**: Built-in code modification capabilities by default
- **Optional Aider Integration**: Use the `--use-aider` flag to leverage aider's specialized code editing abilities

## Next Steps

- Check out the [Installation Guide](/quickstart/installation) to set up RA.Aid
- See [Usage Examples](/usage/modern-web-app) to get started quickly
- Read our [Contributing Guide](/contributing) to get involved
- Join our [Discord Community](https://discord.gg/f6wYbzHYxV) for help and discussions

Ready to revolutionize your development workflow? Let's get started! 🚀
