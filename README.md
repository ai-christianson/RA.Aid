  ```ascii

  ██▀███   ▄▄▄            ▄▄▄       ██▓▓█████▄ 
 ▓██ ▒ ██▒▒████▄         ▒████▄    ▓██▒▒██▀ ██▌
 ▓██ ░▄█ ▒▒██  ▀█▄       ▒██  ▀█▄  ▒██▒░██   █▌
 ▒██▀▀█▄  ░██▄▄▄▄██      ░██▄▄▄▄██ ░██░░▓█▄   ▌
 ░██▓ ▒██▒ ▓█   ▓██▒ ██▓  ▓█   ▓██▒░██░░▒████▓ 
 ░ ▒▓ ░▒▓░ ▒▒   ▓▒█░ ▒▓▒  ▒▒   ▓▒█░░▓   ▒▒▓  ▒ 
   ░▒ ░ ▒░  ▒   ▒▒ ░ ░▒    ▒   ▒▒ ░ ▒ ░ ░ ▒  ▒ 
   ░░   ░   ░   ▒    ░     ░   ▒    ▒ ░ ░ ░  ░ 
    ░           ░  ░  ░        ░  ░ ░     ░    
                      ░                 ░      
```

[![Python Versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-Beta-yellow)]()

# RA.Aid

**RA.Aid is an AI software development agent.**

RA.Aid (ReAct Aid) was made by putting `aider` (https://aider.chat/) in a LangChain ReAct agent loop. This unique combination allows developers to leverage aider's code editing capabilities while benefiting from LangChain's agent-based task execution framework. The tool provides an intelligent assistant that can help with research, planning, and implementation of multi-step development tasks. 

Here's a demo of RA.Aid adding a feature to itself:

<img src="assets/demo-ra-aid-task-1.gif" alt="RA.Aid Demo" autoplay loop style="width: 100%; max-width: 800px;">

> 👋 **Pull requests are very welcome!** Have ideas for how to impove RA.Aid? Don't be shy - your help makes a real difference!
>
> 💬 **Join our Discord community:** [Click here to join](https://discord.gg/f6wYbzHYxV)

⚠️ **IMPORTANT: USE AT YOUR OWN RISK** ⚠️
- This tool **can and will** automatically execute shell commands and make code changes
- The --cowboy-mode flag can be enabled to skip shell command approval prompts
- No warranty is provided, either express or implied
- Always use in version-controlled repositories
- Review proposed changes in your git diff before committing

## Key Features

- **Multi-Step Task Planning**: The agent breaks down complex tasks into discrete, manageable steps and executes them sequentially. This systematic approach ensures thorough implementation and reduces errors.

- **Automated Command Execution**: The agent can run shell commands automatically to accomplish tasks. While this makes it powerful, it also means you should carefully review its actions.

- **Three-Stage Architecture**:
  1. **Research**: Analyzes codebases and gathers context
  2. **Planning**: Breaks down tasks into specific, actionable steps
  3. **Implementation**: Executes each planned step sequentially

What sets RA.Aid apart is its ability to handle complex programming tasks that extend beyond single-shot code edits. By combining research, strategic planning, and implementation into a cohesive workflow, RA.Aid can:

- Break down and execute multi-step programming tasks
- Research and analyze complex codebases to answer architectural questions
- Plan and implement significant code changes across multiple files
- Provide detailed explanations of existing code structure and functionality
- Execute sophisticated refactoring operations with proper planning

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Development Setup](#development-setup)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Three-Stage Architecture**: The workflow consists of three powerful stages:
  1. **Research** 🔍 - Gather and analyze information
  2. **Planning** 📋 - Develop execution strategy
  3. **Implementation** ⚡ - Execute the plan with AI assistance
  
  Each stage is powered by dedicated AI agents and specialized toolsets.
- **Advanced AI Integration**: Built on LangChain and leverages the latest LLMs for natural language understanding and generation.
- **Comprehensive Toolset**:
  - Shell command execution
  - Expert querying system
  - File operations and management
  - Memory management
  - Research and planning tools
  - Code analysis capabilities
- **Interactive CLI Interface**: Simple yet powerful command-line interface for seamless interaction
- **Modular Design**: Structured as a Python package with specialized modules for console output, processing, text utilities, and tools
- **Git Integration**: Built-in support for Git operations and repository management

## Installation

RA.Aid can be installed directly using pip:

```bash
pip install ra-aid
```

### Prerequisites

Before using RA.Aid, you'll need:

1. Python package `aider` installed and available in your PATH:
```bash
pip install aider-chat
```

2. API keys for the required AI services:

```bash
# Set up API keys based on your preferred provider:

# For Anthropic Claude models (recommended)
export ANTHROPIC_API_KEY=your_api_key_here

# For OpenAI models
export OPENAI_API_KEY=your_api_key_here

# For OpenRouter provider (optional)
export OPENROUTER_API_KEY=your_api_key_here

# For OpenAI-compatible providers (optional)
export OPENAI_API_BASE=your_api_base_url
```

Note: The programmer tool (aider) will automatically select its model based on your available API keys:
- If ANTHROPIC_API_KEY is set, it will use Claude models
- If only OPENAI_API_KEY is set, it will use OpenAI models
- You can set multiple API keys to enable different features

You can get your API keys from:
- Anthropic API key: https://console.anthropic.com/
- OpenAI API key: https://platform.openai.com/api-keys
- OpenRouter API key: https://openrouter.ai/keys

## Usage

RA.Aid is designed to be simple yet powerful. Here's how to use it:

```bash
# Basic usage
ra-aid -m "Your task or query here"

# Research-only mode (no implementation)
ra-aid -m "Explain the authentication flow" --research-only
```

### Command Line Options

- `-m, --message`: The task or query to be executed (required)
- `--research-only`: Only perform research without implementation
- `--cowboy-mode`: Skip interactive approval for shell commands
- `--provider`: Specify the model provider (See Model Configuration section)
- `--model`: Specify the model name (See Model Configuration section)
- `--expert-provider`: Specify the provider for the expert tool (defaults to OpenAI)
- `--expert-model`: Specify the model name for the expert tool (defaults to o1-preview for OpenAI)

### Model Configuration

RA.Aid supports multiple AI providers and models. The default model is Anthropic's Claude 3 Sonnet (`claude-3-5-sonnet-20241022`).

The programmer tool (aider) automatically selects its model based on your available API keys. It will use Claude models if ANTHROPIC_API_KEY is set, or fall back to OpenAI models if only OPENAI_API_KEY is available.

Note: The expert tool can be configured to use different providers (OpenAI, Anthropic, OpenRouter) using the --expert-provider flag along with the corresponding EXPERT_*API_KEY environment variables. Each provider requires its own API key set through the appropriate environment variable.

#### Environment Variables

RA.Aid supports multiple providers through environment variables:

- `ANTHROPIC_API_KEY`: Required for the default Anthropic provider
- `OPENAI_API_KEY`: Required for OpenAI provider
- `OPENROUTER_API_KEY`: Required for OpenRouter provider
- `OPENAI_API_BASE`: Required for OpenAI-compatible providers along with `OPENAI_API_KEY`

Expert Tool Environment Variables:
- `EXPERT_OPENAI_API_KEY`: API key for expert tool using OpenAI provider
- `EXPERT_ANTHROPIC_API_KEY`: API key for expert tool using Anthropic provider
- `EXPERT_OPENROUTER_API_KEY`: API key for expert tool using OpenRouter provider  
- `EXPERT_OPENAI_API_BASE`: Base URL for expert tool using OpenAI-compatible provider

You can set these permanently in your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
# Default provider (Anthropic)
export ANTHROPIC_API_KEY=your_api_key_here

# For OpenAI features and expert tool
export OPENAI_API_KEY=your_api_key_here

# For OpenRouter provider
export OPENROUTER_API_KEY=your_api_key_here

# For OpenAI-compatible providers
export OPENAI_API_BASE=your_api_base_url
```

Note: The expert tool defaults to OpenAI's o1-preview model with the OpenAI provider, but this can be configured using the --expert-provider flag along with the corresponding EXPERT_*_KEY environment variables.

#### Examples

1. **Using Anthropic (Default)**
   ```bash
   # Uses default model (claude-3-5-sonnet-20241022)
   ra-aid -m "Your task"

   # Or explicitly specify:
   ra-aid -m "Your task" --provider anthropic --model claude-3-5-sonnet-20241022
   ```

2. **Using OpenAI**
   ```bash
   ra-aid -m "Your task" --provider openai --model gpt-4o
   ```

3. **Using OpenRouter**
   ```bash
   ra-aid -m "Your task" --provider openrouter --model mistralai/mistral-large-2411
   ```

4. **Configuring Expert Provider**

   The expert tool is used by the agent for complex logic and debugging tasks. It can be configured to use different providers (OpenAI, Anthropic, OpenRouter) using the --expert-provider flag along with the corresponding EXPERT_*API_KEY environment variables.

   ```bash
   # Use Anthropic for expert tool
   export EXPERT_ANTHROPIC_API_KEY=your_anthropic_api_key
   ra-aid -m "Your task" --expert-provider anthropic --expert-model claude-3-5-sonnet-20241022

   # Use OpenRouter for expert tool
   export OPENROUTER_API_KEY=your_openrouter_api_key
   ra-aid -m "Your task" --expert-provider openrouter --expert-model mistralai/mistral-large-2411

   # Use default OpenAI for expert tool
   export EXPERT_OPENAI_API_KEY=your_openai_api_key
   ra-aid -m "Your task" --expert-provider openai --expert-model o1-preview
   ```

**Important Notes:**
- Performance varies between models. The default Claude 3 Sonnet model currently provides the best and most reliable results.
- Model configuration is done via command line arguments: `--provider` and `--model`
- The `--model` argument is required for all providers except Anthropic (which defaults to `claude-3-5-sonnet-20241022`)

### Example Tasks

1. Code Analysis:
   ```bash
   ra-aid -m "Explain how the authentication middleware works" --research-only
   ```

2. Complex Changes:
   ```bash
   ra-aid -m "Refactor the database connection code to use connection pooling" --cowboy-mode
   ```

3. Automated Updates:
   ```bash
   ra-aid -m "Update deprecated API calls across the entire codebase" --cowboy-mode
   ```

4. Code Research:
   ```bash
   ra-aid -m "Analyze the current error handling patterns" --research-only
   ```

2. Code Research:
   ```bash
   ra-aid -m "Explain how the authentication middleware works" --research-only
   ```

3. Refactoring:
   ```bash
   ra-aid -m "Refactor the database connection code to use connection pooling" --cowboy-mode
   ```

### Shell Command Automation with Cowboy Mode 🏇

The `--cowboy-mode` flag enables automated shell command execution without confirmation prompts. This is useful for:

- CI/CD pipelines
- Automated testing environments
- Batch processing operations
- Scripted workflows

```bash
ra-aid -m "Update all deprecated API calls" --cowboy-mode
```

**⚠️ Important Safety Notes:**
- Cowboy mode skips confirmation prompts for shell commands
- Always use in version-controlled repositories
- Ensure you have a clean working tree before running
- Review changes in git diff before committing

### Environment Variables

See the [Model Configuration](#model-configuration) section for details on provider-specific environment variables.

## Architecture


RA.Aid implements a three-stage architecture for handling development and research tasks:

1. **Research Stage**: 
   - Gathers information and context
   - Analyzes requirements
   - Identifies key components and dependencies

2. **Planning Stage**:
   - Develops detailed implementation plans
   - Breaks down tasks into manageable steps
   - Identifies potential challenges and solutions

3. **Implementation Stage**:
   - Executes planned tasks
   - Generates code or documentation
   - Performs necessary system operations

### Core Components

- **Console Module** (`console/`): Handles console output formatting and user interaction
- **Processing Module** (`proc/`): Manages interactive processing and workflow control
- **Text Module** (`text/`): Provides text processing and manipulation utilities
- **Tools Module** (`tools/`): Contains various utility tools for file operations, search, and more

## Dependencies

### Core Dependencies
- `langchain-anthropic`: LangChain integration with Anthropic's Claude
- `langgraph`: Graph-based workflow management
- `rich>=13.0.0`: Terminal formatting and output
- `GitPython==3.1.41`: Git repository management
- `fuzzywuzzy==0.18.0`: Fuzzy string matching
- `python-Levenshtein==0.23.0`: Fast string matching
- `pathspec>=0.11.0`: Path specification utilities

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `pytest-timeout>=2.2.0`: Test timeout management

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ai-christianson/ra-aid.git
cd ra-aid
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Run tests:
```bash
python -m pytest
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and commit:
```bash
git commit -m 'Add some feature'
```

4. Push to your fork:
```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Keep commits focused and message clear
- Ensure all tests pass before submitting PR

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 AI Christianson

## Contact

- **Issues**: Please report bugs and feature requests on our [Issue Tracker](https://github.com/ai-christianson/ra-aid/issues)
- **Repository**: [https://github.com/ai-christianson/ra-aid](https://github.com/ai-christianson/ra-aid)
- **Documentation**: [https://github.com/ai-christianson/ra-aid#readme](https://github.com/ai-christianson/ra-aid#readme)
