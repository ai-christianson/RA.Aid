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

<img src="assets/demo.gif" alt="RA.Aid Demo" autoplay loop style="width: 100%; max-width: 800px;">

> 👋 **Pull requests are very welcome!** As a technical founder with limited time, I greatly appreciate any contributions to this repository. Don't be shy - your help makes a real difference!
>
> 💬 **Join our Discord community:** [Click here to join](https://discord.gg/f6wYbzHYxV)

# RA.Aid

RA.Aid (ReAct Aid) is a powerful AI-driven command-line tool that integrates `aider` (https://aider.chat/) within a LangChain ReAct agent loop. This unique combination allows developers to leverage aider's code editing capabilities while benefiting from LangChain's agent-based task execution framework. The tool provides an intelligent assistant that can help with research, planning, and implementation of development tasks.

⚠️ **IMPORTANT: USE AT YOUR OWN RISK** ⚠️
- This tool **can and will** automatically execute shell commands on your system
- Shell commands require interactive approval unless --cowboy-mode is enabled
- The --cowboy-mode flag disables command approval and should be used with extreme caution
- No warranty is provided, either express or implied
- Always review the actions the agent proposes before allowing them to proceed

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
# Required: Set up your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here

# Optional: Set up OpenAI API key if using OpenAI features
export OPENAI_API_KEY=your_api_key_here
```

You can get your API keys from:
- Anthropic API key: https://console.anthropic.com/
- OpenAI API key: https://platform.openai.com/api-keys

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

### Automating Code Changes with Cowboy Mode 🏇

For situations where you need to automate code modifications without manual intervention—such as continuous integration/continuous deployment (CI/CD) pipelines, scripted batch operations, or large-scale refactoring—you can use the `--cowboy-mode` flag. This mode executes commands non-interactively, bypassing the usual confirmation prompts.

```bash
ra-aid -m "Update all deprecated API calls" --cowboy-mode
```

In the example above, the command will automatically find and update all deprecated API calls in your codebase **without** asking for confirmation before each change.

**⚠️ Use with Extreme Caution:** Cowboy mode is a powerful tool that removes safety checks designed to prevent unintended modifications. While it enables efficient automation, it also increases the risk of errors propagating through your codebase. **Ensure you have proper backups or version control in place before using this mode.**

**Appropriate Use Cases for Cowboy Mode:**

- **CI/CD Pipelines:** Automate code changes as part of your deployment process.
- **Scripted Batch Operations:** Apply repetitive changes across multiple files without manual approval.
- **Controlled Environments:** Use in environments where changes can be reviewed and reverted if necessary.

**When Not to Use Cowboy Mode:**

- **Research or Experimental Changes:** When you are exploring solutions and unsure of the outcomes.
- **Critical Codebases Without Backups:** If you don't have a way to revert changes, it's safer to use the interactive mode.

### Environment Variables

RA.Aid uses the following environment variables:

- `ANTHROPIC_API_KEY` (Required): Your Anthropic API key for accessing Claude
- `OPENAI_API_KEY` (Optional): Your OpenAI API key if using OpenAI features

You can set these permanently in your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
export ANTHROPIC_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_api_key_here
```

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
