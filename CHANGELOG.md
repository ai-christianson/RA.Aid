# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.4] 2025-03-24

### Added
- Custom Tools Feature
  - Added support for custom tools with `--custom-tools <path>` CLI flag
  - Implemented MCP (Model-Completion-Protocol) client for integrating external tool providers
  - Created documentation on custom tools usage in `docs/docs/usage/custom-tools.md`
  - Added example code in `examples/custom-tools-mcp/` directory
- API Documentation
  - Added comprehensive OpenAPI documentation for REST API endpoints
  - Implemented API documentation in Docusaurus with new MDX files
  - Added YAML OpenAPI specification file `docs/ra-aid.openapi.yml`
  - Created script to generate OpenAPI documentation automatically
- Session Usage Statistics
  - Added CLI commands for retrieving usage statistics for all sessions and the latest session
  - Enhanced session and trajectory repositories with new methods
  - Moved scripts into proper Python package structure (`ra_aid/scripts/`)
- Web UI Improvements
  - Added new UI components including input box, session screen, and buttons
  - Improved session management UI
  - Enhanced styling and layout

### Changed
- WebSocket Endpoint Migration
  - Migrated WebSocket endpoint from `/ws` to `/v1/ws` to align with REST API endpoint pattern
  - Updated root HTML endpoint to reflect the new WebSocket path
- Project Maintenance
  - Refactored agent creation logic to use model capabilities for selecting agent type
  - Improved model detection and normalization
  - Updated dependencies via uv.lock
  - Fixed various typos and improved prompts

## [0.18.0] 2025-03-19

### Added
- Project State Directory Feature
  - Added `--project-state-dir` parameter to allow customization of where project data is stored
  - Modified database connection, logging, and memory wiping to support custom directories
  - Created comprehensive documentation in docs/docs/configuration/project-state.md
- Ollama Integration
  - Added support for running models locally via Ollama
  - Implemented configuration options including model selection and context window size
  - Added documentation for Ollama in docs/docs/configuration/ollama.md
  - Updated open-models.md to include Ollama as a supported provider
- Web UI and API Progress (partially implemented)
  - Created API endpoints for session management (create, list, retrieve)
  - Added trajectory tracking and visualization
  - Implemented UI components for session management
  - Added server infrastructure for web interface
- Token Usage and Cost Tracking
  - Enhanced trajectory tracking with token counting
  - Added session-level token usage and cost tracking
  - Improved cost calculation and logging

## [0.17.1] 2025-03-13

### Fixed
- Fixed bug with `process_thinking_content` function by moving it from `agent_utils` to `ra_aid.text.processing` module
- Fixed config parameter handling in research request functions
- Updated development setup instructions in README to use `pip install -e \".[dev]\"` instead of `pip install -r requirements-dev.txt`

## [0.17.0] 2025-03-12

### Added
- Added support for think tags in models with the new extract_think_tag function
- Enhanced CiaynAgent and expert tool to extract and display thinking content from <think>...</think> tags
- Added model parameters for think tag support
- Added comprehensive testing for think tag functionality
- Added `--show-thoughts` flag to show thoughts of thinking models
- Added `--show-cost` flag to display cost information during agent operations
- Enhanced cost tracking with AnthropicCallbackHandler for monitoring token usage and costs
- Added Session and Trajectory models to track application state and agent actions
- Added comprehensive environment inventory system for collecting and providing system information to agents
- Added repository implementations for Session and Trajectory models
- Added support for reasoning assistance in research phase
- Added new config parameters for managing cost display and reasoning assistance

### Changed
- Updated langchain/langgraph deps
- Improved trajectory tracking for better debugging and analysis
- Enhanced prompts throughout the system for better performance
- Improved token management with better handling of thinking tokens in Claude models
- Updated project information inclusion in prompts
- Reorganized agent code with better extraction of core functionality
- Refactored anthropic token limiting for better control over token usage

### Fixed
- Fixed binary file detection
- Fixed environment inventory sorting
- Fixed token limiter functionality
- Various test improvements and fixes

## [0.16.1] 2025-03-07

### Changed
- Replaced thread-local storage with contextvars in agent_context.py for better context isolation
- Improved React agent execution with LangGraph's interrupt mechanism
- Enhanced _run_agent_stream function to properly handle agent state and continuation

### Fixed
- Fixed tests to work with the new implementation

## [0.16.0] 2025-03-07

### Added
- Database-backed memory system with SQLite (.ra-aid/pk.db)
- Repository pattern for memory access (KeyFactRepository, KeySnippetRepository, ResearchNoteRepository)
- Memory garbage collection with configurable thresholds
- "--wipe-project-memory" flag to reset memory
- Memory statistics in status panel
- Propagation depth control for agent_should_exit
- Fixed string parameter for ripgrep tool
- Support for Claude 3.7 Sonnet thinking tokens in expert tool

### Changed
- Enhanced file logging with support for .ra-aid/logs/
- Improved CiaynAgent with better tool validation and execution
- Memory-related prompt improvements

### Fixed
- Various bug fixes in tool execution
- Test improvements for memory system

## [0.15.2] - 2025-02-27

### Added
- Added agent_should_exit context functionality with propagation to parent contexts
- Improved agent crash detection with non-propagating crash state
- Enhanced ripgrep tool with better context support
- Improved agent context inheritance
- Added comprehensive test coverage for exit and crash handling

## [0.15.1] - 2025-02-27

### Fixed
- Improved chat prompt to prevent endless loop behavior with sonnet 3.7.

## [0.15.0] - 2025-02-27

### Added
- Added database infrastructure with models, connections, and migrations
- Added agent context system for improved context management
- Added aider-free mode with command line option to disable aider-related functionality
- Added database-related dependencies

### Changed
- Improved file editing tools with enhanced functionality
- Enhanced agent implementation tools with modified return values and logic
- Improved agent tool prompts for better clarity and effectiveness
- Fixed langgraph prebuilt dependency

### Fixed
- Fixed project state detection logic with added tests

## [0.14.9] - 2025-02-25

### Added
- Added binary file detection and filtering to prevent binary files from being added to related files
- Added python-magic dependency for improved binary file detection
- Added support for "thinking" budget parameter for Claude 3.7 Sonnet

### Changed
- Updated dependencies:
  - langchain-anthropic from 0.3.7 to 0.3.8
  - langchain-google-genai from 2.0.10 to 2.0.11
- Improved shell command tool description to recommend keeping commands under 300 words
- Enhanced binary file filtering to include detailed reporting of skipped files
- Updated test assertions to be more flexible with parameter checking

## [0.14.8] - 2025-02-25

### Changed
- Improved programmer.py tool prompts for better clarity on related files visibility
- Enhanced programmer tool to remind users to call emit_related_files on any new files created
- Updated README.md to use media queries for showing different logos based on color scheme preference


## [0.14.7] - 2025-02-25

### Added
- Windows compatibility improvements
  - Add error handling for Windows-specific modules
  - Add Windows-specific tests for compatibility

### Changed
- Improve cross-platform support in interactive.py
- WebUI improvements
  - Improve message display
  - Add syntax highlighting
  - Add animations
- Expert tool prompt improvements

### Fixed
- WebUI improvements
  - Fix WebSocket communication
- Interactive command handling improvements
  - Fix interactive history capture
  - Fix command capture bugs
  - Multiple fixes for interactive command execution on both Linux and Windows
  - Enhance error handling for interactive processes

## [0.14.6] - 2025-02-25

### Added
- Added `--no-git` flag to aider commands to prevent git operations

### Changed
- Updated aider-chat dependency from 0.75 to 0.75.1
- Improved prompts for better tool effectiveness
- Enhanced emit_key_snippet documentation to focus on upcoming work relevance

## [0.14.5] - 2025-02-24

### Changed
- Optimized prompts

## [0.14.4] - 2025-02-24

### Changed
- Updated aider-chat dependency from 0.74.2 to 0.75
- Improved tool calling performance by minimizing tool return values
- Replaced emit_key_snippets with emit_key_snippet for simpler code snippet management
- Simplified return values for multiple tools to improve tool calling accuracy
- Updated tool prompts to remove unnecessary context cleanup references
- Reorganized order of tools in read-only tools list

### Fixed
- Fixed tests to align with updated tool return values
- Updated test assertions to match new simplified tool outputs

## [0.14.3] - 2025-02-24

### Added
- Added support for Claude 3.7 Sonnet model
- Added version display in startup configuration panel

### Changed
- Updated language library dependencies (langgraph, langchain-core, langchain, langchain-openai, langchain-google-genai)
- Changed default Anthropic model from Claude 3.5 Sonnet to Claude 3.7 Sonnet

### Fixed
- Fixed f-string syntax error in write_file.py
- Fixed bug where model selection on Anthropic was always using default instead of respecting user selection
- Fixed Anthropic key error message to reference the correct variable
- Added test for user-specified Anthropic model selection

## [0.14.2] - 2025-02-19

### Added
- Added automatic fallback mechanism to alternative LLM models on consecutive failures
- Added FallbackHandler class to manage tool failures and fallback logic
- Added console notification for tool fallback activation
- Added detailed fallback configuration options in command line arguments
- Added validation for required environment variables for LLM providers

### Changed
- Enhanced CiaynAgent to handle chat history and improve context management
- Improved error handling and logging in fallback mechanism
- Streamlined fallback model selection and invocation process
- Refactored agent stream handling for better clarity
- Reduced maximum tool failures from 3 to 2

### Fixed
- Fixed tool execution error handling and retry logic
- Enhanced error resilience and user experience with fallback handler
- Improved error message formatting and logging
- Updated error handling to include base message for better debugging

## [0.14.1] - 2025-02-13

### Added
- Added expected_runtime_seconds parameter for shell commands with graceful process shutdown
- Added config printing at startup (#88)

### Changed
- Enforce byte limit in interactive commands
- Normalize/dedupe related file paths
- Relax aider version requirement for SWE-bench compatibility 
- Upgrade langchain/langgraph dependencies

### Fixed
- Fixed aider flags (#89)
- Fixed write_file_tool references

## [0.14.0] - 2025-02-12

### Added
- Status panel showing tool/LLM status and outputs
- Automatic detection of OpenAI expert models
- Timeouts on LLM clients

### Changed
- Improved interactive TTY process capture and history handling
- Upgraded langgraph dependencies
- Improved prompts and work logging
- Refined token/bytes ratio handling
- Support default temperature on per-model basis
- Reduced tool count for more reliable tool calling
- Updated logo and branding assets
- Set environment variables to disable common interactive modes

### Fixed
- Various test fixes
- Bug fixes for completion message handling and file content operations
- Interactive command input improvements
- Use reasoning_effort=high for OpenAI expert models
- Do not default to o1 model (#82)
- Make current working directory and date available to more agents

## [0.13.2] - 2025-02-02

- Fix temperature parameter error for expert tool.

## [0.13.1] - 2025-01-31

### Added
- WebUI (#61)
- Support o3-mini

### Changed
- Convert list input to string to handle create-react-agent tool calls correctly (#66)
- Add commands for code checking and fixing using ruff (#63)

### Fixed
- Fix token estimation
- Fix tests
- Prevent duplicate files (#64)
- Ensure default temperature is set correctly for different providers
- Do not incorrectly give temp parameter to expert model
- Correcting URLs that were referencing ai-christianson/ra-aid - should be ai-christianson/RA.Aid (#69)

### Improved
- Integrate litellm to retrieve model token limits for better flexibility (#51)
- Handle user defined test cmd (#59)
- Run tests during Github CICD (#58)
- Refactor models_tokens to be models_params so we can track multiple parameters on a per-model basis.

## [0.13.0] - 2025-01-22

### Added
- Added Deepseek Provider Support and Custom Deepseek Reasoner Chat Model (#50)
- Added Aider config File Argument Support (#43) 
- Added configurable --recursion-limit argument (#46)
- Set Default Max Token Limit with Provider/Model Dictionary (#45)

### Changed
- Updated aider-chat version from 0.69.1 to 0.72.1 (#47)

### Fixed
- Fixed Issue 42 related to Levenshtein (#44)

### Improved
- Various prompt improvements
- Better handling of 429 errors on openrouter
- Improved project info handling and token usage optimization
- Extracted tool reflection functionality
- Improved work log handling
- Added tests for CiaynAgent._does_not_conform_to_pattern

## [0.12.1] - 2025-01-08
- Fix bug where directories are added as related files.

## [0.12.0] - 2025-01-04

### Added
- Google Gemini AI provider support
- Dependency check functionality in ra_aid/dependencies.py
- Test coverage reporting to pytest commands

### Changed
- Updated minimum Python requirement to 3.9
- Updated OpenAI model defaults
- Modified test files to support new Gemini provider
- Updated SWE-bench dataset generation script with UV package management

### Fixed
- Date-based assertions in directory listing tests

## [0.11.3] - 2024-12-30

- MacOS fixes.

## [0.11.2] - 2024-12-30

- Fix SyntaxError: f-string expression part cannot include a backslash.

## [0.11.1] - 2024-12-29

- Improve prompts.
- Fix issue #24.

## [0.11.0] - 2024-12-28

- Add CiaynAgent to support models that do not have, or are not good at, agentic function calling.
- Improve env var validation.
- Add --temperature CLI parameter.

## [0.10.3] - 2024-12-27

- Fix logging on interrupt.
- Fix web research prompt.
- Simplify planning stage by executing tasks directly.
- Make research notes available to more agents/tools.

## [0.10.2] - 2024-12-26

- Add logging.
- Fix bug where anthropic is used in chat mode even if another provider is specified.

## [0.10.0] - 2024-12-24

- Added new web research agent.

## [0.9.1] - 2024-12-23

- Fix ask human multiline continuation.

## [0.9.0] - 2024-12-23

- Improve agent interruption UX by allowing user to specify feedback or exit the program entirely.
- Do not put file ID in file paths when reading for expert context.
- Agents log work internally, improving context information.
- Clear task list when plan is completed.

## [0.8.2] - 2024-12-23

- Optimize first prompt in chat mode to avoid unnecessary LLM call.

## [0.8.1] - 2024-12-22

- Improved prompts.

## [0.8.0] - 2024-12-22

- Chat mode.
- Allow agents to be interrupted.

## [0.7.1] - 2024-12-20

- Fix model parameters.

## [0.7.0] - 2024-12-20

- Make delete_tasks tool available to planning agent.
- Get rid of implementation args as they are not used.
- Improve ripgrep tool status output.
- Added ask_human tool to allow human operator to answer questions asked by the agent.
- Handle keyboard interrupt (ctrl-c.)
- Disable PAGERs for shell commands so agent can work autonomously.
- Reduce model temperatures to 0.
- Update dependencies.

## [0.6.4] - 2024-12-19

- Added monorepo_detected, existing_project_detected, and ui_detected tools so the agent can take specific actions.
- Prompt improvements for real-world projects.
- Fix env var fallback when base key is given, expert and base provider are different, and expert key is missing.

## [0.6.3] - 2024-12-18

- Fix one shot completion signaling.
- Clean up error outputs.
- Update prompt for better performance on large/monorepo projects.
- Update programmer prompt so we don't use it to delete files.

## [0.6.2] - 2024-12-18
- Allow shell commands to be run in read-only mode.
- When asking for shell command approval, allow cowboy mode to be enabled.
- Update prompt to suggest commands be run in non-interactive mode if possible, e.g. using --no-pager git flag.
- Show tool errors in a panel.

## [0.6.1] - 2024-12-17

### Added
- When key snippets are emitted, snippet files are auto added to related files.
- Add base task to research subtask prompt.
- Adjust research prompt to make sure related files are related to the base task, not just the research subtask.
- Track tasks by ID and allow them to be deleted.
- Make one_shot_completed tool available to research agent.
- Make sure file modification tools are not available when research only flag is used.
- Temporarily disable write file/str replace as they do not work as well as just using the programmer tool.

## [0.6.0] - 2024-12-17

### Added
- New `file_str_replace` tool for performing exact string replacements in files with unique match validation
- New `write_file_tool` for writing content to files with rich output formatting and comprehensive error handling
