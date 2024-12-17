# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- When key snippets are emitted, snippet files are auto added to related files.
- Add base task to research subtask prompt.
- Adjust research prompt to make sure related files are related to the base task, not just the research subtask.

## [0.6.0] - 2024-12-17

### Added
- New `file_str_replace` tool for performing exact string replacements in files with unique match validation
- New `write_file_tool` for writing content to files with rich output formatting and comprehensive error handling