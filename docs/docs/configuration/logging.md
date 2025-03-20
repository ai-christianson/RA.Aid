# Logging System

RA.Aid includes a powerful and flexible logging system that helps you troubleshoot issues and monitor the tool's operation. This document explains how to configure and use the logging features effectively.

## Overview

The logging system in RA.Aid provides:

- Configurable logging modes for different use cases
- Multiple log levels for controlling verbosity
- File logging with rotation and backup capabilities
- Pretty console logging with formatted display
- Comprehensive log messages capturing agent activity

## Command Line Options

RA.Aid provides the following command line options to control logging behavior:

### Log Mode

The `--log-mode` option determines where logs are directed:

```bash
ra-aid -m "Add new feature" --log-mode file
```

Available modes:

- `file` (default): Logs are written to both file and console
  - Console shows only warnings and errors
  - File contains all logs at the specified log level
- `console`: Logs are only shown in the console at the specified log level
  - No log files are created

### Log Level

The `--log-level` option controls the verbosity of logging:

```bash
ra-aid -m "Add new feature" --log-level debug
```

Available levels (case-insensitive):

- `debug`: Most verbose, includes detailed debugging information
- `info`: General information about operations
- `warning` (default): Potential issues that might need attention
- `error`: Error conditions that prevent normal operation
- `critical`: Critical errors that may cause the application to terminate

The log level affects different outputs depending on the log mode:
- With `--log-mode=file`: Controls the file logging level, while console still shows only warnings and errors
- With `--log-mode=console`: Controls the console logging level directly

### Pretty Logger

The `--pretty-logger` option enables formatted panel-style logging output:

```bash
ra-aid -m "Add new feature" --pretty-logger
```

When enabled, log messages appear in colored panels with titles indicating the log level:
- 🔥 CRITICAL: Bold red panels for critical errors
- ❌ ERROR: Red panels for errors
- ⚠️ WARNING: Yellow panels for warnings
- ℹ️ INFO: Green panels for informational messages
- 🐞 DEBUG: Blue panels for debug messages

## Log Files

When `--log-mode=file` is used, RA.Aid creates and maintains log files with the following characteristics:

### Location

Log files are stored in the `.ra-aid/logs/` directory in your current working directory:

```
.ra-aid/logs/ra_aid_YYYYMMDD_HHMMSS.log
```

RA.Aid automatically creates this directory if it doesn't exist.

### Project State Directory

By default, logs are stored in the `.ra-aid` directory in your current working directory. However, you can customize this location using the `--project-state-dir` flag:

```bash
ra-aid -m "Your task" --project-state-dir /path/to/custom/directory
```

This will store logs in `/path/to/custom/directory/logs/` instead of the default location.

For more details on customizing the project state directory, see [Project State Directory](./project-state.md).

### Naming Convention

Log files follow a timestamp-based naming pattern:

```
ra_aid_YYYYMMDD_HHMMSS.log
```

Where:
- `YYYYMMDD`: Year, month, and day when the log file was created
- `HHMMSS`: Hour, minute, and second when the log file was created

Example: `ra_aid_20250301_143027.log`

### Log Rotation

RA.Aid uses automatic log rotation to manage log file size and prevent excessive disk usage:

- Maximum file size: 5 MB
- Maximum backup files: 100

When a log file reaches 5 MB, it is renamed with a numeric suffix (e.g., `.1`, `.2`), and a new log file is created. Up to 100 backup files are maintained.

## Examples

### Basic Usage (Default)

Use the default file logging mode with warnings and errors:

```bash
ra-aid -m "Add new feature"
```

### Detailed File Logging

Log everything including debug messages to file (console still shows only warnings+):

```bash
ra-aid -m "Add new feature" --log-level debug
```

### Console-Only Debugging

Get detailed debug logs in the console without creating log files:

```bash
ra-aid -m "Add new feature" --log-mode console --log-level debug
```

### Informational Console Logging

Get informational console output without debug details:

```bash
ra-aid -m "Add new feature" --log-mode console --log-level info
```

### Pretty Logging Output

Use formatted panel-style logging for better readability:

```bash
ra-aid -m "Add new feature" --pretty-logger
```

## Debugging Tips

- For troubleshooting issues, start with `--log-mode console --log-level debug`
- Examine log files in `.ra-aid/logs/` for historical issues
- Use `--pretty-logger` when working with complex tasks for better log clarity
- For production use, the default settings (`--log-mode file --log-level warning`) provide a good balance of information without excessive output

## Log Message Format

Standard log messages follow this format:

```
YYYY-MM-DD HH:MM:SS,MS - logger_name - LEVEL - Message text
```

Example:
```
2025-03-01 14:30:27,123 - ra_aid.agent_utils - WARNING - Command execution timeout after 60 seconds