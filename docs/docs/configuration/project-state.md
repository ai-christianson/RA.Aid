# Project State Directory

## Introduction

RA.Aid maintains a project state directory that stores persistent data including the project knowledge database and log files. By default, this directory is created as `.ra-aid` in your current working directory, but you can customize its location using the `--project-state-dir` flag.

Understanding how the project state directory works is important for:
- Managing project-specific knowledge across sessions
- Configuring logging for troubleshooting and monitoring
- Sharing or backing up project state
- Working with multiple projects or workspaces

## Directory Structure

The project state directory contains:

```
project-state-dir/
├── pk.db           # SQLite database containing project knowledge
└── logs/           # Directory containing log files
    └── ra_aid_YYYYMMDD_HHMMSS.log  # Log files with timestamps
```

### Database File

The `pk.db` file is a SQLite database that stores:
- Key facts about your project
- Code snippets with file paths and line numbers
- Research notes and findings
- Human input history
- Configuration settings

This database is the core of RA.Aid's memory system, allowing it to remember important information across sessions.

### Log Files

The `logs/` directory contains log files that follow a timestamp-based naming pattern:

```
ra_aid_YYYYMMDD_HHMMSS.log
```

These logs capture detailed information about RA.Aid's operations, which is essential for troubleshooting and understanding agent behavior.

## Command Line Configuration

You can specify a custom project state directory using the `--project-state-dir` flag:

```bash
ra-aid -m "Your task" --project-state-dir /path/to/custom/directory
```

This flag accepts an absolute or relative path to the directory where you want RA.Aid to store its state.

### Behavior

When using the `--project-state-dir` flag:

1. RA.Aid will use the specified directory directly (it doesn't create a `.ra-aid` subdirectory within it)
2. If the directory doesn't exist, RA.Aid will attempt to create it
3. Both the knowledge database and logs will be stored in this directory
4. The directory permissions must allow RA.Aid to create and write files

## Use Cases

### Cross-Project Knowledge Sharing

Use a shared project state directory to leverage knowledge across related projects:

```bash
# Working on project A with shared knowledge
ra-aid -m "Implement feature" --project-state-dir ~/shared-ra-aid

# Later, working on project B with the same knowledge base
cd ~/project-B
ra-aid -m "Implement similar feature" --project-state-dir ~/shared-ra-aid
```

### Project-Specific Knowledge Isolation

Keep project knowledge separate for unrelated projects:

```bash
# For project A
cd ~/project-A
ra-aid -m "Work on Project A" --project-state-dir ~/project-A-ra-aid

# For project B
cd ~/project-B
ra-aid -m "Work on Project B" --project-state-dir ~/project-B-ra-aid
```

### Backup and Restore

Easily back up and restore your project state:

```bash
# Back up by copying the directory
cp -r ~/.ra-aid ~/ra-aid-backup-20250319

# Restore from backup
ra-aid -m "Continue work" --project-state-dir ~/ra-aid-backup-20250319
```

### Team Collaboration

Share project knowledge with team members:

```bash
# Export project state to a shared location
ra-aid -m "Document API" --project-state-dir /team/shared/project-knowledge

# Team members can use the same knowledge base
ra-aid -m "Extend API" --project-state-dir /team/shared/project-knowledge
```

### Permission Issues Resolution

Resolve permission issues by specifying a directory with proper access rights:

```bash
# Use a directory with appropriate permissions
ra-aid -m "Fix bug" --project-state-dir ~/user-owned-directory
```

## When to Change the Project State Directory

Consider using a custom project state directory when:

1. **Working Across Multiple Projects**: To share knowledge between related projects or isolate unrelated ones
2. **Permission Issues**: When the default location isn't writable or accessible
3. **Backing Up State**: To create snapshots of project knowledge at specific milestones
4. **Collaboration**: To share project knowledge with team members
5. **Disk Space Management**: To store the potentially large database and log files on a different drive
6. **CI/CD Integration**: To specify a consistent location in automated environments

## Troubleshooting

### Directory Creation Issues

If RA.Aid fails to create the project state directory:

1. Check if you have write permissions to the parent directory
2. Verify the path is valid and accessible
3. Try creating the directory manually before running RA.Aid
4. Review log output for specific error messages

### Database Access Problems

If RA.Aid can't access the database:

1. Ensure the directory exists and is accessible
2. Check file permissions on the `pk.db` file
3. Verify no other processes have locked the database
4. Consider using `--wipe-project-memory` along with `--project-state-dir` to reset the database

## Related Configuration

The project state directory is closely related to two other configuration areas:

- [Memory Management](./memory-management.md): The knowledge database stored in the project state directory is the foundation of RA.Aid's memory system
- [Logging Configuration](./logging.md): Log files stored in the project state directory provide detailed information about RA.Aid's operations

## Examples

### Basic Usage with Custom Directory

```bash
# Use a specific directory for project state
ra-aid -m "Implement feature" --project-state-dir ~/my-project/ra-aid-state
```

### Combining with Memory Management

```bash
# Wipe project memory in a custom directory
ra-aid -m "Fresh start" --project-state-dir ~/custom-state --wipe-project-memory
```

### Combining with Logging Options

```bash
# Custom project state with specific logging configuration
ra-aid -m "Debug issue" --project-state-dir ~/debug-state --log-level debug --pretty-logger
```
