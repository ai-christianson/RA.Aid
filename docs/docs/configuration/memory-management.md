# Memory Management

## Introduction
RA.Aid's memory system enables the agent to persist knowledge across sessions, creating a more efficient and coherent experience. The memory stores key facts, code snippets, and research notes about your project, allowing RA.Aid to build on previous work and avoid redundant operations.

This persistent memory gives RA.Aid the ability to:
- Remember important project facts and architectural decisions
- Store relevant code snippets with their file locations
- Preserve research findings and technical notes
- Build contextual understanding of your project over time

## Database Architecture
RA.Aid's memory is stored in a SQLite database file located in a hidden `.ra-aid` directory within your current working directory. The specific file path is:

```
.ra-aid/pk.db
```

You can customize this location using the `--project-state-dir` flag to store the database in a different directory:

```bash
ra-aid -m "Your task" --project-state-dir /path/to/custom/directory
```

This will store the database at `/path/to/custom/directory/pk.db` instead of the default location. See [Project State Directory](./project-state.md) for more details.

This database implements several key tables:
- `KeyFact`: Stores important facts about your project
- `KeySnippet`: Preserves code snippets with their file paths and line numbers
- `ResearchNote`: Contains detailed research findings
- `HumanInput`: Tracks user inputs to provide context for memory items

All memory items have timestamps (created_at, updated_at) that help with relevance tracking and garbage collection.

The memory system uses thread-local state via Python's contextvars to ensure thread safety when multiple components access the database simultaneously.

## Memory Repositories and Garbage Collection
RA.Aid implements the repository pattern for database access, with specialized repositories for each memory type:
- `KeyFactRepository`
- `KeySnippetRepository`
- `ResearchNoteRepository`

Each repository provides methods to add, query, and delete items while abstracting the underlying database implementation.

To prevent uncontrolled growth of the memory database, RA.Aid implements automatic garbage collection that triggers when specific thresholds are exceeded:
- Facts: > 50 items
- Snippets: > 35 items
- Notes: > 30 items

When garbage collection activates, specialized agents analyze all stored items, considering factors like:
- Relevance to the current task
- Age of the memory items
- Relationship to other items
- Overall importance

Memory items associated with the current human input are protected from garbage collection to preserve context for the active task.

## CLI Commands and Configuration
RA.Aid provides a CLI flag to completely reset the memory database:

```bash
ra-aid --wipe-project-memory [other arguments]
```

This flag deletes the entire `.ra-aid/pk.db` database file, giving you a fresh start with no stored memory.

If you're using a custom project state directory, combine both flags:

```bash
ra-aid --wipe-project-memory --project-state-dir /path/to/custom/directory [other arguments]
```

This will delete the database file at `/path/to/custom/directory/pk.db` instead of the default location.

The memory statistics are displayed in the status panel when you start RA.Aid, showing:
- The number of facts, snippets, and notes currently stored
- A reminder about the `--wipe-project-memory` flag when memory items exist

### When to Wipe Memory
You might want to wipe project memory in these situations:

1. **Major Codebase Changes**: When your project has undergone significant refactoring or structural changes, making the stored memory items obsolete.

2. **Fresh Start**: When beginning a new phase of development and you want to clear out irrelevant historical context.

3. **Incorrect Information**: If the agent has stored incorrect or outdated information that's affecting its performance.

4. **Troubleshooting**: When unexpected behavior might be related to the stored memory items.

## Troubleshooting
Common memory-related issues and their solutions:

### Issue: Agent recalling outdated information
**Solution**: Use `--wipe-project-memory` to reset the memory database, especially after major code changes.

### Issue: Database lockup or corruption
**Solution**: 
1. Ensure RA.Aid has properly shut down before starting a new session
2. If issues persist, use `--wipe-project-memory` to recreate the database
3. Check the logs in `.ra-aid/logs/` for specific errors

### Issue: Memory items seem irrelevant
**Solution**:
- Let the automatic garbage collection work by continuing to use RA.Aid
- For immediate reset, use `--wipe-project-memory`

### Issue: Missing .ra-aid directory
**Solution**: The directory is automatically created when you run RA.Aid. If it's missing, simply run RA.Aid again.

## Examples / Use Cases

### Example 1: Wiping memory after major refactoring

```bash
# After refactoring your project structure
ra-aid --wipe-project-memory -m "Update the authentication system"
```

### Example 2: Starting a new development phase

```bash
# Before starting work on a new major feature
ra-aid --wipe-project-memory -m "Implement payment processing system"
```

### Example 3: Checking memory status without wiping

```bash
# Check the memory statistics in the status panel
ra-aid -m "Show me the project structure"
# Look for the 💾 Memory: X facts, Y snippets, Z notes line
```

### Example 4: Using memory during ongoing development

When working on a complex feature over multiple sessions, memory allows RA.Aid to:
1. Remember architectural decisions from previous sessions
2. Recall the context of partially implemented features
3. Build on previous research without repeating the same queries
4. Maintain awareness of project constraints and requirements