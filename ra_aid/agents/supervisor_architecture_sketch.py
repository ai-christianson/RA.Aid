"""
SUPERVISOR ARCHITECTURE SKETCH
==============================

Key insight: The Supervisor IS the expert.
- Supervisor = expert model (smartest, does planning + decisions)
- Workers = cheaper/faster models (just execute specific tasks)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# =============================================================================
# TASK DEFINITION
# =============================================================================

class TaskType(Enum):
    RESEARCH = "research"
    IMPLEMENT = "implement"
    VERIFY = "verify"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: int
    type: TaskType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)


# =============================================================================
# WORKERS - Cheap/fast models, stateless, do ONE thing
# =============================================================================

class Worker:
    """Base worker - executes a single task and returns."""

    def __init__(self, model, tools: List[str]):
        self.model = model  # Cheaper/faster model
        self.tools = tools

    def execute(self, task: Task, context: str) -> str:
        """
        Execute task with given context. Returns result.

        CANNOT:
        - Spawn other agents
        - Create new tasks
        - Loop indefinitely
        - Make architectural decisions
        """
        raise NotImplementedError


class Researcher(Worker):
    """Gathers information only."""

    def __init__(self, model):
        super().__init__(model, tools=[
            "read_file",
            "list_directory",
            "grep_search",
            "web_search",
        ])

    def execute(self, task: Task, context: str) -> str:
        prompt = f"""
        Task: {task.description}

        Context:
        {context}

        Find the requested information using available tools.
        Return ONLY your findings. No suggestions, no next steps.
        """
        return self._tool_loop(prompt)

    def _tool_loop(self, prompt: str) -> str:
        """Simple tool loop - runs until model says done or max iterations."""
        # Bounded loop with hard iteration limit
        pass


class Implementor(Worker):
    """Makes changes only."""

    def __init__(self, model):
        super().__init__(model, tools=[
            "read_file",
            "write_file",
            "edit_file",
            "run_command",
        ])

    def execute(self, task: Task, context: str) -> str:
        prompt = f"""
        Task: {task.description}

        Context:
        {context}

        Make the required changes.
        Return ONLY what you changed. No research, no questions.
        """
        return self._tool_loop(prompt)

    def _tool_loop(self, prompt: str) -> str:
        pass


# =============================================================================
# SUPERVISOR - The Expert Model
# =============================================================================

class Supervisor:
    """
    The Supervisor IS the expert.

    - Uses the best/smartest model
    - Creates and owns the plan
    - Delegates execution to cheaper workers
    - Makes all decisions
    - Knows when done
    """

    def __init__(self, expert_model, worker_model):
        self.model = expert_model  # The expert - smartest model

        # Workers use cheaper model
        self.researcher = Researcher(worker_model)
        self.implementor = Implementor(worker_model)

        self.tasks: List[Task] = []
        self.context: str = ""

    def run(self, user_request: str) -> str:
        """
        Main entry point. THE ONLY LOOP IN THE SYSTEM.
        """

        # === PHASE 1: PLAN ===
        # Expert creates the plan
        self.tasks = self._create_plan(user_request)
        self._print_plan()

        # === PHASE 2: EXECUTE ===
        # Loop through tasks, delegate to workers
        while task := self._get_next_task():
            task.status = TaskStatus.IN_PROGRESS

            try:
                if task.type == TaskType.RESEARCH:
                    result = self.researcher.execute(task, self.context)
                    self.context += f"\n\n## {task.description}\n{result}"

                elif task.type == TaskType.IMPLEMENT:
                    result = self.implementor.execute(task, self.context)

                elif task.type == TaskType.VERIFY:
                    result = self._verify(task)

                task.result = result
                task.status = TaskStatus.COMPLETED

                # Expert reviews and may adjust plan
                self._review_and_adjust(task)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = str(e)
                # Expert decides: retry, skip, or abort?
                if not self._handle_failure(task):
                    break

        # === PHASE 3: DONE ===
        return self._create_summary()

    def _create_plan(self, user_request: str) -> List[Task]:
        """
        Expert decomposes request into tasks.

        This is where the intelligence lives.
        Single LLM call, structured output.
        """
        prompt = f"""
        User request: {user_request}

        Create a minimal plan to accomplish this.

        Task types:
        - RESEARCH: Gather specific information (before implementing)
        - IMPLEMENT: Make specific changes (after research)
        - VERIFY: Check something works (after implementation)

        Rules:
        - Research FIRST, then implement
        - Be specific - "find X" not "understand the codebase"
        - Minimal tasks - don't over-plan
        - Order by dependencies

        Output format:
        1. [RESEARCH] Find where X is defined
        2. [RESEARCH] Check existing patterns for Y
        3. [IMPLEMENT] Add Z to file A
        4. [VERIFY] Run tests
        """

        response = self.model.invoke(prompt)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> List[Task]:
        """Parse LLM output into Task objects."""
        tasks = []
        # Parse numbered list...
        return tasks

    def _get_next_task(self) -> Optional[Task]:
        """Get next pending task with met dependencies."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                deps_met = all(
                    self.tasks[d].status == TaskStatus.COMPLETED
                    for d in task.dependencies
                )
                if deps_met:
                    return task
        return None

    def _review_and_adjust(self, completed_task: Task) -> None:
        """
        Expert reviews completed task, may adjust plan.

        This is bounded - can only modify remaining tasks.
        Cannot spawn new agent chains.
        """
        # Could: add a task, remove unnecessary task, reorder
        # Single LLM call to review, not a new agent
        pass

    def _handle_failure(self, failed_task: Task) -> bool:
        """
        Expert decides how to handle failure.

        Returns True to continue, False to abort.
        """
        prompt = f"""
        Task failed: {failed_task.description}
        Error: {failed_task.result}

        Options:
        1. RETRY - try the task again
        2. SKIP - mark as skipped, continue with next
        3. ABORT - stop execution

        Which option and why?
        """

        response = self.model.invoke(prompt)
        # Parse response, maybe retry or skip
        return True  # Continue by default

    def _verify(self, task: Task) -> str:
        """Run verification - tests, linting, etc."""
        # Could run shell commands to verify
        pass

    def _print_plan(self) -> None:
        """Display the plan to user."""
        print("\n=== PLAN ===")
        for task in self.tasks:
            print(f"  {task.id}. [{task.type.value}] {task.description}")
        print("============\n")

    def _create_summary(self) -> str:
        """Create final summary."""
        completed = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.tasks if t.status == TaskStatus.FAILED]

        summary = f"Completed {len(completed)}/{len(self.tasks)} tasks.\n\n"
        for task in completed:
            summary += f"✓ {task.description}\n"
        for task in failed:
            summary += f"✗ {task.description}: {task.result}\n"

        return summary


# =============================================================================
# USAGE
# =============================================================================

def main(user_request: str, expert_model, worker_model):
    """
    Clean entry point.

    expert_model: Best model (claude-opus, gpt-4, etc.)
    worker_model: Fast/cheap model (claude-haiku, gpt-3.5, etc.)
    """
    supervisor = Supervisor(expert_model, worker_model)
    return supervisor.run(user_request)


# =============================================================================
# WHY THIS WORKS
# =============================================================================

"""
1. EXPERT IS THE SUPERVISOR
   - Smartest model makes all decisions
   - Cheaper models just execute
   - No wasted tokens on dumb models making decisions

2. ONE LOOP
   - Only Supervisor.run() loops
   - Workers execute() and return
   - No nested loops, no spawn chains

3. PLAN FIRST
   - Expert plans BEFORE any work
   - Tasks are explicit and ordered
   - No "what should I do next?" decisions during execution

4. BOUNDED ADJUSTMENT
   - Expert can adjust plan after each task
   - But only remaining tasks, not new agent chains
   - Adaptation without chaos

5. CLEAR TERMINATION
   - Task list is finite
   - _get_next_task() returns None when done
   - No ambiguity

6. FAILURE HANDLING BY EXPERT
   - When worker fails, expert decides what to do
   - Not the worker deciding to spawn more work
   - Centralized control
"""
