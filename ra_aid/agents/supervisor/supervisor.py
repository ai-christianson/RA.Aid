"""Supervisor - The expert-driven orchestrator.

The Supervisor is the ONLY component that:
- Creates and owns the task list
- Delegates work to workers
- Makes all decisions
- Knows when done

This is THE ONLY LOOP in the system.
"""

import json
import logging
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ra_aid.agents.supervisor.models import Task, TaskType, TaskStatus
from ra_aid.agents.supervisor.workers import Researcher, Implementor, Verifier
from ra_aid.console.formatting import cpm, console_panel

logger = logging.getLogger(__name__)


PLANNING_PROMPT = """You are an expert software architect planning a task.

Given a user request, create a MINIMAL plan to accomplish it.

Task types:
- RESEARCH: Gather specific information (always do this FIRST)
- IMPLEMENT: Make specific code changes (after research)
- VERIFY: Run tests or check that changes work (after implementation)

Rules:
1. Research BEFORE implementing - you need context first
2. Be SPECIFIC - "find where X is defined" not "understand the codebase"
3. Be MINIMAL - don't over-plan, fewer tasks is better
4. Order by dependencies - research before implement, implement before verify

Response types (output ONLY ONE as JSON):

1. Clarification needed:
   {"clarify": "your question here"}

2. Conversational response:
   {"chat": "your response"}

3. Create/update an artifact (commit message, plan, document, code block):
   {"artifact": {"type": "commit_message|plan|document|code", "content": "the full content", "description": "brief description"}}

4. Execute with current artifact (when user approves):
   {"execute_artifact": true}

5. Actionable tasks:
   [{"type": "RESEARCH", "description": "..."}, {"type": "IMPLEMENT", "description": "..."}]

When user asks to draft/write/create something iterable, use artifact response.
When user says "yes", "do it", "commit", etc. to approve an artifact, use execute_artifact.

ONLY output JSON, nothing else."""


class Supervisor:
    """
    The expert-driven orchestrator.

    Uses the best model for planning/decisions.
    Delegates execution to cheaper worker models.
    """

    def __init__(self, expert_model, worker_model):
        """
        Initialize the supervisor.

        Args:
            expert_model: The expert/planning model (e.g., claude-opus, gpt-4)
            worker_model: The worker model (e.g., claude-haiku, gpt-3.5)
        """
        self.expert_model = expert_model
        self.worker_model = worker_model

        # Workers
        self.researcher = Researcher(worker_model)
        self.implementor = Implementor(worker_model)
        self.verifier = Verifier(worker_model)

        # State
        self.tasks: List[Task] = []
        self.context: str = ""  # Accumulated research findings
        self.completed_task_ids: set = set()
        self.conversation_history: List[tuple[str, str]] = []  # (user_msg, assistant_msg) pairs

        # Working artifact - the current thing being iterated on
        self.working_artifact: Optional[dict] = None  # {"type": "commit_message|plan|document|code", "content": "...", "description": "..."}

    def run(self, user_request: str) -> str:
        """
        Main entry point. THE ONLY LOOP IN THE SYSTEM.

        Args:
            user_request: What the user wants to accomplish

        Returns:
            Summary of what was done
        """
        console_panel(f"User Request: {user_request}", title="Supervisor", border_style="blue bold")

        # === PHASE 1: PLAN ===
        console_panel("Creating plan...", title="Phase 1: Planning", border_style="yellow")
        self.tasks, chat_message = self._create_plan(user_request)

        # Handle conversational/clarification responses
        if chat_message:
            console_panel(chat_message, title="💬 Response", border_style="cyan")
            # Store in conversation history for context
            self.conversation_history.append((user_request, chat_message))
            return chat_message

        self._print_plan()

        if not self.tasks:
            return "Could not create a plan for this request."

        # === PHASE 2: EXECUTE ===
        console_panel("Executing tasks...", title="Phase 2: Execution", border_style="green")

        while True:
            task = self._get_next_task()
            if task is None:
                break  # All done!

            task.mark_in_progress()
            self._print_task_start(task)

            try:
                result = self._execute_task(task)
                task.mark_completed(result)
                self.completed_task_ids.add(task.id)
                self._print_task_complete(task)

                # Store task completion in conversation history
                task_summary = f"[Task: {task.description}] Result: {result[:500] if result else 'completed'}"
                self.conversation_history.append((f"Executed: {task.description}", task_summary))

                # Review progress and potentially adjust plan
                if not self._review_progress(user_request):
                    break  # Expert says we're done early

            except Exception as e:
                logger.error(f"Task failed: {e}")
                task.mark_failed(str(e))
                self._print_task_failed(task)

                # Ask expert what to do
                if not self._handle_failure(task):
                    break  # Expert says abort

        # === PHASE 3: DONE ===
        console_panel("Generating summary...", title="Phase 3: Complete", border_style="blue")
        return self._create_summary()

    def _create_plan(self, user_request: str) -> tuple[List[Task], Optional[str]]:
        """Use expert model to create task list.

        Returns:
            Tuple of (tasks, message). If message is set, it's a chat/clarify response.
        """
        # Build context from conversation history
        history_context = ""
        if self.conversation_history:
            history_lines = []
            for user_msg, assistant_msg in self.conversation_history[-5:]:  # Last 5 exchanges
                history_lines.append(f"User: {user_msg}")
                history_lines.append(f"Assistant: {assistant_msg}")
            history_context = f"\n\nRecent conversation:\n" + "\n".join(history_lines)

        # Include accumulated research/task context
        research_context = ""
        if self.context:
            research_context = f"\n\nAccumulated context from previous tasks:\n{self.context[-2000:]}"  # Last 2000 chars

        # Include working artifact if one exists
        artifact_context = ""
        if self.working_artifact:
            artifact_context = f"\n\nCURRENT WORKING ARTIFACT ({self.working_artifact['type']}):\n```\n{self.working_artifact['content']}\n```\nUser may be asking to modify this or approve it."

        messages = [
            SystemMessage(content=PLANNING_PROMPT),
            HumanMessage(content=f"User request: {user_request}{history_context}{research_context}{artifact_context}")
        ]

        response = self.expert_model.invoke(messages)
        content = self._extract_text_content(response.content)

        return self._parse_plan(content)

    def _extract_text_content(self, content) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle list of content blocks (Claude format)
            texts = []
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    texts.append(block['text'])
                elif hasattr(block, 'text'):
                    texts.append(block.text)
            return '\n'.join(texts)
        return str(content)

    def _parse_plan(self, content: str) -> tuple[List[Task], Optional[str]]:
        """Parse JSON plan from expert response.

        Returns:
            Tuple of (tasks, message). If message is set, it's a chat/clarify response.
        """
        try:
            # Extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if "```" in content:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)

            # Try to parse as JSON
            parsed = json.loads(content)

            # Check response type
            if isinstance(parsed, dict):
                if "chat" in parsed:
                    return [], parsed["chat"]
                if "clarify" in parsed:
                    return [], parsed["clarify"]
                if "artifact" in parsed:
                    # Store the artifact for iteration
                    artifact = parsed["artifact"]
                    self.working_artifact = artifact
                    # Format artifact for display
                    artifact_display = f"**{artifact.get('description', 'Draft')}**\n\n```\n{artifact['content']}\n```\n\nReady to proceed, or would you like changes?"
                    return [], artifact_display
                if "execute_artifact" in parsed and self.working_artifact:
                    # Create task to execute with the artifact
                    artifact = self.working_artifact
                    self.working_artifact = None  # Clear after using

                    if artifact["type"] == "commit_message":
                        # Store full message in context for the implementor
                        self.context += f"\n\n## Commit Message to Use:\n{artifact['content']}"
                        return [Task(
                            id=0,
                            type=TaskType.IMPLEMENT,
                            description=f"Run: git add -A && git commit with the commit message from context"
                        )], None
                    elif artifact["type"] == "code":
                        # Store code in context for implementor to write to file
                        description = artifact.get("description", "code changes")
                        self.context += f"\n\n## Code to Apply:\n{description}\n```\n{artifact['content']}\n```"
                        return [Task(
                            id=0,
                            type=TaskType.IMPLEMENT,
                            description=f"Apply the code changes from context: {description}"
                        )], None
                    elif artifact["type"] == "plan":
                        # Convert plan to tasks
                        self.context += f"\n\n## Plan to Execute:\n{artifact['content']}"
                        return [Task(
                            id=0,
                            type=TaskType.RESEARCH,
                            description=f"Begin executing plan: {artifact.get('description', 'the plan')}"
                        )], None
                    else:
                        # Generic execution - store in context
                        self.context += f"\n\n## Artifact to Use ({artifact['type']}):\n{artifact['content']}"
                        return [Task(
                            id=0,
                            type=TaskType.IMPLEMENT,
                            description=f"Execute {artifact['type']}: {artifact.get('description', 'as specified')}"
                        )], None
                # Single task dict? Wrap in list
                if "type" in parsed and "description" in parsed:
                    parsed = [parsed]

            # Should be a list of tasks
            if not isinstance(parsed, list):
                return [], f"I'm not sure how to help with that. Could you rephrase your request?"

            tasks = []
            for i, item in enumerate(parsed):
                task_type = TaskType[item["type"].upper()]
                tasks.append(Task(
                    id=i,
                    type=task_type,
                    description=item["description"],
                ))

            return tasks, None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse plan: {e}")
            logger.error(f"Content was: {content}")
            # Return the raw content as a chat response if it looks conversational
            if any(word in content.lower() for word in ["sorry", "could you", "what", "help", "?"]):
                return [], content
            console_panel(f"Failed to parse plan: {e}\n\nResponse was:\n{content[:300]}...", title="Planning Error", border_style="red")
            return [], None

    def _get_next_task(self) -> Optional[Task]:
        """Get next pending task with satisfied dependencies."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if task.is_ready(self.completed_task_ids):
                    return task
        return None

    def _execute_task(self, task: Task) -> str:
        """Delegate task to appropriate worker."""
        if task.type == TaskType.RESEARCH:
            result = self.researcher.execute(task, self.context)
            # Accumulate research findings
            self.context += f"\n\n## Research: {task.description}\n{result}"
            return result

        elif task.type == TaskType.IMPLEMENT:
            return self.implementor.execute(task, self.context)

        elif task.type == TaskType.VERIFY:
            return self.verifier.execute(task, self.context)

        else:
            raise ValueError(f"Unknown task type: {task.type}")

    def _handle_failure(self, task: Task) -> bool:
        """
        Ask expert what to do about failed task.

        Returns:
            True to continue, False to abort
        """
        messages = [
            SystemMessage(content="You are an expert deciding how to handle a failed task."),
            HumanMessage(content=f"""
Task failed: {task.description}
Error: {task.error}

Options:
1. SKIP - Mark as skipped and continue with remaining tasks
2. ABORT - Stop execution entirely

Respond with just SKIP or ABORT.""")
        ]

        response = self.expert_model.invoke(messages)
        content = self._extract_text_content(response.content).upper()

        if "ABORT" in content:
            return False

        # Default to skip and continue
        task.mark_skipped(f"Skipped due to error: {task.error}")
        self.completed_task_ids.add(task.id)  # Allow dependents to proceed
        return True

    def _review_progress(self, user_request: str) -> bool:
        """
        Review progress after task completion and potentially adjust plan.

        Returns:
            True to continue, False if we're done early
        """
        completed = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]

        # No pending tasks = nothing to review
        if not pending:
            return True

        # Build context for review
        completed_summary = "\n".join([
            f"- [{t.type.value}] {t.description}: {t.result[:200] if t.result else 'done'}..."
            for t in completed[-3:]  # Last 3 completed
        ])
        pending_summary = "\n".join([
            f"- [{t.type.value}] {t.description}"
            for t in pending
        ])

        messages = [
            SystemMessage(content="""You are reviewing progress on a task. Based on what's been completed, decide what to do next.

Options:
1. CONTINUE - proceed with remaining tasks as planned
2. DONE - the goal is already achieved, skip remaining tasks
3. ADD: {"type": "RESEARCH|IMPLEMENT|VERIFY", "description": "..."} - add a new task
4. REMOVE: <task_description> - remove a pending task that's no longer needed

Respond with just one option. Be concise."""),
            HumanMessage(content=f"""
Original request: {user_request}

Completed tasks:
{completed_summary}

Pending tasks:
{pending_summary}

Accumulated context (what we've learned):
{self.context[:1000] if self.context else 'None yet'}

What should we do?""")
        ]

        response = self.expert_model.invoke(messages)
        content = self._extract_text_content(response.content).strip()

        # Parse response
        upper_content = content.upper()

        if upper_content.startswith("DONE"):
            console_panel("Expert determined goal is achieved - skipping remaining tasks",
                         title="📋 Plan Adjusted", border_style="yellow")
            # Mark remaining tasks as skipped
            for t in pending:
                t.mark_skipped("Goal achieved early")
                self.completed_task_ids.add(t.id)
            return False

        if upper_content.startswith("ADD:"):
            try:
                # Parse the new task JSON
                json_str = content[4:].strip()
                if json_str.startswith("{"):
                    new_task_data = json.loads(json_str)
                    new_id = max(t.id for t in self.tasks) + 1
                    new_task = Task(
                        id=new_id,
                        type=TaskType[new_task_data["type"].upper()],
                        description=new_task_data["description"],
                    )
                    self.tasks.append(new_task)
                    console_panel(f"Added: [{new_task.type.value}] {new_task.description}",
                                 title="📋 Task Added", border_style="yellow")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse new task: {e}")

        if upper_content.startswith("REMOVE:"):
            desc_to_remove = content[7:].strip().lower()
            for t in pending:
                if desc_to_remove in t.description.lower():
                    t.mark_skipped("Removed by expert review")
                    self.completed_task_ids.add(t.id)
                    console_panel(f"Removed: [{t.type.value}] {t.description}",
                                 title="📋 Task Removed", border_style="yellow")
                    break

        # Default: CONTINUE
        return True

    def _create_summary(self) -> str:
        """Create final summary of what was done."""
        completed = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.tasks if t.status == TaskStatus.FAILED]
        skipped = [t for t in self.tasks if t.status == TaskStatus.SKIPPED]

        lines = [
            f"## Summary",
            f"Completed: {len(completed)}/{len(self.tasks)} tasks",
            ""
        ]

        if completed:
            lines.append("### Completed")
            for t in completed:
                lines.append(f"- [{t.type.value}] {t.description}")

        if skipped:
            lines.append("\n### Skipped")
            for t in skipped:
                lines.append(f"- [{t.type.value}] {t.description}")

        if failed:
            lines.append("\n### Failed")
            for t in failed:
                lines.append(f"- [{t.type.value}] {t.description}: {t.error}")

        summary = "\n".join(lines)
        cpm(summary, title="Results", border_style="green bold")
        return summary

    def _print_plan(self) -> None:
        """Print the task plan."""
        if not self.tasks:
            console_panel("No tasks created", border_style="red")
            return

        lines = []
        for task in self.tasks:
            lines.append(f"{task.id + 1}. [{task.type.value.upper()}] {task.description}")

        cpm("\n".join(lines), title=f"Plan ({len(self.tasks)} tasks)", border_style="yellow")

    def _print_task_start(self, task: Task) -> None:
        """Print task starting."""
        console_panel(
            f"[{task.type.value.upper()}] {task.description}",
            title=f"Task {task.id + 1}/{len(self.tasks)}",
            border_style="cyan"
        )

    def _print_task_complete(self, task: Task) -> None:
        """Print task completed."""
        result_preview = task.result[:200] + "..." if len(task.result) > 200 else task.result
        console_panel(result_preview, title=f"Task {task.id + 1} Complete", border_style="green")

    def _print_task_failed(self, task: Task) -> None:
        """Print task failed."""
        console_panel(task.error or "Unknown error", title=f"Task {task.id + 1} Failed", border_style="red")


def run_supervisor(user_request: str, expert_model, worker_model) -> str:
    """
    Entry point for supervisor mode.

    Runs in a loop, maintaining context across requests.

    Args:
        user_request: Initial request (can be empty to prompt user)
        expert_model: Model for planning/decisions
        worker_model: Model for worker execution

    Returns:
        Summary of session
    """
    from ra_aid.tools.human import ask_human

    supervisor = Supervisor(expert_model, worker_model)

    # If no initial request, prompt for one
    if not user_request or not user_request.strip():
        user_request = ask_human.invoke({"question": "What would you like help with?"})

    # Main session loop - keep going until user exits
    while True:
        # Run the current request
        result = supervisor.run(user_request)

        # Ask for next request
        console_panel("Ready for next task. Type 'exit' or 'quit' to end session.", border_style="blue")

        # Generate contextually relevant follow-up prompt
        from ra_aid.tools.expert import ask_expert
        completed_tasks = [t for t in supervisor.tasks if t.status.value == "completed"]

        if supervisor.working_artifact:
            # There's an artifact being worked on - ask about it
            artifact_type = supervisor.working_artifact.get("type", "item")
            follow_up_prompt = f"Proceed with this {artifact_type.replace('_', ' ')}? (or request changes)"
        elif completed_tasks:
            # Use expert to suggest contextual follow-up based on completed work
            task_summaries = ", ".join([t.description for t in completed_tasks[-3:]])
            follow_up_prompt = ask_expert.invoke({
                "question": f"Based on completed tasks ({task_summaries}), suggest a brief, natural follow-up question. Keep it under 10 words."
            })
        elif supervisor.conversation_history:
            # No tasks but we had conversation - use that context
            recent_exchange = supervisor.conversation_history[-1]
            user_msg = recent_exchange[0]
            assistant_response = recent_exchange[1][:500] if len(recent_exchange[1]) > 500 else recent_exchange[1]
            follow_up_prompt = ask_expert.invoke({
                "question": f"User asked: '{user_msg}'. You responded: '{assistant_response}'. Based on this exchange, suggest a brief, natural follow-up question (under 10 words)."
            })
        else:
            # Fresh start - use simple prompt
            follow_up_prompt = "What would you like help with?"

        next_request = ask_human.invoke({"question": follow_up_prompt})

        # Check for exit
        if next_request.lower().strip() in ['exit', 'quit', 'q', 'done', 'bye']:
            console_panel("Session ended. Context preserved.", title="Goodbye", border_style="green")
            break

        # Store the follow-up exchange in conversation history
        # This gives context for short responses like "yes", "no", "do it"
        # Format: (what was asked, user's response)
        exchange_context = f"Assistant asked: '{follow_up_prompt}' - User answered: '{next_request}'"
        supervisor.conversation_history.append((exchange_context, "acknowledged"))

        # Continue with next request (context is preserved in supervisor)
        user_request = next_request

    return "Session completed."
