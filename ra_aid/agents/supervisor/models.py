"""Task models for the supervisor architecture."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TaskType(Enum):
    """Types of tasks the supervisor can delegate."""
    RESEARCH = "research"      # Gather information (read-only)
    IMPLEMENT = "implement"    # Make changes (read-write)
    VERIFY = "verify"          # Check work (run tests, validate)


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """A single unit of work to be executed by a worker."""
    id: int
    type: TaskType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    dependencies: List[int] = field(default_factory=list)

    def is_ready(self, completed_ids: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep_id in completed_ids for dep_id in self.dependencies)

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error

    def mark_skipped(self, reason: str) -> None:
        self.status = TaskStatus.SKIPPED
        self.result = reason
