"""Unit tests for the agent_context module."""

import threading
import time


from ra_aid.agent_context import (
    AgentContext,
    agent_context,
    get_completion_message,
    get_current_context,
    is_completed,
    mark_plan_completed,
    mark_task_completed,
    reset_completion_flags,
)


class TestAgentContext:
    """Test cases for the AgentContext class and related functions."""

    def test_context_creation(self):
        """Test creating a new context."""
        context = AgentContext()
        assert context.task_completed is False
        assert context.plan_completed is False
        assert context.completion_message == ""

    def test_context_inheritance(self):
        """Test that child contexts inherit state from parent contexts."""
        parent = AgentContext()
        parent.mark_task_completed("Parent task completed")

        child = AgentContext(parent_context=parent)
        assert child.task_completed is True
        assert child.completion_message == "Parent task completed"

    def test_mark_task_completed(self):
        """Test marking a task as completed."""
        context = AgentContext()
        context.mark_task_completed("Task done")

        assert context.task_completed is True
        assert context.plan_completed is False
        assert context.completion_message == "Task done"

    def test_mark_plan_completed(self):
        """Test marking a plan as completed."""
        context = AgentContext()
        context.mark_plan_completed("Plan done")

        assert context.task_completed is True
        assert context.plan_completed is True
        assert context.completion_message == "Plan done"

    def test_reset_completion_flags(self):
        """Test resetting completion flags."""
        context = AgentContext()
        context.mark_task_completed("Task done")

        context.reset_completion_flags()
        assert context.task_completed is False
        assert context.plan_completed is False
        assert context.completion_message == ""

    def test_is_completed_property(self):
        """Test the is_completed property."""
        context = AgentContext()
        assert context.is_completed is False

        context.mark_task_completed("Task done")
        assert context.is_completed is True

        context.reset_completion_flags()
        assert context.is_completed is False

        context.mark_plan_completed("Plan done")
        assert context.is_completed is True


class TestContextManager:
    """Test cases for the agent_context context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        assert get_current_context() is None

        with agent_context() as ctx:
            assert get_current_context() is ctx
            assert ctx.task_completed is False

        assert get_current_context() is None

    def test_nested_context_managers(self):
        """Test nested context managers."""
        with agent_context() as outer_ctx:
            assert get_current_context() is outer_ctx

            with agent_context() as inner_ctx:
                assert get_current_context() is inner_ctx
                assert inner_ctx is not outer_ctx

            assert get_current_context() is outer_ctx

    def test_context_manager_with_parent(self):
        """Test context manager with explicit parent context."""
        parent = AgentContext()
        parent.mark_task_completed("Parent task")

        with agent_context(parent_context=parent) as ctx:
            assert ctx.task_completed is True
            assert ctx.completion_message == "Parent task"

    def test_context_manager_inheritance(self):
        """Test that nested contexts inherit from outer contexts by default."""
        with agent_context() as outer:
            outer.mark_task_completed("Outer task")

            with agent_context() as inner:
                assert inner.task_completed is True
                assert inner.completion_message == "Outer task"

                inner.mark_plan_completed("Inner plan")

            # Outer context should not be affected by inner context changes
            assert outer.task_completed is True
            assert outer.plan_completed is False
            assert outer.completion_message == "Outer task"


class TestThreadIsolation:
    """Test thread isolation of context variables."""

    def test_thread_isolation(self):
        """Test that contexts are isolated between threads."""
        results = {}

        def thread_func(thread_id):
            with agent_context() as ctx:
                ctx.mark_task_completed(f"Thread {thread_id}")
                time.sleep(0.1)  # Give other threads time to run
                # Store the context's message for verification
                results[thread_id] = get_completion_message()

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own message
        assert results[0] == "Thread 0"
        assert results[1] == "Thread 1"
        assert results[2] == "Thread 2"


class TestUtilityFunctions:
    """Test utility functions that operate on the current context."""

    def test_mark_task_completed_utility(self):
        """Test the mark_task_completed utility function."""
        with agent_context():
            mark_task_completed("Task done via utility")
            assert is_completed() is True
            assert get_completion_message() == "Task done via utility"

    def test_mark_plan_completed_utility(self):
        """Test the mark_plan_completed utility function."""
        with agent_context():
            mark_plan_completed("Plan done via utility")
            assert is_completed() is True
            assert get_completion_message() == "Plan done via utility"

    def test_reset_completion_flags_utility(self):
        """Test the reset_completion_flags utility function."""
        with agent_context():
            mark_task_completed("Task done")
            reset_completion_flags()
            assert is_completed() is False
            assert get_completion_message() == ""

    def test_utility_functions_without_context(self):
        """Test utility functions when no context is active."""
        # These should not raise exceptions even without an active context
        mark_task_completed("No context")
        mark_plan_completed("No context")
        reset_completion_flags()

        # These should have safe default returns
        assert is_completed() is False
        assert get_completion_message() == ""


"""Unit tests for the agent_context module."""





class TestAgentContext:
    """Test cases for the AgentContext class and related functions."""

    def test_context_creation(self):
        """Test creating a new context."""
        context = AgentContext()
        assert context.task_completed is False
        assert context.plan_completed is False
        assert context.completion_message == ""

    def test_context_inheritance(self):
        """Test that child contexts do not inherit completion flags from parent contexts."""
        parent = AgentContext()
        parent.mark_task_completed("Parent task completed")
        child = AgentContext(parent_context=parent)
        assert child.task_completed is False
        assert child.completion_message == ""

    def test_mark_task_completed(self):
        """Test marking a task as completed."""
        context = AgentContext()
        context.mark_task_completed("Task done")
        assert context.task_completed is True
        assert context.plan_completed is False
        assert context.completion_message == "Task done"

    def test_mark_plan_completed(self):
        """Test marking a plan as completed."""
        context = AgentContext()
        context.mark_plan_completed("Plan done")
        assert context.task_completed is True
        assert context.plan_completed is True
        assert context.completion_message == "Plan done"

    def test_reset_completion_flags(self):
        """Test resetting completion flags."""
        context = AgentContext()
        context.mark_task_completed("Task done")
        context.reset_completion_flags()
        assert context.task_completed is False
        assert context.plan_completed is False
        assert context.completion_message == ""

    def test_is_completed_property(self):
        """Test the is_completed property."""
        context = AgentContext()
        assert context.is_completed is False
        context.mark_task_completed("Task done")
        assert context.is_completed is True
        context.reset_completion_flags()
        assert context.is_completed is False
        context.mark_plan_completed("Plan done")
        assert context.is_completed is True


class TestContextManager:
    """Test cases for the agent_context context manager."""

    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        assert get_current_context() is None
        with agent_context() as ctx:
            assert get_current_context() is ctx
            assert ctx.task_completed is False
        assert get_current_context() is None

    def test_nested_context_managers(self):
        """Test nested context managers."""
        with agent_context() as outer_ctx:
            assert get_current_context() is outer_ctx
            with agent_context() as inner_ctx:
                assert get_current_context() is inner_ctx
                assert inner_ctx is not outer_ctx
            assert get_current_context() is outer_ctx

    def test_context_manager_with_parent(self):
        """Test context manager with explicit parent context."""
        parent = AgentContext()
        parent.mark_task_completed("Parent task")
        with agent_context(parent_context=parent) as ctx:
            assert ctx.task_completed is False
            assert ctx.completion_message == ""

    def test_context_manager_inheritance(self):
        """Test that nested contexts do not inherit completion flags from outer contexts."""
        with agent_context() as outer:
            outer.mark_task_completed("Outer task")
            with agent_context() as inner:
                assert inner.task_completed is False
                assert inner.completion_message == ""
                inner.mark_plan_completed("Inner plan")
            # Outer context should not be affected by inner context changes
            assert outer.task_completed is True
            assert outer.plan_completed is False
            assert outer.completion_message == "Outer task"


class TestThreadIsolation:
    """Test thread isolation of context variables."""

    def test_thread_isolation(self):
        """Test that contexts are isolated between threads."""
        results = {}

        def thread_func(thread_id):
            with agent_context() as ctx:
                ctx.mark_task_completed(f"Thread {thread_id}")
                time.sleep(0.1)  # Give other threads time to run
                # Store the context's message for verification
                results[thread_id] = get_completion_message()

        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own message
        assert results[0] == "Thread 0"
        assert results[1] == "Thread 1"
        assert results[2] == "Thread 2"


class TestUtilityFunctions:
    """Test utility functions that operate on the current context."""

    def test_mark_task_completed_utility(self):
        """Test the mark_task_completed utility function."""
        with agent_context():
            mark_task_completed("Task done via utility")
            assert is_completed() is True
            assert get_completion_message() == "Task done via utility"

    def test_mark_plan_completed_utility(self):
        """Test the mark_plan_completed utility function."""
        with agent_context():
            mark_plan_completed("Plan done via utility")
            assert is_completed() is True
            assert get_completion_message() == "Plan done via utility"

    def test_reset_completion_flags_utility(self):
        """Test the reset_completion_flags utility function."""
        with agent_context():
            mark_task_completed("Task done")
            reset_completion_flags()
            assert is_completed() is False
            assert get_completion_message() == ""

    def test_utility_functions_without_context(self):
        """Test utility functions when no context is active."""
        # These should not raise exceptions even without an active context
        mark_task_completed("No context")
        mark_plan_completed("No context")
        reset_completion_flags()
        # These should have safe default returns
        assert is_completed() is False
        assert get_completion_message() == ""
