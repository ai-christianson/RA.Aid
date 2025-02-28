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
    mark_should_exit,
    mark_task_completed,
    reset_completion_flags,
    should_exit,
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


class TestExitPropagation:
    """Test cases for the agent_should_exit flag propagation."""

    def test_mark_should_exit_propagation(self):
        """Test that mark_should_exit propagates to parent contexts."""
        parent = AgentContext()
        child = AgentContext(parent_context=parent)

        # Initially both contexts should have agent_should_exit as False
        assert parent.agent_should_exit is False
        assert child.agent_should_exit is False

        # Mark the child context as should exit
        child.mark_should_exit()

        # Both child and parent should now have agent_should_exit as True
        assert child.agent_should_exit is True
        assert parent.agent_should_exit is True

    def test_nested_should_exit_propagation(self):
        """Test that mark_should_exit propagates through multiple levels of parent contexts."""
        grandparent = AgentContext()
        parent = AgentContext(parent_context=grandparent)
        child = AgentContext(parent_context=parent)

        # Initially all contexts should have agent_should_exit as False
        assert grandparent.agent_should_exit is False
        assert parent.agent_should_exit is False
        assert child.agent_should_exit is False

        # Mark the child context as should exit
        child.mark_should_exit()

        # All contexts should now have agent_should_exit as True
        assert child.agent_should_exit is True
        assert parent.agent_should_exit is True
        assert grandparent.agent_should_exit is True

    def test_context_manager_should_exit_propagation(self):
        """Test that mark_should_exit propagates when using context managers."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both contexts should have agent_should_exit as False
                assert outer.agent_should_exit is False
                assert inner.agent_should_exit is False

                # Mark the inner context as should exit
                inner.mark_should_exit()

                # Both inner and outer should now have agent_should_exit as True
                assert inner.agent_should_exit is True
                assert outer.agent_should_exit is True


class TestCrashPropagation:
    """Test cases for the agent_has_crashed flag non-propagation."""

    def test_mark_agent_crashed_no_propagation(self):
        """Test that mark_agent_crashed does not propagate to parent contexts."""
        parent = AgentContext()
        child = AgentContext(parent_context=parent)

        # Initially both contexts should have agent_has_crashed as False
        assert parent.is_crashed() is False
        assert child.is_crashed() is False

        # Mark the child context as crashed
        child.mark_agent_crashed("Child crashed")

        # Child should be crashed, but parent should not
        assert child.is_crashed() is True
        assert parent.is_crashed() is False
        assert child.agent_crashed_message == "Child crashed"
        assert parent.agent_crashed_message is None

    def test_nested_crash_no_propagation(self):
        """Test that crash states don't propagate through multiple levels of parent contexts."""
        grandparent = AgentContext()
        parent = AgentContext(parent_context=grandparent)
        child = AgentContext(parent_context=parent)

        # Initially all contexts should have agent_has_crashed as False
        assert grandparent.is_crashed() is False
        assert parent.is_crashed() is False
        assert child.is_crashed() is False

        # Mark the child context as crashed
        child.mark_agent_crashed("Child crashed")

        # Only child should be crashed, parent and grandparent should not
        assert child.is_crashed() is True
        assert parent.is_crashed() is False
        assert grandparent.is_crashed() is False
        assert child.agent_crashed_message == "Child crashed"
        assert parent.agent_crashed_message is None
        assert grandparent.agent_crashed_message is None

    def test_context_manager_crash_no_propagation(self):
        """Test that crash state doesn't propagate when using context managers."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both contexts should have agent_has_crashed as False
                assert outer.is_crashed() is False
                assert inner.is_crashed() is False

                # Mark the inner context as crashed
                inner.mark_agent_crashed("Inner crashed")

                # Inner should be crashed, but outer should not
                assert inner.is_crashed() is True
                assert outer.is_crashed() is False
                assert inner.agent_crashed_message == "Inner crashed"
                assert outer.agent_crashed_message is None

    def test_crash_state_not_inherited(self):
        """Test that new child contexts don't inherit crash states from parent contexts."""
        parent = AgentContext()

        # Mark the parent as crashed
        parent.mark_agent_crashed("Parent crashed")
        assert parent.is_crashed() is True

        # Create a child context with the crashed parent as parent_context
        child = AgentContext(parent_context=parent)

        # Child should not be crashed even though parent is
        assert parent.is_crashed() is True
        assert child.is_crashed() is False
        assert parent.agent_crashed_message == "Parent crashed"
        assert child.agent_crashed_message is None


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

    def test_mark_should_exit_utility(self):
        """Test the mark_should_exit utility function."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both contexts should have agent_should_exit as False
                assert should_exit() is False

                # Mark the current context (inner) as should exit
                mark_should_exit()

                # Both inner and outer should now have agent_should_exit as True
                assert should_exit() is True
                assert inner.agent_should_exit is True
                assert outer.agent_should_exit is True
