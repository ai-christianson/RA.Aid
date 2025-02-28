"""Unit tests for crash propagation behavior in agent_context."""


from ra_aid.agent_context import (
    AgentContext,
    agent_context,
    get_crash_message,
    is_crashed,
    mark_agent_crashed,
)


class TestCrashPropagation:
    """Test cases for crash state propagation behavior."""

    def test_mark_agent_crashed_no_propagation(self):
        """Test that mark_agent_crashed does not propagate to parent contexts."""
        parent = AgentContext()
        child = AgentContext(parent_context=parent)

        # Initially both contexts should have is_crashed as False
        assert parent.is_crashed() is False
        assert child.is_crashed() is False

        # Mark the child context as crashed
        child.mark_agent_crashed("Child crashed")

        # Child should be crashed but parent should not
        assert child.is_crashed() is True
        assert child.agent_crashed_message == "Child crashed"
        assert parent.is_crashed() is False
        assert parent.agent_crashed_message is None

    def test_nested_crash_no_propagation(self):
        """Test that crash state doesn't propagate through multiple levels of parent contexts."""
        grandparent = AgentContext()
        parent = AgentContext(parent_context=grandparent)
        child = AgentContext(parent_context=parent)

        # Initially all contexts should have is_crashed as False
        assert grandparent.is_crashed() is False
        assert parent.is_crashed() is False
        assert child.is_crashed() is False

        # Mark the child context as crashed
        child.mark_agent_crashed("Child crashed")

        # Only child should be crashed
        assert child.is_crashed() is True
        assert parent.is_crashed() is False
        assert grandparent.is_crashed() is False

    def test_context_manager_crash_no_propagation(self):
        """Test that crash states don't propagate when using context managers."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both contexts should have is_crashed as False
                assert outer.is_crashed() is False
                assert inner.is_crashed() is False

                # Mark the inner context as crashed
                inner.mark_agent_crashed("Inner crashed")

                # Inner should be crashed but outer should not
                assert inner.is_crashed() is True
                assert outer.is_crashed() is False

    def test_utility_functions_for_crash_state(self):
        """Test utility functions for crash state."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both contexts should have is_crashed as False
                assert is_crashed() is False
                assert get_crash_message() is None

                # Mark the current context (inner) as crashed
                mark_agent_crashed("Utility function crash")

                # Current context should be crashed but outer should not
                assert is_crashed() is True
                assert get_crash_message() == "Utility function crash"
                assert inner.is_crashed() is True
                assert outer.is_crashed() is False
