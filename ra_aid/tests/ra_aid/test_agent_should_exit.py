"""Unit tests for agent_should_exit functionality."""


from ra_aid.agent_context import (
    AgentContext,
    agent_context,
    mark_should_exit,
    should_exit,
)


class TestAgentShouldExit:
    """Test cases for the agent_should_exit flag and related functions."""

    def test_mark_should_exit_basic(self):
        """Test basic mark_should_exit functionality."""
        context = AgentContext()
        assert context.agent_should_exit is False

        context.mark_should_exit()
        assert context.agent_should_exit is True

    def test_should_exit_utility(self):
        """Test the should_exit utility function."""
        with agent_context() as ctx:
            assert should_exit() is False
            mark_should_exit()
            assert should_exit() is True
            assert ctx.agent_should_exit is True

    def test_propagation_to_parent_context(self):
        """Test that mark_should_exit propagates to parent contexts."""
        parent = AgentContext()
        child = AgentContext(parent_context=parent)

        # Mark child as should exit
        child.mark_should_exit()

        # Verify both child and parent are marked
        assert child.agent_should_exit is True
        assert parent.agent_should_exit is True

    def test_nested_context_manager_propagation(self):
        """Test propagation with nested context managers."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both should be False
                assert outer.agent_should_exit is False
                assert inner.agent_should_exit is False

                # Mark inner as should exit
                inner.mark_should_exit()

                # Both should now be True
                assert inner.agent_should_exit is True
                assert outer.agent_should_exit is True
