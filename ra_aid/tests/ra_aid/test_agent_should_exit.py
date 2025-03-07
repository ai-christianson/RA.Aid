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
        """Test that mark_should_exit propagates to parent contexts when specified."""
        parent = AgentContext()
        child = AgentContext(parent_context=parent)

        # Mark child as should exit with propagation to all parents
        child.mark_should_exit(propagation_depth=None)

        # Verify both child and parent are marked
        assert child.agent_should_exit is True
        assert parent.agent_should_exit is True
        
        # Reset for the next test
        parent.agent_should_exit = False
        child.agent_should_exit = False
        
        # Test default behavior (propagation_depth=0)
        child.mark_should_exit()
        
        # Verify only child is marked
        assert child.agent_should_exit is True
        assert parent.agent_should_exit is False

    def test_nested_context_manager_propagation(self):
        """Test propagation with nested context managers."""
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both should be False
                assert outer.agent_should_exit is False
                assert inner.agent_should_exit is False

                # Mark inner as should exit with explicit propagation to all parents
                inner.mark_should_exit(propagation_depth=None)

                # Both should now be True
                assert inner.agent_should_exit is True
                assert outer.agent_should_exit is True
                
        # Test default behavior (propagation_depth=0)
        with agent_context() as outer:
            with agent_context() as inner:
                # Initially both should be False
                assert outer.agent_should_exit is False
                assert inner.agent_should_exit is False

                # Mark inner as should exit with default propagation
                inner.mark_should_exit()

                # Only inner should be True
                assert inner.agent_should_exit is True
                assert outer.agent_should_exit is False
                
    def test_mark_should_exit_propagation_depth(self):
        """Test that mark_should_exit respects propagation depth."""
        # Create a hierarchy of contexts: ctx1 (root) -> ctx2 -> ctx3
        ctx1 = AgentContext()
        ctx2 = AgentContext(parent_context=ctx1)
        ctx3 = AgentContext(parent_context=ctx2)

        # Test case 1: propagation_depth=0 (only marks current context)
        ctx3.mark_should_exit(propagation_depth=0)
        assert ctx3.agent_should_exit is True
        assert ctx2.agent_should_exit is False
        assert ctx1.agent_should_exit is False

        # Reset all contexts
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        ctx3.agent_should_exit = False

        # Test case 2: propagation_depth=1 (marks current context and immediate parent)
        ctx3.mark_should_exit(propagation_depth=1)
        assert ctx3.agent_should_exit is True
        assert ctx2.agent_should_exit is True
        assert ctx1.agent_should_exit is False

        # Reset all contexts
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        ctx3.agent_should_exit = False

        # Test case 3: propagation_depth=2 (marks current context, parent, and grandparent)
        ctx3.mark_should_exit(propagation_depth=2)
        assert ctx3.agent_should_exit is True
        assert ctx2.agent_should_exit is True
        assert ctx1.agent_should_exit is True

        # Reset all contexts
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        ctx3.agent_should_exit = False

        # Test case 4: propagation_depth=None (marks all contexts)
        ctx3.mark_should_exit(propagation_depth=None)
        assert ctx3.agent_should_exit is True
        assert ctx2.agent_should_exit is True
        assert ctx1.agent_should_exit is True
        
        # Reset all contexts
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        ctx3.agent_should_exit = False
        
        # Test case 5: default behavior (propagation_depth=0)
        ctx3.mark_should_exit()  # Default is now 0
        assert ctx3.agent_should_exit is True
        assert ctx2.agent_should_exit is False
        assert ctx1.agent_should_exit is False
        
    def test_helper_mark_should_exit_propagation_depth(self):
        """Test that helper mark_should_exit function respects propagation depth."""
        # Create a hierarchy of contexts
        ctx1 = AgentContext()
        ctx2 = AgentContext(parent_context=ctx1)
        
        # Test with agent_context to set the current context
        with agent_context(ctx2) as current_ctx:
            # Test case 1: propagation_depth=0 (only marks current context)
            mark_should_exit(propagation_depth=0)
            assert current_ctx.agent_should_exit is True
            assert ctx2.agent_should_exit is False  # The context manager creates a new context
            assert ctx1.agent_should_exit is False
            
        # Reset for the next test
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        
        with agent_context(ctx2) as current_ctx:
            # Test case 2: propagation_depth=1 (marks current context and immediate parent)
            mark_should_exit(propagation_depth=1)
            assert current_ctx.agent_should_exit is True
            # The current_ctx's parent is ctx2, so it should be marked
            assert ctx2.agent_should_exit is True
            assert ctx1.agent_should_exit is False
            
        # Reset for the next test
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        
        with agent_context(ctx2) as current_ctx:
            # Test case 3: propagation_depth=2 (marks current context, parent, and grandparent)
            mark_should_exit(propagation_depth=2)
            assert current_ctx.agent_should_exit is True
            assert ctx2.agent_should_exit is True
            assert ctx1.agent_should_exit is True
            
        # Reset for the next test
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        
        with agent_context(ctx2) as current_ctx:
            # Test case 4: propagation_depth=None (marks all contexts)
            mark_should_exit(propagation_depth=None)
            assert current_ctx.agent_should_exit is True
            assert ctx2.agent_should_exit is True
            assert ctx1.agent_should_exit is True
            
        # Reset for the next test
        ctx1.agent_should_exit = False
        ctx2.agent_should_exit = False
        
        with agent_context(ctx2) as current_ctx:
            # Test case 5: default behavior (propagation_depth=0)
            mark_should_exit()  # Default is now 0
            assert current_ctx.agent_should_exit is True
            assert ctx2.agent_should_exit is False
            assert ctx1.agent_should_exit is False
