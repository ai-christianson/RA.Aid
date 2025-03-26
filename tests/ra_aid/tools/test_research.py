
import pytest
from unittest.mock import patch, MagicMock

from ra_aid.tools.research import mark_research_complete_no_implementation_required


class TestResearchTools:
    @patch('ra_aid.tools.research.mark_task_completed')
    @patch('ra_aid.tools.research.mark_should_exit')
    # @patch('ra_aid.tools.research.console_panel') # Removed
    @patch('ra_aid.tools.research.cpm')
    @patch('ra_aid.tools.research.get_trajectory_repository')
    @patch('ra_aid.tools.research.get_human_input_repository')
    def test_mark_research_complete_no_implementation_required(
        self, mock_get_human_input_repo, mock_get_trajectory_repo, 
        mock_cpm, mock_mark_should_exit, mock_mark_task_completed # Removed mock_console_panel
    ):
        # Arrange
        mock_human_input_repo = MagicMock()
        mock_human_input_repo.get_most_recent_id.return_value = 123
        mock_get_human_input_repo.return_value = mock_human_input_repo
        
        mock_trajectory_repo = MagicMock()
        mock_get_trajectory_repo.return_value = mock_trajectory_repo
        
        test_message = "No implementation required because the API already supports this feature."
        
        # Act
        result = mark_research_complete_no_implementation_required(test_message)
        
        # Assert
        mock_mark_task_completed.assert_called_once_with(test_message)
        mock_mark_should_exit.assert_called_once()
        mock_cpm.assert_called_once()
        # mock_console_panel.assert_called_once() # Removed
        mock_trajectory_repo.create.assert_called_once()
        assert "Research task completed" in result
        assert test_message in result
