import fnmatch
import logging
from typing import List, Tuple, Dict, Optional, Any

from fuzzywuzzy import process
from git import Repo, exc
from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.console.formatting import console_panel, cpm
from ra_aid.file_listing import get_all_project_files, FileListerError

console = Console()


def record_trajectory(
    tool_name: str,
    tool_parameters: Dict,
    step_data: Dict,
    record_type: str = "tool_execution",
    is_error: bool = False,
    error_message: Optional[str] = None,
    error_type: Optional[str] = None
) -> None:
    """
    Helper function to record trajectory information, handling the case when repositories are not available.
    
    Args:
        tool_name: Name of the tool
        tool_parameters: Parameters passed to the tool
        step_data: UI rendering data
        record_type: Type of trajectory record
        is_error: Flag indicating if this record represents an error
        error_message: The error message
        error_type: The type/class of the error
    """
    try:
        from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
        from ra_aid.database.repositories.human_input_repository import get_human_input_repository
        
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            step_data=step_data,
            record_type=record_type,
            human_input_id=human_input_id,
            is_error=is_error,
            error_message=error_message,
            error_type=error_type
        )
    except (ImportError, RuntimeError):
        # If either the repository modules can't be imported or no repository is available,
        # just log and continue without recording trajectory
        logging.debug("Skipping trajectory recording: repositories not available")

DEFAULT_EXCLUDE_PATTERNS = [
    "*.pyc",
    "__pycache__/*",
    ".git/*",
    "*.so",
    "*.o",
    "*.class",
]


@tool
def fuzzy_find_project_files(
    search_term: str,
    *,
    repo_path: str = ".",
    threshold: int = 60,
    max_results: int = 10,
    include_paths: List[str] = None,
    exclude_patterns: List[str] = None,
    include_hidden: bool = False,
) -> List[Tuple[str, int]]:
    """Fuzzy find files in a project matching the search term.

    This tool searches for files within a project directory using fuzzy string matching,
    allowing for approximate matches to the search term. It returns a list of matched
    files along with their match scores. Works with both git and non-git repositories.

    Args:
        search_term: String to match against file paths
        repo_path: Path to project directory (defaults to current directory)
        threshold: Minimum similarity score (0-100) for matches (default: 60)
        max_results: Maximum number of results to return (default: 10)
        include_paths: Optional list of path patterns to include in search
        exclude_patterns: Optional list of path patterns to exclude from search
        include_hidden: Whether to include hidden files in search (default: False)

    Returns:
        List of tuples containing (file_path, match_score)

    Raises:
        ValueError: If threshold is not between 0 and 100
        FileListerError: If there's an error accessing or listing files
    """
    # Validate threshold
    if not 0 <= threshold <= 100:
        error_msg = "Threshold must be between 0 and 100"
        
        # Record error in trajectory
        record_trajectory(
            tool_name="fuzzy_find_project_files",
            tool_parameters={
                "search_term": search_term,
                "repo_path": repo_path,
                "threshold": threshold,
                "max_results": max_results,
                "include_paths": include_paths,
                "exclude_patterns": exclude_patterns,
                "include_hidden": include_hidden
            },
            step_data={
                "search_term": search_term,
                "display_title": "Invalid Threshold Value",
                "error_message": error_msg
            },
            record_type="tool_execution",
            is_error=True,
            error_message=error_msg,
            error_type="ValueError"
        )
        
        raise ValueError(error_msg)

    # Handle empty search term as special case
    if not search_term:
        return []

    # Combine default and user-provided exclude patterns
    all_exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + (exclude_patterns or [])
    
    try:
        # Get all project files using the common utility function
        all_files = get_all_project_files(
            repo_path, 
            include_hidden=include_hidden, 
            exclude_patterns=all_exclude_patterns
        )
        
        # Apply include patterns if specified
        if include_paths:
            filtered_files = []
            for pattern in include_paths:
                filtered_files.extend(f for f in all_files if fnmatch.fnmatch(f, pattern))
            all_files = filtered_files

        # Perform fuzzy matching
        matches = process.extract(search_term, all_files, limit=max_results)

        # Filter by threshold
        filtered_matches = [(path, score) for path, score in matches if score >= threshold]

        # Build info panel content
        info_sections = []

        # Search parameters section
        params_section = [
            "## Search Parameters",
            f"**Search Term**: `{search_term}`",
            f"**Directory**: `{repo_path}`",
            f"**Threshold**: {threshold}",
            f"**Max Results**: {max_results}",
            f"**Include Hidden Files**: {include_hidden}",
        ]
        if include_paths:
            params_section.append("\n**Include Patterns**:")
            for pattern in include_paths:
                params_section.append(f"- `{pattern}`")
        if exclude_patterns:
            params_section.append("\n**Exclude Patterns**:")
            for pattern in exclude_patterns:
                params_section.append(f"- `{pattern}`")
        info_sections.append("\n".join(params_section))

        # Results statistics section
        stats_section = [
            "## Results Statistics",
            f"**Total Files Scanned**: {len(all_files)}",
            f"**Matches Found**: {len(filtered_matches)}",
        ]
        info_sections.append("\n".join(stats_section))

        # Top results section
        if filtered_matches:
            results_section = ["## Top Matches"]
            for path, score in filtered_matches[:5]:  # Show top 5 matches
                results_section.append(f"- `{path}` (score: {score})")
            info_sections.append("\n".join(results_section))
        else:
            info_sections.append("## Results\n*No matches found*")

        # Record fuzzy find in trajectory
        record_trajectory(
            tool_name="fuzzy_find_project_files",
            tool_parameters={
                "search_term": search_term,
                "repo_path": repo_path,
                "threshold": threshold,
                "max_results": max_results,
                "include_paths": include_paths,
                "exclude_patterns": exclude_patterns,
                "include_hidden": include_hidden
            },
            step_data={
                "search_term": search_term,
                "display_title": "Fuzzy Find Results",
                "total_files": len(all_files),
                "matches_found": len(filtered_matches)
            },
            record_type="tool_execution"
        )
        
        # Display the panel
        cpm(
            "\n\n".join(info_sections),
            title="🔍 Fuzzy Find Results",
            border_style="bright_blue"
        )

        return filtered_matches
        
    except FileListerError as e:
        error_msg = f"Error listing files: {e}"
        
        # Record error in trajectory
        record_trajectory(
            tool_name="fuzzy_find_project_files",
            tool_parameters={
                "search_term": search_term,
                "repo_path": repo_path,
                "threshold": threshold,
                "max_results": max_results,
                "include_paths": include_paths,
                "exclude_patterns": exclude_patterns,
                "include_hidden": include_hidden
            },
            step_data={
                "search_term": search_term,
                "display_title": "Fuzzy Find Error",
                "error_message": error_msg
            },
            record_type="tool_execution",
            is_error=True,
            error_message=error_msg,
            error_type=type(e).__name__
        )
        
        console.print(f"[bold red]{error_msg}[/bold red]")
        return []
