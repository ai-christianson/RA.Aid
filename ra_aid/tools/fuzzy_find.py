from typing import List, Tuple
import fnmatch
from git import Repo
from fuzzywuzzy import process
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

DEFAULT_EXCLUDE_PATTERNS = [
    '*.pyc',
    '__pycache__/*',
    '.git/*',
    '*.so',
    '*.o',
    '*.class'
]

@tool
def fuzzy_find_project_files(
    search_term: str,
    *,
    repo_path: str = ".",
    threshold: int = 60,
    max_results: int = 10,
    include_paths: List[str] = None,
    exclude_patterns: List[str] = None
) -> List[Tuple[str, int]]:
    """Fuzzy find files in a git repository matching the search term.
    
    This tool searches for files within a git repository using fuzzy string matching,
    allowing for approximate matches to the search term. It returns a list of matched
    files along with their match scores.
    
    Args:
        search_term: String to match against file paths
        repo_path: Path to git repository (defaults to current directory)
        threshold: Minimum similarity score (0-100) for matches (default: 60)
        max_results: Maximum number of results to return (default: 10)
        include_paths: Optional list of path patterns to include in search
        exclude_patterns: Optional list of path patterns to exclude from search
        
    Returns:
        List of tuples containing (file_path, match_score)
        
    Raises:
        InvalidGitRepositoryError: If repo_path is not a git repository
        ValueError: If threshold is not between 0 and 100
    """
    # Validate threshold
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")
        
    # Handle empty search term as special case
    if not search_term:
        return []

    # Initialize repo for normal search
    repo = Repo(repo_path)
    
    # Get all tracked files
    tracked_files = repo.git.ls_files().splitlines()
    
    # Get all untracked files
    untracked_files = repo.untracked_files
    
    # Combine file lists
    all_files = tracked_files + untracked_files
    
    # Apply include patterns if specified
    if include_paths:
        filtered_files = []
        for pattern in include_paths:
            filtered_files.extend(
                f for f in all_files 
                if fnmatch.fnmatch(f, pattern)
            )
        all_files = filtered_files
    
    # Apply exclude patterns
    patterns = DEFAULT_EXCLUDE_PATTERNS + (exclude_patterns or [])
    for pattern in patterns:
        all_files = [
            f for f in all_files 
            if not fnmatch.fnmatch(f, pattern)
        ]
    
    # Perform fuzzy matching
    matches = process.extract(
        search_term,
        all_files,
        limit=max_results
    )
    
    # Filter by threshold
    filtered_matches = [
        (path, score) 
        for path, score in matches 
        if score >= threshold
    ]

    # Build info panel content
    info_sections = []
    
    # Search parameters section
    params_section = [
        "## Search Parameters",
        f"**Search Term**: `{search_term}`",
        f"**Repository**: `{repo_path}`",
        f"**Threshold**: {threshold}",
        f"**Max Results**: {max_results}"
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
        f"**Matches Found**: {len(filtered_matches)}"
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

    # Display the panel
    console.print(Panel(
        Markdown("\n\n".join(info_sections)),
        title="🔍 Fuzzy Find Results",
        border_style="bright_blue"
    ))
    
    return filtered_matches
