import os
import logging
from typing import List

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

logger = logging.getLogger(__name__)

from ..database.repositories.trajectory_repository import get_trajectory_repository
from ..database.repositories.human_input_repository import get_human_input_repository

from ..database.repositories.key_fact_repository import get_key_fact_repository
from ..database.repositories.key_snippet_repository import get_key_snippet_repository
from ..database.repositories.related_files_repository import get_related_files_repository
from ..database.repositories.research_note_repository import get_research_note_repository
from ..database.repositories.config_repository import get_config_repository
from ..llm import initialize_expert_llm
from ..model_formatters import format_key_facts_dict
from ..model_formatters.key_snippets_formatter import format_key_snippets_dict
from ..model_formatters.research_notes_formatter import format_research_notes_dict
from ..models_params import models_params
from ..text.processing import process_thinking_content

console = Console()
_model = None


def get_model():
    global _model
    try:
        if _model is None:
            config_repo = get_config_repository()
            provider = config_repo.get("expert_provider") or config_repo.get("provider")
            model = config_repo.get("expert_model") or config_repo.get("model")
            _model = initialize_expert_llm(provider, model)
    except Exception as e:
        _model = None
        console.print(
            Panel(
                f"Failed to initialize expert model: {e}",
                title="Error",
                border_style="red",
            )
        )
        raise
    return _model


# Keep track of context globally
expert_context = {
    "text": [],  # Additional textual context
    "files": [],  # File paths to include
}


@tool("emit_expert_context")
def emit_expert_context(context: str) -> str:
    """Add context for the next expert question.

    This should be highly detailed contents such as entire sections of source code, etc.

    Do not include your question in the additional context.

    Err on the side of adding more context rather than less, but keep it information dense and under 500 words total.

    You must give the complete contents.

    Expert context will be reset after the ask_expert tool is called.

    Args:
        context: The context to add
    """
    expert_context["text"].append(context)

    # Record expert context in trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="emit_expert_context",
            tool_parameters={"context_length": len(context)},
            step_data={
                "display_title": "Expert Context",
                "context_length": len(context),
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        logger.error(f"Failed to record trajectory: {e}")

    # Create and display status panel
    panel_content = f"Added expert context ({len(context)} characters)"
    console.print(Panel(panel_content, title="Expert Context", border_style="blue"))

    return "Context added."


def read_files_with_limit(file_paths: List[str], max_lines: int = 10000) -> str:
    """Read multiple files and concatenate contents, stopping at line limit.

    Args:
        file_paths: List of file paths to read
        max_lines: Maximum total lines to read (default: 10000)

    Note:
        - Each file's contents will be prefaced with its path as a header
        - Stops reading files when max_lines limit is reached
        - Files that would exceed the line limit are truncated
    """
    total_lines = 0
    contents = []

    for path in file_paths:
        try:
            if not os.path.exists(path):
                console.print(f"Warning: File not found: {path}", style="yellow")
                continue

            with open(path, "r", encoding="utf-8") as f:
                file_content = []
                for i, line in enumerate(f):
                    if total_lines + i >= max_lines:
                        file_content.append(
                            f"\n... truncated after {max_lines} lines ..."
                        )
                        break
                    file_content.append(line)

                if file_content:
                    contents.append(f"\n## File: {path}\n")
                    contents.append("".join(file_content))
                    total_lines += len(file_content)

        except Exception as e:
            console.print(f"Error reading file {path}: {str(e)}", style="red")
            continue

    return "".join(contents)


def read_related_files(file_paths: List[str]) -> str:
    """Read the provided files and return their contents.

    Args:
        file_paths: List of file paths to read

    Returns:
        String containing concatenated file contents, or empty string if no paths
    """
    if not file_paths:
        return ""

    return read_files_with_limit(file_paths, max_lines=10000)


@tool("ask_expert")
def ask_expert(question: str) -> str:
    """Ask a question to an expert AI model.

    Keep your questions specific, but long and detailed.

    You only query the expert when you have a specific question in mind.

    The expert can be extremely useful at logic questions, debugging, and reviewing complex source code, but you must provide all context including source manually.

    The expert can see any key facts and code snippets previously noted, along with any additional context you've provided.
      But the expert cannot see or reason about anything you have not explicitly provided in this way.

    Try to phrase your question in a way that it does not expand the scope of our top-level task.

    The expert can be prone to overthinking depending on what and how you ask it.
    """
    global expert_context

    # Get all content first
    file_paths = list(get_related_files_repository().get_all().values())
    related_contents = read_related_files(file_paths)
    # Get key snippets directly from repository and format using the formatter
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""
    # Get key facts directly from repository and format using the formatter
    try:
        facts_dict = get_key_fact_repository().get_facts_dict()
        key_facts = format_key_facts_dict(facts_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
    # Get research notes directly from repository and format using the formatter
    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""

    # Build display query (just question)
    display_query = "# Question\n" + question

    # Record expert query in trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="ask_expert",
            tool_parameters={"question": question},
            step_data={
                "display_title": "Expert Query",
                "question": question,
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        logger.error(f"Failed to record trajectory: {e}")

    # Show only question in panel
    console.print(
        Panel(Markdown(display_query), title="🤔 Expert Query", border_style="yellow")
    )

    # Clear context after panel display
    expert_context["text"].clear()
    expert_context["files"].clear()

    # Build full query in specified order
    query_parts = []

    if related_contents:
        query_parts.extend(["# Related Files", related_contents])

    if formatted_research_notes:
        query_parts.extend(["# Research Notes", formatted_research_notes])

    if key_snippets and len(key_snippets) > 0:
        query_parts.extend(["# Key Snippets", key_snippets])

    if key_facts and len(key_facts) > 0:
        query_parts.extend(["# Key Facts About This Project", key_facts])

    if expert_context["text"]:
        query_parts.extend(
            ["\n# Additional Context", "\n".join(expert_context["text"])]
        )

    query_parts.extend(["# Question", question])
    query_parts.extend(
        [
            "\n # Addidional Requirements",
            "**DO NOT OVERTHINK**",
            "**DO NOT OVERCOMPLICATE**",
        ]
    )

    # Ensure all elements in query_parts are strings before joining
    query_parts = [str(part) for part in query_parts]
    
    # Join all parts
    full_query = "\n".join(query_parts)

    # Get response using full query
    response = get_model().invoke(full_query)
    
    # Get the content from the response
    content = response.content
    logger.debug(f"Expert response content type: {type(content).__name__}")
    
    # Check if model supports think tags
    config_repo = get_config_repository()
    provider = config_repo.get("expert_provider") or config_repo.get("provider")
    model_name = config_repo.get("expert_model") or config_repo.get("model")
    model_config = models_params.get(provider, {}).get(model_name, {})
    supports_think_tag = model_config.get("supports_think_tag", False)
    supports_thinking = model_config.get("supports_thinking", False)
    
    logger.debug(f"Expert model: {provider}/{model_name}")
    logger.debug(f"Model supports think tag: {supports_think_tag}")
    logger.debug(f"Model supports thinking: {supports_thinking}")
    
    # Process thinking content using the common processing function
    try:
        # Use the process_thinking_content function to handle both string and list responses
        content, thinking = process_thinking_content(
            content=content,
            supports_think_tag=supports_think_tag,
            supports_thinking=supports_thinking,
            panel_title="💭 Thoughts",
            panel_style="yellow",
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"Exception during content processing: {str(e)}")
        raise
    
    # Record expert response in trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="ask_expert",
            tool_parameters={"question": question},
            step_data={
                "display_title": "Expert Response",
                "response_length": len(content),
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        logger.error(f"Failed to record trajectory: {e}")

    # Format and display response
    console.print(
        Panel(Markdown(content), title="Expert Response", border_style="blue")
    )

    return content