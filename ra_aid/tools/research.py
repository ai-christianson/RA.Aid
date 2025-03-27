from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel

from ra_aid.console.formatting import console_panel, cpm
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.human_input_repository import get_human_input_repository
from ra_aid.agent_context import mark_should_exit, mark_task_completed

console = Console()


@tool("existing_project_detected")
def existing_project_detected() -> dict:
    """
    When to call: Once you have confirmed that the current working directory contains project files.
    """
    try:
        # Record detection in trajectory
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="existing_project_detected",
            tool_parameters={},
            step_data={
                "detection_type": "existing_project",
                "display_title": "Existing Project Detected",
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        # Continue even if trajectory recording fails
        console.print(f"Warning: Could not record trajectory: {str(e)}")

    console_panel("📁 Existing Project Detected", border_style="bright_blue", padding=0)
    return {
        "hint": (
            "You are working within an existing codebase that already has established patterns and standards. "
            "Integrate any new functionality by adhering to the project's conventions:\n\n"
            "- Carefully discover existing folder structure, naming conventions, and architecture.\n"
            "- Look very carefully forestablished authentication, authorization, and data handling patterns.\n"
            "- Find detailed and nuanced information about how tests are written and run.\n"
            "- Align with the project's existing CI/CD pipelines, deployment strategies, and versioning schemes.\n"
            "- Find existing logging, error handling, and documentation patterns.\n\n"
            "In short, your goal is to seamlessly fit into the current ecosystem rather than reshape it."
        )
    }


@tool("monorepo_detected")
def monorepo_detected() -> dict:
    """
    When to call: After identifying that multiple packages or modules exist within a single repository.
    """
    try:
        # Record detection in trajectory
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="monorepo_detected",
            tool_parameters={},
            step_data={
                "detection_type": "monorepo",
                "display_title": "Monorepo Detected",
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        # Continue even if trajectory recording fails
        console.print(f"Warning: Could not record trajectory: {str(e)}")

    console_panel("📦 Monorepo Detected", border_style="bright_blue", padding=0)
    return {
        "hint": (
            "You are researching in a monorepo environment that manages multiple packages or services under one roof. "
            "Ensure new work fits cohesively within the broader structure:\n\n"
            "- Search all packages for shared libraries, utilities, and patterns, and reuse them to avoid redundancy.\n"
            "- Find and note coding standards, test strategies, and build processes across all modules.\n"
            "- Find and note existing tooling, scripts, and workflows that have been set up for the monorepo.\n"
            "- Align any new features or changes with overarching CI/CD pipelines and deployment models, ensuring interoperability across all components.\n"
            "- Find and note  existing versioning and release management practices already in place.\n\n"
            "- Pay extra attention to integration nuances such as authentication, authorization, examples of how APIs are called, etc.\n\n"
            "- Find and note specific examples of all of the above.\n\n"
            "- Because you are in a monorepo, you will need to carefully organize your research into focused areas.\n\n"
            "Your goal is to enhance the entire codebase without disrupting its well-established, unified structure."
        )
    }


@tool("ui_detected")
def ui_detected() -> dict:
    """
    When to call: After detecting that the project contains a user interface layer or front-end component.
    """
    try:
        # Record detection in trajectory
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="ui_detected",
            tool_parameters={},
            step_data={
                "detection_type": "ui",
                "display_title": "UI Detected",
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        # Continue even if trajectory recording fails
        console.print(f"Warning: Could not record trajectory: {str(e)}")

    console_panel("🎯 UI Detected", border_style="bright_blue", padding=0)
    return {
        "hint": (
            "You are working with a user interface component where established UI conventions, styles, and frameworks are likely in place. "
            "Any modifications or additions should blend seamlessly with the existing design and user experience:\n\n"
            "- Locate and note existing UI design conventions, including layout, typography, color schemes, and interaction patterns.\n"
            "- Search for and carefully note any integrated UI libraries, components, or styling frameworks already in the project.\n"
            "- UI changes can be challenging to test automatically. If you find tests, note them, otherwise note that this is a UI feature and testing will be requested but not implemented as part of the task.\n"
            "- Find and note established workflows for building, bundling, and deploying the UI layer, ensuring that any new changes do not conflict with the existing pipeline.\n\n"
            "Your goal is to enhance the user interface without disrupting the cohesive look, feel, and functionality already established."
        )
    }


@tool
def mark_research_complete_no_implementation_required(message: str):
    """Mark the current research task as complete with no implementation required.

    Use this when research is complete and it has been determined that no implementation 
    is needed or possible. The agent will exit after calling this tool.

    Args:
        message: Message explaining why no implementation is required.
    """
    # Try to get the latest human input
    human_input_id = None
    try:
        human_input_repo = get_human_input_repository()
        human_input_id = human_input_repo.get_most_recent_id()
    except Exception as e:
        console.print(f"Warning: Could not get human input ID: {str(e)}", style="yellow")

    # Record to trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        trajectory_repo.create(
            tool_name="mark_research_complete_no_implementation_required",
            tool_parameters={"message": message},
            step_data={
                "completion_message": message,
                "display_title": "Research Complete (No Implementation Required)",
            },
            record_type="task_completion",
            human_input_id=human_input_id
        )
    except Exception as e:
        console.print(f"Warning: Could not record trajectory: {str(e)}", style="yellow")
    
    # Mark task as completed
    mark_task_completed(message)
    
    # Print confirmation message to console
    cpm(f"🎯 Research complete (no implementation required): {message}", "Research Complete", "green")
    
    # Signal agent to exit
    mark_should_exit()
    
    return f"Research task completed with no implementation required: {message}"