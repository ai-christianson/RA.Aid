import os
import random
from typing import List

from ..models_params import reasoning_tiers, ReasoningTier

from langchain_core.tools import tool
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..llm import initialize_expert_llm
from .memory import _global_memory, get_memory_value
from ra_aid import models_params
from ..exceptions import ToolExecutionError

console = Console()
_model = None

def get_best_expert_model_by_capabilities(
    provider: str | None = None,
    capabilities: List[str] | None = None,
) -> tuple[str, str]:
    """Find the first model with the specified reasoning tier, capabilities, and specialties.

    Args:
        provider: Optional provider name to filter by (default: None)
        capabilities: List of capabilities to search for (default: None)

    Returns:
        tuple[str, str]: Tuple of (model_name, provider_name), or ("", "") if none found
    """
    if capabilities is None:
        capabilities = []

    for model_name, config in reasoning_tiers.items():
        if all(capability in config["capabilities"] for capability in capabilities):
            model_provider = config.get("provider", "")
            if provider is None or model_provider == provider:
                return model_name, model_provider
    return "", ""

def get_best_expert_model_by_provider(
    provider: str | None = None,
    exclude_models: List[str] | None = None
) -> tuple[str, str]:
    """Find the model with the highest reasoning tier for the specified provider.

    Args:
        provider: Optional provider name to filter by (default: None)
        exclude_models: Optional list of model names to exclude from selection (default: None)

    Returns:
        tuple[str, str]: Tuple of (model_name, provider_name), or ("", "") if none found
    """
    best_tier = -1
    best_model = ""
    best_provider = ""
    
    if exclude_models is None:
        exclude_models = []
    
    for model_name, config in reasoning_tiers.items():
        if model_name in exclude_models:
            continue
            
        model_provider = config.get("provider", "")
        current_tier = config.get("tier", ReasoningTier.BASIC)
        
        if (provider is None or model_provider == provider) and current_tier > best_tier:
            best_tier = current_tier
            best_model = model_name
            best_provider = model_provider
            
    return best_model, best_provider


def get_random_model_from_provider(provider: str) -> str:
    """Get a random model from the given provider."""
    # get random model by provider from models_params
    models = models_params.get(provider, {}).get("models", [])    
    if not models:
        return ""
    return random.choice(models)
    
def check_access_to_model(model_name: str, provider: str) -> bool:
    """Check if the model is accessible."""
    try:
        # Initialize the model
        model = initialize_expert_llm(provider, model_name)
        
        # Check if the model is accessible
        if model.is_accessible():
            return True
        else:
            console.print(Panel(f"Model {model_name} is not accessible", title="Warning", border_style="yellow"))
            return False
    except Exception as e:
        console.print(Panel(f"Error checking model accessibility: {str(e)}", title="Error", border_style="red"))
        return False

# TODO: This is intent of the changes.
# The goal is to dynamically select the best model based on the capabilities needed from the query or the specific task being worked on.
# TODO: replace get_best_expert_model_by_provider with get_best_expert_model_by_capabilities but cant at the moment since we dont have yet the code to inspect the query to determine the capabilities needed.
# Using get_best_expert_model_by_provider for now; considering it hard-coded temporarily while coding.
# The bunch of print statements should be removed once the code is working.
# Current state and concerns:
# 1. The code is successfully iterating through the models for provider.
# 2. But what do we do when we cannot find an expert model? I found at times my API token does not have access to some models, what should happen in this case?
def get_model():
    global _model
    models_to_exclude = set()  # Using a set to ensure uniqueness
    max_retries = 3
    
    try:
        if _model is None:
            expert_provider = _global_memory["config"]["expert_provider"] or "openai"
            auto_select_expert_model = _global_memory["config"]["expert_auto_select_model"] or False
            console.print(Panel(f"INFO: Expert provider: {expert_provider}", title="Info", border_style="white"))
            console.print(Panel(f"INFO: Auto select expert model: {auto_select_expert_model}", title="Info", border_style="white"))

            while True:
                try:
                    # Get the next best model, excluding previously failed ones
                    if auto_select_expert_model:
                        expert_model, provider = get_best_expert_model_by_provider(expert_provider, list(models_to_exclude))
                    else:
                        expert_model = _global_memory["config"]["expert_model"]
                        if not expert_model or expert_model in models_to_exclude:
                            expert_model, provider = get_best_expert_model_by_provider(expert_provider, list(models_to_exclude))

                    # If no model found, return None
                    if not expert_model:
                        console.print(Panel(f"Unable to find expert model for expert provider: {expert_provider}", title="No Expert Model Found", border_style="red"))
                        return None

                    expert_provider = provider or expert_provider
                    console.print(Panel(f"Selected expert model: {expert_model} from {expert_provider}", title="Auto Expert Model Selection", border_style="yellow"))

                    # check access to model
                    if not check_access_to_model(expert_model, expert_provider):
                        console.print(Panel(f"No access to expert model: {expert_model} from {expert_provider}. Fetching new model", title="Access Denied", border_style="red"))
                        models_to_exclude.add(expert_model)
                        continue

                    # Try to initialize the model
                    _model = initialize_expert_llm(expert_provider, expert_model)
                    
                except Exception as e:
                    # Add failed model to exclude set
                    models_to_exclude.add(expert_model)
                    console.print(Panel(f"Failed to initialize model {expert_model}: {e}\nExcluded models: {models_to_exclude}", title="Warning", border_style="yellow", ))

    except Exception as e:
        _model = None
        raise ToolExecutionError(f"Expert model initialization failed: {str(e)}")
        
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

    Err on the side of adding more context rather than less.

    You must give the complete contents.

    Expert context will be reset after the ask_expert tool is called.

    Args:
        context: The context to add
    """
    expert_context["text"].append(context)

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

    The can see any key facts and code snippets previously noted, along with any additional context you've provided.
      But the expert cannot see or reason about anything you have not explicitly provided in this way.

    Try to phrase your question in a way that it does not expand the scope of our top-level task.

    The expert can be prone to overthinking depending on what and how you ask it.
    """
    try:
        model = get_model()
        if model is None:
            raise ToolExecutionError("Failed to initialize expert model")
        global expert_context

        # Get all content first
        file_paths = list(_global_memory["related_files"].values())
        related_contents = read_related_files(file_paths)
        key_snippets = get_memory_value("key_snippets")
        key_facts = get_memory_value("key_facts")
        research_notes = get_memory_value("research_notes")

        # Build display query (just question)
        display_query = "# Question\n" + question

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

        if related_contents:
            query_parts.extend(["# Research Notes", research_notes])

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
            ["\n # Addidional Requirements", "**DO NOT OVERTHINK**", "**DO NOT OVERCOMPLICATE**"]
        )

        # Join all parts
        full_query = "\n".join(query_parts)

        # Get response using full query
        response = model.invoke(full_query)

        # Format and display response
        console.print(
            Panel(Markdown(response.content), title="Expert Response", border_style="blue")
        )

        return response.content
    except Exception as e:
        raise ToolExecutionError(f"Expert consultation failed: {str(e)}")
