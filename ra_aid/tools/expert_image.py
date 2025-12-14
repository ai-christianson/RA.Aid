"""Expert tool with image/screenshot support.

This module provides a tool for asking questions to the expert model
with image attachments (screenshots, diagrams, etc.).
"""

import os
import base64
import logging
import mimetypes
from typing import List, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from rich.console import Console

from ra_aid.console.formatting import console_panel, cpm

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
from .expert import read_related_files, expert_context

logger = logging.getLogger(__name__)
console = Console()

# Cache for expert model
_image_model = None


def get_image_model():
    """Get or initialize the expert model for image analysis."""
    global _image_model
    try:
        if _image_model is None:
            config_repo = get_config_repository()
            provider = config_repo.get("expert_provider") or config_repo.get("provider")
            model = config_repo.get("expert_model") or config_repo.get("model")
            _image_model = initialize_expert_llm(provider, model)
    except Exception as e:
        _image_model = None
        console_panel(
            f"Failed to initialize expert model for image analysis: {e}",
            title="Error",
            border_style="red"
        )
        raise
    return _image_model


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image file and return base64 encoded data with mime type.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_data, mime_type)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file type is not supported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine mime type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Try to infer from extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        mime_type = mime_map.get(ext)
    
    if mime_type not in ['image/png', 'image/jpeg', 'image/gif', 'image/webp']:
        raise ValueError(f"Unsupported image type: {mime_type}. Supported: PNG, JPEG, GIF, WebP")
    
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')
    
    return image_data, mime_type


def build_multimodal_message(text_content: str, image_paths: List[str]) -> HumanMessage:
    """Build a multimodal HumanMessage with text and images.
    
    Args:
        text_content: The text portion of the message
        image_paths: List of paths to image files
        
    Returns:
        HumanMessage with multimodal content
    """
    content = []
    
    # Add text first
    content.append({
        "type": "text",
        "text": text_content
    })
    
    # Add images
    for path in image_paths:
        try:
            image_data, mime_type = load_image_as_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })
            logger.debug(f"Added image: {path} ({mime_type})")
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            content.append({
                "type": "text",
                "text": f"\n[Failed to load image: {path} - {e}]\n"
            })
    
    return HumanMessage(content=content)


@tool("ask_expert_with_image")
def ask_expert_with_image(question: str, image_paths: List[str]) -> str:
    """Ask a question to the expert AI model with image attachments.
    
    Use this tool when you need to analyze screenshots, diagrams, UI mockups,
    error messages, or any visual content. The expert model (Claude Opus) has
    full vision capabilities.
    
    Args:
        question: Your detailed question about the image(s)
        image_paths: List of absolute paths to image files (PNG, JPEG, GIF, WebP)
        
    Example:
        ask_expert_with_image(
            question="What error is shown in this screenshot and how do I fix it?",
            image_paths=["/path/to/screenshot.png"]
        )
    """
    global expert_context
    
    # Validate image paths
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            console_panel(
                f"Image not found: {path}",
                title="Warning",
                border_style="yellow"
            )
    
    if not valid_paths:
        return "Error: No valid image paths provided. Please check the paths and try again."
    
    # Get all context (same as ask_expert)
    file_paths = list(get_related_files_repository().get_all().values())
    related_contents = read_related_files(file_paths)
    
    try:
        key_snippets = format_key_snippets_dict(get_key_snippet_repository().get_snippets_dict())
    except RuntimeError as e:
        logger.error(f"Failed to access key snippet repository: {str(e)}")
        key_snippets = ""
        
    try:
        facts_dict = get_key_fact_repository().get_facts_dict()
        key_facts = format_key_facts_dict(facts_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access key fact repository: {str(e)}")
        key_facts = ""
        
    try:
        repository = get_research_note_repository()
        notes_dict = repository.get_notes_dict()
        formatted_research_notes = format_research_notes_dict(notes_dict)
    except RuntimeError as e:
        logger.error(f"Failed to access research note repository: {str(e)}")
        formatted_research_notes = ""
    
    # Build display query
    display_query = f"# Question (with {len(valid_paths)} image(s))\n{question}\n\nImages: {', '.join(valid_paths)}"
    
    # Record in trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="ask_expert_with_image",
            tool_parameters={"question": question, "image_count": len(valid_paths)},
            step_data={
                "display_title": "Expert Image Query",
                "question": question,
                "image_paths": valid_paths,
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        logger.error(f"Failed to record trajectory: {e}")
    
    # Show panel
    cpm(display_query, title="🖼️ Expert Image Query", border_style="yellow")
    
    # Clear context after panel display
    expert_context["text"].clear()
    expert_context["files"].clear()
    
    # Build full text query
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
        query_parts.extend(["\n# Additional Context", "\n".join(expert_context["text"])])
    
    query_parts.extend(["# Question", question])
    query_parts.extend([
        "\n# Image Analysis Instructions",
        "Please carefully analyze the attached image(s) to answer the question.",
        "Describe what you see and provide specific, actionable guidance.",
    ])
    query_parts.extend([
        "\n# Additional Requirements",
        "**DO NOT OVERTHINK**",
        "**DO NOT OVERCOMPLICATE**",
    ])
    
    query_parts = [str(part) for part in query_parts]
    full_query = "\n".join(query_parts)
    
    # Build multimodal message
    message = build_multimodal_message(full_query, valid_paths)
    
    # Get response
    response = get_image_model().invoke([message])
    content = response.content
    logger.debug(f"Expert image response content type: {type(content).__name__}")
    
    # Process thinking content
    config_repo = get_config_repository()
    provider = config_repo.get("expert_provider") or config_repo.get("provider")
    model_name = config_repo.get("expert_model") or config_repo.get("model")
    model_config = models_params.get(provider, {}).get(model_name, {})
    supports_think_tag = model_config.get("supports_think_tag", False)
    supports_thinking = model_config.get("supports_thinking", False)
    
    try:
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

    # Ensure content is a string (safety net for thinking model responses)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", str(item)))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        content = "\n".join(text_parts)
    
    # Record response in trajectory
    try:
        trajectory_repo = get_trajectory_repository()
        human_input_id = get_human_input_repository().get_most_recent_id()
        trajectory_repo.create(
            tool_name="ask_expert_with_image",
            tool_parameters={"question": question, "image_count": len(valid_paths)},
            step_data={
                "display_title": "Expert Image Response",
                "response_length": len(content),
                "response_content": content,
            },
            record_type="tool_execution",
            human_input_id=human_input_id
        )
    except Exception as e:
        logger.error(f"Failed to record trajectory: {e}")
    
    # Display response
    cpm(content, title="🖼️ Expert Image Response", border_style="blue")
    
    return content
