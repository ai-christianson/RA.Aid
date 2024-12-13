from typing import List
import os
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from ..llm import initialize_expert_llm
from .memory import get_memory_value, get_related_files

console = Console()
_model = None

def get_model():
    global _model
    if _model is None:
        provider = get_memory_value('expert_provider') or 'openai'
        model = get_memory_value('expert_model') or 'o1-preview'
        _model = initialize_expert_llm(provider, model)
    return _model

# Keep track of context globally
expert_context = []

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
        
    Returns:
        Confirmation message
    """
    expert_context.append(context)
    
    # Create and display status panel
    panel_content = f"Added expert context ({len(context)} characters)"
    console.print(Panel(panel_content, title="Expert Context", border_style="blue"))
    
    return f"Context added."

def read_files_with_limit(file_paths: List[str], max_lines: int = 10000) -> str:
    """Read multiple files and concatenate contents, stopping at line limit.
    
    Args:
        file_paths: List of file paths to read
        max_lines: Maximum total lines to read (default: 10000)
        
    Returns:
        String containing concatenated file contents with headers
        
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
                
            with open(path, 'r', encoding='utf-8') as f:
                file_content = []
                for i, line in enumerate(f):
                    if total_lines + i >= max_lines:
                        file_content.append(f"\n... truncated after {max_lines} lines ...")
                        break
                    file_content.append(line)
                
                if file_content:
                    contents.append(f'\n## File: {path}\n')
                    contents.append(''.join(file_content))
                    total_lines += len(file_content)
            
        except Exception as e:
            console.print(f"Error reading file {path}: {str(e)}", style="red")
            continue
            
    return ''.join(contents)

def read_related_files() -> str:
    """Read related files from memory.
    
    Returns:
        String containing concatenated file contents with headers
    """
    related_files = get_related_files()
    if not related_files:
        return ''
    
    return read_files_with_limit(list(related_files), max_lines=10000)

@tool("ask_expert")
def ask_expert(question: str) -> str:
    """Ask a question to an expert AI model.

    Keep your questions specific, but long and detailed.

    You only query the expert when you have a specific question in mind.

    The expert can be extremely useful at logic questions, debugging, and reviewing complex source code, but you must provide all context including source manually.

    The query will automatically include any key facts and code snippets from memory, along with any additional context you've provided.

    Try to phrase your question in a way that it does not expand the scope of our top-level task.

    The expert can be prone to overthinking depending on what and how you ask it.
    
    Args:
        question: The question to ask the expert
        
    Returns:
        The expert's response
    """
    global expert_context
    
    # Build query with context and key facts
    query_parts = []
    
    # Add key facts if they exist
    key_facts = get_memory_value('key_facts')
    if key_facts and len(key_facts) > 0:
        query_parts.append("# Key Facts About This Project")
        query_parts.append(key_facts)
    
    # Add key snippets if they exist
    key_snippets = get_memory_value('key_snippets')
    if key_snippets and len(key_snippets) > 0:
        query_parts.append('# Key Snippets')
        query_parts.append(key_snippets)
    
    # Add other context if it exists
    if expert_context:
        query_parts.append("\n# Additional Context")
        query_parts.append("\n".join(expert_context))
    
    # Add the question
    if query_parts:  # If we have context/facts, add a newline before question
        query_parts.append("\n# Question")
    query_parts.append(question)
    
    # Join all parts
    query = "\n".join(query_parts)
    
    # Display the query in a panel before making the call
    console.print(Panel(
        Markdown(query),
        title="🤔 Expert Query",
        border_style="yellow"
    ))
    
    # Clear context after use (only after successful panel display)
    expert_context.clear()
    
    # Get related file contents and rebuild query with it at the start
    related_contents = read_related_files()
    if related_contents:
        # Create new query_parts with related files at the start
        new_query_parts = ['# Related Files', related_contents]
        new_query_parts.extend(query_parts)
        query_parts = new_query_parts
        query = "\n".join(query_parts)
        
    # Get response
    response = get_model().invoke(query)
    
    # Format and display response
    console.print(Panel(
        Markdown(response.content),
        title="Expert Response",
        border_style="blue"
    ))
    
    return response.content
