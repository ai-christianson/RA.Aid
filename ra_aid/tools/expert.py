from typing import List
import os
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from ..llm import initialize_expert_llm
from .memory import get_memory_value, get_related_files, _global_memory

console = Console()
_model = None

def get_model():
    global _model
    try:
        if _model is None:
            provider = _global_memory['config']['expert_provider'] or 'openai'
            model = _global_memory['config']['expert_model'] or 'o1-preview'
            _model = initialize_expert_llm(provider, model)
    except Exception as e:
        _model = None
        console.print(Panel(f"Failed to initialize expert model: {e}", title="Error", border_style="red"))
        raise
    return _model

# Keep track of context globally
expert_context = {
    'text': [],    # Additional textual context
    'files': []    # File paths to include
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
    expert_context['text'].append(context)
    
    # Create and display status panel
    panel_content = f"Added expert context ({len(context)} characters)"
    console.print(Panel(panel_content, title="Expert Context", border_style="blue"))
    
    return f"Context added."

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

def read_related_files(file_paths: List[str]) -> str:
    """Read the provided files and return their contents.
    
    Args:
        file_paths: List of file paths to read
        
    Returns:
        String containing concatenated file contents, or empty string if no paths
    """
    if not file_paths:
        return ''
    
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
    global expert_context
    
    # Get all content first
    file_paths = expert_context['files'] + list(get_related_files())
    related_contents = read_related_files(file_paths)
    key_snippets = get_memory_value('key_snippets')
    key_facts = get_memory_value('key_facts')
    
    # Build display query (just question)
    display_query = "# Question\n" + question
    
    # Show only question in panel
    console.print(Panel(
        Markdown(display_query),
        title="🤔 Expert Query",
        border_style="yellow"
    ))
    
    # Clear context after panel display
    expert_context['text'].clear()
    expert_context['files'].clear()
    
    # Build full query in specified order
    query_parts = []
    
    if related_contents:
        query_parts.extend(['# Related Files', related_contents])
        
    if key_snippets and len(key_snippets) > 0:
        query_parts.extend(['# Key Snippets', key_snippets])
        
    if key_facts and len(key_facts) > 0:
        query_parts.extend(['# Key Facts About This Project', key_facts])
        
    if expert_context['text']:
        query_parts.extend(['\n# Additional Context', '\n'.join(expert_context['text'])])
        
    query_parts.extend(['# Question', question])
    query_parts.extend(['\n # Addidional Requirements', "Do not expand the scope unnecessarily."])
    
    # Join all parts
    full_query = '\n'.join(query_parts)
    
    # Get response using full query
    response = get_model().invoke(full_query)
    
    # Format and display response
    console.print(Panel(
        Markdown(response.content),
        title="Expert Response",
        border_style="blue"
    ))
    
    return response.content
