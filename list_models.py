#!/usr/bin/env python3
"""Script to list available OpenAI models using the OpenAI API."""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

from openai import OpenAI
from rich.console import Console
from rich.table import Table

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="List available OpenAI models",
        epilog="""
Examples:
  %(prog)s                        # List all models sorted by date
  %(prog)s --type gpt-4          # List only GPT-4 models
  %(prog)s --type tts --sort name # List TTS models sorted by name
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--type",
        choices=["gpt-4", "gpt-3.5", "dall-e", "whisper", "embedding", "tts", "moderation", "all"],
        metavar="TYPE",
        default="all",
        help="Filter models by type (choices: gpt-4, gpt-3.5, dall-e, whisper, embedding, tts, moderation, all)"
    )
    parser.add_argument(
        "--sort",
        choices=["date", "name"],
        metavar="ORDER",
        default="date",
        help="Sort models by date (newest first) or name (alphabetically)"
    )
    return parser.parse_args()

def format_timestamp(timestamp: int) -> str:
    """Convert Unix timestamp to human-readable date."""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    except (TypeError, ValueError):
        return 'N/A'

console = Console()

def get_openai_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        sys.exit(1)
    return api_key

def get_models() -> List[Dict]:
    """Get list of available models from OpenAI API."""
    client = OpenAI(api_key=get_openai_key())
    try:
        models = client.models.list()
        return [model.model_dump() for model in models.data]
    except Exception as e:
        console.print(f"[red]Error fetching models: {str(e)}[/red]")
        sys.exit(1)

def display_models(models: List[Dict], model_type: str = "all", sort_by: str = "date") -> None:
    """Display models in a formatted table."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Created", style="dim")
    table.add_column("Owned By", style="dim")
    table.add_column("Type", style="dim")

    # Filter models by type if specified
    filtered_models = models
    if model_type != "all":
        filtered_models = [
            model for model in models
            if model_type.lower() in model.get('id', '').lower()
        ]

    # Sort models
    if sort_by == "date":
        sorted_models = sorted(filtered_models, key=lambda x: x.get('created', 0), reverse=True)
    else:  # sort by name
        sorted_models = sorted(filtered_models, key=lambda x: x.get('id', '').lower())

    for model in sorted_models:
        # Determine model type based on ID
        model_id = model.get('id', '').lower()
        model_type = 'Unknown'
        if 'gpt-4' in model_id:
            model_type = 'GPT-4'
        elif 'gpt-3.5' in model_id:
            model_type = 'GPT-3.5'
        elif 'dall-e' in model_id:
            model_type = 'DALL-E'
        elif 'whisper' in model_id:
            model_type = 'Whisper'
        elif 'embedding' in model_id:
            model_type = 'Embedding'
        elif 'tts' in model_id:
            model_type = 'Text-to-Speech'
        elif 'moderation' in model_id:
            model_type = 'Moderation'

        # Color code different model types
        type_colors = {
            'GPT-4': 'bright_blue',
            'GPT-3.5': 'blue',
            'DALL-E': 'magenta',
            'Whisper': 'yellow',
            'Embedding': 'cyan',
            'Text-to-Speech': 'green',
            'Moderation': 'red',
            'Unknown': 'dim white'
        }
        
        table.add_row(
            model.get('id', 'N/A'),
            format_timestamp(model.get('created', 0)),
            model.get('owned_by', 'N/A'),
            f"[{type_colors.get(model_type, 'white')}]{model_type}[/]"
        )

    filtered_count = len(sorted_models)
    total_count = len(models)
    count_text = f"{filtered_count} of {total_count}" if model_type != "all" else str(total_count)
    console.print(f"\n[bold green]Available OpenAI Models ({count_text} total):[/bold green]")
    console.print(table)

def main():
    """Main function."""
    try:
        args = parse_args()
        models = get_models()
        display_models(models, args.type, args.sort)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)

if __name__ == "__main__":
    main()