import argparse
import sys
import uuid
from rich.panel import Panel
from rich.console import Console
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from ra_aid.env import validate_environment
from ra_aid.tools.memory import _global_memory, get_related_files, get_memory_value
from ra_aid import print_stage_header, print_task_header, print_error, run_agent_with_retry
from ra_aid.agent_utils import run_research_agent
from ra_aid.prompts import (
    PLANNING_PROMPT,
    IMPLEMENTATION_PROMPT,
    CHAT_PROMPT,
    EXPERT_PROMPT_SECTION_PLANNING,
    EXPERT_PROMPT_SECTION_IMPLEMENTATION,
    HUMAN_PROMPT_SECTION_PLANNING,
    HUMAN_PROMPT_SECTION_IMPLEMENTATION
)
from ra_aid.llm import initialize_llm

from ra_aid.tool_configs import (
    get_planning_tools,
    get_implementation_tools,
    get_chat_tools
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='RA.Aid - AI Agent for executing programming and research tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    ra-aid -m "Add error handling to the database module"
    ra-aid -m "Explain the authentication flow" --research-only
        '''
    )
    parser.add_argument(
        '-m', '--message',
        type=str,
        help='The task or query to be executed by the agent'
    )
    parser.add_argument(
        '--research-only',
        action='store_true',
        help='Only perform research without implementation'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='anthropic',
        choices=['anthropic', 'openai', 'openrouter', 'openai-compatible'],
        help='The LLM provider to use'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='The model name to use (required for non-Anthropic providers)'
    )
    parser.add_argument(
        '--cowboy-mode',
        action='store_true',
        help='Skip interactive approval for shell commands'
    )
    parser.add_argument(
        '--expert-provider',
        type=str,
        default='openai',
        choices=['anthropic', 'openai', 'openrouter', 'openai-compatible'],
        help='The LLM provider to use for expert knowledge queries (default: openai)'
    )
    parser.add_argument(
        '--expert-model',
        type=str,
        help='The model name to use for expert knowledge queries (required for non-OpenAI providers)'
    )
    parser.add_argument(
        '--hil', '-H',
        action='store_true',
        help='Enable human-in-the-loop mode, where the agent can prompt the user for additional information.'
    )
    parser.add_argument(
        '--chat',
        action='store_true',
        help='Enable chat mode with direct human interaction (implies --hil)'
    )
    
    args = parser.parse_args()
    
    # Set hil=True when chat mode is enabled
    if args.chat:
        args.hil = True
    
    # Set default model for Anthropic, require model for other providers
    if args.provider == 'anthropic':
        if not args.model:
            args.model = 'claude-3-5-sonnet-20241022'
    elif not args.model:
        parser.error(f"--model is required when using provider '{args.provider}'")
    
    # Validate expert model requirement
    if args.expert_provider != 'openai' and not args.expert_model:
        parser.error(f"--expert-model is required when using expert provider '{args.expert_provider}'")
    
    return args

# Create console instance
console = Console()

# Create individual memory objects for each agent
research_memory = MemorySaver()
planning_memory = MemorySaver()
implementation_memory = MemorySaver()


def is_informational_query() -> bool:
    """Determine if the current query is informational based on implementation_requested state."""
    return _global_memory.get('config', {}).get('research_only', False) or not is_stage_requested('implementation')

def is_stage_requested(stage: str) -> bool:
    """Check if a stage has been requested to proceed."""
    if stage == 'implementation':
        return _global_memory.get('implementation_requested', False)
    return False

def run_implementation_stage(base_task, tasks, plan, related_files, model, expert_enabled: bool):
    """Run implementation stage with a distinct agent for each task."""
    if not is_stage_requested('implementation'):
        print_stage_header("Implementation Stage Skipped")
        return
        
    print_stage_header("Implementation Stage")
    
    # Get tasks directly from memory, maintaining order by ID
    task_list = [task for _, task in sorted(_global_memory['tasks'].items())]
    
    print_task_header(f"Found {len(task_list)} tasks to implement")
    
    for i, task in enumerate(task_list, 1):
        print_task_header(task)
        
        # Create a unique memory instance for this task
        task_memory = MemorySaver()
        
        # Create a fresh agent for each task
        task_agent = create_react_agent(model, get_implementation_tools(expert_enabled=expert_enabled), checkpointer=task_memory)
        
        # Construct task-specific prompt
        expert_section = EXPERT_PROMPT_SECTION_IMPLEMENTATION if expert_enabled else ""
        human_section = HUMAN_PROMPT_SECTION_IMPLEMENTATION if _global_memory.get('config', {}).get('hil', False) else ""
        task_prompt = (IMPLEMENTATION_PROMPT).format(
            plan=plan,
            key_facts=get_memory_value('key_facts'),
            key_snippets=get_memory_value('key_snippets'),
            task=task,
            related_files="\n".join(related_files),
            base_task=base_task,
            expert_section=expert_section,
            human_section=human_section
        )
        
        # Run agent for this task
        run_agent_with_retry(task_agent, task_prompt, {"configurable": {"thread_id": "abc123"}, "recursion_limit": 100})



def main():
    """Main entry point for the ra-aid command line tool."""
    try:
        args = parse_arguments()
        expert_enabled, expert_missing = validate_environment(args)  # Will exit if main env vars missing
        
        if expert_missing:
            console.print(Panel(
                f"[yellow]Expert tools disabled due to missing configuration:[/yellow]\n" + 
                "\n".join(f"- {m}" for m in expert_missing) +
                "\nSet the required environment variables or args to enable expert mode.",
                title="Expert Tools Disabled",
                style="yellow"
            ))
        
        # Create the base model after validation
        model = initialize_llm(args.provider, args.model)

        # Handle chat mode
        if args.chat:
            print_stage_header("Chat Mode")
            
            # Create chat agent with appropriate tools
            chat_agent = create_react_agent(
                model,
                get_chat_tools(expert_enabled=expert_enabled),
                checkpointer=MemorySaver()
            )
            
            # Run chat agent with CHAT_PROMPT
            config = {
                "configurable": {"thread_id": uuid.uuid4()},
                "recursion_limit": 100,
                "chat_mode": True,
                "cowboy_mode": args.cowboy_mode,
                "hil": True  # Always true in chat mode
            }
            
            # Store config in global memory
            _global_memory['config'] = config
            _global_memory['config']['expert_provider'] = args.expert_provider
            _global_memory['config']['expert_model'] = args.expert_model
            
            # Run chat agent and exit
            run_agent_with_retry(chat_agent, CHAT_PROMPT, config)
            return

        # Validate message is provided
        if not args.message:
            print_error("--message is required")
            sys.exit(1)
            
        base_task = args.message
        config = {
            "configurable": {"thread_id": uuid.uuid4()},
            "recursion_limit": 100,
            "research_only": args.research_only,
            "cowboy_mode": args.cowboy_mode
        }
    
        # Store config in global memory for access by is_informational_query
        _global_memory['config'] = config
    
        # Store expert provider and model in config
        _global_memory['config']['expert_provider'] = args.expert_provider
        _global_memory['config']['expert_model'] = args.expert_model
        
        # Run research stage
        print_stage_header("Research Stage")
        
        run_research_agent(
            base_task,
            model,
            expert_enabled=expert_enabled,
            research_only=args.research_only,
            hil=args.hil,
            memory=research_memory,
            config=config
        )
        
        # Proceed with planning and implementation if not an informational query
        if not is_informational_query():
            print_stage_header("Planning Stage")
            
            # Create planning agent
            planning_agent = create_react_agent(model, get_planning_tools(expert_enabled=expert_enabled), checkpointer=planning_memory)
            
            expert_section = EXPERT_PROMPT_SECTION_PLANNING if expert_enabled else ""
            human_section = HUMAN_PROMPT_SECTION_PLANNING if args.hil else ""
            planning_prompt = PLANNING_PROMPT.format(
                expert_section=expert_section,
                human_section=human_section,
                base_task=base_task,
                research_notes=get_memory_value('research_notes'),
                related_files="\n".join(get_related_files()),
                key_facts=get_memory_value('key_facts'),
                key_snippets=get_memory_value('key_snippets'),
                research_only_note='' if args.research_only else ' Only request implementation if the user explicitly asked for changes to be made.'
            )

            # Run planning agent
            run_agent_with_retry(planning_agent, planning_prompt, config)

            # Run implementation stage with task-specific agents
            run_implementation_stage(
                base_task,
                get_memory_value('tasks'),
                get_memory_value('plan'),
                get_related_files(),
                model,
                expert_enabled=expert_enabled
            )

    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
