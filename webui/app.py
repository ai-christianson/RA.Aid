"""
RA.Aid WebUI - Main Application Module

This module implements the main Streamlit web interface for RA.Aid, an AI Development Assistant.
It handles the following core functionalities:
- WebSocket communication for real-time updates
- Session state management
- API configuration and validation
- Task execution pipeline (Research -> Planning -> Implementation)
- User interface rendering and message handling
"""

import streamlit as st
import threading
from queue import Queue, Empty
import logging
import sys
import subprocess
from datetime import datetime
from webui.socket_interface import SocketInterface
from components.memory import initialize_memory, _global_memory
from components.research import research_component
from components.planning import planning_component
from components.implementation import implementation_component
from webui.config import WebUIConfig, load_environment_status
from ra_aid.logger import logger
from ra_aid.llm import initialize_llm
from ra_aid.agent_utils import (
    run_research_agent,
    run_planning_agent, 
    run_task_implementation_agent
)
import asyncio
import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import requests
from typing import Dict, Any
from litellm import completion

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure logging with a clean, informative format"""
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors and emojis for different log levels"""
        COLORS = {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[32m',   # Green
            'WARNING': '\033[33m', # Yellow
            'ERROR': '\033[31m',   # Red
            'CRITICAL': '\033[41m' # Red background
        }
        EMOJIS = {
            'DEBUG': '🔍',
            'INFO': '📝',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        RESET = '\033[0m'

        def format(self, record):
            # Add emoji and color based on log level
            color = self.COLORS.get(record.levelname, '')
            emoji = self.EMOJIS.get(record.levelname, '')
            reset = self.RESET
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            
            # Format the message
            message = super().format(record)
            return f"{color}{emoji} [{timestamp}] {record.levelname:<8} | {message}{reset}"

    # Create logger
    ui_logger = logging.getLogger('streamlit_ui')
    ui_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    ui_logger.handlers = []
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter('%(message)s'))
    ui_logger.addHandler(console_handler)
    
    return ui_logger

# Initialize logger
ui_logger = setup_logging()

# Initialize global components for WebSocket communication
socket_interface = SocketInterface()
message_queue = Queue()

# Debug environment variables
ui_logger.info("Starting RA.Aid WebUI")
ui_logger.info(f"ANTHROPIC_API_KEY configured: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
ui_logger.info(f"ANTHROPIC_API_KEY value: {os.getenv('ANTHROPIC_API_KEY')[:10]}..." if os.getenv('ANTHROPIC_API_KEY') else "None")

def handle_output(message: dict):
    """
    Callback function to handle messages received from the WebSocket.
    Messages are added to a queue for processing in the main Streamlit loop.
    
    Args:
        message (dict): The message received from the WebSocket server
    """
    ui_logger.debug(f"Received WebSocket message: {message}")
    if isinstance(message, dict):
        # Format the message for display
        if 'content' in message:
            ui_logger.info(f"Processing message: {message.get('type', 'text')}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": message['content'],
                "type": message.get('type', 'text')
            })
        # Add to queue for processing
        message_queue.put(message)

def websocket_thread():
    """
    Background thread function that manages WebSocket connection and message handling.
    """
    try:
        ui_logger.info("Starting WebSocket thread")
        socket_interface.register_handler("message", handle_output)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        ui_logger.info("Attempting WebSocket connection...")
        connected = loop.run_until_complete(socket_interface.connect())
        st.session_state.connected = connected
        
        if connected:
            ui_logger.info("✅ WebSocket connected successfully")
            loop.run_until_complete(socket_interface.setup_handlers())
        else:
            ui_logger.error("❌ Failed to connect to WebSocket server")
    except Exception as e:
        ui_logger.error(f"WebSocket thread error: {str(e)}")
        st.session_state.connected = False

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    Handles:
    - Chat message history
    - Connection status
    - Available AI models
    - WebSocket thread status
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Welcome to RA.Aid! I'm your AI Development Assistant. How can I help you today?",
            "type": "text"
        }]
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'models' not in st.session_state:
        logger.info("Loading available models...")
        st.session_state.models = load_available_models()
        logger.info(f"Available providers in session state: {list(st.session_state.models.keys())}")
    if 'websocket_thread_started' not in st.session_state:
        st.session_state.websocket_thread_started = False

def get_configured_providers():
    """
    Get configured providers from environment variables.
    
    Provider Configuration Pattern:
    {
        'provider_name': {
            'env_key': 'PROVIDER_API_KEY',
            'api_url': 'https://api.provider.com/v1/models',  # None for client library
            'headers': {  # None for client library
                'Authorization': 'Bearer {api_key}',
                'Other-Header': 'value'
            },
            'client_library': True/False,  # Whether to use a client library instead of REST
            'client_class': OpenAI/Anthropic/etc.,  # The client class to use if client_library is True
        }
    }
    """
    PROVIDER_CONFIGS = {
        'openai': {
            'env_key': 'OPENAI_API_KEY',
            'client_library': True,
            'client_class': OpenAI,
            'api_url': None,
            'headers': None
        },
        'anthropic': {
            'env_key': 'ANTHROPIC_API_KEY',
            'client_library': True,
            'client_class': anthropic.Anthropic,
            'api_url': None,
            'headers': None
        },
        'openrouter': {
            'env_key': 'OPENROUTER_API_KEY',
            'client_library': False,
            'api_url': 'https://openrouter.ai/api/v1/models',
            'headers': {
                'Authorization': 'Bearer {api_key}',
                'HTTP-Referer': 'https://github.com/OpenRouterTeam/openrouter-python',
                'X-Title': 'RA.Aid'
            }
        }
    }
    
    # Return only providers that have API keys configured
    return {
        name: config 
        for name, config in PROVIDER_CONFIGS.items() 
        if os.getenv(config['env_key'])
    }

def load_available_models():
    """
    Load and format models from different providers using OpenRouter's pattern.
    
    OpenRouter Pattern:
    {
        "data": [
            {"id": "provider/model-name", ...},  # Key pattern to follow
        ]
    }
    """
    models = {}
    configured_providers = get_configured_providers()
    
    for provider_name, config in configured_providers.items():
        try:
            if config['client_library']:
                # Use client library (e.g., OpenAI)
                client = config['client_class']()
                response = client.models.list()
                data = [{"id": f"{provider_name}/{model.id}"} for model in response.data]
                models[provider_name] = [item['id'] for item in data]
            else:
                # Use REST API
                headers = {
                    k: v.format(api_key=os.getenv(config['env_key']))
                    for k, v in config['headers'].items()
                }
                response = requests.get(config['api_url'], headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if provider_name == 'openrouter':
                        # OpenRouter already includes provider prefix
                        models[provider_name] = [item['id'] for item in data['data']]
                    else:
                        # Add provider prefix for other APIs
                        models[provider_name] = [
                            f"{provider_name}/{model['id']}" 
                            for model in data['data']
                        ]
                else:
                    raise Exception(f"API request failed with status {response.status_code}")
            
            logger.info(f"Loaded {len(models[provider_name])} {provider_name} models")
            
        except Exception as e:
            logger.error(f"Failed to load {provider_name} models: {str(e)}")
            # Fallback models
            if provider_name == 'openai':
                models[provider_name] = ["openai/gpt-4", "openai/gpt-3.5-turbo"]
            elif provider_name == 'anthropic':
                models[provider_name] = ["anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229"]
            elif provider_name == 'openrouter':
                models[provider_name] = ["openai/gpt-4-turbo", "anthropic/claude-3-opus"]
    
    # Log final model counts
    for provider, provider_models in models.items():
        logger.info(f"{provider}: {len(provider_models)} models loaded")
        logger.debug(f"{provider} models: {provider_models}")
    
    return models

def filter_models(models: list, search_query: str) -> list:
    """
    Filter models based on search query.
    Supports searching by provider name or model name.
    
    Args:
        models (list): List of model names
        search_query (str): Search query to filter models
        
    Returns:
        list: Filtered list of models
    """
    if not search_query:
        return models
    
    search_query = search_query.lower()
    return [
        model for model in models 
        if search_query in model.lower() or 
        (
            '/' in model and 
            (search_query in model.split('/')[0].lower() or search_query in model.split('/')[1].lower())
        )
    ]

def render_environment_status():
    """
    Render the environment configuration status in the sidebar.
    Displays the status of each configured API provider (OpenAI, Anthropic, etc.)
    using color-coded indicators (green for configured, red for not configured).
    """
    st.sidebar.subheader("Environment Status")
    env_status = load_environment_status()
    status_text = " | ".join([f"{provider}: {'✓' if status else '×'}" 
                             for provider, status in env_status.items()])
    st.sidebar.caption(f"API Status: {status_text}")

def process_message_queue():
    """
    Process messages from the WebSocket message queue.
    Handles different message types and updates the UI accordingly.
    Messages are processed until the queue is empty.
    """
    try:
        while True:
            message = message_queue.get_nowait()
            if isinstance(message, dict):
                content = message.get('content', '')
                msg_type = message.get('type', 'text')
                
                # Add message to session state if not already there
                if content and not any(m.get('content') == content for m in st.session_state.messages):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content,
                        "type": msg_type
                    })
    except Empty:
        pass

def render_messages():
    """
    Render all messages in the chat interface.
    Handles different message types with appropriate styling.
    """
    if not st.session_state.messages:
        return
        
    for message in st.session_state.messages:
        role = message.get('role', 'system')
        content = message.get('content', '')
        msg_type = message.get('type', 'text')
        
        if not content:
            continue
            
        if role == "user":
            st.markdown(f"**You:** {content}")
        elif role == "assistant":
            if msg_type == 'error':
                st.error(content)
            elif msg_type == 'success':
                st.success(content)
            elif msg_type == 'info':
                st.info(content)
            elif msg_type == 'warning':
                st.warning(content)
            elif msg_type == 'research':
                st.markdown(f"🔍 **Research:**\n{content}")
            elif msg_type == 'plan':
                st.markdown(f"📋 **Plan:**\n{content}")
            elif msg_type == 'implementation':
                st.markdown(f"⚙️ **Implementation:**\n{content}")
            else:
                st.markdown(f"**Assistant:** {content}")
        else:
            st.text(content)

def send_task(task: str, config: dict):
    """
    Send a task to the WebSocket server for processing.
    
    Args:
        task (str): The task description or query
        config (dict): Configuration parameters for task execution
    
    Displays:
        - Success message if task is sent successfully
        - Error message if sending fails or not connected
    """
    if st.session_state.connected:
        try:
            asyncio.run(socket_interface.send_task(task, config))
            st.success("Task sent successfully")
        except Exception as e:
            st.error(f"Failed to send task: {str(e)}")
    else:
        st.error("Not connected to server")

async def research_component(task: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the research stage of RA.Aid."""
    try:
        ui_logger.info("Starting research phase")
        
        # Validate required config fields
        required_fields = ["provider", "model"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        ui_logger.info(f"Initializing LLM: {config['provider']}/{config['model']}")
        model = initialize_llm(config["provider"], config["model"])
        
        _global_memory['config'] = config.copy()
        
        st.session_state.messages.append({
            "role": "assistant",
            "type": "info",
            "content": "🔍 Starting Research Phase..."
        })
        
        ui_logger.info("Running research agent")
        raw_results = await run_research_agent(
            task,
            model,
            expert_enabled=True,
            research_only=getattr(config, "research_only", True),
            hil=getattr(config, "hil", False),
            web_research_enabled=getattr(config, "web_research_enabled", False),
            config=config
        )
        
        ui_logger.debug(f"Research agent raw results: {raw_results}")
        
        if raw_results is None:
            raise ValueError("Research agent returned no results")
            
        results = {
            "success": True,
            "research_notes": [],
            "key_facts": {},
            "related_files": _global_memory.get('related_files', {})
        }
        
        if isinstance(raw_results, str):
            ui_logger.info("Processing research results")
            sections = raw_results.split('\n\n')
            for section in sections:
                if section.startswith('Research Notes:'):
                    ui_logger.debug("Processing research notes")
                    notes = section.replace('Research Notes:', '').strip().split('\n')
                    results['research_notes'].extend([note.strip('- ') for note in notes if note.strip()])
                    if results['research_notes']:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "research",
                            "content": "\n".join([f"- {note}" for note in results['research_notes']])
                        })
                elif section.startswith('Key Facts:'):
                    ui_logger.debug("Processing key facts")
                    facts = section.replace('Key Facts:', '').strip().split('\n')
                    for fact in facts:
                        if ':' in fact:
                            key, value = fact.strip('- ').split(':', 1)
                            results['key_facts'][key.strip()] = value.strip()
                    if results['key_facts']:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "research",
                            "content": "\n".join([f"- **{key}**: {value}" for key, value in results['key_facts'].items()])
                        })
                else:
                    content = section.strip()
                    if content:
                        ui_logger.debug(f"Processing additional content: {content[:100]}...")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": content
                        })
        
        _global_memory['research_notes'] = results['research_notes']
        _global_memory['key_facts'] = results['key_facts']
        _global_memory['implementation_requested'] = False
        
        ui_logger.info("✅ Research phase completed successfully")
        st.session_state.messages.append({
            "role": "assistant",
            "type": "success",
            "content": "✅ Research phase complete"
        })
        
        return results
    except Exception as e:
        ui_logger.error(f"Research Error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "type": "error",
            "content": f"Research failed: {str(e)}"
        })
        raise

def determine_interaction_type(task: str) -> str:
    """
    Determine the type of interaction based on the task description.
    
    Args:
        task (str): The task description
        
    Returns:
        str: The interaction type ('research', 'development', 'conversation', or 'shell')
    """
    # Check for shell command first
    if task.startswith("run "):
        return "shell"
    
    # Keywords that indicate research tasks
    research_keywords = [
        'research', 'find', 'search', 'look up', 'investigate',
        'analyze', 'study', 'explore', 'learn about', 'understand'
    ]
    
    # Keywords that indicate development tasks
    development_keywords = [
        'create', 'build', 'implement', 'develop', 'make',
        'code', 'program', 'write', 'add', 'update', 'modify',
        'fix', 'debug', 'change', 'refactor'
    ]
    
    task_lower = task.lower()
    
    # Check for research keywords
    if any(keyword in task_lower for keyword in research_keywords):
        return 'research'
    
    # Check for development keywords
    if any(keyword in task_lower for keyword in development_keywords):
        return 'development'
    
    # Default to conversation
    return 'conversation'

async def handle_conversation(task: str, config: Dict[str, Any]) -> str:
    """Handle a conversation task by calling the LLM."""
    try:
        # Extract provider and model name from config
        provider, model_name = config.model.split('/')
            
        response = completion(
            model=f"{provider}/{model_name}",  # Format as provider/model
            messages=[{"role": "user", "content": task}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"LLM Error: {str(e)}"
        ui_logger.error(error_message)
        raise Exception(error_message)

async def handle_research_results(results: Dict[str, Any]) -> None:
    """Handle the results from the research phase."""
    if results.get("success"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": results.get("findings", "Research completed successfully"),
            "type": "text"
        })
    else:
        error_message = f"Research failed: {results.get('error', 'Unknown error')}"
        ui_logger.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "type": "error"
        })
        raise Exception(error_message)

async def handle_planning_results(results: Dict[str, Any]) -> None:
    """Handle the results from the planning phase."""
    if results.get("success"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": results.get("plan", "Planning completed successfully"),
            "type": "text"
        })
    else:
        error_message = f"Planning failed: {results.get('error', 'Unknown error')}"
        ui_logger.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "type": "error"
        })
        raise Exception(error_message)

async def handle_implementation_results(results: Dict[str, Any]) -> None:
    """Handle the results from the implementation phase."""
    if results.get("success"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": results.get("result", "Implementation completed successfully"),
            "type": "text"
        })
    else:
        error_message = f"Implementation failed: {results.get('error', 'Unknown error')}"
        ui_logger.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "type": "error"
        })
        raise Exception(error_message)

async def execute_full_pipeline(task: str, config: Dict[str, Any]) -> None:
    """
    Execute the full development pipeline (Research -> Planning -> Implementation).
    
    Args:
        task (str): The task description
        config (Dict[str, Any]): Configuration for the task
    """
    try:
        ui_logger.info("Starting full development pipeline")
        
        # Research phase
        st.session_state.execution_stage = "research"
        with st.spinner("Conducting Research..."):
            research_results = await research_component(task, config)
            await handle_research_results(research_results)
        
        # Planning phase
        st.session_state.execution_stage = "planning"
        with st.spinner("Planning Implementation..."):
            planning_results = await planning_component(task, research_results, config)
            await handle_planning_results(planning_results)
        
        # Implementation phase
        st.session_state.execution_stage = "implementation"
        with st.spinner("Implementing Solution..."):
            implementation_results = await implementation_component(task, planning_results, config)
            await handle_implementation_results(implementation_results)
        
        # Update WebSocket if connected
        if hasattr(socket_interface, 'connected') and socket_interface.connected:
            await socket_interface.send({
                "type": "pipeline_complete",
                "content": "Pipeline execution completed successfully"
            })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Pipeline execution completed successfully",
            "type": "success"
        })
        
    except Exception as e:
        error_message = f"Pipeline error: {str(e)}"
        ui_logger.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "type": "error"
        })
        raise

async def handle_llm_task(task: str, config: Dict[str, Any]) -> None:
    """Handle a task using LLM."""
    try:
        # Prepare model parameters
        # Extract provider and model name from config
        provider, model_name = config.model.split('/')
        
        model_params = {
            "model": f"{provider}/{model_name}",  # Format as provider/model
            "messages": [{"role": "user", "content": task}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # Call LLM
        response = completion(**model_params)
        result = response.choices[0].message.content
        
        # Add response to messages
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": result
        })
        
        return {"success": True, "response": result}
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "type": "error",
            "content": f"LLM Error: {str(e)}"
        })
        return {"success": False, "error": str(e)}

async def handle_task(task: str, config: Dict[str, Any]) -> None:
    """
    Handle a task from the user. This can be either a shell command or an LLM interaction.
    
    Args:
        task (str): The task to handle
        config (Dict[str, Any]): Configuration for the task
    """
    # Initialize session state if needed
    initialize_session_state()
    
    # Add user message to conversation
    st.session_state.messages.append({
        "role": "user",
        "content": task,
        "type": "text"
    })
    
    try:
        interaction_type = determine_interaction_type(task)
        
        if interaction_type == "shell":
            # Handle shell command
            command = task[4:]  # Remove "run " prefix
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.stdout,
                "type": "text"
            })
        elif interaction_type == "conversation":
            response = await handle_conversation(task, config)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "type": "text"
            })
        else:
            await execute_full_pipeline(task, config)
    except Exception as e:
        error_message = f"Error processing task: {str(e)}"
        ui_logger.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message,
            "type": "error"
        })

async def main():
    """
    Main application function that sets up and runs the Streamlit interface.
    
    Handles:
    1. Page configuration and layout
    2. Session state initialization
    3. Memory initialization
    4. Sidebar configuration
        - API key inputs
        - Environment validation
        - Mode selection
        - Model configuration
        - Feature toggles
    5. WebSocket connection management
    6. Task input and execution
    7. Message processing and display
    """
    ui_logger.info("Initializing RA.Aid WebUI")
    
    st.set_page_config(
        page_title="RA.Aid WebUI",
        page_icon="🤖",
        layout="wide"
    )

    st.title("RA.Aid - AI Development Assistant")
    
    ui_logger.info("Initializing core components")
    initialize_session_state()
    initialize_memory()

    with st.sidebar:
        ui_logger.debug("Rendering sidebar configuration")
        st.header("Configuration")
        render_environment_status()
        
        mode = st.radio(
            "Mode",
            ["Research Only", "Full Development"],
            index=0
        )
        ui_logger.info(f"Selected mode: {mode}")

        # Get available providers (only those with valid API keys)
        available_providers = list(st.session_state.models.keys())
        logger.info(f"Models in session state: {st.session_state.models}")
        logger.info(f"Available providers before sorting: {available_providers}")
        available_providers.sort()  # Sort providers alphabetically
        logger.info(f"Available providers after sorting: {available_providers}")
        
        if not available_providers:
            st.error("No API providers configured. Please check your .env file.")
            return

        # Model Configuration
        provider = st.selectbox(
            "Provider",
            available_providers,
            format_func=lambda x: x.capitalize()  # Capitalize provider names
        )
        
        logger.info(f"Selected provider: {provider}")
        if provider:
            logger.info(f"Models available for {provider}: {st.session_state.models.get(provider, [])}")

        # Only show model selection if provider is selected
        if provider:
            available_models = st.session_state.models.get(provider, [])
            
            # Show model count
            st.caption(f"Available models: {len(available_models)}")
            
            # Model selection
            model = st.selectbox(
                "Model",
                available_models,
                format_func=lambda x: x.split('/')[-1] if '/' in x else x
            )
            
            if model:
                st.session_state.selected_model = model
                st.session_state.selected_provider = provider
        
        # Feature toggles
        st.subheader("Features")
        research_only = st.toggle("Research Only Mode", value=True)
        cowboy_mode = st.toggle("Cowboy Mode", value=False)
        hil = st.toggle("Human-in-the-Loop", value=False)
        web_research = st.toggle("Enable Web Research", value=False)
        
        # Create config object
        config = WebUIConfig(
            provider=provider,
            model=model,
            research_only=research_only,
            cowboy_mode=cowboy_mode,
            hil=hil,
            web_research_enabled=web_research
        )
        
        # Store config in memory
        _global_memory['config'] = config
        
        # Display current configuration
        with st.expander("Current Configuration"):
            st.code(str(config))
    
    # Start WebSocket thread if not already started
    if not st.session_state.websocket_thread_started:
        ui_logger.info("Starting WebSocket thread")
        thread = threading.Thread(target=websocket_thread, daemon=True)
        thread.start()
        st.session_state.websocket_thread_started = True
    
    # Process any pending messages
    process_message_queue()
    
    # Display Messages
    render_messages()
    
    # Task Input
    task = st.text_area("Enter your task or question:", height=100)
    
    if st.button("Submit", type="primary"):
        if not task:
            st.warning("Please enter a task or question")
            return
            
        if not provider or not model:
            st.error("Please select a provider and model")
            return
            
        ui_logger.info(f"Processing task: {task}")
        
        # Determine interaction type
        interaction_type = determine_interaction_type(task)
        ui_logger.info(f"Determined interaction type: {interaction_type}")
        
        # Execute task based on interaction type
        if interaction_type == 'conversation':
            st.session_state.execution_stage = "conversation"
            with st.spinner("Processing..."):
                response = await handle_conversation(task, config)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "type": "text"
                })
        else:
            if research_only:
                st.session_state.execution_stage = "research"
                with st.spinner("Conducting Research..."):
                    research_results = await research_component(task, _global_memory['config'])
                    await handle_research_results(research_results)
            else:
                ui_logger.info("Starting full development pipeline")
                # Execute full pipeline (research -> planning -> implementation)
                await execute_full_pipeline(task)
    
    # Process and Display Messages
    process_message_queue()
    render_messages()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        ui_logger.info("Application terminated by user")
    except Exception as e:
        ui_logger.error(f"Application error: {str(e)}")
        raise
