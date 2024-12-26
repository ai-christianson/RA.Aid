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
from webui.socket_interface import SocketInterface
from components.memory import initialize_memory, _global_memory
from components.research import research_component
from components.planning import planning_component
from components.implementation import implementation_component
from webui.config import WebUIConfig, load_environment_status
from ra_aid.logger import logger
import asyncio
import os
import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize global components for WebSocket communication
socket_interface = SocketInterface()
message_queue = Queue()

# Debug environment variables
logger.info("Initial Environment Check:")
logger.info(f"ANTHROPIC_API_KEY present: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
logger.info(f"ANTHROPIC_API_KEY value: {os.getenv('ANTHROPIC_API_KEY')[:10]}..." if os.getenv('ANTHROPIC_API_KEY') else "None")

def handle_output(message: dict):
    """
    Callback function to handle messages received from the WebSocket.
    Messages are added to a queue for processing in the main Streamlit loop.
    
    Args:
        message (dict): The message received from the WebSocket server
    """
    message_queue.put(message)

def websocket_thread():
    """
    Background thread function that manages WebSocket connection and message handling.
    Handles:
    - Server connection with retry logic
    - Message handler registration
    - Asyncio event loop management
    - Connection status updates
    """
    try:
        # Register the message handler for incoming WebSocket messages
        socket_interface.register_handler("message", handle_output)
        
        # Setup asyncio event loop for WebSocket operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Attempt connection with retry logic
        connected = loop.run_until_complete(socket_interface.connect_server())
        st.session_state.connected = connected
        
        if connected:
            logger.info("WebSocket connected successfully.")
        else:
            logger.error("Failed to connect to WebSocket server.")
        
        # Start listening for incoming messages
        loop.run_until_complete(socket_interface.setup_handlers())
    except Exception as e:
        logger.error(f"WebSocket thread encountered an error: {str(e)}")
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
        st.session_state.messages = []
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
    status_text = " | ".join([f"{provider}: {'✓' if status else '��'}" 
                             for provider, status in env_status.items()])
    st.sidebar.caption(f"API Status: {status_text}")

def process_message_queue():
    """
    Process messages from the WebSocket message queue.
    Handles different message types (error, standard) and updates the UI accordingly.
    Messages are processed until the queue is empty.
    """
    while True:
        try:
            message = message_queue.get_nowait()
            if isinstance(message, dict):
                if message.get('type') == 'error':
                    st.session_state.messages.append({'type': 'error', 'content': message.get('content', 'An error occurred.')})
                else:
                    st.session_state.messages.append(message)
        except Empty:
            break

def render_messages():
    """
    Render all messages in the chat interface.
    Handles different message types with appropriate styling:
    - Error messages are displayed with error styling
    - Standard messages are displayed normally
    """
    for message in st.session_state.messages:
        if message.get('type') == 'error':
            st.error(message['content'])
        else:
            st.write(message['content'])

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

def main():
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
    # Configure the Streamlit page
    st.set_page_config(
        page_title="RA.Aid WebUI",
        page_icon="🤖",
        layout="wide"
    )

    st.title("RA.Aid - AI Development Assistant")
    
    # Initialize core components
    initialize_session_state()
    initialize_memory()

    # Sidebar Configuration Section
    with st.sidebar:
        st.header("Configuration")
        
        # Minimal environment status
        render_environment_status()
        
        # Mode Selection
        mode = st.radio(
            "Mode",
            ["Research Only", "Full Development"],
            index=0
        )

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
            
            # Group models by sub-provider (for OpenRouter and similar cases)
            def group_models(models):
                grouped = {}
                for model in models:
                    provider_name = model.split('/')[0]
                    if provider_name not in grouped:
                        grouped[provider_name] = []
                    grouped[provider_name].append(model)
                return grouped
            
            # Get grouped models if needed
            if provider == 'openrouter':
                grouped_models = group_models(available_models)
                # Flatten but keep grouping order
                model_list = []
                for provider_name in sorted(grouped_models.keys()):
                    model_list.extend(sorted(grouped_models[provider_name], reverse=True))
                available_models = model_list
            else:
                available_models = sorted(available_models, reverse=True)
            
            # Integrated search and select with model grouping display
            model = st.selectbox(
                "Select a model",
                available_models,
                index=0 if available_models else None,
                format_func=lambda x: x.split('/')[-1] if '/' in x else x,  # Show only model name in dropdown
                placeholder="Start typing to search models...",
                help="Type to quickly filter models"
            )
            
            # Show model info if selected
            if model:
                # Show provider/model hierarchy
                provider_name, model_name = model.split('/') if '/' in model else (provider, model)
                st.caption(f"Provider: {provider_name}")
                
                model_info = {
                    "openai/gpt-4-turbo": "Latest GPT-4 model with improved performance",
                    "openai/gpt-4": "Most capable GPT-4 model",
                    "anthropic/claude-3-opus": "Most capable Claude model",
                    "anthropic/claude-3-sonnet": "Balanced Claude model",
                    "google/gemini-pro": "Google's latest language model",
                    "meta-llama/llama-2-70b-chat": "Meta's largest open LLM",
                    "mistral/mixtral-8x7b": "Mistral's mixture of experts model"
                }
                if model in model_info:
                    st.caption(f"Description: {model_info[model]}")
                else:
                    st.caption(f"Model: {model_name}")
        else:
            model = ""

        # Feature Toggles
        st.subheader("Features")
        cowboy_mode = st.checkbox("Cowboy Mode", help="Skip interactive approval for shell commands")
        hil_mode = st.checkbox("Human-in-the-Loop", help="Enable human interaction during execution")
        web_research = st.checkbox("Enable Web Research")
    
    # Display WebSocket Connection Status
    if st.session_state.connected:
        st.sidebar.success("Connected to server")
    else:
        st.sidebar.warning("Not connected to server")

    # Initialize WebSocket Connection
    if not st.session_state.websocket_thread_started:
        thread = threading.Thread(target=websocket_thread, daemon=True)
        thread.start()
        st.session_state.websocket_thread_started = True

    # Task Input and Execution Section
    task = st.text_area("Enter your task or query:", height=150)
    
    if st.button("Start"):
        if not task.strip():
            st.error("Please enter a valid task or query.")
            return

        # Update global memory with current configuration
        _global_memory['config'] = {
            "provider": provider,
            "model": model,
            "research_only": mode == "Research Only",
            "cowboy_mode": cowboy_mode,
            "hil": hil_mode,
            "web_research_enabled": web_research
        }

        # Execute Task Pipeline
        # 1. Research Phase
        st.session_state.execution_stage = "research"
        with st.spinner("Conducting Research..."):
            research_results = research_component(task, _global_memory['config'])
            st.session_state.research_results = research_results

        # 2. Planning Phase (if not research-only mode)
        if mode != "Research Only" and research_results.get("success"):
            st.session_state.execution_stage = "planning"
            with st.spinner("Planning Implementation..."):
                planning_results = planning_component(task, _global_memory['config'])
                st.session_state.planning_results = planning_results

            # 3. Implementation Phase
            if planning_results.get("success"):
                st.session_state.execution_stage = "implementation"
                with st.spinner("Implementing Changes..."):
                    implementation_results = implementation_component(
                        task,
                        st.session_state.research_results,
                        st.session_state.planning_results,
                        _global_memory['config']
                    )
    
    # Process and Display Messages
    process_message_queue()
    render_messages()

if __name__ == "__main__":
    main()