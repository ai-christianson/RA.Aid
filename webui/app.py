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

# Initialize SocketInterface and Message Queue
socket_interface = SocketInterface()
message_queue = Queue()

def handle_output(message: dict):
    """Callback to handle messages from the WebSocket."""
    message_queue.put(message)

def websocket_thread():
    """Thread to manage WebSocket connection and message handling."""
    try:
        # Register the message handler
        socket_interface.register_handler("message", handle_output)
        
        # Connect to the server with retries
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connected = loop.run_until_complete(socket_interface.connect_server())
        st.session_state.connected = connected
        if connected:
            logger.info("WebSocket connected successfully.")
        else:
            logger.error("Failed to connect to WebSocket server.")
        
        # Start listening to incoming messages
        loop.run_until_complete(socket_interface.setup_handlers())
    except Exception as e:
        logger.error(f"WebSocket thread encountered an error: {str(e)}")
        st.session_state.connected = False

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'models' not in st.session_state:
        st.session_state.models = load_available_models()
    if 'websocket_thread_started' not in st.session_state:
        st.session_state.websocket_thread_started = False

def load_available_models():
    """Load available models from APIs."""
    models = {
        'openai': [],
        'anthropic': []
    }
    
    env_status = load_environment_status()
    
    # Load OpenAI models
    if env_status.get('openai'):
        try:
            client = OpenAI()
            response = client.models.list()
            models['openai'] = [model.id for model in response.data if model.id.startswith('gpt')]
        except Exception as e:
            logger.error(f"Failed to load OpenAI models: {str(e)}")
            # Fallback to default models
            models['openai'] = ["gpt-4", "gpt-3.5-turbo"]
    
    # Load Anthropic models
    if env_status.get('anthropic'):
        try:
            client = anthropic.Anthropic()
            response = client.models.list()
            models['anthropic'] = [model.id for model in response.data]
        except Exception as e:
            logger.error(f"Failed to load Anthropic models: {str(e)}")
            # Fallback to default models
            models['anthropic'] = ["claude-3-opus", "claude-3-sonnet", "claude-2"]
    
    return models

def render_environment_status():
    """Render environment status section."""
    st.sidebar.subheader("Environment Status")
    env_status = load_environment_status()
    
    for provider, status in env_status.items():
        if status:
            st.sidebar.success(f"{provider.capitalize()} configured")
        else:
            st.sidebar.error(f"{provider.capitalize()} not configured")

def process_message_queue():
    """Process messages from the queue."""
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
    """Render messages in the chat interface."""
    for message in st.session_state.messages:
        if message.get('type') == 'error':
            st.error(message['content'])
        else:
            st.write(message['content'])

def send_task(task: str, config: dict):
    """Send task to the server."""
    if st.session_state.connected:
        try:
            asyncio.run(socket_interface.send_task(task, config))
            st.success("Task sent successfully")
        except Exception as e:
            st.error(f"Failed to send task: {str(e)}")
    else:
        st.error("Not connected to server")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="RA.Aid WebUI",
        page_icon="🤖",
        layout="wide"
    )

    st.title("RA.Aid - AI Development Assistant")
    
    # Initialize session state and memory
    initialize_session_state()
    initialize_memory()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys
        anthropic_key = st.text_input("Anthropic API Key", type="password")
        openai_key = st.text_input("OpenAI API Key", type="password")
        tavily_key = st.text_input("Tavily API Key", type="password")
        
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        # Validate environment
        config_validation = load_environment_status()
        render_environment_status()
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["Research Only", "Full Development"],
            index=0
        )

        # Model configuration
        provider = st.selectbox(
            "Provider",
            ["anthropic", "openai", "openrouter", "openai-compatible"]
        )

        model = st.text_input(
            "Model Name",
            value="claude-3-opus" if provider == "anthropic" else "gpt-4" if provider == "openai" else ""
        )

        # Features
        st.subheader("Features")
        cowboy_mode = st.checkbox("Cowboy Mode", help="Skip interactive approval for shell commands")
        hil_mode = st.checkbox("Human-in-the-Loop", help="Enable human interaction during execution")
        web_research = st.checkbox("Enable Web Research", value=config_validation.get("web_research_enabled", False))
    
    # Display connection status
    if st.session_state.connected:
        st.sidebar.success("Connected to server")
    else:
        st.sidebar.warning("Not connected to server")

    # Start WebSocket connection in a separate thread
    if not st.session_state.websocket_thread_started:
        thread = threading.Thread(target=websocket_thread, daemon=True)
        thread.start()
        st.session_state.websocket_thread_started = True

    # Main content area
    task = st.text_area("Enter your task or query:", height=150)
    
    if st.button("Start"):
        if not task.strip():
            st.error("Please enter a valid task or query.")
            return

        # Update global memory configuration
        _global_memory['config'] = {
            "provider": provider,
            "model": model,
            "research_only": mode == "Research Only",
            "cowboy_mode": cowboy_mode,
            "hil": hil_mode,
            "web_research_enabled": web_research
        }

        # Research stage
        st.session_state.execution_stage = "research"
        with st.spinner("Conducting Research..."):
            research_results = research_component(task, _global_memory['config'])
            st.session_state.research_results = research_results

        if mode != "Research Only" and research_results.get("success"):
            # Planning stage
            st.session_state.execution_stage = "planning"
            with st.spinner("Planning Implementation..."):
                planning_results = planning_component(task, _global_memory['config'])
                st.session_state.planning_results = planning_results

            if planning_results.get("success"):
                # Implementation stage
                st.session_state.execution_stage = "implementation"
                with st.spinner("Implementing Changes..."):
                    implementation_results = implementation_component(
                        task,
                        st.session_state.research_results,
                        st.session_state.planning_results,
                        _global_memory['config']
                    )
    
    # Process and render messages
    process_message_queue()
    render_messages()

if __name__ == "__main__":
    main()