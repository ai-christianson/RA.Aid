# WebUI Implementation Documentation

## Overview

This document outlines the current implementation of the RA.Aid WebUI, which uses Streamlit for the user interface and a Socket Interface for backend communication.

## Core Components

### 1. Memory Management (`memory.py`)

The memory management system provides a centralized store for various types of data used throughout the application.

#### Key Components:

- **Global Memory Store**: A dictionary that maintains different types of data:
  ```python
  _global_memory = {
      'research_notes': [],
      'plans': [],
      'tasks': {},  # Dict[int, str] - ID to task mapping
      'key_facts': {},  # Dict[int, str] - ID to fact mapping
      'key_snippets': {},  # Dict[int, SnippetInfo] - ID to snippet mapping
      'related_files': {},  # Dict[int, str] - ID to filepath mapping
      # ... other memory components
  }
  ```

- **SnippetInfo Type Definition**:
  ```python
  class SnippetInfo(TypedDict):
      filepath: str
      line_number: int
      snippet: str
      source: str
      relevance: float
      description: Optional[str]
  ```

#### Key Functions:

1. **`emit_key_snippets`**:
   - Stores code snippets in memory
   - Automatically tracks related files
   - Displays formatted snippets with source location and description
   - Returns confirmation of storage

2. **`delete_key_snippets`**:
   - Removes specified snippets by their IDs
   - Handles non-existent IDs gracefully
   - Returns confirmation of deletion

3. **`emit_related_files`**:
   - Tracks files referenced in the codebase
   - Prevents duplicate entries
   - Returns file IDs for reference

### 2. Socket Interface (`socket_interface.py`)

The Socket Interface manages real-time communication between the Streamlit frontend and the backend server.

#### Key Features:

- Asynchronous communication using `socketio.AsyncClient`
- Connection management with retry capabilities
- Message queue handling
- Event handling for various socket events

#### Core Methods:

1. **Connection Management**:
   ```python
   async def connect_server(self) -> bool:
       # Establishes connection to backend server
       # Returns success/failure status

   async def disconnect(self) -> None:
       # Safely disconnects from server
   ```

2. **Task Communication**:
   ```python
   async def send_task(self, task: str, config: Dict[str, Any]) -> bool:
       # Sends task to backend for processing
       # Returns success/failure status
   ```

3. **Message Handling**:
   ```python
   def register_handler(self, event: str, handler: Callable) -> None:
       # Registers custom event handlers
   ```

### 3. Streamlit Application (`app.py`)

The main application interface built with Streamlit.

#### Key Features:

- Environment status monitoring
- Task submission interface
- Real-time message display
- Configuration management

#### Core Functions:

1. **Session State Management**:
   ```python
   def initialize_session_state():
       # Sets up initial session state
       # Manages connection status and message queue

   def process_message_queue():
       # Processes and displays incoming messages
   ```

2. **UI Components**:
   ```python
   def render_environment_status():
       # Displays API key status and environment info

   def render_config_section():
       # Renders configuration options
   ```

3. **Task Management**:
   ```python
   def main_send_task():
       # Handles task submission
       # Manages socket communication
   ```

## Testing

The implementation includes comprehensive unit tests for all components:

1. **Memory Tests**:
   - Snippet management
   - File tracking
   - State persistence

2. **Socket Interface Tests**:
   - Connection handling
   - Message processing
   - Error scenarios

3. **Application Tests**:
   - UI component rendering
   - State management
   - Configuration handling

## Usage

The WebUI can be started using the provided run script:

```bash
./run_tests.sh  # For running tests
streamlit run webui/app.py  # For starting the application
```

## Current Test Coverage

- `app.py`: 93% coverage
- `config.py`: 100% coverage
- `logger.py`: 96% coverage
- `socket_interface.py`: 86% coverage

## Future Improvements

1. Increase test coverage for `socket_interface.py`
2. Enhance error handling and recovery mechanisms
3. Implement additional UI features for better user experience
4. Add more comprehensive logging and monitoring
