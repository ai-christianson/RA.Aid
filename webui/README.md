# RA.Aid WebUI

The web user interface component of RA.Aid, providing a Streamlit-based interface for interacting with the AI Development Assistant.

## Directory Structure

```
webui/
├── app.py              # Main Streamlit application
├── config.py           # Configuration management
├── logger.py           # Logging setup and utilities
├── server.py           # WebSocket server implementation
├── socket_interface.py # WebSocket client interface
├── tests/             # Test suite
│   ├── test_app.py    # Tests for main application
│   ├── test_config.py # Tests for configuration
│   ├── test_logger.py # Tests for logging
│   └── test_socket_interface.py # Tests for WebSocket interface
└── logs/              # Log files directory
```

## Components

- **Main Application (`app.py`)**: Implements the Streamlit web interface and core functionality
- **Configuration (`config.py`)**: Handles environment variables and application settings
- **Logger (`logger.py`)**: Provides structured logging with colored output
- **WebSocket Server (`server.py`)**: Implements real-time communication server
- **Socket Interface (`socket_interface.py`)**: Client-side WebSocket handling

## Testing

Run tests with:
```bash
pytest webui/tests/ -v
```

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run webui/app.py
```

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OPENROUTER_API_KEY`: OpenRouter API key

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting changes 