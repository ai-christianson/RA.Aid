import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from websockets.exceptions import WebSocketException
import asyncio
from webui.socket_interface import SocketInterface

@pytest.fixture
def socket_interface():
    """Create a socket interface instance"""
    return SocketInterface()

@pytest.fixture
def mock_websocket():
    """Create a mock websocket"""
    mock = AsyncMock()
    mock.close = AsyncMock()
    mock.send = AsyncMock()
    mock.recv = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_connect_server_success(socket_interface, mock_websocket):
    """Test successful server connection"""
    async def mock_connect(*args, **kwargs):
        return mock_websocket

    with patch('websockets.connect', side_effect=mock_connect):
        connected = await socket_interface.connect_server()
        assert connected is True
        assert socket_interface.websocket is mock_websocket

@pytest.mark.asyncio
async def test_connect_server_failure(socket_interface):
    """Test server connection failure"""
    with patch('websockets.connect', side_effect=WebSocketException("Failed")):
        connected = await socket_interface.connect_server(max_retries=1)
        assert connected is False
        assert socket_interface.websocket is None

@pytest.mark.asyncio
async def test_send_task_success(socket_interface, mock_websocket):
    """Test successful task sending"""
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True  # Set connected state
    task = "test task"
    config = {"provider": "test", "model": "test-model"}

    result = await socket_interface.send_task(task, config)
    assert result is True
    mock_websocket.send.assert_called_once()

@pytest.mark.asyncio
async def test_send_task_not_connected(socket_interface):
    """Test task sending when not connected"""
    socket_interface.websocket = None
    socket_interface.connected = False
    task = "test task"
    config = {"provider": "test", "model": "test-model"}

    result = await socket_interface.send_task(task, config)
    assert result is False

@pytest.mark.asyncio
async def test_send_task_connection_error(socket_interface, mock_websocket):
    """Test task sending with connection error"""
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    mock_websocket.send.side_effect = WebSocketException("Send failed")
    task = "test task"
    config = {"provider": "test", "model": "test-model"}

    result = await socket_interface.send_task(task, config)
    assert result is False

def test_register_handler(socket_interface):
    """Test handler registration"""
    mock_handler = MagicMock()
    socket_interface.register_handler("test_event", mock_handler)
    assert "test_event" in socket_interface.handlers
    assert socket_interface.handlers["test_event"] == mock_handler

@pytest.mark.asyncio
async def test_setup_handlers_success(socket_interface, mock_websocket):
    """Test successful handler setup"""
    socket_interface.websocket = mock_websocket
    mock_websocket.recv.return_value = '{"type": "test_event", "data": "test data"}'
    mock_handler = AsyncMock()
    socket_interface.register_handler("test_event", mock_handler)

    # Mock the websocket's __aiter__ to return one message then raise StopAsyncIteration
    mock_websocket.__aiter__.return_value = [mock_websocket.recv.return_value].__iter__()

    await socket_interface.setup_handlers()
    mock_handler.assert_called_once_with({"type": "test_event", "data": "test data"})

@pytest.mark.asyncio
async def test_setup_handlers_unknown_type(socket_interface, mock_websocket):
    """Test handler setup with unknown message type"""
    socket_interface.websocket = mock_websocket
    mock_websocket.recv.return_value = '{"type": "unknown_event", "data": "test data"}'
    mock_handler = AsyncMock()
    socket_interface.register_handler("test_event", mock_handler)

    # Mock the websocket's __aiter__ to return one message then raise StopAsyncIteration
    mock_websocket.__aiter__.return_value = [mock_websocket.recv.return_value].__iter__()

    await socket_interface.setup_handlers()
    mock_handler.assert_not_called()

@pytest.mark.asyncio
async def test_setup_handlers_invalid_message(socket_interface, mock_websocket):
    """Test handler setup with invalid message format"""
    socket_interface.websocket = mock_websocket
    invalid_messages = [
        'not json',
        '{}',  # missing type
        '{"type": ""}',  # empty type
        '{"type": null}',  # null type
    ]

    for message in invalid_messages:
        mock_websocket.recv.return_value = message
        # Mock the websocket's __aiter__ to return one message then raise StopAsyncIteration
        mock_websocket.__aiter__.return_value = [message].__iter__()
        await socket_interface.setup_handlers()  # Should not raise any exception

@pytest.mark.asyncio
async def test_disconnect_success(socket_interface, mock_websocket):
    """Test successful disconnection"""
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    await socket_interface.disconnect()
    mock_websocket.close.assert_called_once()
    assert socket_interface.websocket is None
    assert socket_interface.connected is False

@pytest.mark.asyncio
async def test_disconnect_no_websocket(socket_interface):
    """Test disconnection when no websocket exists"""
    socket_interface.websocket = None
    socket_interface.connected = False
    await socket_interface.disconnect()  # Should not raise any exception
    assert socket_interface.connected is False

@pytest.mark.asyncio
async def test_disconnect_error(socket_interface, mock_websocket):
    """Test disconnection with error"""
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    mock_websocket.close.side_effect = WebSocketException("Close failed")
    await socket_interface.disconnect()  # Should not raise any exception
    assert socket_interface.websocket is None
    assert socket_interface.connected is False 