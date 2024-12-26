import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json
from websockets.exceptions import WebSocketException

@pytest.fixture
def socket_interface():
    from webui.socket_interface import SocketInterface
    return SocketInterface()

@pytest.fixture
def mock_websocket():
    mock = AsyncMock()
    mock.__aiter__.return_value = []
    return mock

@pytest.mark.asyncio
async def test_connect_server_success(socket_interface):
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = AsyncMock()
        result = await socket_interface.connect_server()
        assert result is True
        assert socket_interface.connected is True
        mock_connect.assert_called_once_with("ws://localhost:8765")

@pytest.mark.asyncio
async def test_connect_server_failure(socket_interface):
    with patch('websockets.connect', side_effect=WebSocketException("Connection failed")):
        result = await socket_interface.connect_server(max_retries=1)
        assert result is False
        assert socket_interface.connected is False

@pytest.mark.asyncio
async def test_disconnect(socket_interface, mock_websocket):
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    await socket_interface.disconnect()
    mock_websocket.close.assert_awaited_once()
    assert socket_interface.connected is False
    assert socket_interface.websocket is None

@pytest.mark.asyncio
async def test_send_task_success(socket_interface, mock_websocket):
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    task = "test task"
    config = {"model": "test-model"}
    
    result = await socket_interface.send_task(task, config)
    assert result is True
    mock_websocket.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_task_not_connected(socket_interface):
    socket_interface.connected = False
    task = "test task"
    config = {"model": "test-model"}
    
    result = await socket_interface.send_task(task, config)
    assert result is False

@pytest.mark.asyncio
async def test_setup_handlers(socket_interface, mock_websocket):
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    
    handler = AsyncMock()
    socket_interface.register_handler("test", handler)
    
    mock_websocket.__aiter__.return_value = [
        '{"type": "test", "content": "test message"}'
    ]
    
    await socket_interface.setup_handlers()
    handler.assert_awaited_once_with({"type": "test", "content": "test message"})

@pytest.mark.asyncio
async def test_setup_handlers_invalid_json(socket_interface, mock_websocket):
    socket_interface.websocket = mock_websocket
    socket_interface.connected = True
    
    handler = AsyncMock()
    socket_interface.register_handler("test", handler)
    
    mock_websocket.__aiter__.return_value = ["invalid json"]
    
    await socket_interface.setup_handlers()
    handler.assert_not_awaited()

@pytest.mark.asyncio
async def test_concurrent_connections(socket_interface):
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = AsyncMock()
        
        # First connection should succeed
        result1 = await socket_interface.connect_server()
        assert result1 is True
        
        # Second connection should not create a new connection
        result2 = await socket_interface.connect_server()
        assert result2 is False
        
        mock_connect.assert_called_once_with("ws://localhost:8765") 