import pytest
from unittest.mock import Mock, AsyncMock, patch
import websockets
from webui.socket_interface import SocketInterface

@pytest.fixture
def socket_interface():
    return SocketInterface("ws://test.com")

@pytest.mark.asyncio
async def test_connect_success(socket_interface):
    """Test successful connection"""
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        result = await socket_interface.connect()
        assert result is True
        assert socket_interface.connected is True
        mock_connect.assert_called_once_with("ws://test.com")

@pytest.mark.asyncio
async def test_connect_retries(socket_interface):
    """Test connection retries"""
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = [
            websockets.exceptions.WebSocketException("Failed"),
            websockets.exceptions.WebSocketException("Failed"),
            AsyncMock()  # Succeed on third try
        ]
        
        result = await socket_interface.connect(max_retries=3)
        assert result is True
        assert mock_connect.call_count == 3

@pytest.mark.asyncio
async def test_connect_failure(socket_interface):
    """Test connection failure"""
    with patch('websockets.connect', side_effect=websockets.exceptions.WebSocketException("Failed")):
        result = await socket_interface.connect(max_retries=2)
        assert result is False
        assert socket_interface.connected is False

@pytest.mark.asyncio
async def test_send_success(socket_interface):
    """Test successful message sending"""
    # Setup mock websocket
    mock_websocket = AsyncMock()
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_websocket
        
        # Connect first
        await socket_interface.connect()
        
        # Test sending message
        message = {"type": "test", "content": "test message"}
        result = await socket_interface.send(message)
        
        assert result is True
        mock_websocket.send.assert_called_once()

@pytest.mark.asyncio
async def test_send_not_connected(socket_interface):
    """Test sending message when not connected"""
    message = {"type": "test", "content": "test message"}
    result = await socket_interface.send(message)
    assert result is False

@pytest.mark.asyncio
async def test_disconnect(socket_interface):
    """Test disconnection"""
    # Setup mock websocket
    mock_websocket = AsyncMock()
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_websocket
        
        # Connect first
        await socket_interface.connect()
        assert socket_interface.connected is True
        
        # Test disconnect
        await socket_interface.disconnect()
        assert socket_interface.connected is False
        assert socket_interface.websocket is None
        mock_websocket.close.assert_called_once()

def test_register_handler(socket_interface):
    """Test handler registration"""
    async def test_handler(data):
        pass
        
    socket_interface.register_handler("test", test_handler)
    assert "test" in socket_interface.handlers
    assert socket_interface.handlers["test"] == test_handler

@pytest.mark.asyncio
async def test_setup_handlers(socket_interface):
    """Test handler setup"""
    # Setup mock websocket
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = [
        '{"type": "test", "data": "test_data"}'
    ]
    
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = mock_websocket
        
        # Connect first
        await socket_interface.connect()
        
        # Register mock handler
        mock_handler = AsyncMock()
        socket_interface.register_handler("test", mock_handler)
        
        # Run handlers
        await socket_interface.setup_handlers()
        
        # Verify handler was called
        mock_handler.assert_called_once()
