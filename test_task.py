import socketio
import time

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')
    
    # Send a task
    task_data = {
        "task": "hello",
        "config": {
            "provider": "openai",
            "model": "gpt-4",
            "expert_provider": "openai",
            "expert_model": "gpt-4",
            "cowboy_mode": False,
            "hil": True,
            "web_research_enabled": False
        }
    }
    sio.emit('task', task_data)

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.event
def output(data):
    print('Received output:', data)

# Connect to the server
try:
    sio.connect('http://localhost:5001')
    time.sleep(5)  # Wait for responses
    sio.disconnect()
except Exception as e:
    print('Error:', e) 