class RAWebUI {
    constructor() {
        this.messageHistory = [];
        this.connectionAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.setupElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }

    setupElements() {
        this.userInput = document.getElementById('user-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');
        this.streamOutput = document.getElementById('stream-output');
        this.historyList = document.getElementById('history-list');
        
        // Disable send button initially
        this.sendButton.disabled = true;
    }

    setupEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    async connectWebSocket() {
        // Don't try to reconnect if we've exceeded the maximum attempts
        if (this.connectionAttempts >= this.maxReconnectAttempts) {
            this.appendMessage(
                'Maximum reconnection attempts reached. Please refresh the page.',
                'error'
            );
            return;
        }

        try {
            // Get the server port from the meta tag
            const serverPort = document.querySelector('meta[name="server-port"]')?.content || '8080';
            
            // Construct WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.hostname}:${serverPort}/ws`;
            console.log('Attempting to connect to WebSocket URL:', wsUrl);
            
            // Close existing connection if any
            if (this.ws) {
                this.ws.close();
            }

            // Create new WebSocket connection
            console.log('Creating new WebSocket connection...');
            this.ws = new WebSocket(wsUrl);
            this.connectionAttempts++;

            // Setup WebSocket event handlers
            this.ws.onopen = () => {
                console.log('WebSocket connection established successfully');
                this.connectionAttempts = 0; // Reset counter on successful connection
                this.sendButton.disabled = false;
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.sendButton.disabled = true;

                // Only attempt reconnect if not a normal closure and within retry limits
                if (event.code !== 1000 && this.connectionAttempts < this.maxReconnectAttempts) {
                    const delay = Math.min(1000 * Math.pow(2, this.connectionAttempts), 10000);
                    this.appendMessage(
                        `Connection lost. Reconnecting in ${delay/1000} seconds...`,
                        'error'
                    );
                    setTimeout(() => this.connectWebSocket(), delay);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleServerMessage(data);
                } catch (error) {
                    console.error('Error parsing message:', error);
                    this.appendMessage('Error processing server message', 'error');
                }
            };

        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.appendMessage(
                `Connection error: ${error.message}. Retrying...`,
                'error'
            );
            
            // Attempt to reconnect with exponential backoff
            const delay = Math.min(1000 * Math.pow(2, this.connectionAttempts), 10000);
            setTimeout(() => this.connectWebSocket(), delay);
        }
    }

    handleServerMessage(data) {
        if (data.type === 'stream_start') {
            this.streamOutput.textContent = '';
            this.streamOutput.style.display = 'block';
        } else if (data.type === 'stream_end') {
            this.streamOutput.style.display = 'none';
            this.addToHistory(data.request);
            this.sendButton.disabled = false;
        } else if (data.type === 'chunk') {
            this.handleChunk(data.chunk);
        }
    }

    handleChunk(chunk) {
        if (chunk.agent && chunk.agent.messages) {
            chunk.agent.messages.forEach(msg => {
                if (msg.content) {
                    if (Array.isArray(msg.content)) {
                        msg.content.forEach(content => {
                            if (content.type === 'text' && content.text.trim()) {
                                this.appendMessage(content.text.trim(), 'system');
                            }
                        });
                    } else if (msg.content.trim()) {
                        this.appendMessage(msg.content.trim(), 'system');
                    }
                }
            });
        } else if (chunk.tools && chunk.tools.messages) {
            chunk.tools.messages.forEach(msg => {
                if (msg.status === 'error' && msg.content) {
                    this.appendMessage(msg.content.trim(), 'error');
                }
            });
        }
    }

    appendMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = content;
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    addToHistory(request) {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.textContent = request.slice(0, 50) + (request.length > 50 ? '...' : '');
        historyItem.title = request;
        historyItem.addEventListener('click', () => {
            this.userInput.value = request;
            this.userInput.focus();
        });
        this.historyList.insertBefore(historyItem, this.historyList.firstChild);
        this.messageHistory.push(request);
    }

    sendMessage() {
        console.log('Send button clicked');
        const message = this.userInput.value.trim();
        console.log('Message content:', message);
        
        if (!message) {
            console.log('Message is empty, not sending');
            return;
        }

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket is not connected');
            this.appendMessage('Error: Not connected to server. Please wait...', 'error');
            return;
        }

        try {
            console.log('Sending message to server');
            this.appendMessage(message, 'user');
            const payload = { type: 'request', content: message };
            console.log('Payload:', payload);
            
            this.ws.send(JSON.stringify(payload));
            console.log('Message sent successfully');
            
            this.userInput.value = '';
            this.sendButton.disabled = true;
        } catch (error) {
            console.error('Error sending message:', error);
            this.appendMessage(`Error sending message: ${error.message}`, 'error');
            this.sendButton.disabled = false;
        }
    }
}

// Initialize the UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.raWebUI = new RAWebUI();
});