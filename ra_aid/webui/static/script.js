class WebSocketHandler {
    constructor() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    initialize() {
        // Store DOM elements as instance variables
        this.messageInput = document.getElementById('user-input');
        this.sendButton = document.getElementById('send-button');
        this.clearButton = document.getElementById('clear-button');
        this.streamOutput = document.getElementById('stream-output');

        // Validate required elements exist
        if (!this.messageInput || !this.sendButton || !this.streamOutput) {
            console.error('Required elements not found:', {
                messageInput: !!this.messageInput,
                sendButton: !!this.sendButton,
                streamOutput: !!this.streamOutput
            });
            return;
        }

        // Remove hidden class if present
        this.streamOutput.classList.remove('hidden');
        
        // Initialize WebSocket
        this.connectWebSocket();

        // Add event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.clearButton?.addEventListener('click', () => this.clearConversation());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        console.log('WebSocketHandler initialized with elements:', {
            messageInput: this.messageInput,
            sendButton: this.sendButton,
            streamOutput: this.streamOutput
        });
    }

    connectWebSocket() {
        try {
            const wsUrl = `ws://${window.location.host}/ws`;
            console.log('Attempting to connect to WebSocket URL:', wsUrl);

            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connection established successfully');
                this.sendButton.disabled = false;
            };

            this.ws.onclose = () => {
                console.log('WebSocket connection closed');
                this.sendButton.disabled = true;
                // Try to reconnect after a delay
                setTimeout(() => this.connectWebSocket(), 2000);
            };

            this.ws.onmessage = (event) => {
                try {
                    console.log('Raw WebSocket message:', event.data);
                    const message = JSON.parse(event.data);
                    console.log('Parsed WebSocket message:', message);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error handling message:', error);
                    this.appendOutput({
                        content: `Error: ${error.message}`,
                        status: 'error'
                    });
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.sendButton.disabled = true;
                this.appendOutput({
                    content: 'Connection error. Retrying...',
                    status: 'error'
                });
            };
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.appendOutput({
                content: `Connection error: ${error.message}`,
                status: 'error'
            });
        }
    }

    handleMessage(message) {
        switch (message.type) {
            case 'stream_start':
                this.handleStreamStart();
                break;
            case 'chunk':
                this.handleChunk(message.chunk);
                break;
            case 'stream_end':
                this.handleStreamEnd();
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }

    handleStreamStart() {
        console.log('Stream starting');
        this.clearStreamOutput();
        this.appendOutput({
            content: 'Starting new conversation...',
            status: 'info'
        });
    }

    handleStreamEnd() {
        console.log('Stream ending');
        this.appendOutput({
            content: 'Conversation complete.',
            status: 'success'
        });
        this.messageInput.disabled = false;
        this.sendButton.disabled = false;
    }

    handleChunk(chunk) {
        console.log(' Processing chunk:', chunk);
        if (chunk.agent && chunk.agent.messages) {
            chunk.agent.messages.forEach(message => {
                console.log(' Processing agent message:', message);
                console.log(' Adding agent message:', message.content);
                this.appendOutput(message);
            });
        }
    }

    clearStreamOutput() {
        console.log('Clearing stream output');
        while (this.streamOutput.firstChild) {
            this.streamOutput.removeChild(this.streamOutput.firstChild);
        }
        console.log('Stream output cleared');
    }

    clearConversation() {
        this.clearStreamOutput();
        this.appendOutput({
            content: 'Conversation cleared.',
            status: 'info'
        });
    }

    appendOutput(message) {
        console.log(' Appending output:', message);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.status || 'info'}`;

        const contentSpan = document.createElement('span');
        
        // Convert ANSI escape codes to HTML
        let content = message.content;
        content = this.convertAnsiToHtml(content);
        
        // Check for code blocks and apply syntax highlighting
        if (content.includes('```')) {
            content = this.highlightCodeBlocks(content);
        }
        
        contentSpan.innerHTML = content;
        messageDiv.appendChild(contentSpan);
        
        this.streamOutput.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    convertAnsiToHtml(text) {
        // ANSI color codes to CSS classes
        const ansiToClass = {
            '[94m': '<span class="text-blue-400">',
            '[1;32m': '<span class="text-green-400 font-bold">',
            '[0m': '</span>'
        };

        // Replace ANSI codes with HTML
        let html = text;
        for (const [ansi, htmlClass] of Object.entries(ansiToClass)) {
            html = html.replaceAll('\u001b' + ansi, htmlClass);
        }
        return html;
    }

    highlightCodeBlocks(content) {
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        return content.replace(codeBlockRegex, (match, lang, code) => {
            const language = lang || 'plaintext';
            const highlighted = hljs.highlight(code.trim(), { language }).value;
            return `<pre><code class="language-${language}">${highlighted}</code></pre>`;
        });
    }

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        console.log('Sending message:', message);
        this.ws.send(JSON.stringify({
            type: "request",
            content: message
        }));
        this.messageInput.value = '';
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;
    }
}

// Initialize WebSocket handler when the page loads
window.addEventListener('load', () => {
    new WebSocketHandler();
});