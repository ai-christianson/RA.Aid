/**
 * Represents the connection state of the WebSocket.
 */
export enum WebSocketState {
  CONNECTING = 'CONNECTING',
  OPEN = 'OPEN',
  CLOSING = 'CLOSING',
  CLOSED = 'CLOSED',
  RECONNECTING = 'RECONNECTING', // Used in Task 3
}

/**
 * Configuration for the WebSocketConnection.
 */
export interface WebSocketConfig {
  url: string;

  // Reconnection options (Task 3)
  autoReconnect?: boolean; // Default: true
  reconnectIntervalMs?: number; // Initial delay, e.g., 1000ms
  maxReconnectIntervalMs?: number; // Maximum delay, e.g., 30000ms
  reconnectDecay?: number; // Backoff factor, e.g., 1.5
  maxReconnectAttempts?: number | null; // e.g., 5, null for infinite

  // Heartbeat options (Task 2)
  heartbeatIntervalMs?: number; // e.g., 30000ms (30 seconds)
  heartbeatTimeoutMs?: number; // e.g., 5000ms (5 seconds)
  pingMessage?: string | (() => string); // Default: 'ping'
  pongMessage?: string; // Default: 'pong'

  // Custom message handler (Task 1 - Frontend Update)
  onMessage?: (data: any) => void;
}

/**
 * Manages a WebSocket connection, including state, heartbeat, and automatic reconnection.
 */
export class WebSocketConnection {
  private ws: WebSocket | null = null;
  private state: WebSocketState = WebSocketState.CLOSED;
  private config: WebSocketConfig;
  private _onMessage: ((data: any) => void) | undefined; // Task 1: Add handler member

  // Heartbeat properties (Task 2)
  private heartbeatIntervalTimer: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeoutTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly heartbeatIntervalMs: number | undefined;
  private readonly heartbeatTimeoutMs: number;
  private readonly pingMessage: string | (() => string);
  private readonly pongMessage: string;

  // Reconnection properties (Task 3)
  private reconnectAttempts: number = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionalClose: boolean = false;
  private readonly autoReconnect: boolean;
  private readonly reconnectIntervalMs: number;
  private readonly maxReconnectIntervalMs: number;
  private readonly reconnectDecay: number;
  private readonly maxReconnectAttempts: number | null;


  constructor(config: WebSocketConfig) {
    this.config = config;
    this._onMessage = config.onMessage; // Task 1: Store handler

    // Initialize heartbeat config with defaults (Task 2)
    this.heartbeatIntervalMs = config.heartbeatIntervalMs;
    this.heartbeatTimeoutMs = config.heartbeatTimeoutMs ?? 5000;
    this.pingMessage = config.pingMessage ?? 'ping';
    this.pongMessage = config.pongMessage ?? 'pong';

    // Initialize reconnection config with defaults (Task 3)
    this.autoReconnect = config.autoReconnect ?? true;
    this.reconnectIntervalMs = config.reconnectIntervalMs ?? 1000;
    this.maxReconnectIntervalMs = config.maxReconnectIntervalMs ?? 30000;
    this.reconnectDecay = config.reconnectDecay ?? 1.5;
    this.maxReconnectAttempts = config.maxReconnectAttempts === undefined ? null : config.maxReconnectAttempts; // Allow null for infinite

    console.log('WebSocketConnection initialized with config:', {
      ...config,
      // Log resolved heartbeat defaults
      heartbeatIntervalMs: this.heartbeatIntervalMs,
      heartbeatTimeoutMs: this.heartbeatTimeoutMs,
      pingMessage: typeof this.pingMessage === 'function' ? '[Function]' : this.pingMessage,
      pongMessage: this.pongMessage,
      // Log resolved reconnect defaults
      autoReconnect: this.autoReconnect,
      reconnectIntervalMs: this.reconnectIntervalMs,
      maxReconnectIntervalMs: this.maxReconnectIntervalMs,
      reconnectDecay: this.reconnectDecay,
      maxReconnectAttempts: this.maxReconnectAttempts,
      // Indicate if handler is present
      onMessage: this._onMessage ? '[Function]' : undefined,
    });
  }

  /**
   * Returns the current connection state.
   */
  public getState(): WebSocketState {
    return this.state;
  }

  /**
   * Initiates the WebSocket connection or reconnection attempt.
   */
  public connect(): void {
    // Don't attempt to connect if already open or connecting
    if (this.state === WebSocketState.OPEN || this.state === WebSocketState.CONNECTING) {
      console.log(`[WebSocket] Connect called but state is already ${this.state}. Aborting.`);
      return;
    }

    // If this is *not* a reconnection attempt, reset flags
    if (this.state !== WebSocketState.RECONNECTING) {
        this.intentionalClose = false;
        this.reconnectAttempts = 0;
    }

    this.state = WebSocketState.CONNECTING;
    console.log(`[WebSocket] Attempting to connect to ${this.config.url}... (Attempt: ${this.reconnectAttempts + 1})`);

    try {
      // Clear any lingering timers *before* creating a new connection
      this._clearTimers();
      this.ws = new WebSocket(this.config.url);
      this.ws.onopen = this._handleOpen.bind(this);
      this.ws.onmessage = this._handleMessage.bind(this);
      this.ws.onerror = this._handleError.bind(this);
      this.ws.onclose = this._handleClose.bind(this);
    } catch (error) {
      console.error('[WebSocket] Instantiation failed:', error);
      this.state = WebSocketState.CLOSED; // Ensure state is CLOSED on instant failure
      // Trigger reconnection logic directly if instantiation fails and autoReconnect is enabled
      this._handleClose(new CloseEvent('error', { reason: 'WebSocket instantiation failed' }));
    }
  }

  /**
   * Closes the WebSocket connection intentionally and prevents automatic reconnection.
   * @param code Optional WebSocket close code.
   * @param reason Optional reason string.
   */
  public close(code?: number, reason?: string): void {
    if (this.state === WebSocketState.CLOSING || this.state === WebSocketState.CLOSED) {
       console.log(`[WebSocket] Close called but state is already ${this.state}. Aborting.`);
      return;
    }

    console.log(`[WebSocket] Intentional close requested.`);
    this.intentionalClose = true; // Mark as intentional *before* closing
    this._clearTimers(); // Stop heartbeats AND any pending reconnection attempts

    if (!this.ws) {
        console.log(`[WebSocket] Close called but no WebSocket instance exists. Setting state to CLOSED.`);
        this.state = WebSocketState.CLOSED;
        return;
    }

    this.state = WebSocketState.CLOSING;
    console.log(`[WebSocket] Closing connection with code=${code}, reason=${reason}`);
    try {
        // Use default code 1000 (Normal Closure) if none provided
        this.ws.close(code ?? 1000, reason);
    } catch (error) {
        console.error('[WebSocket] Error during close:', error);
        // Force state to CLOSED just in case 'onclose' never fires.
        this.state = WebSocketState.CLOSED;
        this.ws = null; // Ensure instance is released
    }
  }

  // --- Private Event Handlers ---


  private _handleOpen(event: Event): void {
    this.state = WebSocketState.OPEN;
    this.reconnectAttempts = 0; // Reset reconnect attempts on successful connection (Task 3)
    console.log('[WebSocket] Connection established successfully.');
    this._startHeartbeat(); // Start heartbeat on successful connection (Task 2)
  }

  private _handleMessage(event: MessageEvent): void {
    const messageData = event.data;

    // Handle Pong messages (Task 2)
    if (messageData === this.pongMessage) {
        //console.log('[WebSocket] Received Pong.'); // Can be noisy
        this._clearHeartbeatTimeout(); // Reset the timeout timer upon receiving a pong
        return; // Don't pass pong messages to the application's message handler
    }

    // Task 1: Call registered message handler or log if none exists
    if (this._onMessage) {
        try {
            // Attempt to parse if it looks like JSON
            let parsedData = messageData;
            if (typeof messageData === 'string') {
                try {
                    parsedData = JSON.parse(messageData);
                } catch (e) {
                    // Keep as string if parsing fails, maybe log warning
                    // console.warn('[WebSocket] Received non-JSON string message:', messageData);
                }
            }
            this._onMessage(parsedData); // Pass potentially parsed data
        } catch (error) {
            console.error('[WebSocket] Error in onMessage handler:', error);
        }
    } else {
        // Log if no handler is registered and it's not a pong
        console.log('[WebSocket] Received unhandled message:', messageData);
    }
  }

  private _handleError(event: Event): void {
    // Log the error, but rely on onclose for state changes and reconnection logic
    console.error('[WebSocket] Error:', event);
  }

  private _handleClose(event: CloseEvent): void {
    const previousState = this.state;
    // Only change state if not already intentionally closing or closed
    if (this.state !== WebSocketState.CLOSING && this.state !== WebSocketState.CLOSED) {
        this.state = WebSocketState.CLOSED; // Set to closed initially before attempting reconnect
    }

    // Always clear timers when the connection is definitively closed
    this._clearTimers();

    console.log(
      `[WebSocket] Connection closed. Code: ${event.code}, Reason: "${event.reason}", Was Clean: ${event.wasClean}, Previous State: ${previousState}, Intentional: ${this.intentionalClose}`
    );

    // Invalidate the WebSocket instance and remove listeners
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      this.ws = null;
    }

    // Initiate Reconnection Logic (Task 3)
    if (this.intentionalClose) {
        console.log('[WebSocket] Intentional close, not reconnecting.');
        this.state = WebSocketState.CLOSED; // Ensure final state is CLOSED
        return;
    }

    if (!this.autoReconnect) {
        console.log('[WebSocket] Auto-reconnect disabled.');
         this.state = WebSocketState.CLOSED; // Ensure final state is CLOSED
        return;
    }

    if (this.maxReconnectAttempts !== null && this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.log(`[WebSocket] Max reconnect attempts (${this.maxReconnectAttempts}) reached. Stopping.`);
        this.state = WebSocketState.CLOSED; // Ensure final state is CLOSED
        return;
    }

    // Calculate delay with exponential backoff and jitter
    let delay = this.reconnectIntervalMs * Math.pow(this.reconnectDecay, this.reconnectAttempts);
    delay = Math.min(delay, this.maxReconnectIntervalMs);
    const jitter = delay * 0.2 * (Math.random() - 0.5); // +/- 10% jitter
    const finalDelay = Math.max(0, delay + jitter); // Ensure delay is not negative

    this.reconnectAttempts++;
    this.state = WebSocketState.RECONNECTING; // Set state *before* scheduling timeout

    console.log(`[WebSocket] Reconnecting: Attempt ${this.reconnectAttempts}${this.maxReconnectAttempts ? `/${this.maxReconnectAttempts}` : ''}. Retrying in ${finalDelay.toFixed(0)}ms...`);

    this.reconnectTimer = setTimeout(() => {
        // Check state again before connecting, in case close() was called during the timeout
        if (this.state === WebSocketState.RECONNECTING) {
            this.connect();
        } else {
            console.log(`[WebSocket] Reconnect timer fired, but state changed to ${this.state}. Aborting reconnect.`);
        }
    }, finalDelay);
  }

  // --- Private Heartbeat Methods (Task 2) ---


  private _startHeartbeat(): void {
    this._clearHeartbeatInterval(); // Clear existing interval timer just in case

    if (typeof this.heartbeatIntervalMs === 'number' && this.heartbeatIntervalMs > 0) {
      //console.log(`[WebSocket] Starting heartbeat interval (${this.heartbeatIntervalMs}ms).`);
      this.heartbeatIntervalTimer = setInterval(() => {
        this._sendPing();
      }, this.heartbeatIntervalMs);
    } else {
        //console.log('[WebSocket] Heartbeat interval not configured or invalid.');
    }
  }

  private _sendPing(): void {
    if (this.state !== WebSocketState.OPEN) {
      // Stop sending pings if the connection is no longer open
      this._clearHeartbeatInterval();
      return;
    }

    // Clear previous timeout timer before setting a new one
    this._clearHeartbeatTimeout();

    this.heartbeatTimeoutTimer = setTimeout(() => {
      this._handleHeartbeatTimeout();
    }, this.heartbeatTimeoutMs);

    try {
        const message = typeof this.pingMessage === 'function' ? this.pingMessage() : this.pingMessage;
        this.ws?.send(message);
    } catch (error) {
        console.error('[WebSocket] Failed to send ping:', error);
        // The timeout mechanism will handle the lack of pong
    }
  }

  private _handleHeartbeatTimeout(): void {
    if (this.state !== WebSocketState.OPEN) {
        return;
    }
    console.warn(`[WebSocket] Heartbeat timeout after ${this.heartbeatTimeoutMs}ms. No Pong received. Closing connection.`);
    this.heartbeatTimeoutTimer = null; // Ensure timer ID is cleared
    // Force close the connection. The `_handleClose` handler will manage state and cleanup.
    this.ws?.close(1001, "Heartbeat timeout");
  }

  private _clearHeartbeatInterval(): void {
    if (this.heartbeatIntervalTimer !== null) {
      clearInterval(this.heartbeatIntervalTimer);
      this.heartbeatIntervalTimer = null;
    }
  }

  private _clearHeartbeatTimeout(): void {
      if (this.heartbeatTimeoutTimer !== null) {
        clearTimeout(this.heartbeatTimeoutTimer);
        this.heartbeatTimeoutTimer = null;
      }
  }

  // Centralized timer clearing
  private _clearTimers(): void {
    this._clearHeartbeatInterval();
    this._clearHeartbeatTimeout();

    // Clear reconnection timer (Task 3)
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
