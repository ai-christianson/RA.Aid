# Web UI

RA.Aid includes a modern web-based user interface that allows you to interact with the agent through your browser. This interface offers a convenient alternative to the command-line experience, with real-time streaming of agent output and an intuitive chat-like interaction.

## Launching the Web UI

To start RA.Aid with the web interface, use the `--server` flag:

```bash
# Start with default settings (0.0.0.0:1818)
ra-aid --server
```

This will launch the web server and provide a URL you can open in your browser.

## Accessing the Web UI

By default, the web UI is accessible at:

```
http://localhost:1818
```

### Customizing Host and Port

You can customize the host and port using the `--server-host` and `--server-port` flags:

```bash
# Specify custom host and port
ra-aid --server --server-host 127.0.0.1 --server-port 3000
```

Command line options for the web interface:
- `--server`: Launch the server with web interface
- `--server-host`: Host to listen on (default: 0.0.0.0)
- `--server-port`: Port to listen on (default: 1818)

## Features

The web interface provides a modern, intuitive experience with the following features:

- **Beautiful Dark-Themed Interface**: A clean, modern design optimized for code and long-form text
- **Real-Time Streaming**: Watch the agent's responses, tool executions, and thought processes as they happen
- **Session Management**: View and manage multiple agent sessions
- **Request History**: Browse through your previous requests and quickly resubmit them
- **Trajectory Visualization**: See the agent's reasoning process and actions in real-time
- **Responsive Design**: Works on all devices, from desktop to mobile
- **Automatic Reconnection**: Seamlessly reconnects if your connection drops

## Using the Web UI

After opening the web UI in your browser, you'll see:

1. **Left Sidebar**: Shows your session history and provides navigation
2. **Main Content Area**: Displays the agent's output in real-time
3. **New Session Button**: Click the new session button to start a new session

To use the web UI:

1. Type your task or query in the input box at the bottom of the new session screen
2. Click the start session button
3. Watch as RA.Aid processes your request in real-time
4. Review the output and interact further as needed

## Troubleshooting

- If you can't access the web UI, ensure the port isn't blocked by a firewall
- If you need to access the UI from another machine, use `--server-host 0.0.0.0` and access via the server's IP address
- For SSL/TLS support, you'll need to configure a reverse proxy like Nginx in front of the RA.Aid server