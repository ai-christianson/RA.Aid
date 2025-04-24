import asyncio
import os

import threading
import logging

logger = logging.getLogger(__name__)
from typing import Any, List

from langchain_core.tools import BaseTool
from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter


class MCPUseClientSync:
    """Synchronous wrapper around ``mcp_use.MCPClient``.

    ``mcp_use`` is fully async; RA.Aid’s tool-pipeline expects plain
    synchronous ``langchain_core.tools.BaseTool`` instances.  This wrapper
    starts a background event-loop, initializes the MCP client and converts
    every server tool to a ``StructuredTool`` via the official
    Note:
        Using MCP servers often requires external dependencies managed by that
        server. For example, using the Context7 server requires ``npx`` (Node.js)
        to be available in the environment to run ``@upstash/context7-mcp``.

    ``LangChainAdapter``.
    """

    def __init__(self, config: str | dict[str, Any]):
        """Create the client.

        Args:
            config: Either a path to a JSON config file **or** a config dict
                    compatible with ``MCPClient``.
        """
        self.loop = asyncio.new_event_loop()
        self._loop_started = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        if not self._loop_started.wait(timeout=10): # Wait up to 10s
            # Log and re-raise to ensure __main__ knows init failed
            err_msg = "Background event loop failed to start within 10s"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        self._client: MCPClient | None = None
        self._tools: List[BaseTool] = []
        self._active_server_names: List[str] = [] # Track successfully initialized
        self._closed = False

        fut = asyncio.run_coroutine_threadsafe(
            self._setup_client(config), self.loop
        )
        fut.result()  # propagate any exceptions synchronously

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def get_tools_sync(self) -> List[BaseTool]:
        """Return the converted LangChain tools (synchronous)."""
        return self._tools

    def get_active_server_names(self) -> List[str]:
        """Return the names of MCP servers that initialized successfully."""
        return self._active_server_names

    def close(self) -> None:
        """Close all MCP sessions and stop the event-loop. Idempotent."""
        if getattr(self, "_closed", False):
            return # Already closed

        logger.debug("Closing MCPUseClientSync...")
        if self._client is not None:
            try:
                # Ensure the loop is running before scheduling
                if self.loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self._client.close_all_sessions(), self.loop
                    )
                    # Wait with a timeout, but don't block indefinitely if loop is stuck
                    fut.result(timeout=10) 
                else:
                    logger.warning("MCP client loop not running during close.")
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for MCP client sessions to close.")
            except Exception as e:
                logger.error(f"Error closing MCP client sessions: {e}")

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        self.thread.join(timeout=10) # Wait up to 10s for thread
        if self.thread.is_alive():
             logger.warning("MCP client background thread did not exit cleanly.")

        self._closed = True
        logger.debug("MCPUseClientSync closed.")

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._loop_started.set() # Signal that the loop is ready
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
            logger.debug("MCP client event loop closed.")


    async def _setup_client(self, config: str | dict[str, Any]):
        """Initialize client, sessions, tools, and track active servers."""
        try:
            if isinstance(config, str):
                client = MCPClient.from_config_file(config)
            else:
                client = MCPClient.from_dict(config)
            self._client = client
        except Exception as e:
            logger.error(f"MCPClient creation failed: {e}", exc_info=True)
            # Re-raise to be caught by the caller in __main__
            raise

        # Create sessions individually to catch errors per server
        server_names = self._client.get_server_names()
        active_sessions = {}
        initialization_errors = {}

        for name in server_names:
            try:
                logger.info(f"Initializing MCP session for server: {name}")
                session = await self._client.create_session(name, auto_initialize=True)
                active_sessions[name] = session
                self._active_server_names.append(name)
                logger.info(f"MCP session for '{name}' initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize MCP session for server '{name}': {e}", exc_info=True)
                initialization_errors[name] = str(e)
                # Hint for tree-sitter common issues
                if name == "tree_sitter":
                     logger.error("Tree-sitter server failed. This might be due to missing C/C++ build tools or Tree-sitter parser build errors.")
        
        if initialization_errors:
             logger.warning(f"Some MCP servers failed to initialize: {list(initialization_errors.keys())}")
             # Continue with successfully initialized servers

        # Get tools ONLY from active sessions
        adapter = LangChainAdapter()
        all_tools = []
        for name, session in active_sessions.items():
            try:
                session_tools = await adapter.create_tools_from_session(session)
                all_tools.extend(session_tools)
                logger.debug(f"Loaded {len(session_tools)} tools from active server '{name}'")
            except Exception as e:
                 logger.error(f"Failed to get tools from active MCP session '{name}': {e}", exc_info=True)
        
        self._tools = all_tools
        logger.info(f"Total MCP tools loaded from active servers: {len(self._tools)}")

        # Attempt to auto-register the current project with tree-sitter if it's active
        if "tree_sitter" in self._active_server_names:
            await self._auto_register_tree_sitter(self._client)
        else:
             logger.debug("Skipping tree-sitter auto-registration as server is not active.")

    async def _auto_register_tree_sitter(self, client: MCPClient):
        """Automatically register the current project if tree-sitter server is active."""
        # Check if tree_sitter session exists and is active first
        try:
            ts_session = client.get_session("tree_sitter")
            if not ts_session or not ts_session.connector:
                logger.debug("Tree-sitter MCP session not found or inactive.")
                return
        except ValueError:
            logger.debug("Tree-sitter MCP server ('tree_sitter') not configured.")
            return
        except Exception as e:
             logger.error(f"Error getting tree-sitter session: {e}", exc_info=True)
             return

        # Now check if the tool exists among loaded tools
        register_tool = next((t for t in self._tools if t.name == "register_project_tool"), None)
        if not register_tool:
            logger.warning("Tree-sitter server is active, but register_project_tool is missing.")
            return

        # Proceed with registration
        try:
            project_path = os.getcwd()
            project_name = os.path.basename(project_path)
            logger.info(f"Auto-registering project '{project_name}' at '{project_path}' with tree-sitter server.")
            
            # Call the tool via its coroutine
            await register_tool.acall(path=project_path, name=project_name)
            logger.info(f"Project '{project_name}' registered with tree-sitter.")

        except Exception as e:
            logger.error(f"Failed to auto-register project with tree-sitter during tool call: {e}", exc_info=True)

