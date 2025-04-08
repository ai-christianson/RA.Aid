import React, { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback
import { createPortal } from 'react-dom';
import { PanelLeft, Plus, X, MoreHorizontal } from 'lucide-react';
import {
  Button,
  Layout,
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from './ui';
import { SessionDrawer } from './SessionDrawer';
import { SessionList } from './SessionList';
import { TrajectoryPanel } from './TrajectoryPanel';
import { InputSection } from './InputSection';
import { useSessionStore, useClientConfigStore, useTrajectoryStore } from '../store';
import { Trajectory, safeBackendToTrajectory, BackendTrajectory } from '../models/trajectory'; // Added Trajectory models
import { WebSocketConnection, WebSocketConfig } from '../websocket/connection';
import logoBlack from '../assets/logo-black-transparent.png';
import logoWhite from '../assets/logo-white-transparent.gif';
import { SessionStatus } from '../models/session'; // <-- Import SessionStatus

/**
 * DefaultAgentScreen component
 *
 * Main application screen for displaying agent sessions and their trajectories.
 * Handles state management, responsive design, and UI interactions.
 */
export const DefaultAgentScreen: React.FC = () => {
  // State for drawer open/close
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // State for theme (dark is default)
  const [isDarkTheme, setIsDarkTheme] = useState(true);

  // WebSocket connection management
  const wsConnectionRef = useRef<WebSocketConnection | null>(null);
  const { host, port } = useClientConfigStore();

  const addOrUpdateTrajectory = useTrajectoryStore((state) => state.addOrUpdateTrajectory);
  const updateSessionStatus = useSessionStore((state) => state.updateSessionStatus); // <-- Get action from store

  const handleWebSocketMessage = useCallback((messageData: any) => {
    console.log('[DefaultAgentScreen] handleWebSocketMessage received:', messageData);

    if (typeof messageData !== 'object' || messageData === null) {
      console.warn('[DefaultAgentScreen] Received non-object message:', messageData);
      return;
    }

    if (messageData.type === 'trajectory' && messageData.payload) {
      console.log('[DefaultAgentScreen] Received trajectory message:', messageData.payload);
      const backendTrajectory = messageData.payload as BackendTrajectory;
      const convertedTrajectory = safeBackendToTrajectory(backendTrajectory);

      if (convertedTrajectory) {
        console.log('[DefaultAgentScreen] Converted trajectory, updating store:', convertedTrajectory);
        addOrUpdateTrajectory(convertedTrajectory);
      } else {
        console.error('[DefaultAgentScreen] Failed to convert backend trajectory:', backendTrajectory);
      }
    } else if (messageData.type === 'session_update' && messageData.payload) { // <-- Add handler for session_update
      console.log('[DefaultAgentScreen] Received session_update message:', messageData.payload);
      const sessionPayload = messageData.payload as { id: number; status: string /* other fields */ }; // <-- Change id type to number
      // Basic validation for status before calling store
      if (sessionPayload.id && sessionPayload.status && ['pending', 'running', 'completed', 'error'].includes(sessionPayload.status)) {
         console.log(`[DefaultAgentScreen] Processing session_update for ${sessionPayload.id} with status ${sessionPayload.status}`)
         // Ensure ID is passed as number as expected by the store
         updateSessionStatus(sessionPayload.id, sessionPayload.status as SessionStatus); // <-- Remove String() conversion
      } else {
         console.warn("[DefaultAgentScreen] Received invalid session_update payload:", sessionPayload);
      }
    } else if (messageData.type) {
       console.log(`[DefaultAgentScreen] Received non-trajectory/session_update message type: ${messageData.type}`);
       // Handle other message types here if needed in the future
    } else {
        console.warn('[DefaultAgentScreen] Received message without a type:', messageData);
    }
  }, [addOrUpdateTrajectory, updateSessionStatus]); // <-- Add updateSessionStatus to dependency array

  // Establish WebSocket connection on mount
  useEffect(() => {
    // Prevent multiple connections
    if (wsConnectionRef.current) return;

    const url = `ws://${host}:${port}/v1/ws`;
    const config: WebSocketConfig = {
      url,
      onMessage: handleWebSocketMessage, // Pass the memoized handler
      // Use default heartbeat/reconnection settings from WebSocketConnection
    };

    console.log(`Attempting WebSocket connection to ${url} with message handler`);
    wsConnectionRef.current = new WebSocketConnection(config);
    wsConnectionRef.current.connect();

    // Cleanup function on component unmount
    return () => {
      if (wsConnectionRef.current) {
        console.log('Closing WebSocket connection');
        wsConnectionRef.current.close();
        wsConnectionRef.current = null;
      }
    };
  // }, [host, port, handleWebSocketMessage]); // Reconnect if host/port changes, recreate if handler changes
  // NOTE: Including handleWebSocketMessage might cause excessive reconnects if it's redefined often.
  // Let's only depend on host and port for reconnection, but keep the handler up-to-date via ref or ensure stability.
  // Using useCallback above helps ensure stability if dependencies are correct.
  }, [host, port, handleWebSocketMessage]); // <-- Keep handleWebSocketMessage if stable via useCallback

  // Handle message submission for existing sessions
  const handleSubmit = async (message: string) => {
    if (!selectedSessionId || !message.trim()) return;

    try {
      // TODO: Implement reply to existing session via WebSocket
      // This will need a specific message format or API call
      console.log('Message submitted to existing session:', message, 'sessionId:', selectedSessionId);
      // Example: wsConnectionRef.current?.send(JSON.stringify({ type: 'reply', sessionId: selectedSessionId, message }));

      return true; // Success (assuming optimistic update or WS confirmation)
    } catch (error) {
      console.error("Error handling message submission:", error);
      return false; // Failure
    }
  };

  // Get session store data
  const {
    sessions,
    selectedSessionId,
    selectSession,
    fetchSessions,
    isLoading,
    error,
    newSession,
    startNewSession,
    cancelNewSession,
    updateNewSessionMessage, // Likely remove if using InputSection state
    submitNewSession // Likely trigger via WebSocket in InputSection
    // updateSessionStatus is already obtained above
  } = useSessionStore();

  // Fetch initial sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Set up theme on component mount
  useEffect(() => {
    const isDark = setupTheme();
    setIsDarkTheme(isDark);
  }, []);

  // Close drawer when window resizes to desktop width
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768 && isDrawerOpen) {
        setIsDrawerOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isDrawerOpen]);

  // Handle session selection
  const handleSessionSelect = (sessionId: string) => {
    selectSession(sessionId);
    setIsDrawerOpen(false); // Close drawer on selection (mobile)
  };

  // Toggle theme function
  const toggleTheme = () => {
    const newIsDark = !isDarkTheme;
    setIsDarkTheme(newIsDark);
    if (newIsDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', newIsDark ? 'dark' : 'light');
  };

  // Get selected session name
  const selectedSession = sessions.find(s => s.id === selectedSessionId);
  const sessionName = selectedSession?.name || 'Unknown';

  // Render header content
  const headerContent = (
    <div className="w-full flex items-center justify-between h-full px-4">
      <div className="flex-initial">
        <img
          src={isDarkTheme ? logoWhite : logoBlack}
          alt="RA.Aid Logo"
          className="h-8"
        />
      </div>
      <div className="flex-initial ml-auto">
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label={isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDarkTheme ? (
            // Sun icon
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5" /><line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" /><line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" /><line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" /><line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" /></svg>
          ) : (
            // Moon icon
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
          )}
        </Button>
      </div>
    </div>
  );

  // Sidebar content with sessions list
  const sidebarContent = (
    <div className="h-full flex flex-col p-4">
      <SessionList
        sessions={sessions}
        onSelectSession={handleSessionSelect}
        currentSessionId={selectedSessionId || undefined}
        className="flex-1 pr-1 -mr-1"
        isLoading={isLoading}
        error={error}
        onRefresh={fetchSessions}
      />
    </div>
  );

  // Render drawer
  const drawerContent = (
    <SessionDrawer
      sessions={sessions}
      currentSessionId={selectedSessionId || undefined}
      onSelectSession={handleSessionSelect}
      isOpen={isDrawerOpen}
      onClose={() => setIsDrawerOpen(false)}
    />
  );

  // Render main content based on the state
  const mainContent = selectedSessionId ? (
    // Existing session view
    <div className="flex flex-col h-full w-full">
      <div className="flex-1 overflow-auto w-full">
        {/* Session title with minimal spacing */}
        <div className="px-6 pt-4 pb-2 border-b border-border/30">
          <h2 className="text-xl font-medium">{sessionName}</h2>
        </div>
        {/* Trajectory panel with consistent spacing */}
        <TrajectoryPanel
          sessionId={selectedSessionId}
          addBottomPadding={true}
          customClassName="px-6 pt-3 pb-4" // Reduced top padding to minimize gap
        />
      </div>
    </div>
  ) : newSession ? (
    // New session composition view
    <div className="flex flex-col h-full w-full">
      <div className="flex-1 overflow-auto w-full">
        {/* Session title with minimal spacing */}
        <div className="px-6 pt-4 pb-2 border-b border-border/30">
          <h2 className="text-xl font-medium">Create New Session</h2>
        </div>
        <div className="px-6 pt-3 pb-4">
          <p className="text-muted-foreground mb-6">
            Type your message in the input box below to start a new conversation with the agent.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div className="p-4 rounded-md border border-border bg-background/50">
              <h4 className="text-sm font-medium mb-2">Research Mode</h4>
              <p className="text-xs text-muted-foreground">
                The agent will gather information about your request and provide a summary
                without implementing any solutions.
              </p>
            </div>
            <div className="p-4 rounded-md border border-border bg-background/50">
              <h4 className="text-sm font-medium mb-2">Implementation Mode</h4>
              <p className="text-xs text-muted-foreground">
                The agent will analyze your request, create a plan, and implement a solution
                based on your requirements.
              </p>
            </div>
          </div>
        </div>
      </div>
      <InputSection
        isNewSession={true}
        isDrawerOpen={isDrawerOpen}
        // Pass WebSocket sending capability if InputSection needs it for new sessions
        // sendWebSocketMessage={(msg) => wsConnectionRef.current?.send(JSON.stringify(msg))}
      />
    </div>
  ) : (
    // No session selected view
    <div className="flex items-center justify-center h-full">
      <p className="text-muted-foreground">Select a session or start a new one</p>
    </div>
  );

  // Floating action button component
  const FloatingActionButton = ({ onClick }: { onClick: () => void }) => {
    const [mounted, setMounted] = useState(false);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
      setMounted(true);
      const checkMobile = () => setIsMobile(window.innerWidth < 768);
      checkMobile();
      window.addEventListener('resize', checkMobile);
      return () => {
        setMounted(false);
        window.removeEventListener('resize', checkMobile);
      };
    }, []);

    // Removed isInputVisible calculation
    const buttonPosition = "bottom-4"; // Always position at bottom-4 when rendered
    const buttonStyle = "p-2 rounded-md shadow-md bg-zinc-800/90 hover:bg-zinc-700 text-zinc-100 flex items-center justify-center border border-zinc-700 dark:border-zinc-600";

    if (!mounted || newSession) return null; // Don't show if creating new session

    return createPortal(
      <div className={`fixed ${buttonPosition} right-4 z-[80] flex space-x-2`}>
        {isMobile && (
          <Button variant="default" size="sm" onClick={onClick} aria-label="Toggle sessions panel" className={buttonStyle}>
            <PanelLeft className="h-5 w-5" />
          </Button>
        )}
        <Button variant="default" size="sm" onClick={startNewSession} aria-label="Create new session" className={buttonStyle}>
          <Plus className="h-5 w-5" />
        </Button>
      </div>,
      document.body
    );
  };

  return (
    <>
      <Layout
        header={headerContent}
        sidebar={sidebarContent}
        drawer={drawerContent}
      >
        {mainContent}
      </Layout>
      <FloatingActionButton onClick={() => setIsDrawerOpen(true)} />
    </>
  );
};

// Helper function for theme setup
const setupTheme = () => {
  const storedTheme = localStorage.getItem('theme');
  const isDark = storedTheme ? storedTheme === 'dark' : true; // Default to dark
  if (isDark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
  return isDark;
};
