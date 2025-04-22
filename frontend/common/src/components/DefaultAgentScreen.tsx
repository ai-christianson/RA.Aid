
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { PanelLeft, Plus } from 'lucide-react';
import {
  Button,
  Layout,
} from './ui';
import { SessionDrawer } from './SessionDrawer';
import { SessionList } from './SessionList';
import { TrajectoryPanel } from './TrajectoryPanel';
import { InputSection } from './InputSection';
import { useSessionStore, useClientConfigStore, useTrajectoryStore } from '../store';
import { BackendTrajectory, safeBackendToTrajectory } from '../models/trajectory';
import { WebSocketConnection, WebSocketConfig } from '../websocket/connection';
import logoBlack from '../assets/logo-black-transparent.png';
import logoWhite from '../assets/logo-white-transparent.gif';
import { AgentSession, SessionStatus, safeBackendToAgentSession } from '../models/session';

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

  // Refs and state for autoscroll
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState<boolean>(false);
  const isUserScrolledUpRef = useRef<boolean>(false); // Ref to keep latest value across effects

  // Scroll behaviour threshold (in pixels)
  const BOTTOM_THRESHOLD = 10;     // Considered back at bottom when within this distance / threshold for scroll up


  // WebSocket connection management
  const wsConnectionRef = useRef<WebSocketConnection | null>(null);
  const { host, port } = useClientConfigStore();

  const addOrUpdateTrajectory = useTrajectoryStore((state) => state.addOrUpdateTrajectory);
  const updateSessionStatus = useSessionStore((state) => state.updateSessionStatus);
  const updateSessionDetails = useSessionStore((state) => state.updateSessionDetails); // <-- Get the new action
  const trajectories = useTrajectoryStore((state) => state.trajectories); // Get trajectories for autoscroll effect

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
        // Autoscroll is handled by the useEffect below
      } else {
        console.error('[DefaultAgentScreen] Failed to convert backend trajectory:', backendTrajectory);
      }
    } else if (messageData.type === 'session_update' && messageData.payload) {
      console.log('[DefaultAgentScreen] Received session_update message:', messageData.payload);
      const sessionPayload = messageData.payload as { id: number; status: string };
      if (
        sessionPayload.id && typeof sessionPayload.id === 'number' &&
        sessionPayload.status && ['pending', 'running', 'completed', 'error'].includes(sessionPayload.status)
      ) {
         console.log(`[DefaultAgentScreen] Processing session_update for ${sessionPayload.id} with status ${sessionPayload.status}`)
         updateSessionStatus(sessionPayload.id, sessionPayload.status as SessionStatus);
      } else {
         console.warn("[DefaultAgentScreen] Received invalid session_update payload:", sessionPayload);
      }
    } else if (messageData.type === 'session_details_update' && messageData.payload) { // <-- Handle new type
      console.log('[DefaultAgentScreen] Received session_details_update message:', messageData.payload);
      // Payload should be a BackendSession object
      const backendSession = messageData.payload; // Assuming payload is directly the BackendSession
      const convertedSession = safeBackendToAgentSession(backendSession); // Convert backend format to frontend format

      if (convertedSession) {
          console.log(`[DefaultAgentScreen] Processing session_details_update for ${convertedSession.id}`);
          updateSessionDetails(convertedSession); // Update store with the full session details
      } else {
          console.warn("[DefaultAgentScreen] Received invalid session_details_update payload or conversion failed:", backendSession);
      }
    } else if (messageData.type) {
       console.log(`[DefaultAgentScreen] Received unhandled message type: ${messageData.type}`);
    } else {
        console.warn('[DefaultAgentScreen] Received message without a type:', messageData);
    }
  }, [addOrUpdateTrajectory, updateSessionStatus, updateSessionDetails]); // <-- Add updateSessionDetails to dependencies

  // Establish WebSocket connection on mount
  useEffect(() => {
    if (wsConnectionRef.current) return;

    const url = `ws://${host}:${port}/v1/ws`;
    const config: WebSocketConfig = {
      url,
      onMessage: handleWebSocketMessage,
    };

    console.log(`Attempting WebSocket connection to ${url} with message handler`);
    wsConnectionRef.current = new WebSocketConnection(config);
    wsConnectionRef.current.connect();

    return () => {
      if (wsConnectionRef.current) {
        console.log('Closing WebSocket connection');
        wsConnectionRef.current.close();
        wsConnectionRef.current = null;
      }
    };
  }, [host, port, handleWebSocketMessage]);

  // Get session store data needed for this component
  const {
    sessions,
    selectedSessionId,
    selectSession,
    fetchSessions,
    isLoading,
    error,
    newSession,
    startNewSession, // <-- Get startNewSession
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

  // Keep ref in sync with state
  useEffect(() => {
    isUserScrolledUpRef.current = isUserScrolledUp;
  }, [isUserScrolledUp]);

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

  // Autoscroll Effect: Scrolls down when new trajectories arrive and user hasn't scrolled up
  useEffect(() => {
    const element = scrollContainerRef.current;
    if (!element) return;

    // Autoscroll debug log removed

    if (!isUserScrolledUpRef.current) {
      element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    }
  }, [trajectories]); // Only run when new trajectories arrive

  // Scroll Event Listener Effect: Detects manual scrolling and toggles autoscroll
  useEffect(() => {
    const handleScroll = () => {
      const element = scrollContainerRef.current;
      if (!element) return;

      const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;

      // Scroll debug log removed

      if (distanceFromBottom > BOTTOM_THRESHOLD) {
        if (!isUserScrolledUpRef.current) {
          console.log('[DefaultAgentScreen] User scrolled up – autoscroll disabled');
          isUserScrolledUpRef.current = true;
          setIsUserScrolledUp(true);
        }
      } else {
        if (isUserScrolledUpRef.current) {
          console.log('[DefaultAgentScreen] Back at bottom – autoscroll enabled');
          isUserScrolledUpRef.current = false;
          setIsUserScrolledUp(false);
        }
      }
    };

    const element = scrollContainerRef.current;
    if (element) {
      element.addEventListener('scroll', handleScroll, { passive: true });
      // Scroll listener added log removed
    }

    // Cleanup function
    return () => {
      if (element) {
        element.removeEventListener('scroll', handleScroll);
        // Scroll listener removed log removed
      }
    };
  }, []); // Attach listener once on mount

  // --- Ctrl+Space Shortcut Implementation ---
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.code === 'Space') {
        event.preventDefault(); // Prevent default browser space action
        console.log('Ctrl+Space pressed, starting new session');
        startNewSession();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    console.log('[DefaultAgentScreen] Ctrl+Space listener added');

    // Cleanup function
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      console.log('[DefaultAgentScreen] Ctrl+Space listener removed');
    };
  }, [startNewSession]); // Add startNewSession as dependency


  // Handle session selection - Accepts number
  const handleSessionSelect = (sessionId: number) => {
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

  // --- Determine selected session and completion status ---
  const selectedSession = sessions.find(s => s.id === selectedSessionId);
  const isCompleted = selectedSession?.status === 'completed';
  const sessionName = selectedSession?.name || 'Unknown'; // Rely on the session name from the store

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
        sessions={sessions} // Pass the sessions from the store
        onSelectSession={handleSessionSelect}
        currentSessionId={selectedSessionId}
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
      sessions={sessions} // Pass the sessions from the store
      currentSessionId={selectedSessionId}
      onSelectSession={handleSessionSelect}
      isOpen={isDrawerOpen}
      onClose={() => setIsDrawerOpen(false)}
    />
  );

  // Render main content based on the state
  const mainContent = selectedSessionId !== null ? (
    // Existing session view
    <div className="flex flex-col h-full w-full">
      {/* Assign the ref to the scrollable container */}
      <div ref={scrollContainerRef} className="flex-1 overflow-auto w-full">
        {/* Session title with minimal spacing */}
        <div className="px-6 pt-4 pb-2 border-b border-border/30 sticky top-0 bg-background z-10"> {/* Added sticky positioning and background */}
          <h2 className="text-xl font-medium">{sessionName}</h2> {/* Name comes directly from selectedSession */}
        </div>
        {/* Trajectory panel with consistent spacing */}
        <TrajectoryPanel
          sessionId={selectedSessionId}
          addBottomPadding={true}
          customClassName="px-6 pt-3 pb-4" // Reduced top padding to minimize gap
        />
      </div>
      {isCompleted && (
        <div className="text-center py-4 text-muted-foreground border-t border-border/30"> {/* Added border-t */}
          All done! Press <kbd className="px-1.5 py-0.5 border rounded bg-muted font-mono text-xs">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 border rounded bg-muted font-mono text-xs">Space</kbd> to start a new session.
        </div>
      )}
      {/* Input section for existing session (if needed) */}
      {/* <InputSection onSubmit={handleSubmit} /> */}
    </div>
  ) : newSession ? (
    // New session composition view
    <div className="flex flex-col h-full w-full">
      <div className="flex-1 overflow-auto w-full">
        {/* Session title */}
        <div className="px-6 pt-4 pb-2 border-b border-border/30">
          <h2 className="text-xl font-medium">Create New Session</h2>
        </div>
        <div className="px-6 pt-3 pb-4">
          <p className="text-muted-foreground mb-6">
            Type your message in the input box below to start a new conversation with the agent.
          </p>
          {/* Informational cards */}
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
      {/* Input section for new session */}
      <InputSection
        isNewSession={true}
        isDrawerOpen={isDrawerOpen}
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

    const buttonPosition = "bottom-4";
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

// Note: setupTheme moved to the top for better organization
