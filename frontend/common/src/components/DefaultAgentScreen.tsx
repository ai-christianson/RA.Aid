import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { PanelLeft, Plus, X } from 'lucide-react';
import { 
  Button,
  Layout
} from './ui';
import { SessionDrawer } from './SessionDrawer';
import { SessionList } from './SessionList';
import { TrajectoryPanel } from './TrajectoryPanel';
import { InputSection } from './InputSection';
import { useSessionStore } from '../store';
import logoBlack from '../assets/logo-black-transparent.png';
import logoWhite from '../assets/logo-white-transparent.gif';

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
  
  // Handle message submission for existing sessions
  const handleSubmit = async (message: string) => {
    if (!selectedSessionId || !message.trim()) return;
    
    try {
      // TODO: Implement reply to existing session
      // This will need a different endpoint for continuing conversations
      console.log('Message submitted to existing session:', message, 'sessionId:', selectedSessionId);
      
      // Refresh sessions to get updated data
      await fetchSessions();
      
      return true; // Success
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
    updateNewSessionMessage,
    submitNewSession
  } = useSessionStore();
  
  // Fetch sessions on component mount
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
      // Check if we're at desktop size (corresponds to md: breakpoint in Tailwind)
      if (window.innerWidth >= 768 && isDrawerOpen) {
        setIsDrawerOpen(false);
      }
    };

    // Add event listener
    window.addEventListener('resize', handleResize);
    
    // Clean up event listener on component unmount
    return () => window.removeEventListener('resize', handleResize);
  }, [isDrawerOpen]);
  
  // Handle session selection
  const handleSessionSelect = (sessionId: string) => {
    selectSession(sessionId);
    setIsDrawerOpen(false); // Close drawer on selection (mobile)
  };
  
  // Handle new session message submit - no longer needed as we handle this directly in the form
  // This function is kept for compatibility but is no longer used in the UI
  const handleNewSessionSubmit = (message: string) => {
    if (!message.trim()) return;
    
    updateNewSessionMessage(message);
    submitNewSession();
  };
  
  // Toggle theme function
  const toggleTheme = () => {
    const newIsDark = !isDarkTheme;
    setIsDarkTheme(newIsDark);
    
    // Update document element class
    if (newIsDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    
    // Save to localStorage
    localStorage.setItem('theme', newIsDark ? 'dark' : 'light');
  };
  
  // Render header content
  const headerContent = (
    <div className="w-full flex items-center justify-between h-full px-4">
      <div className="flex-initial">
        {/* Use the appropriate logo based on theme */}
        <img 
          src={isDarkTheme ? logoWhite : logoBlack} 
          alt="RA.Aid Logo" 
          className="h-8"
        />
      </div>
      
      <div className="flex-initial ml-auto">
        {/* Theme toggle button */}
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={toggleTheme}
          aria-label={isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDarkTheme ? (
            // Sun icon for light mode toggle
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="20" 
              height="20" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="5" />
              <line x1="12" y1="1" x2="12" y2="3" />
              <line x1="12" y1="21" x2="12" y2="23" />
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
              <line x1="1" y1="12" x2="3" y2="12" />
              <line x1="21" y1="12" x2="23" y2="12" />
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
            </svg>
          ) : (
            // Moon icon for dark mode toggle
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="20" 
              height="20" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </Button>
      </div>
    </div>
  );

  // Sidebar content with sessions list
  const sidebarContent = (
    <div className="h-full flex flex-col px-4 py-3">
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
      <div className="flex-1 overflow-auto w-full px-4">
        <h2 className="text-xl font-semibold mb-4">
          Session: {sessions.find(s => s.id === selectedSessionId)?.name || 'Unknown'}
        </h2>
        <TrajectoryPanel 
          sessionId={selectedSessionId}
          addBottomPadding={true}
        />
      </div>
      <InputSection 
        sessionId={parseInt(selectedSessionId)}
        onSubmit={handleSubmit}
        isDrawerOpen={isDrawerOpen}
      />
    </div>
  ) : newSession ? (
    // New session composition view
    <div className="flex flex-col h-full w-full">
      <div className="flex-1 overflow-auto w-full px-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Create New Session</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={cancelNewSession}
            aria-label="Cancel new session"
            disabled={newSession.isSubmitting}
          >
            <X className="h-4 w-4 mr-1" />
            Cancel
          </Button>
        </div>
        
        {newSession.error && (
          <div className="bg-destructive/10 text-destructive p-3 rounded-md mb-4">
            {newSession.error}
          </div>
        )}
        
        <p className="text-muted-foreground mb-4">
          Type your message to start a new conversation with the agent.
        </p>
        
        <div className="w-full mb-8">
          <textarea
            value={newSession.message}
            onChange={(e) => updateNewSessionMessage(e.target.value)}
            placeholder="What would you like help with today?"
            className="flex w-full resize-none rounded-lg border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            rows={6}
            disabled={newSession.isSubmitting}
          />
          
          <div className="flex justify-end space-x-2 mt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => submitNewSession(true)}
              disabled={!newSession.message.trim() || newSession.isSubmitting}
            >
              {newSession.isSubmitting ? (
                <span className="flex items-center">
                  <span className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Processing...
                </span>
              ) : "Research Only"}
            </Button>
            <Button
              type="button"
              onClick={() => submitNewSession()}
              disabled={!newSession.message.trim() || newSession.isSubmitting}
              className="min-w-[100px]"
            >
              {newSession.isSubmitting ? (
                <span className="flex items-center">
                  <span className="h-4 w-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Creating...
                </span>
              ) : "Create Session"}
            </Button>
          </div>
        </div>
      </div>
      {/* Removed InputSection from new session view to avoid duplicate inputs */}
    </div>
  ) : (
    // No session selected view
    <div className="flex items-center justify-center h-full">
      <p className="text-muted-foreground">Select a session to view details</p>
    </div>
  );

  // Floating action button component that uses Portal to render at document body level
  const FloatingActionButton = ({ onClick }: { onClick: () => void }) => {
    // Only render the portal on the client side, not during SSR
    const [mounted, setMounted] = useState(false);
    const [isMobile, setIsMobile] = useState(false);
    
    useEffect(() => {
      setMounted(true);
      
      const checkMobile = () => {
        setIsMobile(window.innerWidth < 768);
      };
      
      // Initial check
      checkMobile();
      
      // Add event listener for resize
      window.addEventListener('resize', checkMobile);
      
      // Cleanup
      return () => {
        setMounted(false);
        window.removeEventListener('resize', checkMobile);
      };
    }, []);
    
    // Determine if the input section should be visible
    const isInputVisible = (selectedSessionId || (newSession && !newSession.isSubmitting)) 
                           && !(isMobile && isDrawerOpen);
    
    // Button position logic:
    // - When input is visible: position just above the input (104px from bottom)
    // - When input is not visible: position at bottom of screen
    const buttonPosition = isInputVisible ? "bottom-[104px]" : "bottom-4";
    
    const buttonStyle = "p-2 rounded-md shadow-md bg-zinc-800/90 hover:bg-zinc-700 text-zinc-100 flex items-center justify-center border border-zinc-700 dark:border-zinc-600";
    
    if (!mounted) return null;

    return createPortal(
      <div className={`fixed ${buttonPosition} right-4 z-50 flex space-x-2`}>
        {/* Panel toggle button - only shown on mobile */}
        {isMobile && (
          <Button
            variant="default"
            size="sm"
            onClick={onClick}
            aria-label="Toggle sessions panel"
            className={buttonStyle}
          >
            <PanelLeft className="h-5 w-5" />
          </Button>
        )}
        {/* New session button - disabled when a new session is already being submitted */}
        <Button
          variant="default"
          size="sm"
          onClick={startNewSession}
          aria-label="Create new session"
          disabled={newSession?.isSubmitting}
          className={buttonStyle}
        >
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
  // Check if theme preference is stored in localStorage
  const storedTheme = localStorage.getItem('theme');
  
  // Default to dark mode unless explicitly set to light
  const isDark = storedTheme ? storedTheme === 'dark' : true;
  
  // Apply theme to document
  if (isDark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
  
  return isDark;
};