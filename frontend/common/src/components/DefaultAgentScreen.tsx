import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { PanelLeft } from 'lucide-react';
import { 
  Button,
  Layout
} from './ui';
import { SessionDrawer } from './SessionDrawer';
import { SessionList } from './SessionList';
import { TrajectoryPanel } from './TrajectoryPanel'; // Replace TimelineFeed import
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
  
  // Get session store data
  const { 
    sessions, 
    selectedSessionId, 
    selectSession, 
    fetchSessions,
    isLoading,
    error
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

  // Render main content with TrajectoryPanel instead of TimelineFeed
  const mainContent = (
    selectedSessionId ? (
      <>
        <h2 className="text-xl font-semibold mb-4">
          Session: {sessions.find(s => s.id === selectedSessionId)?.name || 'Unknown'}
        </h2>
        <TrajectoryPanel 
          sessionId={selectedSessionId}
        />
      </>
    ) : (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Select a session to view details</p>
      </div>
    )
  );

  // Floating action button component that uses Portal to render at document body level
  const FloatingActionButton = ({ onClick }: { onClick: () => void }) => {
    // Only render the portal on the client side, not during SSR
    const [mounted, setMounted] = useState(false);
    
    useEffect(() => {
      setMounted(true);
      return () => setMounted(false);
    }, []);
    
    const button = (
      <Button
        variant="default"
        size="icon"
        onClick={onClick}
        aria-label="Toggle sessions panel"
        className="h-14 w-14 rounded-full shadow-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-100 flex items-center justify-center border-2 border-zinc-700 dark:border-zinc-600"
      >
        <PanelLeft className="h-6 w-6" />
      </Button>
    );
    
    const container = (
      <div className="fixed bottom-6 right-6 z-[9999] md:hidden" style={{ pointerEvents: 'auto' }}>
        {button}
      </div>
    );
    
    // Return null during SSR, or the portal on the client
    return mounted ? createPortal(container, document.body) : null;
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