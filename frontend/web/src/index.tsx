import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { 
  Button,
  SessionDrawer,
  SessionSidebar,
  TimelineFeed,
  getSampleAgentSessions,
  getSampleAgentSteps
} from '@ra-aid/common';
import { Layout } from './components/Layout';
// The CSS import happens through the common package's index.ts

// Theme management helper function
const setupTheme = () => {
  // Check if theme preference is stored in localStorage
  const storedTheme = localStorage.getItem('theme');
  
  // Default to dark mode unless explicitly set to light
  const isDark = storedTheme ? storedTheme === 'dark' : true;
  
  // Apply theme class to document element (html) for better CSS specificity
  if (isDark) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
  
  // Store the current theme preference
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  
  return isDark;
};

const App = () => {
  // State for drawer open/close
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  
  // State for selected session
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  
  // State for theme (dark is default)
  const [isDarkTheme, setIsDarkTheme] = useState(true);
  
  // Get sample data
  const sessions = getSampleAgentSessions();
  const allSteps = getSampleAgentSteps();
  
  // Set up theme on component mount
  useEffect(() => {
    const isDark = setupTheme();
    setIsDarkTheme(isDark);
  }, []);
  
  // Set initial selected session if none selected
  useEffect(() => {
    if (!selectedSessionId && sessions.length > 0) {
      setSelectedSessionId(sessions[0].id);
    }
  }, [sessions, selectedSessionId]);
  
  // Filter steps for selected session
  const selectedSessionSteps = selectedSessionId 
    ? allSteps.filter(step => sessions.find(s => s.id === selectedSessionId)?.steps.some(s => s.id === step.id))
    : [];
  
  // Handle session selection
  const handleSessionSelect = (sessionId: string) => {
    setSelectedSessionId(sessionId);
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
    <div className="flex justify-between items-center h-full px-4">
      <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 inline-block text-transparent bg-clip-text">
        RA-Aid
      </h1>
      
      <div className="flex items-center gap-2">
        {/* Theme toggle button */}
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={toggleTheme}
          aria-label={isDarkTheme ? "Switch to light mode" : "Switch to dark mode"}
          className="mr-2"
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
        
        {/* Mobile drawer toggle - show only on small screens */}
        <div className="md:hidden">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsDrawerOpen(true)}
            aria-label="Open menu"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-menu"><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/></svg>
          </Button>
        </div>
      </div>
    </div>
  );

  // Render sidebar content
  const sidebarContent = (
    <SessionSidebar 
      sessions={sessions}
      currentSessionId={selectedSessionId || undefined}
      onSelectSession={handleSessionSelect}
    />
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

  // Render main content
  const mainContent = (
    selectedSessionId ? (
      <>
        <h2 className="text-xl font-semibold mb-4">
          Session: {sessions.find(s => s.id === selectedSessionId)?.name || 'Unknown'}
        </h2>
        <TimelineFeed 
          steps={selectedSessionSteps}
        />
      </>
    ) : (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Select a session to view details</p>
      </div>
    )
  );
  
  return (
    <Layout
      header={headerContent}
      sidebar={sidebarContent}
      drawer={drawerContent}
    >
      {mainContent}
    </Layout>
  );
};

// Initialize theme before rendering the app
setupTheme();

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);