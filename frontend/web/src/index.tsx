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
// The CSS import happens through the common package's index.ts

const App = () => {
  // State for drawer open/close
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  
  // State for selected session
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  
  // Get sample data
  const sessions = getSampleAgentSessions();
  const allSteps = getSampleAgentSteps();
  
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
  
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col dark">
      {/* Header */}
      <header className="border-b border-border py-4 px-4 sticky top-0 z-10 bg-background">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 inline-block text-transparent bg-clip-text">
            RA-Aid
          </h1>
          
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
      </header>
      
      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Desktop sidebar - hidden on mobile */}
        <SessionSidebar 
          sessions={sessions}
          currentSessionId={selectedSessionId || undefined}
          onSelectSession={handleSessionSelect}
          className="shrink-0"
        />
        
        {/* Mobile drawer */}
        <SessionDrawer 
          sessions={sessions}
          currentSessionId={selectedSessionId || undefined}
          onSelectSession={handleSessionSelect}
        />
        
        {/* Main content area */}
        <main className="flex-1 overflow-auto p-4">
          {selectedSessionId ? (
            <>
              <h2 className="text-xl font-semibold mb-4">
                Session: {sessions.find(s => s.id === selectedSessionId)?.name || 'Unknown'}
              </h2>
              <TimelineFeed 
                steps={selectedSessionSteps} 
                maxHeight="calc(100vh - 14rem)"
              />
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Select a session to view details</p>
            </div>
          )}
        </main>
      </div>
      
      <footer className="border-t border-border py-4 px-4 text-center text-muted-foreground text-sm">
        <p>Built with shadcn/ui components from the RA-Aid common package</p>
      </footer>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);