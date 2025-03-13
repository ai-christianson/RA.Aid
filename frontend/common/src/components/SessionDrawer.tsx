import React from 'react';
import { Menu } from 'lucide-react';
import { 
  Sheet, 
  SheetTrigger, 
  SheetContent, 
  SheetHeader, 
  SheetTitle, 
  SheetClose 
} from './ui/sheet';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { AgentSession, getSampleAgentSessions } from '../utils/sample-data';

interface SessionDrawerProps {
  onSelectSession?: (sessionId: string) => void;
  currentSessionId?: string;
  sessions?: AgentSession[];
}

export const SessionDrawer: React.FC<SessionDrawerProps> = ({ 
  onSelectSession, 
  currentSessionId,
  sessions = getSampleAgentSessions()
}) => {
  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Format timestamp
  const formatDate = (date: Date) => {
    return date.toLocaleDateString([], { 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle navigation</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[85%] sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Sessions</SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-5rem)] mt-6">
          <div className="space-y-4">
            {sessions.map((session) => (
              <SheetClose key={session.id} asChild>
                <button
                  onClick={() => onSelectSession?.(session.id)}
                  className={`w-full flex items-start p-3 text-left rounded-md transition-colors hover:bg-accent/50 ${
                    currentSessionId === session.id ? 'bg-accent' : ''
                  }`}
                >
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(session.status)} mt-1.5 mr-3 flex-shrink-0`} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{session.name}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {session.steps.length} steps • {formatDate(session.updated)}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      <span className="capitalize">{session.status}</span>
                    </div>
                  </div>
                </button>
              </SheetClose>
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
};