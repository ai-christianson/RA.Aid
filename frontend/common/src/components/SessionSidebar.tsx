import React from 'react';
import { ScrollArea } from './ui/scroll-area';
import { AgentSession } from '../utils/types';
import { getSampleAgentSessions } from '../utils/sample-data';

interface SessionSidebarProps {
  onSelectSession?: (sessionId: string) => void;
  currentSessionId?: string;
  sessions?: AgentSession[];
  className?: string;
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({ 
  onSelectSession, 
  currentSessionId,
  sessions = getSampleAgentSessions(),
  className = ''
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
    <div className={`flex flex-col h-full ${className}`}>
      <div className="p-4 border-b border-border">
        <h3 className="font-medium text-lg">Sessions</h3>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {sessions.map((session) => (
            <button
              key={session.id}
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
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};