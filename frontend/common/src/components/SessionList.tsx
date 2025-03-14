import React from 'react';
import { ScrollArea } from './ui/scroll-area';
import { AgentSession } from '../utils/types';
import { getSampleAgentSessions } from '../utils/sample-data';

interface SessionListProps {
  onSelectSession?: (sessionId: string) => void;
  currentSessionId?: string;
  sessions?: AgentSession[];
  className?: string;
  wrapperComponent?: React.ElementType;
  closeAction?: React.ReactNode;
}

export const SessionList: React.FC<SessionListProps> = ({ 
  onSelectSession, 
  currentSessionId,
  sessions = getSampleAgentSessions(),
  className = '',
  wrapperComponent: WrapperComponent = 'button',
  closeAction
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
    <ScrollArea className={`${className}`}>
      <div className="px-2 py-2 space-y-3 w-full">
        {sessions.map((session) => {
          const buttonContent = (
            <>
              <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor(session.status)} mt-1.5 mr-2 flex-shrink-0`} />
              <div className="flex-1 min-w-0 w-full overflow-hidden">
                <div className="font-medium truncate max-w-full">{session.name}</div>
                <div className="text-xs text-muted-foreground mt-0.5 truncate">
                  {session.steps.length} steps • {formatDate(session.updated)}
                </div>
                <div className="text-xs text-muted-foreground mt-0.5 truncate">
                  <span className="capitalize">{session.status}</span>
                </div>
              </div>
            </>
          );

          return React.createElement(
            WrapperComponent,
            {
              key: session.id,
              onClick: () => onSelectSession?.(session.id),
              className: `w-full flex items-start px-2 py-2 text-left rounded-md transition-colors hover:bg-accent/50 ${
                currentSessionId === session.id ? 'bg-accent' : ''
              }`
            },
            closeAction ? (
              <>
                {buttonContent}
                {React.cloneElement(closeAction as React.ReactElement, { 
                  onClick: (e: React.MouseEvent) => {
                    e.stopPropagation();
                    onSelectSession?.(session.id);
                  }
                })}
              </>
            ) : buttonContent
          );
        })}
      </div>
    </ScrollArea>
  );
};
