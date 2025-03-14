import React from 'react';
import { AgentSession } from '../utils/types';
import { getSampleAgentSessions } from '../utils/sample-data';
import { SessionList } from './SessionList';

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
  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="p-4 border-b border-border">
        <h3 className="font-medium text-lg">Sessions</h3>
      </div>
      <SessionList
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={onSelectSession}
        className="flex-1"
      />
    </div>
  );
};