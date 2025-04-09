import React from 'react';
import { AgentSession } from '../models/session'; // Changed import source
import { SessionList } from './SessionList';

interface SessionSidebarProps {
  onSelectSession?: (sessionId: number) => void; // Changed to number
  currentSessionId?: number | null | undefined; // Changed to number | null | undefined
  sessions?: AgentSession[];
  className?: string;
  isLoading?: boolean; // Added for SessionList
  error?: string | null; // Added for SessionList
  onRefresh?: () => void; // Added for SessionList
}

export const SessionSidebar: React.FC<SessionSidebarProps> = ({
  onSelectSession,
  currentSessionId,
  sessions = [], // Default to empty array
  className = '',
  isLoading,
  error,
  onRefresh
}) => {
  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Removed header - SessionList handles loading/error/empty states */}
      <SessionList
        sessions={sessions}
        currentSessionId={currentSessionId} // Passed number type
        onSelectSession={onSelectSession} // Passed function expecting number
        className="flex-1 p-4" // Added padding
        isLoading={isLoading} // Pass through loading state
        error={error} // Pass through error state
        onRefresh={onRefresh} // Pass through refresh handler
      />
    </div>
  );
};