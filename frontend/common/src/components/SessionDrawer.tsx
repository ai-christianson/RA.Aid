import React from 'react';
import { 
  Sheet, 
  SheetContent, 
  SheetHeader, 
  SheetTitle, 
  SheetClose 
} from './ui/sheet';
import { AgentSession } from '../utils/types';
import { getSampleAgentSessions } from '../utils/sample-data';
import { SessionList } from './SessionList';

interface SessionDrawerProps {
  onSelectSession?: (sessionId: string) => void;
  currentSessionId?: string;
  sessions?: AgentSession[];
  isOpen?: boolean;
  onClose?: () => void;
}

export const SessionDrawer: React.FC<SessionDrawerProps> = ({ 
  onSelectSession, 
  currentSessionId,
  sessions = getSampleAgentSessions(),
  isOpen = false,
  onClose
}) => {
  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent 
        side="left" 
        className="w-full sm:max-w-md border-r border-border p-4"
      >
        <SheetHeader className="px-2">
          <SheetTitle>Sessions</SheetTitle>
        </SheetHeader>
        <SessionList
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={onSelectSession}
          className="h-[calc(100vh-9rem)] mt-4"
          wrapperComponent={SheetClose}
        />
      </SheetContent>
    </Sheet>
  );
};