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
    <Sheet open={isOpen} onOpenChange={onClose} modal={false}>
      <SheetContent 
        side="left" 
        className="w-full sm:max-w-md border-r border-border p-0"
      >
        <div className="h-full flex flex-col">
          <SheetHeader className="px-4 py-4 flex-shrink-0">
            <SheetTitle>Sessions</SheetTitle>
          </SheetHeader>
          <div className="flex-1 overflow-y-auto p-4">
            <SessionList
              sessions={sessions}
              currentSessionId={currentSessionId}
              onSelectSession={onSelectSession}
              wrapperComponent={SheetClose}
            />
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};