import React from 'react';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetClose
} from './ui/sheet';
import { AgentSession } from '../models/session'; // Changed import source
import { getSampleAgentSessions } from '../utils/sample-data';
import { SessionList } from './SessionList';

interface SessionDrawerProps {
  onSelectSession?: (sessionId: number) => void; // Changed from string to number
  currentSessionId?: number | null | undefined; // Changed from string to number | null | undefined
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
              currentSessionId={currentSessionId} // Prop type now matches SessionList
              onSelectSession={onSelectSession} // Prop type now matches SessionList
              wrapperComponent={SheetClose}
            />
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};