import React from 'react';
import { AgentSession } from '../models/session'; // Correct import source
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';

interface SessionListProps {
  onSelectSession?: (sessionId: number) => void;
  currentSessionId?: number | null | undefined;
  sessions?: AgentSession[]; // Uses the correct AgentSession interface
  className?: string;
  wrapperComponent?: React.ElementType;
  closeAction?: React.ReactNode;
  isLoading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
}

export const SessionList: React.FC<SessionListProps> = ({
  onSelectSession,
  currentSessionId,
  sessions = [], // Default to empty array instead of sample data
  className = '',
  wrapperComponent: WrapperComponent = 'button',
  closeAction,
  isLoading = false,
  error = null,
  onRefresh
}) => {

  // Get status color based on SessionStatus type
  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'running':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      case 'pending':
        return 'bg-yellow-500';
      default: // unknown or undefined
        return 'bg-gray-500';
    }
  };

  // Format timestamp
  const formatDate = (dateValue: Date | string | null | undefined): string => {
    if (!dateValue) {
      return '-';
    }
    let date: Date;
    try {
      // Attempt to create a Date object if it's not already one
      date = dateValue instanceof Date ? dateValue : new Date(dateValue);
      // Check if the resulting date is valid
      if (isNaN(date.getTime())) {
        console.warn("Invalid date value received in formatDate:", dateValue);
        return 'Invalid Date';
      }
    } catch (e) {
      console.error("Error parsing date in formatDate:", dateValue, e);
      return 'Invalid Date';
    }

    return date.toLocaleDateString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className={className}>
      {/* Error state */}
      {error && (
        <Card className="mb-3 mx-3 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
          <CardContent className="p-3 text-xs text-red-600 dark:text-red-400">
            <div className="flex flex-col space-y-2">
              <p>{error}</p>
              {onRefresh && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRefresh}
                  className="self-start mt-1 text-xs h-7"
                >
                  Try Again
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="space-y-2 px-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse bg-accent/50 rounded-md h-[72px] p-3">
              <div className="flex items-start">
                <div className="w-2.5 h-2.5 rounded-full bg-gray-300 dark:bg-gray-600 mt-1.5 mr-3"></div>
                <div className="flex-1">
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-2/3 mb-2"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-1"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && sessions.length === 0 && (
        <div className="px-3">
          <Card className="bg-accent/30">
            <CardContent className="p-4 text-center text-muted-foreground text-sm">
              <p>No sessions found</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Session list */}
      {!isLoading && sessions.length > 0 && (
        <div className="space-y-1.5">
          {sessions.map((session) => {
            // Ensure session properties are accessed safely
            const sessionId = session?.id;
            const sessionName = session?.name || 'Unnamed Session';
            const sessionStatus = session?.status;
            const sessionUpdatedAt = session?.updatedAt;

            if (sessionId === undefined || sessionId === null) {
              console.warn("Rendering session with missing ID:", session);
              return null; // Skip rendering if ID is missing
            }

            const buttonContent = (
              <>
                <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor(sessionStatus)} mt-1.5 mr-3 flex-shrink-0`} />
                <div className="flex-1 min-w-0 pr-1">
                  <div className="font-medium text-sm+ break-words">{sessionName}</div>
                  <div className="text-xs text-muted-foreground mt-1 break-words">
                    {formatDate(sessionUpdatedAt)} {/* Use the correct property name */}
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5 break-words">
                    <span className="capitalize">{sessionStatus || 'Unknown'}</span>
                  </div>
                </div>
              </>
            );

            return React.createElement(
              WrapperComponent,
              {
                key: sessionId,
                onClick: () => onSelectSession?.(sessionId),
                className: `w-full flex items-start px-3 py-2.5 text-left rounded-md transition-colors hover:bg-accent/50 ${
                  currentSessionId === sessionId ? 'bg-accent' : ''
                }`
              },
              closeAction ? (
                <>
                  {buttonContent}
                  <div className="ml-2 flex-shrink-0 self-center">
                    {React.cloneElement(closeAction as React.ReactElement, {
                      onClick: (e: React.MouseEvent) => {
                        e.stopPropagation();
                        onSelectSession?.(sessionId);
                      }
                    })}
                  </div>
                </>
              ) : buttonContent
            );
          })}
        </div>
      )}
    </div>
  );
};
