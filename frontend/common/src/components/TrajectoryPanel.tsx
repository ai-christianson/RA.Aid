import React, { useEffect } from 'react';
import {
  ToolExecutionTrajectory,
  MemoryOperationTrajectory,
  StageTransitionTrajectory,
  InfoTrajectory,
  GenericTrajectory
} from './trajectories';
import { useTrajectoryStore, useSessionStore } from '../store'; // <-- Import useSessionStore
import { Trajectory } from '../models/trajectory';
import { Loader2 } from 'lucide-react'; // <-- Import spinner icon
import { ScrollArea } from './ui/scroll-area'; // Import ScrollArea if needed for structure

interface TrajectoryPanelProps {
  /**
   * The ID of the session to display trajectories for
   */
  sessionId: string;

  /**
   * Optional maximum height for the container
   */
  maxHeight?: string;

  /**
   * Whether to add bottom padding to prevent content from being hidden
   * behind a fixed-position input section at the bottom of the screen.
   * Set to true when used in the main content area.
   */
  addBottomPadding?: boolean;

  /**
   * Optional custom className to override default styling
   */
  customClassName?: string;
}

/**
 * TrajectoryPanel component
 *
 * Displays a timeline of agent trajectories for a specific session.
 * Fetches trajectories using the trajectory store and renders them
 * using appropriate components based on their record type.
 * Also displays a running indicator based on session status.
 */
export const TrajectoryPanel: React.FC<TrajectoryPanelProps> = ({
  sessionId,
  maxHeight,
  addBottomPadding = false, // Default to false when not specified
  customClassName = ''
}) => {
  // Get trajectory store data
  const {
    trajectories,
    isLoading: trajectoriesLoading, // Rename to avoid conflict
    error: trajectoriesError,       // Rename to avoid conflict
    fetchSessionTrajectories
  } = useTrajectoryStore();

  // Get session store data
  const { sessions, selectedSessionId } = useSessionStore(); // <-- Get sessions and selectedId

  // Find current session and status
  const currentSession = sessions.find(s => s.id === selectedSessionId);
  const isRunning = currentSession?.status === 'running'; // <-- Check if session is running

  // Fetch trajectories when sessionId changes
  useEffect(() => {
    if (sessionId) {
      // Fetch immediately
      fetchSessionTrajectories(parseInt(sessionId));

      // Optional: Refetch periodically if status is running?
      // const intervalId = setInterval(() => {
      //   const updatedSession = useSessionStore.getState().sessions.find(s => s.id === sessionId);
      //   if (updatedSession?.status === 'running') {
      //     fetchSessionTrajectories(parseInt(sessionId));
      //   } else {
      //     clearInterval(intervalId); // Stop polling if not running
      //   }
      // }, 5000); // e.g., every 5 seconds

      // return () => clearInterval(intervalId); // Cleanup interval on unmount/change
    }
  }, [sessionId, fetchSessionTrajectories]); // Dependency array

  // Render loading state (for initial trajectory fetch)
  if (trajectoriesLoading) {
    return (
      <div className={`w-full rounded-md bg-background p-6 text-center ${customClassName}`}>
        <div className="animate-spin inline-block w-6 h-6 border-2 border-current border-t-transparent text-primary rounded-full mb-2" aria-hidden="true"></div>
        <p className="text-muted-foreground">Loading session data...</p>
        <p className="text-xs text-muted-foreground mt-2">This may take a moment...</p>
      </div>
    );
  }

  // Render error state (for trajectory fetch)
  if (trajectoriesError) {
    return (
      <div className={`w-full rounded-md bg-background p-6 text-center border border-red-300 dark:border-red-800 ${customClassName}`}>
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mx-auto mb-2 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="text-red-800 dark:text-red-200 mb-1">Failed to load session trajectories</p>
        <p className="text-xs text-muted-foreground">{trajectoriesError}</p>
      </div>
    );
  }

  // Render the appropriate component based on record type
  const renderTrajectory = (trajectory: Trajectory) => {
    switch (trajectory.recordType) {
      case 'tool_execution':
        return <ToolExecutionTrajectory key={trajectory.id} trajectory={trajectory} />;
      case 'memory_operation':
        return <MemoryOperationTrajectory key={trajectory.id} trajectory={trajectory} />;
      case 'stage_transition':
        return <StageTransitionTrajectory key={trajectory.id} trajectory={trajectory} />;
      case 'info':
        return <InfoTrajectory key={trajectory.id} trajectory={trajectory} />;
      default:
        // console.warn("Rendering GenericTrajectory for unknown type:", trajectory.recordType, trajectory);
        return <GenericTrajectory key={trajectory.id} trajectory={trajectory} />;
    }
  };

  // Render trajectories or empty/running state
  return (
    <div className={`flex flex-col h-full w-full bg-background ${customClassName}`}>
      {trajectories.length === 0 && !isRunning && !trajectoriesLoading && !trajectoriesError && (
        // True Empty State (session finished/error/pending with no output yet)
        <div className="flex-grow flex flex-col justify-center items-center p-6 text-center border border-dashed border-border rounded-md">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-muted-foreground">No activity recorded for this session yet.</p>
          {currentSession?.status === 'completed' && <p className="text-xs text-muted-foreground mt-1">Session completed.</p>}
          {currentSession?.status === 'error' && <p className="text-xs text-red-500 mt-1">Session ended with an error.</p>}
          {currentSession?.status === 'pending' && <p className="text-xs text-muted-foreground mt-1">Session is pending.</p>}
        </div>
      )}

      {trajectories.length === 0 && isRunning && !trajectoriesLoading && !trajectoriesError && (
         // Empty State but Running
         <div className="flex-grow flex flex-col justify-center items-center p-6 text-center">
           <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
           <span className="mt-3 text-muted-foreground">Agent is running...</span>
           <p className="text-xs text-muted-foreground mt-1">Waiting for first output.</p>
         </div>
      )}

      {trajectories.length > 0 && (
        // Render Trajectories List
        // Use ScrollArea for consistent scrolling behavior if needed
        // <ScrollArea className={`flex-grow ${addBottomPadding ? 'pb-32' : ''}`} style={{ maxHeight: maxHeight || undefined }}>
        <div className={`flex-grow space-y-4 overflow-auto ${addBottomPadding ? 'pb-32' : ''}`} style={{ maxHeight: maxHeight || undefined }}>
            {trajectories.map(renderTrajectory)}
            {/* Add Spinner at the end of the list if running */}
            {isRunning && (
              <div className="flex justify-center items-center pt-4 pb-2"> {/* Added pb-2 */}
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /> {/* Slightly smaller */}
                <span className="ml-2 text-sm text-muted-foreground">Agent is running...</span> {/* Slightly smaller text */}
              </div>
            )}
        </div>
        // </ScrollArea>
      )}
    </div>
  );
};
