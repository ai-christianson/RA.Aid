import React, { useEffect } from 'react';
import { 
  ToolExecutionTrajectory, 
  MemoryOperationTrajectory, 
  StageTransitionTrajectory, 
  InfoTrajectory,
  GenericTrajectory
} from './trajectories';
import { useTrajectoryStore } from '../store';
import { Trajectory } from '../models/trajectory';

interface TrajectoryPanelProps {
  /**
   * The ID of the session to display trajectories for
   */
  sessionId: string;
  
  /**
   * Optional maximum height for the container
   */
  maxHeight?: string;
}

/**
 * TrajectoryPanel component
 * 
 * Displays a timeline of agent trajectories for a specific session.
 * Fetches trajectories using the trajectory store and renders them
 * using appropriate components based on their record type.
 */
export const TrajectoryPanel: React.FC<TrajectoryPanelProps> = ({
  sessionId,
  maxHeight
}) => {
  // Log sessionId being passed to the component
  console.log('TrajectoryPanel - sessionId:', sessionId);
  // Get trajectory store data
  const { 
    trajectories, 
    isLoading, 
    error, 
    fetchSessionTrajectories 
  } = useTrajectoryStore();
  
  // Log trajectories array from the store
  console.log('TrajectoryPanel - trajectories from store:', trajectories);
  
  // Fetch trajectories when sessionId changes
  useEffect(() => {
    if (sessionId) {
      console.log('TrajectoryPanel - fetching trajectories for sessionId:', parseInt(sessionId));
      fetchSessionTrajectories(parseInt(sessionId));
    }
  }, [sessionId, fetchSessionTrajectories]);
  
  // Log the trajectories array
  console.log('TrajectoryPanel - trajectories to display:', trajectories);
  
  // Render loading state
  if (isLoading) {
    console.log('TrajectoryPanel - rendering state: LOADING');
    return (
      <div className="w-full rounded-md bg-background p-6 text-center">
        <div className="animate-spin inline-block w-6 h-6 border-2 border-current border-t-transparent text-primary rounded-full mb-2" aria-hidden="true"></div>
        <p className="text-muted-foreground">Loading trajectories...</p>
      </div>
    );
  }
  
  // Render error state
  if (error) {
    console.log('TrajectoryPanel - rendering state: ERROR', error);
    return (
      <div className="w-full rounded-md bg-background p-6 text-center border border-red-300 dark:border-red-800">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mx-auto mb-2 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="text-red-800 dark:text-red-200 mb-1">Failed to load trajectories</p>
        <p className="text-xs text-muted-foreground">{error}</p>
      </div>
    );
  }
  
  // Render empty state
  if (trajectories.length === 0) {
    console.log('TrajectoryPanel - rendering state: EMPTY (no trajectories)');
    return (
      <div className="w-full rounded-md bg-background p-6 text-center border border-dashed border-border">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-muted-foreground">No trajectories to display</p>
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
        return <GenericTrajectory key={trajectory.id} trajectory={trajectory} />;
    }
  };
  
  // Render trajectories
  console.log('TrajectoryPanel - rendering state: WITH DATA', trajectories.length, 'trajectories');
  return (
    <div className="w-full rounded-md bg-background">
      <div 
        className="px-3 py-3 space-y-4 overflow-auto" 
        style={{ maxHeight: maxHeight || undefined }}
      >
        {trajectories.map(renderTrajectory)}
      </div>
    </div>
  );
};