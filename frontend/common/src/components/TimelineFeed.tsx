import React, { useMemo } from 'react';
import { TimelineStep } from './TimelineStep';
import { AgentStep } from '../utils/types';

interface TimelineFeedProps {
  steps: AgentStep[];
  maxHeight?: string;
}

export const TimelineFeed: React.FC<TimelineFeedProps> = ({
  steps,
  maxHeight
}) => {
  // Always use 'desc' (newest first) sort order
  const sortOrder = 'desc';
  
  // Sort steps with newest first (desc order)
  const sortedSteps = useMemo(() => {
    return [...steps].sort((a, b) => {
      return b.timestamp.getTime() - a.timestamp.getTime();
    });
  }, [steps]);

  return (
    <div className="w-full rounded-md bg-background">
      <div 
        className="px-3 py-3 space-y-4 overflow-auto" 
        style={{ maxHeight: maxHeight || undefined }}
      >
        {sortedSteps.length > 0 ? (
          sortedSteps.map((step) => (
            <TimelineStep key={step.id} step={step} />
          ))
        ) : (
          <div className="text-center text-muted-foreground py-12 border border-dashed border-border rounded-md">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p>No steps to display</p>
          </div>
        )}
      </div>
    </div>
  );
}