
import React from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Trajectory } from '../models/trajectory';
import { TaskCompletedTrajectory } from './trajectories/TaskCompletedTrajectory';
import { PlanCompletedTrajectory } from './trajectories/PlanCompletedTrajectory';
// Import other specific trajectory components if needed
// import { GenericTrajectory } from './trajectories/GenericTrajectory'; 
// import { ToolExecutionTrajectory } from './trajectories/ToolExecutionTrajectory'; 

interface TimelineStepProps {
  trajectory: Trajectory;
}

export const TimelineStep: React.FC<TimelineStepProps> = ({ trajectory }) => {

  // Render specific components for certain record types
  if (trajectory.record_type === 'task_completion') {
    return <TaskCompletedTrajectory trajectory={trajectory} />;
  }
  if (trajectory.record_type === 'plan_completion') {
    return <PlanCompletedTrajectory trajectory={trajectory} />;
  }

  // --- Fallback rendering for other record types using the original structure ---

  // Get icon based on record type (adjust as needed)
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'tool_execution':
        return '🛠️';
      case 'thinking':
        return '💭';
      case 'planning': // Assuming a planning stage might exist
        return '📝';
      case 'implementation': // Assuming an implementation stage
        return '💻';
      case 'user_input':
        return '👤';
      case 'stage_transition':
          return '🔄';
      case 'key_fact':
        return '🔑';
      case 'key_snippet':
        return '📄';
      case 'research_note':
        return '🔍';
      default:
        return '▶️'; // Generic/Unknown
    }
  };

  // Format timestamp
  const formatTime = (timestamp: string) => {
    // Assuming timestamp is an ISO string like "2023-10-27T10:30:00.123Z"
    try {
      return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return 'Invalid Date';
    }
  };

  // Try to get a sensible title
  const getTitle = () => {
    return trajectory.step_data?.display_title || trajectory.record_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  // Try to get a preview content
  const getContentPreview = () => {
    if (trajectory.record_type === 'thinking' && trajectory.step_data?.thought) {
      return trajectory.step_data.thought;
    } 
    // Add more specific previews if needed
    return JSON.stringify(trajectory.step_data).substring(0, 100);
  };

  // Get full content for collapsible section
  const getFullContent = () => {
      // Provide more structured content based on type if needed
      // For now, just stringify step_data
      return <pre className="whitespace-pre-wrap break-all">{JSON.stringify(trajectory.step_data, null, 2)}</pre>;
  }

  // Status indicator removed as Trajectory doesn't have a direct status field

  return (
    <Collapsible className="w-full mb-5 border border-border rounded-md overflow-hidden shadow-sm hover:shadow-md transition-all duration-200">
      <CollapsibleTrigger className="w-full flex items-center justify-between p-4 text-left hover:bg-accent/30 cursor-pointer group">
        <div className="flex items-center space-x-3 min-w-0 flex-1 pr-3">
          {/* Status indicator removed */}
          <div className="flex-shrink-0 text-lg group-hover:scale-110 transition-transform">{getTypeIcon(trajectory.record_type)}</div>
          <div className="min-w-0 flex-1">
            <div className="font-medium text-foreground break-words">{getTitle()}</div>
            <div className="text-sm text-muted-foreground line-clamp-2">
              {getContentPreview()}
              {(getContentPreview()?.length ?? 0) > 100 ? '...' : ''}
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground flex flex-col items-end flex-shrink-0 min-w-[70px] text-right">
          <span className="font-medium">{formatTime(trajectory.created)}</span>
          {trajectory.duration_ms != null && (
            <span className="mt-1 px-2 py-0.5 bg-secondary/50 rounded-full">
              {(trajectory.duration_ms / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="p-5 bg-card/50 border-t border-border">
          <div className="text-sm break-words text-foreground leading-relaxed">
            {getFullContent()}
          </div>
          {trajectory.duration_ms != null && (
            <div className="mt-4 pt-3 border-t border-border/50">
              <div className="text-xs text-muted-foreground flex items-center">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-3.5 w-3.5 mr-1" 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor" 
                  strokeWidth={2}
                >
                  <circle cx="12" cy="12" r="10" />
                  <polyline points="12 6 12 12 16 14" />
                </svg>
                Duration: {(trajectory.duration_ms / 1000).toFixed(1)} seconds
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};
