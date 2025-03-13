import React from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { AgentStep } from '../utils/sample-data';

interface TimelineStepProps {
  step: AgentStep;
}

export const TimelineStep: React.FC<TimelineStepProps> = ({ step }) => {
  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'in-progress':
        return 'bg-blue-500';
      case 'error':
        return 'bg-red-500';
      case 'pending':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Get icon based on step type
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'tool-execution':
        return '🛠️';
      case 'thinking':
        return '💭';
      case 'planning':
        return '📝';
      case 'implementation':
        return '💻';
      case 'user-input':
        return '👤';
      default:
        return '▶️';
    }
  };

  // Format timestamp
  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Collapsible className="w-full mb-4 border border-border rounded-md overflow-hidden transition-all duration-200">
      <CollapsibleTrigger className="w-full flex items-center justify-between p-3 text-left hover:bg-accent/50 cursor-pointer">
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full ${getStatusColor(step.status)} mr-3`} />
          <div className="mr-2">{getTypeIcon(step.type)}</div>
          <div>
            <div className="font-medium">{step.title}</div>
            <div className="text-sm text-muted-foreground truncate max-w-xs">
              {step.type === 'tool-execution' ? 'Run tool' : step.content.substring(0, 60)}
              {step.content.length > 60 ? '...' : ''}
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground flex flex-col items-end">
          <span>{formatTime(step.timestamp)}</span>
          {step.duration && (
            <span className="mt-1">{(step.duration / 1000).toFixed(1)}s</span>
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="p-4 bg-card border-t border-border">
          <div className="text-sm whitespace-pre-wrap">
            {step.content}
          </div>
          {step.duration && (
            <div className="mt-3 pt-3 border-t border-border">
              <div className="text-xs text-muted-foreground">
                Duration: {(step.duration / 1000).toFixed(1)} seconds
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};