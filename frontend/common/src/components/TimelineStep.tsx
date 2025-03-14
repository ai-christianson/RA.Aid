import React from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { AgentStep } from '../utils/types';

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
    <Collapsible className="w-full mb-5 border border-border rounded-md overflow-hidden shadow-sm hover:shadow-md transition-all duration-200">
      <CollapsibleTrigger className="w-full flex items-center justify-between p-4 text-left hover:bg-accent/30 cursor-pointer group">
        <div className="flex items-center space-x-3 min-w-0 flex-1 pr-3">
          <div className={`flex-shrink-0 w-3 h-3 rounded-full ${getStatusColor(step.status)} ring-1 ring-ring/20`} />
          <div className="flex-shrink-0 text-lg group-hover:scale-110 transition-transform">{getTypeIcon(step.type)}</div>
          <div className="min-w-0 flex-1">
            <div className="font-medium text-foreground break-words">{step.title}</div>
            <div className="text-sm text-muted-foreground line-clamp-2">
              {step.type === 'tool-execution' ? 'Run tool' : step.content.substring(0, 60)}
              {step.content.length > 60 ? '...' : ''}
            </div>
          </div>
        </div>
        <div className="text-xs text-muted-foreground flex flex-col items-end flex-shrink-0 min-w-[70px] text-right">
          <span className="font-medium">{formatTime(step.timestamp)}</span>
          {step.duration && (
            <span className="mt-1 px-2 py-0.5 bg-secondary/50 rounded-full">
              {(step.duration / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="p-5 bg-card/50 border-t border-border">
          <div className="text-sm break-words text-foreground leading-relaxed">
            {step.content}
          </div>
          {step.duration && (
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
                Duration: {(step.duration / 1000).toFixed(1)} seconds
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};