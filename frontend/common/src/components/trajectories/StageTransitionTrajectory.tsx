import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/collapsible';
import { Trajectory } from '../../models/trajectory';

interface StageTransitionTrajectoryProps {
  trajectory: Trajectory;
}

export const StageTransitionTrajectory: React.FC<StageTransitionTrajectoryProps> = ({ trajectory }) => {
  // Extract relevant data
  const stepData = trajectory.stepData || {};
  const title = stepData.display_title || 'Stage Transition';
  const description = stepData.description || '';
  const stage = stepData.stage || '';
  const fromStage = stepData.from_stage || '';
  const toStage = stepData.to_stage || '';
  const isError = trajectory.isError;
  
  // Define stage icons mapping - similar to backend console formatting
  const getStageIcon = (stageName: string): string => {
    const stageIcons: Record<string, string> = {
      'research_stage': '🔎',
      'planning_stage': '📝',
      'implementation_stage': '🔧',
      'chat_mode': '💬',
    };
    
    return stageIcons[stageName] || '🚀';
  };
  
  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Get the appropriate icon
  const stageIcon = getStageIcon(stage);

  return (
    <Collapsible className="w-full border border-border rounded-md overflow-hidden shadow-sm hover:shadow-md transition-all duration-200">
      <CollapsibleTrigger className="w-full text-left hover:bg-accent/30 cursor-pointer">
        <CardHeader className="py-3 px-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full bg-green-500 ring-1 ring-ring/20`} />
              <div className="flex-shrink-0 text-lg">{stageIcon}</div>
              <CardTitle className="text-base font-medium">
                {title}
              </CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              {formatTime(trajectory.created)}
            </div>
          </div>
        </CardHeader>
      </CollapsibleTrigger>
      
      <CollapsibleContent>
        <CardContent className="py-3 px-4 border-t border-border bg-card/50">
          {description && (
            <div className="mb-4">
              <p className="text-sm text-foreground whitespace-pre-wrap">{description}</p>
            </div>
          )}
          
          <div className="flex items-center space-x-2 text-sm">
            <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 rounded-md flex items-center">
              <span className="mr-1">{stageIcon}</span> {title}
            </span>
          </div>
          
          {isError && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold mb-2 text-red-500">Error:</h4>
              <pre className="text-xs bg-red-50 dark:bg-red-900/20 p-2 rounded-md text-red-800 dark:text-red-200 overflow-auto max-h-60">
                {trajectory.errorMessage || 'Unknown error'}
                {trajectory.errorType && ` (${trajectory.errorType})`}
              </pre>
            </div>
          )}
          
          {trajectory.currentCost !== null && trajectory.currentCost !== undefined && (
            <div className="mt-3 pt-3 border-t border-border/50 text-xs text-muted-foreground">
              <span className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Cost: ${trajectory.currentCost.toFixed(6)}
              </span>
            </div>
          )}
        </CardContent>
      </CollapsibleContent>
    </Collapsible>
  );
};