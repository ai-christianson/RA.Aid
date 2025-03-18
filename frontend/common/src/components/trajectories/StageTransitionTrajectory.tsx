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
  const title = stepData.title || 'Stage Transition';
  const description = stepData.description || '';
  const fromStage = stepData.from_stage || '';
  const toStage = stepData.to_stage || '';
  const isError = trajectory.isError;
  
  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Collapsible className="w-full border border-border rounded-md overflow-hidden shadow-sm hover:shadow-md transition-all duration-200">
      <CollapsibleTrigger className="w-full text-left hover:bg-accent/30 cursor-pointer">
        <CardHeader className="py-3 px-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full bg-purple-500 ring-1 ring-ring/20`} />
              <div className="flex-shrink-0 text-lg">🔄</div>
              <CardTitle className="text-base font-medium">
                {title}
              </CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              {formatTime(trajectory.created)}
            </div>
          </div>
          <div className="text-sm text-muted-foreground mt-1 line-clamp-2">
            {fromStage && toStage ? `${fromStage} → ${toStage}` : description}
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
          
          {(fromStage || toStage) && (
            <div className="flex items-center space-x-2 text-sm">
              {fromStage && <span className="px-2 py-1 bg-muted rounded-md">{fromStage}</span>}
              {fromStage && toStage && (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              )}
              {toStage && <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200 rounded-md">{toStage}</span>}
            </div>
          )}
          
          {isError && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold mb-2 text-red-500">Error:</h4>
              <pre className="text-xs bg-red-50 dark:bg-red-900/20 p-2 rounded-md text-red-800 dark:text-red-200 overflow-auto max-h-60">
                {trajectory.errorMessage || 'Unknown error'}
                {trajectory.errorType && ` (${trajectory.errorType})`}
              </pre>
            </div>
          )}
          
          {trajectory.currentCost !== null && (
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