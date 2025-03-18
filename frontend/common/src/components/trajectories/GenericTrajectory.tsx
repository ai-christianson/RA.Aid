import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/collapsible';
import { Trajectory } from '../../models/trajectory';

interface GenericTrajectoryProps {
  trajectory: Trajectory;
}

export const GenericTrajectory: React.FC<GenericTrajectoryProps> = ({ trajectory }) => {
  // Format data for display
  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  // Extract relevant data
  const recordType = trajectory.recordType;
  const toolName = trajectory.toolName;
  const toolParameters = trajectory.toolParameters || {};
  const toolResult = trajectory.toolResult || {};
  const stepData = trajectory.stepData || {};
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
              <div className={`w-3 h-3 rounded-full ${isError ? 'bg-red-500' : 'bg-gray-500'} ring-1 ring-ring/20`} />
              <div className="flex-shrink-0 text-lg">📋</div>
              <CardTitle className="text-base font-medium">
                {recordType}
              </CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              {formatTime(trajectory.created)}
            </div>
          </div>
          <div className="text-sm text-muted-foreground mt-1 line-clamp-2">
            {toolName}
            {stepData.display && typeof stepData.display === 'string' && `: ${stepData.display}`}
          </div>
        </CardHeader>
      </CollapsibleTrigger>
      
      <CollapsibleContent>
        <CardContent className="py-3 px-4 border-t border-border bg-card/50">
          {Object.keys(stepData).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold mb-2">Step Data:</h4>
              <pre className="text-xs bg-muted p-2 rounded-md overflow-auto max-h-60">
                {formatValue(stepData)}
              </pre>
            </div>
          )}
          
          {Object.keys(toolParameters).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold mb-2">Parameters:</h4>
              <pre className="text-xs bg-muted p-2 rounded-md overflow-auto max-h-60">
                {formatValue(toolParameters)}
              </pre>
            </div>
          )}
          
          {(!isError && Object.keys(toolResult).length > 0) && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Result:</h4>
              <pre className="text-xs bg-muted p-2 rounded-md overflow-auto max-h-60">
                {formatValue(toolResult)}
              </pre>
            </div>
          )}
          
          {isError && (
            <div>
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
                Cost: ${trajectory.currentCost?.toFixed(6) || 'N/A'}
              </span>
              {(trajectory.inputTokens || trajectory.outputTokens) && (
                <span className="flex items-center mt-1">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                  </svg>
                  Tokens: {trajectory.inputTokens || 0} in / {trajectory.outputTokens || 0} out
                </span>
              )}
            </div>
          )}
        </CardContent>
      </CollapsibleContent>
    </Collapsible>
  );
};