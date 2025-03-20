import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/collapsible';
import { Trajectory } from '../../models/trajectory';

interface MemoryOperationTrajectoryProps {
  trajectory: Trajectory;
}

export const MemoryOperationTrajectory: React.FC<MemoryOperationTrajectoryProps> = ({ trajectory }) => {
  // Format memory operation data for display
  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  // Get memory operation type from tool name
  const getOperationType = (toolName: string): string => {
    const types: Record<string, string> = {
      'emit_key_facts': 'Store Key Facts',
      'emit_key_snippet': 'Store Code Snippet',
      'emit_research_note': 'Store Research Note',
      'read_key_facts': 'Retrieve Key Facts',
      'read_key_snippets': 'Retrieve Code Snippets',
      'read_research_notes': 'Retrieve Research Notes',
    };
    
    return types[toolName] || toolName.replace(/_/g, ' ');
  };

  // Extract relevant data
  const toolName = trajectory.toolName;
  const toolParameters = trajectory.toolParameters || {};
  const toolResult = trajectory.toolResult || {};
  const stepData = trajectory.stepData || {};
  const operationType = getOperationType(toolName);
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
              <div className={`w-3 h-3 rounded-full ${isError ? 'bg-red-500' : 'bg-blue-500'} ring-1 ring-ring/20`} />
              <div className="flex-shrink-0 text-lg">💾</div>
              <CardTitle className="text-base font-medium">
                {operationType}
              </CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              {formatTime(trajectory.created)}
            </div>
          </div>
          {stepData.display && (
            <div className="text-sm text-muted-foreground mt-1 line-clamp-2">
              {typeof stepData.display === 'string' ? stepData.display : JSON.stringify(stepData.display)}
            </div>
          )}
        </CardHeader>
      </CollapsibleTrigger>
      
      <CollapsibleContent>
        <CardContent className="py-3 px-4 border-t border-border bg-card/50">
          {Object.keys(toolParameters).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold mb-2">Memory Data:</h4>
              <pre className="text-xs bg-muted p-2 rounded-md overflow-auto max-h-60">
                {Object.entries(toolParameters).map(([key, value]) => (
                  <div key={key} className="mb-1">
                    <span className="text-blue-600 dark:text-blue-400">{key}:</span> {formatValue(value)}
                  </div>
                ))}
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