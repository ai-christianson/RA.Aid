
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/collapsible';
import { Trajectory } from '../../models/trajectory';

interface ProjectStatusTrajectoryProps {
  trajectory: Trajectory;
}

export const ProjectStatusTrajectory: React.FC<ProjectStatusTrajectoryProps> = ({ trajectory }) => {
  const stepData = trajectory.stepData || {};
  const displayTitle = stepData.display_title || 'Project Status';
  const projectStatus = stepData.project_status === 'new' ? 'New/Empty Project' : 'Existing Project';
  const fileCount = stepData.file_count;
  const totalFiles = stepData.total_files;

  // Format timestamp to HH:MM AM/PM
  const formattedTime = new Date(trajectory.created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <Collapsible
      defaultOpen={true}
      className="w-full border border-border rounded-md overflow-hidden shadow-sm hover:shadow-md transition-all duration-200"
    >
      <CollapsibleTrigger className="w-full text-left hover:bg-accent/30 cursor-pointer">
        <CardHeader className="py-3 px-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              {/* Placeholder icon - replace if a specific one is desired */}
              <div className={`w-3 h-3 rounded-full bg-blue-500 ring-1 ring-ring/20`} />
              <div className="flex-shrink-0 text-lg">📊</div>
              <CardTitle className="text-base font-medium">
                {displayTitle}
              </CardTitle>
            </div>
            <div className="text-xs text-muted-foreground">
              {formattedTime}
            </div>
          </div>
        </CardHeader>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <CardContent className="py-3 px-4 border-t border-border bg-card/50">
          <div className="text-sm space-y-1">
            <p><strong>Status:</strong> {projectStatus}</p>
            {fileCount !== undefined && totalFiles !== undefined && (
               <p><strong>Files Scanned:</strong> {fileCount} out of {totalFiles} total project files.</p>
            )}
             {fileCount !== undefined && totalFiles === undefined && (
               <p><strong>Files Found:</strong> {fileCount}</p>
            )}
          </div>

          {/* Optional: Display cost if available */}
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
