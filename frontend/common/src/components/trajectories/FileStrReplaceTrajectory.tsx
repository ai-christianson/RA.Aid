import React from 'react';
import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Replace } from 'lucide-react'; // Using Replace icon as suggested

export const FileStrReplaceTrajectory: React.FC<{ trajectory: Trajectory }> = ({ trajectory }) => {
  const { filepath = 'N/A', old_str = 'N/A', new_str = 'N/A', count = 0 } = trajectory.stepData || {};
  const displayTitle = trajectory.displayTitle || `Replaced string in ${filepath}`;

  // Basic rendering, will refine later
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center">
            <Replace className="h-4 w-4 text-muted-foreground mr-2 text-blue-500" />
            {displayTitle}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-xs text-muted-foreground space-y-1">
            <p>File: {filepath}</p>
            <p>Old: <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">{old_str}...</span></p>
            <p>New: <span className="font-mono bg-gray-100 dark:bg-gray-700 px-1 rounded">{new_str}...</span></p>
            <Badge variant="secondary">{count} replacement(s)</Badge>
        </div>
      </CardContent>
    </Card>
  );
};

export default FileStrReplaceTrajectory;

