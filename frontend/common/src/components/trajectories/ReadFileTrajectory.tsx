import React from 'react';
import { Trajectory } from '../../models/trajectory';
import {
  Card,
  CardHeader,
} from '../ui/card'; // Removed CardContent import

interface ReadFileTrajectoryProps {
  trajectory: Trajectory;
}

export const ReadFileTrajectory: React.FC<ReadFileTrajectoryProps> = ({
  trajectory,
}) => {
  const { line_count, total_bytes, filepath } = trajectory.stepData || {};
  const timestamp = trajectory.created; // Use created timestamp

  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { // Format time consistently
        hour: '2-digit',
        minute: '2-digit',
      })
    : 'Invalid Date';

  const displayPath = filepath || 'Unknown file';
  const displayLineCount = line_count !== undefined ? line_count : '?';
  const displayTotalBytes = total_bytes !== undefined ? total_bytes : '?';

  return (
    <Card>
      {/* Updated CardHeader layout */}
      <CardHeader className="py-3 px-4">
        <div className="flex justify-between items-center">
          {/* Group icon and descriptive text */}
          <div className="flex items-center space-x-2"> {/* Added grouping div */}
            <span className="mr-2">📄</span> {/* Icon */}
            {/* Descriptive text moved from CardContent */}
            <span>
              Read <strong className="font-semibold">{displayLineCount}</strong> lines (<strong className="font-semibold">{displayTotalBytes}</strong> bytes) from <em className="italic">{displayPath}</em>
            </span>
          </div>
          {/* Timestamp remains on the right */}
          <div className="text-xs text-muted-foreground">
            {formattedTime}
          </div>
        </div>
      </CardHeader>
      {/* CardContent removed */}
    </Card>
  );
};
