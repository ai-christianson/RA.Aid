import React from 'react';
import { Trajectory } from '../../models/trajectory';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/card';
import { Badge } from '../ui/badge';
import { FileText } from 'lucide-react'; // Or another suitable icon

interface FileWriteTrajectoryProps {
  trajectory: Trajectory;
}

export const FileWriteTrajectory: React.FC<FileWriteTrajectoryProps> = ({ trajectory }) => {
  const filepath = trajectory.stepData?.filepath || 'N/A';
  const bytesWritten = trajectory.stepData?.bytes_written;

  return (
    <Card className="mb-4 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800/50">
      <CardHeader className="flex flex-row items-center space-x-3 pb-2">
        <FileText className="h-5 w-5 text-green-600 dark:text-green-400" />
        <CardTitle className="text-base font-medium text-green-700 dark:text-green-300">File Written</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-muted-foreground pl-10">
        <p className="mb-1">Wrote to file: <Badge variant="secondary">{filepath}</Badge></p>
        {bytesWritten !== undefined && (
          <p>Bytes Written: <span className="font-semibold">{bytesWritten}</span></p>
        )}
        {bytesWritten === undefined && (
           <p className="text-xs text-orange-500">Bytes written information not available.</p>
        )}
      </CardContent>
    </Card>
  );
};
