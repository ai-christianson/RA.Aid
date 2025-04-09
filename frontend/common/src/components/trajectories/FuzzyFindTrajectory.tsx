import React from 'react';
import { Trajectory } from '../../models/trajectory'; // Corrected relative path
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';
import { Search } from 'lucide-react'; // Import an icon if desired

interface FuzzyFindStepData {
  search_term: string;
  repo_path?: string; // Optional for now, might add later
  threshold?: number;
  max_results?: number;
  include_hidden?: boolean;
  include_paths?: string[];
  exclude_patterns?: string[];
  matches?: [string, number][];
  total_files?: number;
  matches_found?: number;
  error_message?: string;
  error_type?: string;
  display_title?: string; // "Fuzzy Find Results" or "Fuzzy Find Error"
}

interface FuzzyFindTrajectoryProps {
  trajectory: Trajectory<FuzzyFindStepData>;
}

export const FuzzyFindTrajectory: React.FC<FuzzyFindTrajectoryProps> = ({ trajectory }) => {
  const stepData = trajectory.stepData;
  const isError = !!stepData?.error_message;
  const title = stepData?.display_title || (isError ? 'Fuzzy Find Error' : 'Fuzzy Find Results');
  const searchTerm = stepData?.search_term || 'N/A';
  const formattedTime = new Date(trajectory.created).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <Card className="mb-4">
      <Collapsible defaultOpen={true}>
        <CollapsibleTrigger asChild>
          {/* Updated Header: Padding changed, hover removed, flex layout added, CardTitle replaced with span, timestamp added */}
          <CardHeader className="py-3 px-4 cursor-pointer rounded-t-lg">
            <div className="flex justify-between items-center">
              {/* Left side: Icon and Title */}
              <div className="flex items-center space-x-3">
                 <Search className="h-4 w-4 text-muted-foreground" /> {/* Optional: Added Search icon */}
                <span>
                  {title}: <span className="font-mono text-sm bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded">{searchTerm}</span>
                </span>
              </div>
              {/* Right side: Timestamp */}
              <div className="flex items-center space-x-2">
                <div className="text-xs text-muted-foreground">
                  {formattedTime}
                </div>
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="p-4 text-sm">
            {isError ? (
              <div className="text-red-600 dark:text-red-400">
                <p><strong>Error Type:</strong> {stepData?.error_type || 'Unknown Error'}</p>
                <p><strong>Message:</strong> {stepData?.error_message}</p>
              </div>
            ) : (
              <div>
                <div className="mb-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {/* Search Parameters */}
                  {stepData?.threshold !== undefined && (
                    <p><strong>Threshold:</strong> {stepData.threshold}</p>
                  )}
                  {stepData?.max_results !== undefined && (
                     <p><strong>Max Results:</strong> {stepData.max_results}</p>
                  )}
                   {stepData?.include_hidden !== undefined && (
                     <p><strong>Include Hidden:</strong> {stepData.include_hidden ? 'Yes' : 'No'}</p>
                  )}
                   {/* Add include_paths and exclude_patterns if needed */}
                </div>

                {/* Result Statistics */}
                {(stepData?.matches_found !== undefined && stepData?.total_files !== undefined) && (
                  <div className="mb-3 text-xs">
                    <p><strong>Matches Found:</strong> {stepData.matches_found} / <strong>Total Files Searched:</strong> {stepData.total_files}</p>
                  </div>
                )}

                {/* Match List */}
                {stepData?.matches && stepData.matches.length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-2 text-sm">Top Matches:</h4>
                    <ScrollArea className="h-40 border rounded-md p-2 bg-gray-50 dark:bg-gray-900">
                      <ul>
                        {stepData.matches.map(([path, score], index) => (
                          <li key={index} className="flex justify-between items-center mb-1 text-xs">
                            <span className="font-mono break-all">{path}</span>
                            <Badge variant="secondary">{score}</Badge>
                          </li>
                        ))}
                      </ul>
                    </ScrollArea>
                  </div>
                )}
                 {stepData?.matches && stepData.matches.length === 0 && (
                    <p className="text-gray-500 dark:text-gray-400">No matches found.</p>
                 )}

              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};
