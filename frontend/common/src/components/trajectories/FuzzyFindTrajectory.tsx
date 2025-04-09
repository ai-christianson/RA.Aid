import React from 'react';
import { Trajectory } from '../../models/trajectory'; // Corrected relative path
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { Badge } from '../ui/badge';
import { ScrollArea } from '../ui/scroll-area';

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

  return (
    <Card className="mb-4">
      <Collapsible defaultOpen={true}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer p-4 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-t-lg">
            <CardTitle className="text-lg font-medium">
              {title}: <span className="font-mono text-sm bg-gray-100 dark:bg-gray-700 px-1 py-0.5 rounded">{searchTerm}</span>
            </CardTitle>
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
                  {/* Search Parameters - Add more as needed from stepData */}
                  {stepData?.threshold !== undefined && (
                    <p><strong>Threshold:</strong> {stepData.threshold}</p>
                  )}
                  {stepData?.max_results !== undefined && (
                     <p><strong>Max Results:</strong> {stepData.max_results}</p>
                  )}
                   {stepData?.include_hidden !== undefined && (
                     <p><strong>Include Hidden:</strong> {stepData.include_hidden ? 'Yes' : 'No'}</p>
                  )}
                   {/* Add include_paths and exclude_patterns if they are included in stepData */}
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
