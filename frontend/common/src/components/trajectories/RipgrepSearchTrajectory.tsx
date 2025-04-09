
import React from 'react';
import { AnsiUp } from 'ansi_up/ansi_up';
import { Search } from 'lucide-react';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';

interface RipgrepSearchTrajectoryProps {
  trajectory: Trajectory;
}

const ansiUp = new AnsiUp();

export const RipgrepSearchTrajectory: React.FC<RipgrepSearchTrajectoryProps> = ({ trajectory }) => {
  const { stepData, toolResult, created } = trajectory;
  const searchPattern = stepData?.search_pattern ?? 'unknown pattern';
  const includePaths = stepData?.include_paths ?? [];
  const output = toolResult?.output ?? '(No output)';
  const success = toolResult?.success ?? false;
  const returnCode = toolResult?.return_code ?? 'N/A';

  // Format timestamp
  const formattedTime = created
    ? new Date(created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Invalid Date';

  // Convert ANSI to HTML
  const htmlOutput = ansiUp.ansi_to_html(output);

  return (
    <Card className="mb-4">
      <Collapsible defaultOpen={false}>
        <CollapsibleTrigger asChild>
            <CardHeader className="py-3 px-4 cursor-pointer hover:bg-muted/50">
              <div className="flex justify-between items-center">
                {/* Left side: Icon and summary */}
                <div className="flex items-center space-x-3"> {/* Removed text-sm font-medium */}
                  <Search className="h-4 w-4 text-muted-foreground" />
                  <span>
                      Searched for <b>{searchPattern}</b> {/* Added "Searched for " */}
                      {includePaths.length > 0 && (
                          <>
                              <span className="mx-1">in</span>
                              <i>{includePaths.join(', ')}</i>
                          </>
                      )}
                  </span>
                </div>
                {/* Right side: Timestamp */}
                <div className="text-xs text-muted-foreground">
                  {formattedTime}
                </div>
              </div>
            </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
            <CardContent className="pt-0 pb-3 px-4">
              {/* Display the formatted output */}
              <pre
                className="text-xs bg-background p-3 rounded-md overflow-auto border border-border"
                dangerouslySetInnerHTML={{ __html: htmlOutput }}
              />
              {/* Display exit code if not successful */}
              {!success && (
                <p className="text-xs text-red-500 mt-2">
                  Search failed with exit code: {returnCode}
                </p>
              )}
            </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};
