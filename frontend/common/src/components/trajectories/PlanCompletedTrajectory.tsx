
import React from 'react';
import { ClipboardCheck } from 'lucide-react'; // Import ClipboardCheck
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SyntaxHighlighter from 'react-syntax-highlighter';
// Using a common style, adjust if needed
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { MarkdownCodeBlock } from '../ui/MarkdownCodeBlock'; // Import shared code block component

interface PlanCompletedTrajectoryProps { // Renamed interface
  trajectory: Trajectory;
}

export const PlanCompletedTrajectory: React.FC<PlanCompletedTrajectoryProps> = ({ trajectory }) => { // Renamed component
  const { stepData, created } = trajectory;
  const message = stepData?.completion_message ?? '(No completion message)';
  // Set initial state to true so the collapsible is open by default
  const [isOpen, setIsOpen] = React.useState(true);

  // Format timestamp to HH:mm
  const formattedTime = created
    ? new Date(created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Invalid Date';

  // Custom components for ReactMarkdown using shared MarkdownCodeBlock
  const components: Components = {
    code: MarkdownCodeBlock, // Use the shared component
  };

  return (
    <Card className="mb-4">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="py-3 px-4 cursor-pointer hover:bg-muted/50">
            <div className="flex justify-between items-center">
              {/* Left side: Icon and title */}
              <div className="flex items-center space-x-3">
                <ClipboardCheck className="h-4 w-4 text-blue-500" /> {/* Using ClipboardCheck and changed color for distinction */}
                <span>Plan Executed</span> {/* Updated title */}
              </div>
              {/* Right side: Timestamp */}
              <div className="flex items-center space-x-2">
                <div className="text-xs text-muted-foreground">
                  {formattedTime}
                </div>
                {/* No Chevron icon here */}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="pt-0 pb-3 px-4">
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={components} // Use the defined custom components
              >
                {message}
              </ReactMarkdown>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};
