
import React from 'react';
import { CheckCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Import the new shared component
import { MarkdownCodeBlock } from '../ui/MarkdownCodeBlock';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent } from '../ui/collapsible';
import { formatTimestamp } from '../../utils/index'; // Assuming formatTimestamp is here
import SyntaxHighlighter from 'react-syntax-highlighter';

interface TaskCompletedTrajectoryProps {
  trajectory: Trajectory;
}

export const TaskCompletedTrajectory: React.FC<TaskCompletedTrajectoryProps> = ({ trajectory }) => {
  const { stepData, created } = trajectory;
  const message = stepData?.completion_message ?? '(No completion message)';
  // Set initial state to true so the collapsible is open by default
  const [isOpen, setIsOpen] = React.useState(true);

  // Format timestamp
  const formattedTime = formatTimestamp(created);

  
  // Custom components for ReactMarkdown
  const components: Components = {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    code({ className, children, ...props }) { // Removed node and inline destructuring
      const match = /language-(\w+)/.exec(className || '');
      const codeString = String(children).replace(/\n$/, ''); // Ensure children is a string

      // Check only for match to determine if it's a highlighted block
      if (match) {
        // For block code with language
        return (
          <SyntaxHighlighter
            style={vscDarkPlus}
            language={match[1]}
            PreTag="div"
          >
            {codeString}
          </SyntaxHighlighter>
        );
      } else {
        // For inline code or block code without language
        return (
          <code className={className}>
            {children}
          </code>
        );
      }
    },
  }

  return (
    <Card className="mb-4">
      {/* Collapsible is kept for structure but not user-interactive for collapsing */}
      <Collapsible open={isOpen} /* onOpenChange removed */ >
        {/* No CollapsibleTrigger, CardHeader is rendered directly */}
        <CardHeader className="py-3 px-4">
          <div className="flex justify-between items-center">
            {/* Left side: Icon and title */}
            <div className="flex items-center space-x-3">
              <CheckCircle className="h-4 w-4 text-green-500" /> {/* Using CheckCircle */}
              <span>Task Completed</span> {/* Static title */}
            </div>
            {/* Right side: Timestamp */}
            <div className="flex items-center space-x-2">
              <div className="text-xs text-muted-foreground">
                {formattedTime}
              </div>
            </div>
          </div>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="pt-0 pb-3 px-4">
            {/* Wrap ReactMarkdown with a div and apply classes here */}
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown
                // className prop removed from ReactMarkdown component itself
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
