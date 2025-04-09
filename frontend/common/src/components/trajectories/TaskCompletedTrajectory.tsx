
import React from 'react';
import { CheckCircle } from 'lucide-react';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SyntaxHighlighter from 'react-syntax-highlighter';
// Using a common style, adjust if needed
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible'; // Import CollapsibleTrigger

interface TaskCompletedTrajectoryProps {
  trajectory: Trajectory;
}

export const TaskCompletedTrajectory: React.FC<TaskCompletedTrajectoryProps> = ({ trajectory }) => {
  const { stepData, created } = trajectory;
  const message = stepData?.completion_message ?? '(No completion message)';
  // Set initial state to true so the collapsible is open by default
  const [isOpen, setIsOpen] = React.useState(true);

  // Format timestamp to HH:mm
  const formattedTime = created
    ? new Date(created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Invalid Date';

  // Custom components for ReactMarkdown (handles code blocks and inline code)
  const components: Components = {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    code({ className, children, ...props }) { // Removed node and inline destructuring
      const match = /language-(\w+)/.exec(className || '');
      // Correctly handle children which might be an array
      const childrenArray = React.Children.toArray(children);
      const codeString = childrenArray.map(child =>
        typeof child === 'string' ? child : '' // Handle potential non-string children if necessary, though typically it's string or array of strings
      ).join('').replace(/\n$/, '');


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
          <code className={className || 'inline-code'}>
            {children}
          </code>
        );
      }
    },
  }

  return (
    <Card className="mb-4">
      {/* Use Collapsible structure like ResearchNotes, but without visible trigger icon */}
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        {/* Wrap CardHeader in CollapsibleTrigger */}
        <CollapsibleTrigger asChild>
          {/* Ensure no explicit chevron icon is added here */}
          <CardHeader className="py-3 px-4 cursor-pointer hover:bg-muted/50">
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
                {/* No Chevron icon here */}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="pt-0 pb-3 px-4">
            {/* Wrap ReactMarkdown with a div and apply classes here */}
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
