
import React from 'react';
// Changed icon import from BookText to ClipboardList
import { ClipboardList, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// Using a common style, adjust if needed
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';

interface TaskTrajectoryProps {
  trajectory: Trajectory;
}

// Renamed component from ResearchNotesTrajectory to TaskTrajectory
export const TaskTrajectory: React.FC<TaskTrajectoryProps> = ({ trajectory }) => {
  // Updated data extraction logic: Use step_data.task instead of toolParameters.notes
  const { stepData, created } = trajectory;
  const taskContent = stepData?.task ?? '(No task specification)';
  // Set initial state to true so the collapsible is open by default
  const [isOpen, setIsOpen] = React.useState(true);

  // Format timestamp
  const formattedTime = created
    ? new Date(created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Invalid Date';

  // Custom components for ReactMarkdown (kept the same for code highlighting)
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
      <Collapsible open={isOpen} onOpenChange={setIsOpen} defaultOpen={true}> {/* Ensure defaultOpen is true */}
        <CollapsibleTrigger asChild>
          <CardHeader className="py-3 px-4 cursor-pointer hover:bg-muted/50">
            <div className="flex justify-between items-center">
              {/* Left side: Icon and summary */}
              <div className="flex items-center space-x-3">
                {/* Changed icon from BookText to ClipboardList */}
                <ClipboardList className="h-4 w-4 text-muted-foreground" />
                {/* Updated title from "Research Notes" to "Task Specification" */}
                <span>Task Specification</span>
              </div>
              {/* Right side: Timestamp */}
              <div className="flex items-center space-x-2">
                <div className="text-xs text-muted-foreground">
                  {formattedTime}
                </div>
                {/* Chevron logic can be kept or removed based on preference, keeping for consistency */}
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="pt-0 pb-3 px-4">
            {/* Wrap ReactMarkdown with a div and apply classes here */}
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown
                // className prop removed from ReactMarkdown component itself
                remarkPlugins={[remarkGfm]}
                components={components} // Use the defined custom components
              >
                {/* Changed variable from notes to taskContent */}
                {taskContent}
              </ReactMarkdown>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};
