
import React from 'react';
import { BookText, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// Using a common style, adjust if needed
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { Trajectory } from '../../models/trajectory';
import { Card, CardContent, CardHeader } from '../ui/card';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';

interface ResearchNotesTrajectoryProps {
  trajectory: Trajectory;
}

export const ResearchNotesTrajectory: React.FC<ResearchNotesTrajectoryProps> = ({ trajectory }) => {
  const { toolParameters, created } = trajectory;
  const notes = toolParameters?.notes ?? '(No research notes)';
  // Set initial state to true so the collapsible is open by default
  const [isOpen, setIsOpen] = React.useState(true);

  // Format timestamp
  const formattedTime = created
    ? new Date(created).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Invalid Date';

  // Removed firstLine calculation, using static title now

  return (
    <Card className="mb-4">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="py-3 px-4 cursor-pointer hover:bg-muted/50">
            <div className="flex justify-between items-center">
              {/* Left side: Icon and summary */}
              <div className="flex items-center space-x-3">
                <BookText className="h-4 w-4 text-muted-foreground" />
                {/* Use static title */}
                <span>Research Notes</span>
              </div>
              {/* Right side: Timestamp and Chevron */}
              <div className="flex items-center space-x-2">
                <div className="text-xs text-muted-foreground">
                  {formattedTime}
                </div>
                {isOpen ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
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
                components={{
                  // eslint-disable-next-line @typescript-eslint/no-unused-vars
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        {...props}
                        style={vscDarkPlus} // Use the imported style
                        language={match[1]}
                        PreTag="div"
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      // For inline code or code blocks without a language, DO NOT pass the className prop
                      // to the native <code> element as react-markdown v9+ might handle it differently
                      // or throw an error. The outer div's prose class handles basic code styling.
                      <code {...props}>
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {notes}
              </ReactMarkdown>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
};
