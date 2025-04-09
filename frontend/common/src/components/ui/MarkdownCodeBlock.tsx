// In frontend/common/src/components/ui/MarkdownCodeBlock.tsx
import React from 'react';
// It's crucial to import the type correctly. If this specific path doesn't work,
// check the installed react-markdown version's structure.
// Using CodeProps from react-markdown directly as the specific subpath might be unstable
import { CodeProps } from 'react-markdown/lib/ast-to-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// Ensure this style is appropriate or adjust as needed for the project's theme
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'; // Using VSC Dark Plus theme

export const MarkdownCodeBlock: React.FC<CodeProps> = ({ node, inline, className, children, ...props }) => {
  // 1. Robustly extract the code string
  let codeString = '';
  if (Array.isArray(children)) {
    codeString = children.join(''); // Join array elements
  } else if (typeof children === 'string') {
    codeString = children; // Use string directly
  } else if (children) {
    // Fallback for unexpected types, attempt string conversion
    codeString = String(children);
  } else {
      codeString = ''; // Handle null or undefined children
  }

  // 2. Clean up trailing newline, often added by markdown processors
  codeString = codeString.replace(/\n$/, '');

  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : undefined;

  // Check if node has properties indicating it's a code block (might depend on markdown processor)
  // Simple check: if it's not inline and has a language, treat as block
  const isCodeBlock = !inline && language;

  return isCodeBlock ? ( // Apply syntax highlighting only for non-inline blocks with a language
    <SyntaxHighlighter
      style={vscDarkPlus} // Use your theme
      language={language}
      PreTag="div" // Use div for block display
      {...props} // Pass down other props
    >
      {codeString}
    </SyntaxHighlighter>
  ) : (
    // For inline code or blocks without a specific language, render a simple <code> tag
    <code className={className} {...props}> {/* Pass down className and other props */}
      {codeString} {/* Use the processed string */}
    </code>
  );
};
