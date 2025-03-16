/**
 * Sample data utility for agent UI components demonstration
 */

import { AgentStep, AgentSession } from './types';

/**
 * Returns an array of sample agent steps
 */
export function getSampleAgentSteps(): AgentStep[] {
  return [
    {
      id: "step-1",
      timestamp: new Date(Date.now() - 30 * 60000), // 30 minutes ago
      status: 'completed',
      type: 'planning',
      title: 'Initial Planning',
      content: 'I need to analyze the codebase structure to understand the existing components and their relationships.',
      duration: 5200
    },
    {
      id: "step-2",
      timestamp: new Date(Date.now() - 25 * 60000), // 25 minutes ago
      status: 'completed',
      type: 'tool-execution',
      title: 'List Directory Structure',
      content: 'Executing: list_directory_tree(path="src/", max_depth=2)\n\n📁 /project/src/\n├── 📁 components/\n│   ├── 📁 ui/\n│   └── App.tsx\n├── 📁 utils/\n└── index.tsx',
      duration: 1800
    },
    {
      id: "step-3",
      timestamp: new Date(Date.now() - 20 * 60000), // 20 minutes ago
      status: 'completed',
      type: 'thinking',
      title: 'Component Analysis',
      content: 'Based on the directory structure, I see that the UI components are organized in a dedicated folder. I should examine the existing component patterns before implementing new ones.',
      duration: 3500
    },
    {
      id: "step-4",
      timestamp: new Date(Date.now() - 15 * 60000), // 15 minutes ago
      status: 'completed',
      type: 'tool-execution',
      title: 'Read Component Code',
      content: 'Executing: read_file_tool(filepath="src/components/ui/Button.tsx")\n\n```tsx\nimport { cn } from "../../utils";\n\nexport interface ButtonProps {\n  // Component props...\n}\n\nexport function Button({ children, ...props }: ButtonProps) {\n  // Component implementation...\n}\n```',
      duration: 2100
    },
    {
      id: "step-5",
      timestamp: new Date(Date.now() - 10 * 60000), // 10 minutes ago
      status: 'completed',
      type: 'implementation',
      title: 'Creating NavBar Component',
      content: 'I\'m creating a NavBar component following the design system patterns:\n\n```tsx\nimport { cn } from "../../utils";\n\nexport interface NavBarProps {\n  // New component props...\n}\n\nexport function NavBar({ ...props }: NavBarProps) {\n  // New component implementation...\n}\n```',
      duration: 6800
    },
    {
      id: "step-6",
      timestamp: new Date(Date.now() - 5 * 60000), // 5 minutes ago
      status: 'in-progress',
      type: 'implementation',
      title: 'Styling Timeline Component',
      content: 'Currently working on styling the Timeline component to match the design system:\n\n```tsx\n// Work in progress...\nexport function Timeline({ steps, ...props }: TimelineProps) {\n  // Current implementation...\n}\n```',
    },
    {
      id: "step-7",
      timestamp: new Date(Date.now() - 2 * 60000), // 2 minutes ago
      status: 'error',
      type: 'tool-execution',
      title: 'Running Tests',
      content: 'Error executing: run_shell_command(command="npm test")\n\nTest failed: TypeError: Cannot read property \'steps\' of undefined',
      duration: 3200
    },
    {
      id: "step-8",
      timestamp: new Date(), // Now
      status: 'pending',
      type: 'planning',
      title: 'Next Steps',
      content: 'Need to plan the implementation of the SessionDrawer component...',
    }
  ];
}

/**
 * Returns an array of sample agent sessions
 */
export function getSampleAgentSessions(): AgentSession[] {
  const steps = getSampleAgentSteps();
  
  return [
    {
      id: "session-1",
      name: "UI Component Implementation",
      created: new Date(Date.now() - 35 * 60000), // 35 minutes ago
      updated: new Date(), // Now
      status: 'active',
      steps: steps
    },
    {
      id: "session-2",
      name: "API Integration",
      created: new Date(Date.now() - 2 * 3600000), // 2 hours ago
      updated: new Date(Date.now() - 30 * 60000), // 30 minutes ago
      status: 'completed',
      steps: [
        {
          id: "other-step-1",
          timestamp: new Date(Date.now() - 2 * 3600000), // 2 hours ago
          status: 'completed',
          type: 'planning',
          title: 'API Integration Planning',
          content: 'Planning the integration with the backend API...',
          duration: 4500
        },
        {
          id: "other-step-2",
          timestamp: new Date(Date.now() - 1.5 * 3600000), // 1.5 hours ago
          status: 'completed',
          type: 'implementation',
          title: 'Implementing API Client',
          content: 'Creating API client with fetch utilities...',
          duration: 7200
        },
        {
          id: "other-step-3",
          timestamp: new Date(Date.now() - 1 * 3600000), // 1 hour ago
          status: 'completed',
          type: 'tool-execution',
          title: 'Testing API Endpoints',
          content: 'Running tests against API endpoints...',
          duration: 5000
        }
      ]
    },
    {
      id: "session-3",
      name: "Bug Fixes",
      created: new Date(Date.now() - 5 * 3600000), // 5 hours ago
      updated: new Date(Date.now() - 4 * 3600000), // 4 hours ago
      status: 'error',
      steps: [
        {
          id: "bug-step-1",
          timestamp: new Date(Date.now() - 5 * 3600000), // 5 hours ago
          status: 'completed',
          type: 'planning',
          title: 'Bug Analysis',
          content: 'Analyzing reported bugs from issue tracker...',
          duration: 3600
        },
        {
          id: "bug-step-2",
          timestamp: new Date(Date.now() - 4.5 * 3600000), // 4.5 hours ago
          status: 'error',
          type: 'implementation',
          title: 'Fixing Authentication Bug',
          content: 'Error: Unable to resolve dependency conflict with auth package',
          duration: 2500
        }
      ]
    }
  ];
}