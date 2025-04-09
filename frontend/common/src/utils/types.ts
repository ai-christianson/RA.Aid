/**
 * Common types for agent UI components
 */

/**
 * Represents a single step in the agent process
 * NOTE: This might be deprecated in favor of the Trajectory model.
 */
export interface AgentStep {
  id: string;
  timestamp: Date;
  status: 'completed' | 'in-progress' | 'error' | 'pending';
  type: 'tool-execution' | 'thinking' | 'planning' | 'implementation' | 'user-input';
  title: string;
  content: string;
  duration?: number; // in milliseconds
}

/**
 * Represents a session with multiple steps
 * DEPRECATED: Use AgentSession from '../models/session' instead.
 */
// export interface AgentSession {
//   id: string;
//   name: string;
//   created: Date;
//   updated: Date;
//   status: 'active' | 'completed' | 'error';
//   steps: AgentStep[];
// }
