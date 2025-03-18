/**
 * Session Store
 *
 * Zustand store for managing agent sessions. Provides state and actions for:
 * - Fetching sessions (with fallback to sample data)
 * - Selecting a session
 * - Creating a new session
 */

import { create } from 'zustand';
import { AgentSession } from '../utils/types';
import { getSampleAgentSessions } from '../utils/sample-data';
import {
  AgentSessionSchema,
  validateAgentSession,
  backendToAgentSession,
  safeBackendToAgentSession
} from '../models/session';
import { useClientConfigStore } from './clientConfigStore';

/**
 * Session store state interface
 */
interface SessionState {
  /**
   * Array of available sessions
   */
  sessions: AgentSession[];
  
  /**
   * Currently selected session ID
   */
  selectedSessionId: string | null;
  
  /**
   * Loading state for sessions
   */
  isLoading: boolean;
  
  /**
   * Error message if session loading failed
   */
  error: string | null;
}

/**
 * Session store actions interface
 */
interface SessionActions {
  /**
   * Fetch sessions, with fallback to sample data if the API fails
   */
  fetchSessions: () => Promise<void>;
  
  /**
   * Select a session by ID
   */
  selectSession: (sessionId: string | null) => void;
  
  /**
   * Create a new session
   */
  createSession: (sessionData: Partial<AgentSession>) => Promise<AgentSession>;
}

/**
 * Combined session store type
 */
type SessionStore = SessionState & SessionActions;

/**
 * Zustand store for session management
 */
export const useSessionStore = create<SessionStore>((set, get) => ({
  sessions: [],
  selectedSessionId: null,
  isLoading: false,
  error: null,
  
  /**
   * Fetch sessions, with fallback to sample data if the API fails
   */
  fetchSessions: async () => {
    set({ isLoading: true, error: null });
    
    try {
      // Get the host and port from the client config store
      const { host, port } = useClientConfigStore.getState();
      
      // Fetch sessions from the API
      const response = await fetch(`http://${host}:${port}/v1/session`);
      if (!response.ok) throw new Error(`Failed to fetch sessions: ${response.statusText}`);
      
      const data = await response.json();
      const sessions = data.items.map(safeBackendToAgentSession);
      
      set({ 
        sessions,
        isLoading: false,
        // Auto-select the first session if none is selected
        selectedSessionId: get().selectedSessionId || (sessions.length > 0 ? sessions[0].id : null)
      });
    } catch (error) {
      console.error('Error fetching sessions:', error);
      set({ 
        error: error instanceof Error ? error.message : 'Failed to fetch sessions',
        isLoading: false,
        // Fallback to sample data on error
        sessions: getSampleAgentSessions()
      });
    }
  },
  
  /**
   * Select a session by ID
   */
  selectSession: (sessionId: string | null) => {
    set({ selectedSessionId: sessionId });
  },
  
  /**
   * Create a new session
   */
  createSession: async (sessionData: Partial<AgentSession>) => {
    // Generate default session object
    const now = new Date();
    const newSession: AgentSession = {
      id: `session-${Date.now()}`, // Temporary ID, would be replaced by API response
      name: sessionData.name || 'New Session',
      created: now,
      updated: now,
      status: 'active',
      steps: [],
      ...sessionData
    };
    
    try {
      // Validate the new session
      const validatedSession = validateAgentSession(newSession);
      
      // In the future, this would send the session to the API
      // const response = await fetch('/api/v1/session', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(agentSessionToBackend(validatedSession))
      // });
      // if (!response.ok) throw new Error('Failed to create session');
      // const data = await response.json();
      // const createdSession = safeBackendToAgentSession(data);
      
      // For now, just add to the local state
      set(state => ({ 
        sessions: [...state.sessions, validatedSession],
        selectedSessionId: validatedSession.id
      }));
      
      return validatedSession;
    } catch (error) {
      console.error('Error creating session:', error);
      throw error;
    }
  }
}));