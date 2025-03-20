/**
 * Session Store
 *
 * Zustand store for managing agent sessions. Provides state and actions for:
 * - Fetching sessions (with fallback to sample data)
 * - Selecting a session
 * - Creating a new session
 * - Managing new session state
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
import { spawnAgent, ApiError } from '../utils/api';

/**
 * Interface for tracking the state of a message being composed for a new session
 */
interface NewSessionState {
  /**
   * Message content being composed
   */
  message: string;
  
  /**
   * Whether the message is currently being submitted
   */
  isSubmitting: boolean;
  
  /**
   * Error message if submission failed
   */
  error: string | null;
}

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
   * State for a new session being composed (null if not composing)
   */
  newSession: NewSessionState | null;
  
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
  
  /**
   * Start composing a new session
   */
  startNewSession: () => void;
  
  /**
   * Cancel composing a new session
   */
  cancelNewSession: () => void;
  
  /**
   * Update the message for a new session
   */
  updateNewSessionMessage: (message: string) => void;
  
  /**
   * Submit a new session message to create a real session
   */
  submitNewSession: (researchOnly?: boolean) => Promise<void>;
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
  newSession: null,
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
    set({ 
      selectedSessionId: sessionId,
      // Clear new session state when selecting an existing session
      newSession: sessionId ? null : get().newSession
    });
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
  },
  
  /**
   * Start composing a new session
   */
  startNewSession: () => {
    set({
      newSession: {
        message: '',
        isSubmitting: false,
        error: null
      },
      selectedSessionId: null
    });
  },
  
  /**
   * Cancel composing a new session
   */
  cancelNewSession: () => {
    set({ newSession: null });
    
    // Select the first session if available
    const sessions = get().sessions;
    if (sessions.length > 0) {
      set({ selectedSessionId: sessions[0].id });
    }
  },
  
  /**
   * Update the message for a new session
   */
  updateNewSessionMessage: (message: string) => {
    const currentNewSession = get().newSession;
    
    if (currentNewSession) {
      set({
        newSession: {
          ...currentNewSession,
          message,
          error: null
        }
      });
    }
  },
  
  /**
   * Submit a new session message to create a real session
   * 
   * @param researchOnly - Whether to use research-only mode
   */
  submitNewSession: async (researchOnly = false) => {
    const currentNewSession = get().newSession;
    
    if (!currentNewSession) {
      return;
    }
    
    // If message is empty, set an error and return
    if (!currentNewSession.message.trim()) {
      set({
        newSession: {
          ...currentNewSession,
          error: 'Message cannot be empty'
        }
      });
      return;
    }
    
    // Set isSubmitting to true
    set({
      newSession: {
        ...currentNewSession,
        isSubmitting: true,
        error: null
      }
    });
    
    try {
      // Call the spawn agent API
      const data = await spawnAgent({
        message: currentNewSession.message,
        research_only: researchOnly
      });
      
      // Refresh the sessions to get the newly created one
      await get().fetchSessions();
      
      // Select the new session
      set({
        selectedSessionId: data.session_id,
        newSession: null
      });
      
    } catch (error) {
      console.error('Error submitting new session:', error);
      
      set({
        newSession: {
          ...currentNewSession,
          isSubmitting: false,
          error: error instanceof Error ? error.message : 'Failed to submit message'
        }
      });
    }
  }
}));