/**
 * Session Store
 *
 * Zustand store for managing agent sessions. Provides state and actions for:
 * - Fetching sessions (with fallback to sample data)
 * - Selecting a session
 * - Creating a new session (primarily via API submission)
 * - Updating session status based on real-time updates
 * - Managing new session input state
 */

import { create } from 'zustand';
import { getSampleAgentSessions } from '../utils/sample-data';
import {
  AgentSession, // Import the updated interface
  SessionStatus, // Import the status type
  validateAgentSession,
  safeBackendToAgentSession,
  BackendSession // Import BackendSession if needed for createSession update
} from '../models/session';
import { useClientConfigStore } from './clientConfigStore';
import { spawnAgent, ApiError } from '../utils/api'; // Keep ApiError if used

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
  sessions: AgentSession[]; // Uses the updated AgentSession from models

  /**
   * Currently selected session ID (now a number)
   */
  selectedSessionId: number | null;

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
   * Select a session by ID (now accepts number)
   */
  selectSession: (sessionId: number | null) => void;

  /**
   * Create a new session (Note: Primarily handled by submitNewSession + fetch)
   * This function might be deprecated or used only for optimistic UI updates.
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
   * Submit a new session message to create a real session via API
   */
  submitNewSession: (researchOnly?: boolean) => Promise<void>;

  /**
   * Update the status of a specific session (e.g., based on WebSocket message)
   */
  updateSessionStatus: (sessionId: number, status: SessionStatus) => void;

  /**
   * Update the full details of a specific session (e.g., based on WebSocket message)
   */
  updateSessionDetails: (updatedSession: AgentSession) => void; // New action
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
      // Ensure API endpoint returns the new 'status' field
      const response = await fetch(`http://${host}:${port}/v1/session`);
      if (!response.ok) throw new Error(`Failed to fetch sessions: ${response.statusText}`);

      const data = await response.json();
      // safeBackendToAgentSession now handles the status mapping
      const sessions: AgentSession[] = data.items.map(safeBackendToAgentSession);

      set((state) => ({
        sessions,
        isLoading: false,
        // Auto-select the first session if none is selected (uses number ID)
        selectedSessionId: state.selectedSessionId ?? (sessions.length > 0 ? sessions[0].id : null)
      }));
    } catch (error) {
      console.error('Error fetching sessions:', error);
      // TODO: Update sample data generation if necessary to include status and number ID
      // Note: Sample data generation is now problematic due to type mismatches.
      // Consider removing sample data fallback or creating a compatible sample generator.
      // For now, let's return an empty array on fetch error.
      // const sampleSessions = getSampleAgentSessions().map(s => ({
      //     ...s,
      //     id: parseInt(s.id.replace('sample-',''), 10), // Convert sample string ID to number
      //     status: 'unknown' as SessionStatus, // Add default status to sample data
      //     commandLine: 'Sample Command', // Add missing fields if needed
      //     createdAt: s.created,
      //     updatedAt: s.updated,
      // }));
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch sessions',
        isLoading: false,
        // Fallback to empty array on error instead of potentially incompatible sample data
        sessions: []
      });
    }
  },

  /**
   * Select a session by ID (accepts number)
   */
  selectSession: (sessionId: number | null) => {
    set((state) => ({
      selectedSessionId: sessionId,
      // Clear new session state when selecting an existing session
      newSession: sessionId !== null ? null : state.newSession // Clear if sessionId is not null
    }));
  },

  /**
   * Create a new session (Primarily for optimistic UI, might be removed later)
   * Note: This creates a local-only session object. Real sessions are created via submitNewSession.
   */
  createSession: async (sessionData: Partial<AgentSession>) => {
    const now = new Date();
    // Use a temporary negative ID for local-only sessions to avoid collision with backend IDs
    const tempId = -Date.now();

    // Define default values for an AgentSession
    const defaults = {
      name: 'New Optimistic Session',
      commandLine: 'N/A',
      createdAt: now,
      updatedAt: now,
      status: 'pending' as SessionStatus,
    };

    // Construct the new session ensuring the temporary ID is used
    const newSession: AgentSession = {
      ...defaults,
      ...sessionData, // Apply overrides from sessionData
      id: tempId, // Ensure ID is always the tempId, overriding any ID in sessionData
    };

    // No backend call here, just update local state optimistically
    set(state => ({
      sessions: [newSession, ...state.sessions], // Add to the top
      selectedSessionId: newSession.id // Select the new optimistic session
    }));

    console.warn("Created optimistic session locally:", newSession);
    return newSession; // Return the created optimistic session
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
      selectedSessionId: null // Deselect any current session
    });
  },

  /**
   * Cancel composing a new session
   */
  cancelNewSession: () => {
    set({ newSession: null });

    // Select the first session if available after cancelling
    const sessions = get().sessions;
    if (sessions.length > 0) {
      set({ selectedSessionId: sessions[0].id }); // Select first session by number ID
    }
  },

  /**
   * Update the message for a new session
   */
  updateNewSessionMessage: (message: string) => {
    set((state) => state.newSession ? {
        newSession: {
            ...state.newSession,
            message,
            error: null // Clear error on typing
        }
    } : {});
  },

  /**
   * Submit a new session message to create a real session via API
   *
   * @param researchOnly - Whether to use research-only mode
   */
  submitNewSession: async (researchOnly = false) => {
    const currentNewSession = get().newSession;

    if (!currentNewSession || currentNewSession.isSubmitting) {
      return; // Don't submit if not in new session state or already submitting
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
      }); // Assuming spawnAgent returns { session_id: number, ... }

      // Validate the session_id from the response
      if (typeof data?.session_id !== 'number') {
        throw new Error('Invalid response from spawnAgent: missing or invalid session_id');
      }
      const newSessionId: number = data.session_id;

      // Refresh the sessions to get the newly created one with its status
      // This fetch should now get the session with the name derived from the human input
      await get().fetchSessions();

      // Select the new session using the validated ID (number)
      set({
        selectedSessionId: newSessionId,
        newSession: null // Clear new session state after successful submission
      });

      // NO LONGER NEED TO RELY ON FETCH. The backend should send a session_details_update message.

    } catch (error) {
      console.error('Error submitting new session:', error);
      const currentNewSessionState = get().newSession; // Get potentially updated state
      if (currentNewSessionState) { // Check if still in new session state
          set({
            newSession: {
              ...currentNewSessionState,
              isSubmitting: false,
              error: error instanceof Error ? error.message : 'Failed to submit message'
            }
          });
      }
    }
  },

  /**
   * Update the status of a specific session
   */
  updateSessionStatus: (sessionId: number, status: SessionStatus) => {
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === sessionId ? { ...session, status: status, updatedAt: new Date() } : session // Also update updatedAt timestamp
      ),
    }));
    // Optional: Log the update
    console.log(`Store updated session ${sessionId} status to ${status}`);
  },

  /**
   * Update the full details of a specific session
   */
  updateSessionDetails: (updatedSession: AgentSession) => {
    // Ensure the incoming data is a valid AgentSession
    if (!validateAgentSession(updatedSession)) {
        console.warn('[SessionStore] Received invalid session details payload:', updatedSession);
        return;
    }

    set((state) => ({
        sessions: state.sessions.map((session) =>
            session.id === updatedSession.id ? updatedSession : session
        ),
    }));
    console.log(`Store updated session ${updatedSession.id} details. New name: ${updatedSession.name}`);
  },

}));
