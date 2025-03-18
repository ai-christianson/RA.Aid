/**
 * Trajectory Store
 *
 * Zustand store for managing agent trajectory data. Provides state and actions for:
 * - Fetching all trajectories (note: this may need to be implemented differently based on API)
 * - Fetching trajectories for a specific session
 * - Clearing trajectory data
 */

import { create } from 'zustand';
import { useClientConfigStore } from './clientConfigStore';
import { 
  Trajectory, 
  safeBackendToTrajectory,
  getSampleTrajectory
} from '../models/trajectory';

/**
 * Trajectory store state interface
 */
export interface TrajectoryState {
  /**
   * Array of trajectories
   */
  trajectories: Trajectory[];
  
  /**
   * Loading state for trajectories
   */
  isLoading: boolean;
  
  /**
   * Error message if trajectory loading failed
   */
  error: string | null;
}

/**
 * Trajectory store actions interface
 */
export interface TrajectoryActions {
  /**
   * Fetch all trajectories
   * 
   * Note: The backend may not have a direct endpoint to fetch all trajectories.
   * This method is provided for completeness but may not work as expected without
   * a proper backend endpoint.
   */
  fetchTrajectories: () => Promise<void>;
  
  /**
   * Fetch trajectories for a specific session
   * @param sessionId - ID of the session to fetch trajectories for
   */
  fetchSessionTrajectories: (sessionId: number) => Promise<void>;
  
  /**
   * Clear all trajectory data
   */
  clearTrajectories: () => void;
}

/**
 * Combined trajectory store type
 */
export type TrajectoryStore = TrajectoryState & TrajectoryActions;

/**
 * Zustand store for trajectory management
 */
export const useTrajectoryStore = create<TrajectoryStore>((set) => ({
  trajectories: [],
  isLoading: false,
  error: null,
  
  /**
   * Fetch all trajectories
   * 
   * Note: The backend may not have a direct endpoint to fetch all trajectories.
   * This method attempts to use a hypothetical endpoint and falls back to clearing the data.
   */
  fetchTrajectories: async () => {
    set({ isLoading: true, error: null });
    
    try {
      // Get the host and port from the client config store
      const { host, port } = useClientConfigStore.getState();
      
      // Note: There doesn't seem to be an endpoint to fetch all trajectories directly
      // This is a placeholder that attempts to use a hypothetical endpoint
      const response = await fetch(`http://${host}:${port}/v1/trajectory`);
      if (!response.ok) {
        throw new Error(`Failed to fetch trajectories: ${response.statusText}`);
      }
      
      const data = await response.json();
      const trajectories = data.map(safeBackendToTrajectory).filter(Boolean) as Trajectory[];
      
      set({ 
        trajectories,
        isLoading: false
      });
    } catch (error) {
      console.error('Error fetching trajectories:', error);
      set({ 
        error: error instanceof Error 
          ? error.message 
          : 'Failed to fetch trajectories - endpoint may not exist',
        isLoading: false,
        // Clear trajectories on error
        trajectories: []
      });
    }
  },
  
  /**
   * Fetch trajectories for a specific session
   * @param sessionId - ID of the session to fetch trajectories for
   */
  fetchSessionTrajectories: async (sessionId: number) => {
    console.log(`[TrajectoryStore] Fetching trajectories for session ID: ${sessionId}`);
    set({ isLoading: true, error: null });
    
    try {
      // Get the host and port from the client config store
      const { host, port } = useClientConfigStore.getState();
      
      // Fetch trajectories for the session from the API
      const url = `http://${host}:${port}/v1/session/${sessionId}/trajectory`;
      console.log(`[TrajectoryStore] Making API request to: ${url}`);
      
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to fetch trajectories for session ${sessionId}: ${response.statusText}`);
      
      const data = await response.json();
      console.log('[TrajectoryStore] Raw API response data:', data);
      
      const trajectories = data.map(safeBackendToTrajectory).filter(Boolean) as Trajectory[];
      console.log(`[TrajectoryStore] Converted trajectories (${trajectories.length}):`, trajectories);
      
      set({ 
        trajectories,
        isLoading: false
      });
    } catch (error) {
      console.error(`[TrajectoryStore] ERROR fetching trajectories for session ${sessionId}:`, error);
      console.error('[TrajectoryStore] Error details:', {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : 'No stack trace'
      });
      
      set({ 
        error: error instanceof Error ? error.message : `Failed to fetch trajectories for session ${sessionId}`,
        isLoading: false,
        // Clear trajectories on error
        trajectories: []
      });
    }
  },
  
  /**
   * Clear all trajectory data
   */
  clearTrajectories: () => {
    set({
      trajectories: [],
      error: null
    });
  }
}));