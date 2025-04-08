/**
 * Trajectory Store
 *
 * Zustand store for managing agent trajectory data. Provides state and actions for:
 * - Fetching all trajectories (note: this may need to be implemented differently based on API)
 * - Fetching trajectories for a specific session
 * - Clearing trajectory data
 * - Adding or updating a single trajectory (for real-time updates)
 */

import { create } from 'zustand';
import { useClientConfigStore } from './clientConfigStore';
import {
  Trajectory,
  safeBackendToTrajectory,
  getSampleTrajectory // Assuming this might be used elsewhere, keep it for now
} from '../models/trajectory';

/**
 * Trajectory store state interface
 */
export interface TrajectoryState {
  /**
   * Array of trajectories, potentially sorted by ID or another criterion.
   */
  trajectories: Trajectory[];

  /**
   * Loading state for trajectories (usually for initial fetch)
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
   * Fetch all trajectories (potentially problematic, see implementation notes).
   */
  fetchTrajectories: () => Promise<void>;

  /**
   * Fetch trajectories for a specific session via HTTP.
   * @param sessionId - ID of the session to fetch trajectories for.
   */
  fetchSessionTrajectories: (sessionId: number) => Promise<void>;

  /**
   * Clear all trajectory data.
   */
  clearTrajectories: () => void;

  /**
   * Adds a new trajectory or updates an existing one based on its ID.
   * Intended for use with WebSocket updates. Maintains sorted order by ID.
   * @param trajectory - The trajectory object to add or update.
   */
  addOrUpdateTrajectory: (trajectory: Trajectory) => void; // Task 3: Add signature
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
   * Fetch all trajectories (Potentially deprecated or non-functional)
   */
  fetchTrajectories: async () => {
    set({ isLoading: true, error: null });
    try {
      const { host, port } = useClientConfigStore.getState();
      const response = await fetch(`http://${host}:${port}/v1/trajectory`); // Hypothetical endpoint
      if (!response.ok) throw new Error(`Failed to fetch trajectories: ${response.statusText}`);
      const data = await response.json();
      const trajectories = data.map(safeBackendToTrajectory).filter(Boolean) as Trajectory[];
      set({ trajectories, isLoading: false });
    } catch (error) {
      console.error('Error fetching trajectories:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch trajectories - endpoint may not exist',
        isLoading: false,
        trajectories: [] // Clear on error
      });
    }
  },

  /**
   * Fetch trajectories for a specific session via HTTP.
   */
  fetchSessionTrajectories: async (sessionId: number) => {
    console.log(`[TrajectoryStore] Fetching trajectories for session ID: ${sessionId}`);
    set({ isLoading: true, error: null, trajectories: [] }); // Clear previous trajectories on new fetch
    try {
      const { host, port } = useClientConfigStore.getState();
      const url = `http://${host}:${port}/v1/session/${sessionId}/trajectory`;
      console.log(`[TrajectoryStore] Making API request to: ${url}`);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to fetch trajectories for session ${sessionId}: ${response.statusText} (Status: ${response.status})`);

      // Check for empty response before parsing JSON
      const responseText = await response.text();
      if (!responseText) {
         console.log(`[TrajectoryStore] Received empty response for session ${sessionId}.`);
         set({ trajectories: [], isLoading: false });
         return;
      }

      const data = JSON.parse(responseText); // Parse text after ensuring it's not empty
      console.log('[TrajectoryStore] Raw API response data:', data);

      if (!Array.isArray(data)) {
         throw new Error(`Invalid API response: Expected an array, got ${typeof data}`);
      }

      const trajectories = data.map(safeBackendToTrajectory).filter(Boolean) as Trajectory[];
      // Ensure sorting after fetch as well
      trajectories.sort((a, b) => a.id - b.id);
      console.log(`[TrajectoryStore] Converted & sorted trajectories (${trajectories.length}):`, trajectories);

      set({ trajectories, isLoading: false });
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
        trajectories: [] // Clear on error
      });
    }
  },

  /**
   * Clear all trajectory data.
   */
  clearTrajectories: () => {
    set({
      trajectories: [],
      error: null,
      isLoading: false // Also reset loading state if cleared manually
    });
  },

  /**
   * Task 3: Implement addOrUpdateTrajectory action
   * Adds a new trajectory or updates an existing one, maintaining sort order by ID.
   */
  addOrUpdateTrajectory: (trajectory: Trajectory) => set((state) => {
    // Defensive check: Ensure the input has a valid ID.
    if (typeof trajectory?.id !== 'number') {
        console.warn('[TrajectoryStore] addOrUpdateTrajectory called with invalid trajectory (missing or invalid ID):', trajectory);
        return {}; // No state change
    }

    const index = state.trajectories.findIndex((t) => t.id === trajectory.id);
    let newTrajectories: Trajectory[];

    if (index > -1) {
      // Update existing trajectory
      console.log(`[TrajectoryStore] Updating trajectory ID: ${trajectory.id}`);
      newTrajectories = state.trajectories.map((t, i) => i === index ? trajectory : t);
    } else {
      // Add new trajectory
      console.log(`[TrajectoryStore] Adding new trajectory ID: ${trajectory.id}`);
      newTrajectories = [...state.trajectories, trajectory];
      // Sort only when adding a new item to maintain order
      newTrajectories.sort((a, b) => a.id - b.id);
    }

    return { trajectories: newTrajectories };
  }),

}));
