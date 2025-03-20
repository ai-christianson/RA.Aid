/**
 * Client Configuration Store
 *
 * Zustand store for managing client configuration. Provides state and actions for:
 * - Fetching server configuration
 * - Updating configuration values
 * - Resetting to default values
 */

import { create } from 'zustand';
import {
  ClientConfig,
  ClientConfigSchema,
  validateClientConfig
} from '../models/clientConfig';

/**
 * Client configuration state interface
 */
interface ClientConfigState {
  /**
   * Server host address
   */
  host: string;
  
  /**
   * Server port number
   */
  port: number;
  
  /**
   * Loading state for configuration
   */
  isLoading: boolean;
  
  /**
   * Error message if configuration loading failed
   */
  error: string | null;
}

/**
 * Client configuration actions interface
 */
interface ClientConfigActions {
  /**
   * Update configuration values
   * 
   * @param config - Partial configuration object with values to update
   */
  updateConfig: (config: Partial<ClientConfig>) => void;
  
  /**
   * Fetch configuration from the server's /config endpoint
   */
  fetchConfig: () => Promise<void>;
  
  /**
   * Reset configuration to default values (localhost:1818)
   */
  resetConfig: () => void;
}

/**
 * Combined client configuration store type
 */
type ClientConfigStore = ClientConfigState & ClientConfigActions;

/**
 * Default configuration values
 */
const DEFAULT_CONFIG = ClientConfigSchema.parse({
  host: 'localhost',
  port: 1818
});

/**
 * Zustand store for client configuration
 */
export const useClientConfigStore = create<ClientConfigStore>((set, get) => ({
  ...DEFAULT_CONFIG,
  isLoading: false,
  error: null,
  
  /**
   * Update configuration values
   */
  updateConfig: (config) => {
    try {
      const validatedConfig = validateClientConfig({
        ...get(),
        ...config
      });
      
      set((state) => ({
        ...state,
        host: validatedConfig.host,
        port: validatedConfig.port,
        error: null
      }));
    } catch (error) {
      console.error('Error updating configuration:', error);
      set({ 
        error: error instanceof Error ? error.message : 'Invalid configuration'
      });
    }
  },
  
  /**
   * Fetch configuration from the server's /config endpoint
   */
  fetchConfig: async () => {
    set({ isLoading: true, error: null });
    
    try {
      const { host, port } = get();
      const response = await fetch(`http://${host}:${port}/config`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch configuration: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Validate the received configuration
      const validatedConfig = validateClientConfig({
        host: data.host,
        port: data.port
      });
      
      set({ 
        host: validatedConfig.host,
        port: validatedConfig.port,
        isLoading: false
      });
    } catch (error) {
      console.error('Error fetching configuration:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch configuration',
        isLoading: false
      });
    }
  },
  
  /**
   * Reset configuration to default values
   */
  resetConfig: () => {
    set({
      ...DEFAULT_CONFIG,
      isLoading: false,
      error: null
    });
  }
}));