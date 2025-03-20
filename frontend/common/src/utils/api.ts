/**
 * API utilities for making requests to the RA.Aid server
 */

import { useClientConfigStore } from '../store/clientConfigStore';

/**
 * Interface for spawn agent request parameters
 */
export interface SpawnAgentParams {
  /**
   * Message content to send
   */
  message: string;
  
  /**
   * Whether to use research-only mode
   */
  research_only?: boolean;
}

/**
 * Interface for spawn agent response
 */
export interface SpawnAgentResponse {
  /**
   * ID of the created session
   */
  session_id: string;
  
  /**
   * Status message
   */
  status: string;
}

/**
 * Error class for API errors
 */
export class ApiError extends Error {
  /**
   * HTTP status code
   */
  statusCode: number;
  
  /**
   * Response object
   */
  response: Response;
  
  constructor(message: string, statusCode: number, response: Response) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.response = response;
  }
}

/**
 * Makes a request to spawn a new agent with the given message
 * 
 * @param params - Request parameters
 * @returns Promise with the spawn agent response
 * @throws ApiError if the request fails
 */
export async function spawnAgent(params: SpawnAgentParams): Promise<SpawnAgentResponse> {
  const { host, port } = useClientConfigStore.getState();
  
  try {
    const response = await fetch(`http://${host}:${port}/v1/spawn-agent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: params.message,
        research_only: params.research_only || false
      })
    });
    
    if (!response.ok) {
      throw new ApiError(
        `Failed to spawn agent: ${response.statusText}`,
        response.status,
        response
      );
    }
    
    const data = await response.json();
    return data as SpawnAgentResponse;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    
    throw new Error(
      error instanceof Error 
        ? `Failed to spawn agent: ${error.message}` 
        : 'Failed to spawn agent: Unknown error'
    );
  }
}