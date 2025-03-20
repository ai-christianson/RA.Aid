/**
 * Session model with Zod validation
 * 
 * This module provides:
 * 1. Schema definitions for backend and frontend session models
 * 2. Type definitions using Zod inference
 * 3. Conversion functions between backend and frontend formats
 * 4. Validation functions
 */

import { z } from 'zod';
import { AgentSession, AgentStep } from '../utils/types';

/**
 * Helper function to create a resilient datetime schema that:
 * - Accepts null or undefined values
 * - Handles invalid date formats gracefully
 * - Returns null for invalid dates instead of throwing
 */
const resilientDatetime = () => 
  z.union([
    z.string(),
    z.null(),
    z.undefined()
  ])
  .transform(val => {
    // Handle null or undefined
    if (val === null || val === undefined) {
      return null;
    }
    
    try {
      // Attempt to parse the date
      const date = new Date(val);
      
      // Check if the date is valid
      if (isNaN(date.getTime())) {
        return null;
      }
      
      // Convert to ISO string for consistency
      return date.toISOString();
    } catch {
      return null;
    }
  });

/**
 * Schema for the backend session model format received from the API
 */
export const BackendSessionSchema = z.object({
  id: z.number().optional(),
  created_at: resilientDatetime(),
  updated_at: resilientDatetime(),
  start_time: resilientDatetime(),
  command_line: z.string().optional(),
  program_version: z.string().optional(),
  // Create a refined schema for machine_info that properly transforms string to object
  machine_info: z.union([
    z.record(z.string(), z.any()),  // Already an object
    z.string().transform((str) => {
      try {
        return str ? JSON.parse(str) : null;
      } catch (e) {
        console.error('Failed to parse machine_info JSON:', e);
        return null;
      }
    }),
    z.null(),
    z.undefined()
  ]).nullable().optional(),
  display_name: z.string().optional(),
});

/**
 * Type for the backend session model format, inferred from the schema
 */
export type BackendSession = z.infer<typeof BackendSessionSchema>;

/**
 * Zod schema for validating AgentStep objects
 */
export const AgentStepSchema: z.ZodType<AgentStep> = z.object({
  id: z.string(),
  timestamp: z.date(),
  status: z.enum(['completed', 'in-progress', 'error', 'pending']),
  type: z.enum(['tool-execution', 'thinking', 'planning', 'implementation', 'user-input']),
  title: z.string(),
  content: z.string(),
  duration: z.number().optional(),
});

/**
 * Zod schema for validating AgentSession objects
 */
export const AgentSessionSchema: z.ZodType<AgentSession> = z.object({
  id: z.string(),
  name: z.string(),
  created: z.date(),
  updated: z.date(),
  status: z.enum(['active', 'completed', 'error']),
  steps: z.array(AgentStepSchema),
});

/**
 * Converts a backend session model to a frontend AgentSession model
 * 
 * @param backendSession The backend session model to convert
 * @returns The converted frontend AgentSession model
 */
export function backendToAgentSession(backendSession: BackendSession): AgentSession {
  // Create default date objects for null/undefined datetime fields
  const now = new Date();
  
  return {
    id: backendSession.id !== undefined ? backendSession.id.toString() : '',
    name: backendSession.display_name || 'Unnamed Session',
    created: backendSession.created_at ? new Date(backendSession.created_at) : now,
    updated: backendSession.updated_at ? new Date(backendSession.updated_at) : now,
    status: 'active', // Default status since backend doesn't have this
    steps: [], // Default empty steps since backend doesn't have this
  };
}

/**
 * Converts a frontend AgentSession model to a backend session model
 * 
 * @param agentSession The frontend AgentSession model to convert
 * @returns The converted backend session model
 */
export function agentSessionToBackend(agentSession: AgentSession): BackendSession {
  const id = parseInt(agentSession.id, 10);
  
  return {
    id: !isNaN(id) ? id : undefined,
    created_at: agentSession.created.toISOString(),
    updated_at: agentSession.updated.toISOString(),
    start_time: agentSession.created.toISOString(), // Using created as start_time
    display_name: agentSession.name,
    // Other fields are optional
  };
}

/**
 * Validates a backend session object against the schema
 * 
 * @param data The data to validate
 * @returns The validated backend session
 * @throws If validation fails
 */
export function validateBackendSession(data: unknown): BackendSession {
  try {
    return BackendSessionSchema.parse(data);
  } catch (error) {
    // Log the specific error for debugging
    if (error instanceof z.ZodError) {
      console.error('Backend session validation failed:', error.format());
      
      // Safely check for machine_info errors in a type-safe way
      const formattedError = error.format();
      const fieldErrors = formattedError as Record<string, unknown>;
      
      if ('machine_info' in fieldErrors && typeof data === 'object' && data !== null) {
        const typedData = data as Record<string, unknown>;
        if ('machine_info' in typedData) {
          console.warn('Machine info parsing issue, received type:', typeof typedData.machine_info);
        }
      }
    }
    throw error;
  }
}

/**
 * Validates an agent session object against the schema
 * 
 * @param data The data to validate
 * @returns The validated agent session
 * @throws If validation fails
 */
export function validateAgentSession(data: unknown): AgentSession {
  return AgentSessionSchema.parse(data);
}

/**
 * Safely converts backend session data to an agent session
 * 
 * Validates the backend data before conversion and throws if invalid
 * 
 * @param data The backend data to convert
 * @returns The converted agent session
 * @throws If validation fails
 */
export function safeBackendToAgentSession(data: unknown): AgentSession {
  const validatedBackend = validateBackendSession(data);
  return backendToAgentSession(validatedBackend);
}

/**
 * Safely converts an agent session to backend format
 * 
 * Validates the agent session before conversion and throws if invalid
 * 
 * @param data The agent session data to convert
 * @returns The converted backend session
 * @throws If validation fails
 */
export function safeAgentSessionToBackend(data: unknown): BackendSession {
  const validatedSession = validateAgentSession(data);
  return agentSessionToBackend(validatedSession);
}