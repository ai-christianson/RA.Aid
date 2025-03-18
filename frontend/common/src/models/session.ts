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
 * Schema for the backend session model format received from the API
 */
export const BackendSessionSchema = z.object({
  id: z.number().optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  start_time: z.string().datetime(),
  command_line: z.string().optional(),
  program_version: z.string().optional(),
  machine_info: z.record(z.string(), z.any()).optional(),
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
  return {
    id: backendSession.id !== undefined ? backendSession.id.toString() : '',
    name: backendSession.display_name || 'Unnamed Session',
    created: new Date(backendSession.created_at),
    updated: new Date(backendSession.updated_at),
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
  return BackendSessionSchema.parse(data);
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