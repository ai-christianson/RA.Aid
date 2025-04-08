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
// TODO: Reconcile AgentSession import/definition with the schema below.
// For now, we focus on the schema-defined types.
// import { AgentSession as UtilAgentSession, AgentStep } from '../utils/types';

// Define the possible status values for a session
export type SessionStatus = 'pending' | 'running' | 'completed' | 'error' | 'unknown';
const sessionStatusEnum = z.enum(['pending', 'running', 'completed', 'error', 'unknown']);

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
        console.warn(`Invalid date received: ${val}`);
        return null;
      }

      // Convert to ISO string for consistency
      return date.toISOString();
    } catch {
      console.warn(`Error parsing date: ${val}`);
      return null;
    }
  });

/**
 * Schema for the backend session model format received from the API
 */
export const BackendSessionSchema = z.object({
  id: z.number(), // Assuming ID is always present after creation
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
  status: z.string().optional(), // Add status field from backend
});

/**
 * Type for the backend session model format, inferred from the schema
 */
export type BackendSession = z.infer<typeof BackendSessionSchema>;


// Note: AgentStep and the original AgentSession (from utils/types) are not modified here
// as the task focuses on the Session model itself and its status.
// If AgentStep needs modification later, it can be done separately.

/**
 * Zod schema for validating AgentStep objects
 * Keeping this definition as it was, although it's not directly used in this task's scope.
 */
// Define AgentStep type for AgentSessionSchema below
type AgentStep = {
  id: string;
  timestamp: Date;
  status: 'completed' | 'in-progress' | 'error' | 'pending';
  type: 'tool-execution' | 'thinking' | 'planning' | 'implementation' | 'user-input';
  title: string;
  content: string;
  duration?: number;
};
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
 * Frontend representation of a Session, derived from BackendSession
 * This interface is primarily used in the Zustand store and UI components.
 */
export interface AgentSession {
  id: number; // Use number for consistency with backend ID type
  name: string;
  commandLine: string;
  createdAt: Date;
  updatedAt: Date | null; // Allow null if updated_at is not set
  status: SessionStatus; // Use the defined SessionStatus type
  // Removed 'steps' as it's managed by Trajectory store now.
}

/**
 * Zod schema for validating the *structure* of AgentSession objects used in the frontend.
 * Note: This doesn't parse/transform dates like BackendSessionSchema does.
 * It assumes dates are already Date objects.
 */
export const AgentSessionSchema: z.ZodType<AgentSession> = z.object({
  id: z.number(),
  name: z.string(),
  commandLine: z.string(),
  createdAt: z.date(),
  updatedAt: z.date().nullable(),
  status: sessionStatusEnum, // Validate against the SessionStatus enum
});


/**
 * Converts a backend session model to a frontend AgentSession model
 *
 * @param backendSession The backend session model to convert
 * @returns The converted frontend AgentSession model
 */
export function backendToAgentSession(backendSession: BackendSession): AgentSession {
  const now = new Date(); // Use for fallback if created_at is missing

  // Validate and determine status
  let status: SessionStatus = 'unknown'; // Default status
  if (backendSession.status && ['pending', 'running', 'completed', 'error'].includes(backendSession.status)) {
    status = backendSession.status as SessionStatus;
  } else if (backendSession.status) {
     console.warn(`Received unknown session status: ${backendSession.status} for session ${backendSession.id}`);
  } else {
    // If status is null/undefined from backend, maybe default to pending or unknown?
    status = 'pending'; // Let's default to pending if not provided
  }


  return {
    id: backendSession.id,
    name: backendSession.display_name || `Session ${backendSession.id}`,
    commandLine: backendSession.command_line || 'N/A',
    createdAt: backendSession.created_at ? new Date(backendSession.created_at) : now,
    updatedAt: backendSession.updated_at ? new Date(backendSession.updated_at) : null,
    status: status,
  };
}

/**
 * Converts a frontend AgentSession model to a partial backend session model
 * (Only includes fields relevant for potential updates, not a full representation)
 *
 * @param agentSession The frontend AgentSession model to convert
 * @returns A partial BackendSession model suitable for updates
 */
export function agentSessionToPartialBackend(agentSession: AgentSession): Partial<BackendSession> {
  return {
    id: agentSession.id,
    updated_at: agentSession.updatedAt?.toISOString() ?? new Date().toISOString(), // Use current time if null
    display_name: agentSession.name,
    status: agentSession.status,
    // We don't typically send command_line, created_at back
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
    // Ensure ID is present and a number before parsing the rest
     const idCheck = z.object({ id: z.number() }).safeParse(data);
     if (!idCheck.success) {
         console.error("Backend session validation failed: Missing or invalid 'id'", idCheck.error.format());
         throw new Error("Invalid backend session data: Missing or invalid 'id'");
     }
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
    } else {
        console.error("Unexpected validation error:", error);
    }
    throw error; // Re-throw the original error
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
    try {
      return AgentSessionSchema.parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
             console.error('Agent session validation failed:', error.format());
        } else {
            console.error("Unexpected validation error:", error);
        }
        throw error; // Re-throw the original error
    }
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
 * Safely converts an agent session to partial backend format
 *
 * Validates the agent session before conversion and throws if invalid
 *
 * @param data The agent session data to convert
 * @returns The converted partial backend session
 * @throws If validation fails
 */
export function safeAgentSessionToPartialBackend(data: unknown): Partial<BackendSession> {
  const validatedSession = validateAgentSession(data);
  return agentSessionToPartialBackend(validatedSession);
}
