/**
 * Trajectory model with Zod validation
 * 
 * This module provides:
 * 1. Schema definitions for backend and frontend trajectory models
 * 2. Type definitions using Zod inference
 * 3. Conversion functions between backend and frontend formats
 * 4. Validation functions
 */

import { z } from 'zod';

/**
 * Schema for the backend trajectory model format received from the API
 */
export const BackendTrajectorySchema = z.object({
  id: z.number().optional(),
  created_at: z.string(),
  updated_at: z.string(),
  human_input_id: z.number().nullable().optional(),
  tool_name: z.string(),
  tool_parameters: z.record(z.any()).nullable().optional(),
  tool_result: z.record(z.any()).nullable().optional(),
  step_data: z.record(z.any()).nullable().optional(),
  record_type: z.string(),
  current_cost: z.number().nullable().optional(),
  input_tokens: z.number().nullable().optional(),
  output_tokens: z.number().nullable().optional(),
  is_error: z.boolean().default(false),
  error_message: z.string().nullable().optional(),
  error_type: z.string().nullable().optional(),
  error_details: z.record(z.any()).nullable().optional(),
  session_id: z.number().nullable().optional(),
});

/**
 * Type for the backend trajectory model format, inferred from the schema
 */
export type BackendTrajectory = z.infer<typeof BackendTrajectorySchema>;

/**
 * Schema for the frontend trajectory model
 */
export const TrajectorySchema = z.object({
  id: z.string().optional(),
  created: z.string(),
  updated: z.string(),
  toolName: z.string(),
  toolParameters: z.record(z.any()).nullable().optional(),
  toolResult: z.record(z.any()).nullable().optional(),
  stepData: z.record(z.any()).nullable().optional(),
  recordType: z.string(),
  currentCost: z.number().nullable().optional(),
  inputTokens: z.number().nullable().optional(),
  outputTokens: z.number().nullable().optional(),
  isError: z.boolean().default(false),
  errorMessage: z.string().nullable().optional(),
  errorType: z.string().nullable().optional(),
  errorDetails: z.record(z.any()).nullable().optional(),
  sessionId: z.string().nullable().optional(),
});

/**
 * Type for the frontend trajectory model, inferred from the schema
 */
export type Trajectory = z.infer<typeof TrajectorySchema>;

/**
 * Converts a backend trajectory model to a frontend trajectory model
 * 
 * @param backendTrajectory The backend trajectory model to convert
 * @returns The converted frontend trajectory model
 */
export function backendToTrajectory(backendTrajectory: BackendTrajectory): Trajectory {
  return {
    id: backendTrajectory.id?.toString(),
    created: backendTrajectory.created_at,
    updated: backendTrajectory.updated_at,
    toolName: backendTrajectory.tool_name,
    toolParameters: backendTrajectory.tool_parameters,
    toolResult: backendTrajectory.tool_result,
    stepData: backendTrajectory.step_data,
    recordType: backendTrajectory.record_type,
    currentCost: backendTrajectory.current_cost,
    inputTokens: backendTrajectory.input_tokens,
    outputTokens: backendTrajectory.output_tokens,
    isError: backendTrajectory.is_error || false,
    errorMessage: backendTrajectory.error_message,
    errorType: backendTrajectory.error_type,
    errorDetails: backendTrajectory.error_details,
    sessionId: backendTrajectory.session_id?.toString(),
  };
}

/**
 * Converts a frontend trajectory model to a backend trajectory model
 * 
 * @param trajectory The frontend trajectory model to convert
 * @returns The converted backend trajectory model
 */
export function trajectoryToBackend(trajectory: Trajectory): BackendTrajectory {
  return {
    id: trajectory.id ? parseInt(trajectory.id) : undefined,
    created_at: trajectory.created,
    updated_at: trajectory.updated,
    tool_name: trajectory.toolName,
    tool_parameters: trajectory.toolParameters,
    tool_result: trajectory.toolResult,
    step_data: trajectory.stepData,
    record_type: trajectory.recordType,
    current_cost: trajectory.currentCost,
    input_tokens: trajectory.inputTokens,
    output_tokens: trajectory.outputTokens,
    is_error: trajectory.isError,
    error_message: trajectory.errorMessage,
    error_type: trajectory.errorType,
    error_details: trajectory.errorDetails,
    session_id: trajectory.sessionId ? parseInt(trajectory.sessionId) : null,
  };
}

/**
 * Validates a backend trajectory object against the schema
 * 
 * @param data The data to validate
 * @returns The validated backend trajectory
 * @throws If validation fails
 */
export function validateBackendTrajectory(data: unknown): BackendTrajectory {
  return BackendTrajectorySchema.parse(data);
}

/**
 * Validates a frontend trajectory object against the schema
 * 
 * @param data The data to validate
 * @returns The validated frontend trajectory
 * @throws If validation fails
 */
export function validateTrajectory(data: unknown): Trajectory {
  return TrajectorySchema.parse(data);
}

/**
 * Safely converts backend trajectory data to a frontend trajectory
 * 
 * Validates the backend data before conversion and handles errors
 * 
 * @param data The backend data to convert
 * @returns The converted trajectory or null if validation fails
 */
export function safeBackendToTrajectory(data: unknown): Trajectory | null {
  try {
    const validatedBackend = validateBackendTrajectory(data);
    return backendToTrajectory(validatedBackend);
  } catch (error) {
    console.error('Failed to validate backend trajectory:', error);
    return null;
  }
}

/**
 * Safely validates if an object is a valid trajectory
 * 
 * @param data The data to validate
 * @returns true if the data is a valid trajectory, false otherwise
 */
export function isValidTrajectory(data: unknown): data is Trajectory {
  try {
    validateTrajectory(data);
    return true;
  } catch {
    return false;
  }
}

/**
 * Creates a sample trajectory for testing or placeholder purposes
 * 
 * @returns A sample trajectory object
 */
export function getSampleTrajectory(): Trajectory {
  return {
    id: '1',
    created: new Date().toISOString(),
    updated: new Date().toISOString(),
    toolName: 'sample_tool',
    toolParameters: { param1: 'value1', param2: 'value2' },
    toolResult: { result: 'success' },
    stepData: { display: 'Sample tool execution' },
    recordType: 'tool_execution',
    currentCost: 0.001,
    inputTokens: 100,
    outputTokens: 50,
    isError: false,
    sessionId: '1',
  };
}