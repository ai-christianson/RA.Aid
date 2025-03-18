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
 * 
 * Uses union types to accept both JSON strings and records for certain fields
 */
export const BackendTrajectorySchema = z.object({
  id: z.number().optional(),
  created_at: z.string(),
  updated_at: z.string(),
  human_input_id: z.number().nullable().optional(),
  tool_name: z.string(),
  // Allow either a string (JSON) or a record for object fields
  tool_parameters: z.union([
    z.string(),
    z.record(z.any())
  ]).nullable().optional(),
  tool_result: z.union([
    z.string(),
    z.record(z.any())
  ]).nullable().optional(),
  step_data: z.union([
    z.string(),
    z.record(z.any())
  ]).nullable().optional(),
  record_type: z.string(),
  current_cost: z.number().nullable().optional(),
  input_tokens: z.number().nullable().optional(),
  output_tokens: z.number().nullable().optional(),
  is_error: z.boolean().default(false),
  error_message: z.string().nullable().optional(),
  error_type: z.string().nullable().optional(),
  error_details: z.union([
    z.string(),
    z.record(z.any())
  ]).nullable().optional(),
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
 * Handles JSON string parsing for object fields if they are still strings after validation
 * 
 * @param backendTrajectory The backend trajectory model to convert
 * @returns The converted frontend trajectory model
 */
export function backendToTrajectory(backendTrajectory: BackendTrajectory): Trajectory {
  // Helper function to parse JSON string or return the value if it's already an object
  const parseJsonField = (field: string | Record<string, any> | null | undefined): Record<string, any> | null | undefined => {
    if (field === null || field === undefined) {
      return field;
    }
    
    if (typeof field === 'string') {
      try {
        return JSON.parse(field);
      } catch (error) {
        console.error('Error parsing JSON string in backendToTrajectory:', error);
        return {}; // Return empty object as fallback
      }
    }
    
    return field; // Already an object
  };

  return {
    id: backendTrajectory.id?.toString(),
    created: backendTrajectory.created_at,
    updated: backendTrajectory.updated_at,
    toolName: backendTrajectory.tool_name,
    toolParameters: parseJsonField(backendTrajectory.tool_parameters),
    toolResult: parseJsonField(backendTrajectory.tool_result),
    stepData: parseJsonField(backendTrajectory.step_data),
    recordType: backendTrajectory.record_type,
    currentCost: backendTrajectory.current_cost,
    inputTokens: backendTrajectory.input_tokens,
    outputTokens: backendTrajectory.output_tokens,
    isError: backendTrajectory.is_error || false,
    errorMessage: backendTrajectory.error_message,
    errorType: backendTrajectory.error_type,
    errorDetails: parseJsonField(backendTrajectory.error_details),
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
  // Pre-process the data to parse JSON strings
  const preprocessed = data as Record<string, any>;
  
  // Create a copy to avoid mutating the original
  const processedData = { ...preprocessed };
  
  // Parse JSON fields if they are strings
  if (processedData.tool_parameters && typeof processedData.tool_parameters === 'string') {
    try {
      processedData.tool_parameters = JSON.parse(processedData.tool_parameters);
    } catch (error) {
      console.error('Failed to parse tool_parameters JSON string:', error);
    }
  }
  
  if (processedData.tool_result && typeof processedData.tool_result === 'string') {
    try {
      processedData.tool_result = JSON.parse(processedData.tool_result);
    } catch (error) {
      console.error('Failed to parse tool_result JSON string:', error);
    }
  }
  
  if (processedData.step_data && typeof processedData.step_data === 'string') {
    try {
      processedData.step_data = JSON.parse(processedData.step_data);
    } catch (error) {
      console.error('Failed to parse step_data JSON string:', error);
    }
  }
  
  if (processedData.error_details && typeof processedData.error_details === 'string') {
    try {
      processedData.error_details = JSON.parse(processedData.error_details);
    } catch (error) {
      console.error('Failed to parse error_details JSON string:', error);
    }
  }
  
  return BackendTrajectorySchema.parse(processedData);
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
 * Validates the backend data before conversion and handles errors.
 * Includes pre-processing for JSON string fields.
 * 
 * @param data The backend data to convert
 * @returns The converted trajectory or null if validation fails
 */
export function safeBackendToTrajectory(data: unknown): Trajectory | null {
  // Log the raw data being passed in for conversion
  console.log('Converting backend trajectory data:', JSON.stringify(data, null, 2));
  
  // Pre-check if JSON string fields exist to help with debugging
  if (data && typeof data === 'object') {
    const rawData = data as Record<string, any>;
    
    // Log the data types of relevant fields to help with debugging
    console.log('Field data types:', {
      tool_parameters: rawData.tool_parameters && typeof rawData.tool_parameters,
      tool_result: rawData.tool_result && typeof rawData.tool_result,
      step_data: rawData.step_data && typeof rawData.step_data,
      error_details: rawData.error_details && typeof rawData.error_details
    });
    
    // Show raw string content for string fields that should be objects
    if (rawData.tool_parameters && typeof rawData.tool_parameters === 'string') {
      console.log('tool_parameters string content:', rawData.tool_parameters);
    }
    if (rawData.step_data && typeof rawData.step_data === 'string') {
      console.log('step_data string content:', rawData.step_data);
    }
  }
  
  try {
    // Add detailed logging around validation
    try {
      // validateBackendTrajectory now handles JSON string parsing
      const validatedBackend = validateBackendTrajectory(data);
      // Log the validated backend data
      console.log('Successfully validated backend trajectory:', JSON.stringify(validatedBackend, null, 2));
      
      // Convert and log the result
      const result = backendToTrajectory(validatedBackend);
      console.log('Successfully converted to frontend trajectory:', JSON.stringify(result, null, 2));
      
      return result;
    } catch (validationError) {
      // Detailed validation error logging
      console.error('Validation error details:', validationError);
      
      // Try to get more specific information about the validation error
      if (validationError instanceof z.ZodError) {
        console.error('Zod validation errors:', JSON.stringify(validationError.errors, null, 2));
        
        // Log the specific fields that failed validation
        validationError.errors.forEach(err => {
          console.error(`Field "${err.path.join('.')}": ${err.message}`);
        });
      }
      
      throw validationError; // Re-throw for the outer catch
    }
  } catch (error) {
    console.error('Failed to validate or convert backend trajectory:', error);
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