/**
 * Client configuration model with Zod validation
 * 
 * This module provides:
 * 1. Schema definition for client configuration
 * 2. Type definition using Zod inference
 * 3. Validation functions for client configuration
 */

import { z } from 'zod';

/**
 * Schema for client configuration
 */
export const ClientConfigSchema = z.object({
  host: z.string().default('localhost'),
  port: z.number().default(1818),
});

/**
 * Type for client configuration, inferred from the schema
 */
export type ClientConfig = z.infer<typeof ClientConfigSchema>;

/**
 * Validates a client configuration object against the schema
 * 
 * @param data The data to validate
 * @returns The validated client configuration
 * @throws If validation fails
 */
export function validateClientConfig(data: unknown): ClientConfig {
  return ClientConfigSchema.parse(data);
}

/**
 * Safely validates a client configuration object against the schema
 * 
 * @param data The data to validate
 * @returns Object with success flag and result/error
 */
export function safeValidateClientConfig(data: unknown): { 
  success: boolean; 
  data?: ClientConfig; 
  error?: z.ZodError 
} {
  try {
    const validatedConfig = validateClientConfig(data);
    return { success: true, data: validatedConfig };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { success: false, error };
    }
    throw error; // Re-throw if it's not a Zod error
  }
}