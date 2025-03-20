import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges class names with Tailwind CSS classes
 * Combines clsx for conditional logic and tailwind-merge for handling conflicting tailwind classes
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Export API utilities
export * from './api';

// Note: Sample data functions and types are now exported directly from the root index.ts
// to avoid circular references