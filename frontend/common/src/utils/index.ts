import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merges class names with Tailwind CSS classes
 * Combines clsx for conditional logic and tailwind-merge for handling conflicting tailwind classes
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Formats a timestamp into a readable string (YYYY-MM-DD HH:mm:ss).
 * @param timestamp - The date string or Date object to format.
 * @returns The formatted timestamp string, or an empty string if input is invalid.
 */
export function formatTimestamp(timestamp: string | Date | undefined | null): string {
  if (!timestamp) {
    return "";
  }

  try {
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;

    if (isNaN(date.getTime())) {
      // Handle invalid date strings
      return "Invalid Date";
    }

    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0'); // Months are 0-indexed
    const day = date.getDate().toString().padStart(2, '0');
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  } catch (error) {
    console.error("Error formatting timestamp:", error);
    return "Formatting Error"; // Return an error indicator
  }
}


// Export API utilities
export * from './api';

// Note: Sample data functions and types are now exported directly from the root index.ts
// to avoid circular references