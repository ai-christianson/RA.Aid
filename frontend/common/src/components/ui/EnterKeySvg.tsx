import React from 'react';

/**
 * Interface for the EnterKeySvg component props.
 * Extends standard SVG props to allow className, width, height, etc.
 */
interface EnterKeySvgProps extends React.SVGProps<SVGSVGElement> {
  // No custom props needed for this simple icon
}

/**
 * A React component that renders an inline SVG icon representing
 * a standard computer Enter/Return key symbol (↵).
 * It uses `currentColor` for styling via CSS/Tailwind text color classes.
 * Path data is inspired by common 'corner-down-left' icons (e.g., Feather Icons).
 */
export const EnterKeySvg: React.FC<EnterKeySvgProps> = ({
  className,
  width = "1em", // Default width to match typical text size
  height = "1em", // Default height to match typical text size
  ...props // Spread remaining props onto the SVG element
}) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={width}
      height={height}
      viewBox="0 0 24 24" // Standard viewBox for consistency
      fill="none" // No fill for line icons
      stroke="currentColor" // Use text color for the stroke
      strokeWidth="2" // Standard stroke width
      strokeLinecap="round" // Rounded line endings
      strokeLinejoin="round" // Rounded line joins
      className={className} // Apply passed className
      aria-hidden="true" // Hide from screen readers if decorative
      focusable="false" // Prevent focusing if decorative
      {...props} // Apply any other passed SVG props
    >
      {/* Polyline for the arrowhead part */}
      <polyline points="9 10 4 15 9 20"></polyline>
      {/* Path for the main L-shape line */}
      <path d="M20 4v7a4 4 0 0 1-4 4H4"></path>
    </svg>
  );
};
