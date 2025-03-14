import React, { ReactNode } from 'react';
import { Button } from './button';

export interface FloatingActionButtonProps {
  icon: ReactNode;
  onClick: () => void;
  ariaLabel?: string;
  className?: string;
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
}

/**
 * FloatingActionButton component
 * 
 * A button typically used for primary actions on mobile layouts
 * Designed to be used with the Layout component's floatingAction prop
 */
export const FloatingActionButton: React.FC<FloatingActionButtonProps> = ({
  icon,
  onClick,
  ariaLabel = 'Action button',
  className = '',
  variant = 'default'
}) => {
  return (
    <Button
      variant={variant}
      size="icon"
      onClick={onClick}
      aria-label={ariaLabel}
      className={`h-14 w-14 rounded-full shadow-xl bg-blue-600 hover:bg-blue-700 text-white flex items-center justify-center border-2 border-white dark:border-gray-800 ${className}`}
    >
      {icon}
    </Button>
  );
};
