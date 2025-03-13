// Entry point for @ra-aid/common package
import './styles/globals.css';

// Export utility functions
export { cn } from './lib/utils';

// Export UI components
export { Button, buttonVariants } from './components/ui/button';
export { 
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent 
} from './components/ui/card';
export { Toggle, toggleVariants } from './components/ui/toggle';

// Export theme components
export { ThemeProvider, useTheme } from './components/theme/theme-provider';
export { ThemeToggle } from './components/theme/theme-toggle';

// Export demo component
export { ShadcnDemo } from './components/ShadcnDemo';

// Legacy exports
export const hello = (): void => {
  console.log("Hello from @ra-aid/common");
};