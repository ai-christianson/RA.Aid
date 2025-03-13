// Entry point for @ra-aid/common package
import './styles/global.css';
// Export utility functions
export * from './utils';
// Export all UI components
export * from './components/ui';
// Export the hello function (temporary example)
export const hello = () => {
    console.log("Hello from @ra-aid/common");
};
