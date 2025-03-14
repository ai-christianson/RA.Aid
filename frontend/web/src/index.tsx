import React from 'react';
import ReactDOM from 'react-dom/client';
import { DefaultAgentScreen } from '@ra-aid/common';

/**
 * Main application entry point
 * Simply renders the DefaultAgentScreen component from the common package
 */
const App = () => {
  return <DefaultAgentScreen />;
};

// Mount the app to the root element
const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);