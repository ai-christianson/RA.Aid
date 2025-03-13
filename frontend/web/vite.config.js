import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Ensure that Vite treats symlinked packages as local, so HMR works correctly.
    alias: {
      '@ra-aid/common': path.resolve(__dirname, '../common/src')
    }
  },
  server: {
    watch: {
      // Watch for changes in the common package.
      // This pattern forces Vite to notice file changes in the shared library.
      paths: ['../common/src/**']
    }
  },
  css: {
    // PostCSS configuration is loaded from postcss.config.js
    // This ensures proper processing of Tailwind directives
    devSourcemap: true,
  }
});