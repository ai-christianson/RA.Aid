import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';

// Get all component files from common package
const commonSrcDir = path.resolve(__dirname, '../common/src');

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Direct alias to the source directory
      '@ra-aid/common': path.resolve(__dirname, '../common/src')
    },
    preserveSymlinks: true
  },
  optimizeDeps: {
    // Exclude the common package from optimization so it can trigger hot reload
    exclude: ['@ra-aid/common']
  },
  server: {
    hmr: true,
    watch: {
      usePolling: true,
      interval: 100,
      // Make sure to explicitly NOT ignore the common package
      ignored: [
        '**/node_modules/**',
        '**/dist/**',
        '!**/common/src/**'
      ]
    }
  },
  build: {
    commonjsOptions: {
      transformMixedEsModules: true
    }
  }
});