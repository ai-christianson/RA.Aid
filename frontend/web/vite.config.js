import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
  ],
  resolve: {
    // Point to the source files instead of dist for development
    alias: {
      '@ra-aid/common': path.resolve(__dirname, '../common/src')
    }
  },
  optimizeDeps: {
    // Force Vite to include these dependencies in its optimization
    include: ['@ra-aid/common'],
    // Tell Vite to respect our aliased packages instead of using node_modules for them
    esbuildOptions: {
      preserveSymlinks: true,
    }
  },
  server: {
    hmr: {
      // More verbose logging for HMR
      overlay: true,
    },
    watch: {
      // Watch for changes in the common package
      paths: ['../common/src/**'],
      // Ensure changes in source files trigger a reload
      usePolling: true,
    }
  },
  css: {
    // PostCSS configuration is loaded from postcss.config.js
    // This ensures proper processing of Tailwind directives
    devSourcemap: true,
  },
  build: {
    // When building for production, we need to make sure the common package is built too
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  }
});