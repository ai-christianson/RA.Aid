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
  css: {
    // Enable PostCSS processing
    postcss: './postcss.config.js',
  },
  server: {
    watch: {
      // Watch for changes in the common package, including style files
      // This pattern forces Vite to notice file changes in the shared library
      paths: [
        '../common/src/**',
        '../common/src/styles/**'
      ]
    }
  },
  // Ensure CommonJS modules are properly processed (for Tailwind and PostCSS)
  optimizeDeps: {
    include: ['@ra-aid/common'],
  }
});