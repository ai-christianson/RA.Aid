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
      // Adjust the pattern if your common package layout is different.
      paths: ['../common/src/**']
    }
  }
});
