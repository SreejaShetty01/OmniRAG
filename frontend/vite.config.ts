import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': process.env.VITE_API_PROXY ?? 'http://localhost:8000', '/health': process.env.VITE_API_PROXY ?? 'http://localhost:8000' } },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.ts' },
});
