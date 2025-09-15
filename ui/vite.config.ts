import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// import vitePluginRequire from 'vite-plugin-require';
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // vitePluginRequire,
  ],
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    host: true,
  },
  define: {
    'process.env': {
        UI_PORT: (process.env.UI_PORT || ''),
        UI_SECURE_PORT: (process.env.UI_SECURE_PORT || ''),
        UI_APP_DOMAIN_NAME: (process.env.UI_APP_DOMAIN_NAME || ''),
        VITE_API_BASE_URL: (process.env.UI_API_BASE_URL || ''),
        VITE_DEBUG: (process.env.UI_DEBUG || ''),
        VITE_REMOTE_REPO_URL: (process.env.REMOTE_REPO_URL || ''),
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})