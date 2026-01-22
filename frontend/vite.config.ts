import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      '/upload': 'http://localhost:8000',
      '/generate': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
      '/assets': 'http://localhost:8000',
    }
  }
})
