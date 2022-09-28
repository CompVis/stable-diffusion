import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import eslint from 'vite-plugin-eslint';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const common = {
    plugins: [react(), eslint()],
    server: {
      proxy: {
        '/outputs': {
          target: 'http://localhost:9090/outputs',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/outputs/, ''),
        },
      },
    },
    build: {
      target: 'esnext',
      chunkSizeWarningLimit: 1500, // we don't really care about chunk size
    },
  };
  if (mode == 'development') {
    return {
      ...common,
      build: {
        ...common.build,
        // sourcemap: true, // this can be enabled if needed, it adds ovwer 15MB to the commit
      },
    };
  } else {
    return {
      ...common,
    };
  }
});
