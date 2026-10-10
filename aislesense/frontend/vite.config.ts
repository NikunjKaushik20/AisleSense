/// <reference types="vitest" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` proxies the API to the running backend (python run.py ...).
// Set AISLE_API_PORT when the backend is not on 8000.
// Prod: `npm run build` -> dist/, served by aisle/api/static.py at http://127.0.0.1:8000/.
export default defineConfig(({ mode }) => {
  // loadEnv also reads AISLE_* variables from the shell environment.
  const apiPort = loadEnv(mode, ".", "AISLE_").AISLE_API_PORT || "8000";
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": `http://127.0.0.1:${apiPort}`,
        "/ws": { target: `ws://127.0.0.1:${apiPort}`, ws: true },
      },
    },
    build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 900 },
    test: {
      environment: "jsdom",
      include: ["tests/**/*.test.{ts,tsx}"],
      setupFiles: ["tests/setup.ts"],
      globals: true,
    },
  };
});
