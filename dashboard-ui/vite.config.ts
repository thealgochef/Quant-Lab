import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        // Suppress ECONNABORTED noise from React StrictMode double-mount
        on: {
          proxyReqWs: (_proxyReq: unknown, _req: unknown, socket: import("net").Socket) => {
            socket.on("error", () => {});
          },
        },
      },
    },
  },
});
