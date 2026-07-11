import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    environmentMatchGlobs: [
      ["**/*.test.tsx", "jsdom"],
      ["src/features/workbench/state/graph-monitor/use-graph-view-state.test.ts", "jsdom"],
      ["src/features/workbench/state/target/_inspection-preview-races.test.ts", "jsdom"],
      ["src/features/workbench/state/logs/use-log-query-cache.test.ts", "jsdom"],
      ["src/features/workbench/state/use-workbench-queries.test.ts", "jsdom"],
      ["src/lib/api-origin-lock.test.ts", "jsdom"],
      ["src/lib/api.test.ts", "jsdom"],
    ],
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
    fileParallelism: false,
  },
  resolve: {
    alias: {
      "@": new URL("./src", import.meta.url).pathname,
    },
  },
});
