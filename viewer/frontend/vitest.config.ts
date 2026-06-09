import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    environmentMatchGlobs: [
      ["**/*.test.tsx", "jsdom"],
      ["src/features/viewer/state/graph-monitor/use-graph-view-state.test.ts", "jsdom"],
      ["src/features/viewer/state/graph-monitor/use-preview-inspection.test.ts", "jsdom"],
      ["src/features/viewer/state/logs/use-log-query-cache.test.ts", "jsdom"],
      ["src/features/viewer/state/target/use-target-overrides.test.ts", "jsdom"],
      ["src/features/viewer/state/use-viewer-queries.test.ts", "jsdom"],
      ["src/lib/api.test.ts", "jsdom"],
      ["src/lib/auth-token.test.ts", "jsdom"],
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
