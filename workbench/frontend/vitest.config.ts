import react from "@vitejs/plugin-react";
import { configDefaults, defineConfig } from "vitest/config";

const jsdomTypeScriptTests = [
  "src/features/workbench/state/graph-monitor/use-graph-view-state.test.ts",
  "src/features/workbench/state/target/_inspection-preview-races.test.ts",
  "src/features/workbench/state/logs/use-log-query-cache.test.ts",
  "src/features/workbench/state/model-package/use-model-package-metadata.test.ts",
  "src/lib/api-origin-lock.test.ts",
  "src/lib/api.test.ts",
];

export default defineConfig({
  plugins: [react()],
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: "node",
          environment: "node",
          include: ["**/*.test.ts", "**/*.test.mjs"],
          exclude: [
            ...configDefaults.exclude,
            ...jsdomTypeScriptTests,
            "scripts/runtime-paths.test.mjs",
          ],
        },
      },
      {
        extends: true,
        test: {
          name: "jsdom",
          environment: "jsdom",
          include: ["**/*.test.tsx", ...jsdomTypeScriptTests],
          exclude: configDefaults.exclude,
        },
      },
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
