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
          sequence: { groupOrder: 0 },
          include: ["**/*.test.ts", "**/*.test.mjs"],
          exclude: [
            ...configDefaults.exclude,
            ...jsdomTypeScriptTests,
            "tests/workbench-app/**",
            "scripts/runtime-paths.test.mjs",
          ],
        },
      },
      {
        extends: true,
        test: {
          name: "jsdom",
          environment: "jsdom",
          sequence: { groupOrder: 0 },
          include: ["**/*.test.tsx", ...jsdomTypeScriptTests],
          exclude: [...configDefaults.exclude, "tests/workbench-app/**"],
        },
      },
      {
        extends: true,
        test: {
          name: "workbench-app",
          environment: "jsdom",
          sequence: { groupOrder: 1 },
          include: ["tests/workbench-app/**/*.test.tsx"],
          exclude: configDefaults.exclude,
          fileParallelism: false,
        },
      },
    ],
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
    fileParallelism: true,
    // Full Workbench suites mount large app trees; bounded concurrency keeps
    // parallel execution faster without multiplying jsdom memory across every CPU.
    maxWorkers: 4,
  },
  resolve: {
    alias: {
      "@": new URL("./src", import.meta.url).pathname,
    },
  },
});
