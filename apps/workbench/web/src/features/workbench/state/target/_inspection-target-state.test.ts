import { describe, expect, it } from "vitest";
import {
  createInspectionTargetLifecycleState,
  inspectionTargetLifecycleReducer,
  type InspectionTargetLifecycleEvent,
  type InspectionTargetLifecycleState,
} from "@/features/workbench/state/target/_inspection-target-state";

function initial(
  overrides: Parameters<typeof createInspectionTargetLifecycleState>[0] = {
    modelPackage: { modelType: "linears", model: "linear" },
    preset: "baseline",
  },
) {
  return createInspectionTargetLifecycleState(overrides);
}

function transition(
  state: InspectionTargetLifecycleState,
  event: InspectionTargetLifecycleEvent,
) {
  return inspectionTargetLifecycleReducer(state, event);
}

describe("Inspection target lifecycle reducer", () => {
  it("owns the complete preset, Snapshot, and historical Run transition matrix", () => {
    let state = initial();
    state = transition(state, {
      type: "metadata-refreshed",
      preset: "baseline",
      experimentTask: "image-classification",
      datasets: ["Mnist"],
      expectedRevision: 0,
    });
    expect(state).toMatchObject({
      modelPackage: { modelType: "linears", model: "linear" },
      selectedPreset: "baseline",
      target: { kind: "preset", preset: "baseline" },
      experimentTask: "image-classification",
      datasets: ["Mnist"],
      runtimeDefaults: { preset: {}, active: {} },
    });

    state = transition(state, {
      type: "runtime-defaults-edited",
      presetRuntimeDefaults: { hidden_size: "128" },
      activeRuntimeDefaults: { hidden_size: "128" },
    });
    state = transition(state, {
      type: "snapshot-selected",
      snapshotId: "snapshot-fast",
      preset: "fast",
      runtimeDefaults: { hidden_size: "256" },
    });
    expect(state).toMatchObject({
      selectedPreset: "fast",
      target: {
        kind: "snapshot",
        snapshotId: "snapshot-fast",
        preset: "fast",
      },
      runtimeDefaults: {
        preset: { hidden_size: "128" },
        active: { hidden_size: "256" },
      },
    });

    state = transition(state, {
      type: "historical-run-selected",
      run: {
        runId: "run-fast",
        experiment: "exp-linear",
        preset: "fast",
        dataset: "FashionMnist",
        experimentTask: "fashion-classification",
      },
      catalogPreset: "fast",
      catalogDataset: "FashionMnist",
    });
    expect(state).toMatchObject({
      selectedPreset: "fast",
      target: { kind: "historical-run", run: { runId: "run-fast" } },
      experimentTask: "fashion-classification",
      datasets: ["FashionMnist"],
      runtimeDefaults: { preset: {}, active: {} },
    });

    state = transition(state, { type: "preset-selected", preset: "baseline" });
    expect(state).toMatchObject({
      selectedPreset: "baseline",
      target: { kind: "preset", preset: "baseline" },
      runtimeDefaults: { preset: {}, active: {} },
    });
  });

  it("keeps browsing-only task changes from replacing a complete historical Run", () => {
    const historical = transition(initial(), {
      type: "historical-run-selected",
      run: {
        runId: "run-fast",
        experiment: "exp-linear",
        preset: "fast",
        dataset: "FashionMnist",
      },
    });
    const next = transition(historical, {
      type: "experiment-task-selected",
      experimentTask: "fashion-classification",
      datasets: ["FashionMnist"],
    });

    expect(next.target).toBe(historical.target);
    expect(next.transition).toBe(historical.transition);
    expect(next.experimentTask).toBe("fashion-classification");
  });

  it("ignores stale query-driven repairs after a newer semantic intent", () => {
    const selected = transition(initial(), {
      type: "preset-selected",
      preset: "fast",
    });
    const staleMetadata = transition(selected, {
      type: "metadata-refreshed",
      preset: "baseline",
      experimentTask: "image-classification",
      datasets: ["Mnist"],
      expectedRevision: selected.transition.revision - 1,
    });
    const staleDefaults = transition(selected, {
      type: "runtime-defaults-normalized",
      presetRuntimeDefaults: { hidden_size: "64" },
      activeRuntimeDefaults: { hidden_size: "64" },
      expectedRevision: selected.transition.revision - 1,
    });

    expect(staleMetadata).toBe(selected);
    expect(staleDefaults).toBe(selected);
  });

  it("restores or falls back from persisted Snapshots as one atomic lifecycle event", () => {
    let state = initial({
      modelPackage: { modelType: "linears", model: "linear" },
      preset: "fast",
      restorePersistedTarget: true,
      restoreSnapshotId: "snapshot-fast",
    });
    expect(state.restoration).toEqual({
      phase: "restoring-browser",
      requestedSnapshotId: "snapshot-fast",
    });

    state = transition(state, {
      type: "restoration-snapshot-resolved",
      snapshotId: "snapshot-fast",
      preset: "fast",
      runtimeDefaults: { hidden_size: "256" },
    });
    expect(state).toMatchObject({
      target: { kind: "snapshot", snapshotId: "snapshot-fast" },
      restoration: { phase: "settled", requestedSnapshotId: "" },
      runtimeDefaults: { active: { hidden_size: "256" } },
    });

    state = transition(state, { type: "connection-reset" });
    expect(state).toMatchObject({
      target: { kind: "snapshot", snapshotId: "snapshot-fast" },
      restoration: {
        phase: "rebuilding-connection",
        requestedSnapshotId: "snapshot-fast",
      },
      connectionRevision: 1,
    });

    state = transition(state, { type: "missing-snapshot-fallback" });
    expect(state).toMatchObject({
      target: { kind: "preset", preset: "fast" },
      restoration: { phase: "settled", requestedSnapshotId: "" },
      runtimeDefaults: { active: {} },
    });
  });

  it("records exactly one revision and cause per semantic command", () => {
    let state = initial();
    const commands: InspectionTargetLifecycleEvent[] = [
      {
        type: "runtime-defaults-edited",
        presetRuntimeDefaults: { hidden_size: "128" },
        activeRuntimeDefaults: { hidden_size: "128" },
      },
      { type: "inspection-refreshed" },
      { type: "runtime-defaults-reset" },
      {
        type: "model-package-selected",
        modelPackage: { modelType: "linears", model: "linear_adaptive" },
      },
    ];

    for (const [index, command] of commands.entries()) {
      state = transition(state, command);
      expect(state.transition).toEqual({
        revision: index + 1,
        cause:
          command.type === "inspection-refreshed"
            ? "inspection-refreshed"
            : "target-changed",
      });
    }
  });
});
