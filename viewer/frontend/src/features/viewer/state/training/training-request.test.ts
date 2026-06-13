import { describe, expect, it } from "vitest";
import {
  buildTrainingJobRequest,
  buildTrainingRunPlanRequest,
} from "@/features/viewer/state/training/training-request";
import {
  type ConfigOverrides,
  type ConfigField,
  type TrainingRunPlan,
} from "@/lib/api";
import {
  buildConfigSnapshotRunPlan,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
import {
  buildEffectiveOverrides,
  buildTrainingSearchPayload,
  searchOverrideConflictKeys,
  type TrainingSearchState,
} from "@/lib/training-search";

const summary = {
  totalRuns: 1,
  completedRuns: 0,
  runningRuns: 0,
  pendingRuns: 1,
  failedRuns: 0,
  cancelledRuns: 0,
  skippedRuns: 0,
  totalEpochs: 10,
  completedEpochs: 0,
  remainingEpochs: 10,
};

function runPlan(overrides: ConfigOverrides = {}): TrainingRunPlan {
  return {
    model: "linears/linear",
    preset: "baseline",
    presets: ["baseline"],
    datasets: ["Mnist"],
    overrides,
    search: null,
    logFolder: "runs",
    isRandomSearch: false,
    runs: [],
    summary,
  };
}

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key,
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "General",
    type: overrides.type ?? "int",
    default: overrides.default ?? 10,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function snapshot(overrides: Partial<ConfigSnapshot> & Pick<ConfigSnapshot, "id">): ConfigSnapshot {
  return {
    id: overrides.id,
    name: overrides.name ?? overrides.id,
    model: overrides.model ?? "linears/linear",
    preset: overrides.preset ?? "baseline",
    overrides: overrides.overrides ?? {},
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
  };
}

describe("training requests", () => {
  it("builds run-plan requests from the selected model, presets, datasets, and overrides", () => {
    expect(
      buildTrainingRunPlanRequest({
        canPlan: true,
        selectedModel: "linears/linear",
        selectedPreset: "baseline",
        selectedTrainingPresets: ["baseline", "fast"],
        selectedDatasets: ["Mnist", "FashionMnist"],
        effectiveOverrides: { hidden_size: "128" },
        logFolder: "runs",
      }),
    ).toEqual({
      model: "linears/linear",
      preset: "baseline",
      presets: ["baseline", "fast"],
      datasets: ["Mnist", "FashionMnist"],
      overrides: { hidden_size: "128" },
      logFolder: "runs",
    });
  });

  it("builds Training Job requests with monitors and the current run plan", () => {
    expect(
      buildTrainingJobRequest({
        selectedModel: "linears/linear",
        selectedPreset: "baseline",
        selectedTrainingPresets: ["baseline"],
        selectedDatasets: ["Mnist"],
        effectiveOverrides: { hidden_size: "128" },
        logFolder: "runs",
        selectedMonitors: ["linear_layers", "gradient_norm"],
        runPlan: runPlan({ hidden_size: "128" }),
      }),
    ).toMatchObject({
      model: "linears/linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: { hidden_size: "128" },
      logFolder: "runs",
      monitors: ["linear_layers", "gradient_norm"],
      runPlan: { overrides: { hidden_size: "128" } },
    });
  });

  it("omits overrides controlled by search axes and includes the search payload", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {
        hidden_size: [128, 256],
        dropout: [0.1],
      },
      randomSamples: 10,
    };
    const overrides = {
      hidden_size: "64",
      learning_rate: "0.001",
    };
    const effectiveOverrides = buildEffectiveOverrides(overrides, search);
    const searchPayload = buildTrainingSearchPayload(search);

    expect(searchOverrideConflictKeys(overrides, search)).toEqual(["hidden_size"]);
    expect(effectiveOverrides).toEqual({ learning_rate: "0.001" });
    expect(
      buildTrainingRunPlanRequest({
        canPlan: true,
        selectedModel: "linears/linear",
        selectedPreset: "baseline",
        selectedTrainingPresets: ["baseline"],
        selectedDatasets: ["Mnist"],
        effectiveOverrides,
        logFolder: "search_runs",
        searchPayload,
      }),
    ).toEqual({
      model: "linears/linear",
      preset: "baseline",
      presets: ["baseline"],
      datasets: ["Mnist"],
      overrides: { learning_rate: "0.001" },
      logFolder: "search_runs",
      search: {
        mode: "grid",
        values: {
          hidden_size: [128, 256],
          dropout: [0.1],
        },
      },
    });
  });

  it("submits mixed Config Snapshot run plans without direct overrides or search payloads", () => {
    const snapshotRunPlan = buildConfigSnapshotRunPlan({
      model: "linears/linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline", "fast"],
      selectedDatasets: ["Mnist", "FashionMnist"],
      snapshots: [
        snapshot({
          id: "wide",
          name: "Wide",
          preset: "baseline",
          overrides: { hidden_size: "256" },
        }),
        snapshot({
          id: "dropout",
          name: "Dropout",
          preset: "fast",
          overrides: { dropout: "0.2" },
        }),
      ],
      fields: [field({ key: "hidden_size" }), field({ key: "dropout" })],
      logFolder: "snapshots",
    });

    const request = buildTrainingJobRequest({
      selectedModel: "linears/linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline", "fast"],
      selectedDatasets: ["Mnist", "FashionMnist"],
      effectiveOverrides: {},
      logFolder: "snapshots",
      selectedMonitors: ["linear_layers"],
      runPlan: snapshotRunPlan,
    });

    expect(request?.overrides).toEqual({});
    expect(request).not.toHaveProperty("search");
    expect(request?.monitors).toEqual(["linear_layers"]);
    expect(request?.runPlan?.search).toBeNull();
    expect(request?.runPlan?.summary.totalRuns).toBe(8);
    expect(request?.runPlan?.runs.map((run) => run.snapshotName)).toEqual([
      undefined,
      undefined,
      undefined,
      undefined,
      "Wide",
      "Wide",
      "Dropout",
      "Dropout",
    ]);
    expect(request?.runPlan?.runs[0]).not.toHaveProperty("snapshotId");
    expect(request?.runPlan?.runs[0].overrides).toEqual({});
    expect(request?.runPlan?.runs[4]).toMatchObject({
      snapshotId: "wide",
      snapshotName: "Wide",
      overrides: { hidden_size: "256" },
    });
  });
});
