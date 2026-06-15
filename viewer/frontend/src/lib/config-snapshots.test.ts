import { describe, expect, it } from "vitest";
import {
  buildConfigSnapshotRunPlan,
  createConfigSnapshot,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
import { type ConfigField } from "@/lib/api";

const fields: ConfigField[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    flag: "--hidden-dim",
    label: "Hidden Dim",
    section: "Model",
    type: "int",
    default: 64,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "num_layers",
    configKey: "NUM_LAYERS",
    flag: "--num-layers",
    label: "Num Layers",
    section: "Model",
    type: "int",
    default: 2,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "activation",
    configKey: "ACTIVATION",
    flag: "--activation",
    label: "Activation",
    section: "Model",
    type: "str",
    default: "RELU",
    nullable: false,
    choices: ["RELU", "GELU"],
    locked: false,
  },
  {
    key: "num_epochs",
    configKey: "NUM_EPOCHS",
    flag: "--num-epochs",
    label: "Epochs",
    section: "Training",
    type: "int",
    default: 10,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "layer_norm",
    configKey: "LAYER_NORM",
    flag: "--layer-norm",
    label: "Layer Norm",
    section: "Preset",
    type: "bool",
    default: false,
    nullable: false,
    choices: [],
    locked: true,
    lockedValue: true,
    lockedReason: "Preset controlled",
  },
];

function makeSnapshot(overrides: Record<string, string>, name = "snapshot") {
  const result = createConfigSnapshot({
    id: `snap-${name}`,
    name,
    model: "linear",
    preset: "baseline",
    fields,
    overrides,
    snapshots: [],
    createdAt: "2026-06-04T00:00:00.000Z",
  });
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.snapshot;
}

describe("config snapshots", () => {
  it("rejects default-equivalent snapshots", () => {
    const result = createConfigSnapshot({
      id: "snap-default",
      name: "default",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "64" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Change at least one non-default field before adding a snapshot.",
    });
  });

  it("rejects empty snapshot names", () => {
    const result = createConfigSnapshot({
      id: "snap-empty-name",
      name: "   ",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Config snapshot name cannot be empty.",
    });
  });

  it("rejects duplicate config identity and duplicate names", () => {
    const existing = makeSnapshot({ hidden_dim: "128" }, "same name");
    const duplicate = createConfigSnapshot({
      id: "snap-dup",
      name: "different name",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [existing],
      createdAt: "2026-06-04T00:00:00.000Z",
    });
    const duplicateNameDifferentConfig = createConfigSnapshot({
      id: "snap-name-dup",
      name: " Same Name ",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "256" },
      snapshots: [existing],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(duplicate).toMatchObject({
      ok: false,
      error: "A snapshot with these config values already exists.",
    });
    expect(duplicateNameDifferentConfig).toMatchObject({
      ok: false,
      error: "A snapshot with this name already exists.",
    });
  });

  it("allows edit validation to exclude the snapshot being updated", () => {
    const existing = makeSnapshot({ hidden_dim: "128" }, "existing");

    const nameResult = validateConfigSnapshotName({
      model: "linear",
      preset: "baseline",
      name: " existing ",
      snapshots: [existing],
      excludeSnapshotId: existing.id,
    });
    const configResult = validateConfigSnapshotCandidate({
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [existing],
      excludeSnapshotId: existing.id,
    });

    expect(nameResult).toMatchObject({ ok: true, name: "existing" });
    expect(configResult.ok).toBe(true);
  });

  it("rejects locked-field overrides defensively", () => {
    const result = createConfigSnapshot({
      id: "snap-locked",
      name: "locked",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { layer_norm: "true" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Snapshots cannot include preset-locked fields: Layer Norm.",
    });
  });

  it("builds preset-only run plans when no snapshots are selected", () => {
    const plan = buildConfigSnapshotRunPlan({
      model: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline", "fast"],
      selectedDatasets: ["Mnist"],
      snapshots: [],
      fields,
      logFolder: "presets",
    });

    expect(plan?.presets).toEqual(["baseline", "fast"]);
    expect(plan?.summary.totalRuns).toBe(2);
    expect(plan?.runs.map((run) => run.preset)).toEqual(["baseline", "fast"]);
    expect(plan?.runs.every((run) => !("snapshotId" in run))).toBe(true);
  });

  it("builds snapshot-only run plans from selected snapshot records", () => {
    const snapshots: ConfigSnapshot[] = [
      makeSnapshot({ hidden_dim: "128", num_epochs: "3" }, "wide"),
    ];

    const plan = buildConfigSnapshotRunPlan({
      model: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: [],
      selectedDatasets: ["Mnist"],
      snapshots,
      fields,
      logFolder: "snapshots",
    });

    expect(plan?.preset).toBe("baseline");
    expect(plan?.presets).toEqual(["baseline"]);
    expect(plan?.summary.totalRuns).toBe(1);
    expect(plan?.runs[0]).toMatchObject({
      preset: "baseline",
      snapshotId: "snap-wide",
      snapshotName: "wide",
      overrides: { hidden_dim: "128", num_epochs: "3" },
    });
  });

  it("builds mixed default and snapshot run plans across selected datasets", () => {
    const snapshots: ConfigSnapshot[] = [
      makeSnapshot({ hidden_dim: "128", num_epochs: "3" }, "wide"),
      makeSnapshot({ num_layers: "4" }, "deep"),
    ];

    const plan = buildConfigSnapshotRunPlan({
      model: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline"],
      selectedDatasets: ["Mnist", "Cifar10"],
      snapshots,
      fields,
      logFolder: "snapshots",
    });

    expect(plan?.search).toBeNull();
    expect(plan?.summary.totalRuns).toBe(6);
    expect(plan?.summary.remainingEpochs).toBe(46);
    expect(plan?.runs.slice(0, 2).map((run) => run.snapshotName)).toEqual([
      undefined,
      undefined,
    ]);
    expect(plan?.runs.slice(0, 2).map((run) => run.overrides)).toEqual([{}, {}]);
    expect(plan?.runs[0]).not.toHaveProperty("snapshotId");
    expect(plan?.runs[0]).not.toHaveProperty("snapshotName");
    expect(plan?.runs[0].changes).toEqual([]);
    expect(plan?.runs[0].command).toBe(
      "source experiment.sh linear --preset baseline --datasets Mnist --logdir snapshots",
    );
    expect(plan?.runs.map((run) => run.snapshotName)).toEqual([
      undefined,
      undefined,
      "wide",
      "wide",
      "deep",
      "deep",
    ]);
    expect(plan?.runs[2].changes).toEqual([
      {
        key: "hidden_dim",
        label: "Hidden Dim",
        value: "128",
        source: "override",
      },
      {
        key: "num_epochs",
        label: "Epochs",
        value: "3",
        source: "override",
      },
    ]);
    expect(plan?.runs[2].command).toContain("--logdir snapshots");
    expect(plan?.runs[2].command).toContain("--config --hidden-dim 128");
    expect(plan?.runs[2].command).not.toContain("wide");
    expect(plan?.runs[2].command).not.toContain("snap-wide");
  });

  it("keeps selected snapshots when their source preset is not selected", () => {
    const snapshots: ConfigSnapshot[] = [
      {
        ...makeSnapshot({ hidden_dim: "128" }, "fast-wide"),
        preset: "fast",
      },
    ];

    const plan = buildConfigSnapshotRunPlan({
      model: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline"],
      selectedDatasets: ["Mnist"],
      snapshots,
      fields,
      logFolder: "snapshots",
    });

    expect(plan?.preset).toBe("baseline");
    expect(plan?.presets).toEqual(["baseline", "fast"]);
    expect(plan?.summary.totalRuns).toBe(2);
    expect(plan?.runs[0]).toMatchObject({
      preset: "baseline",
      overrides: {},
    });
    expect(plan?.runs[1]).toMatchObject({
      preset: "fast",
      snapshotId: "snap-fast-wide",
      snapshotName: "fast-wide",
      overrides: { hidden_dim: "128" },
    });
  });

  it("uses the first selected training preset when the target primary is deselected", () => {
    const snapshots: ConfigSnapshot[] = [
      {
        ...makeSnapshot({ hidden_dim: "128" }, "fast-wide"),
        preset: "fast",
      },
    ];

    const plan = buildConfigSnapshotRunPlan({
      model: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["fast"],
      selectedDatasets: ["Mnist"],
      snapshots,
      fields,
      logFolder: "snapshots",
    });

    expect(plan?.preset).toBe("fast");
    expect(plan?.presets).toEqual(["fast"]);
    expect(plan?.runs[0].preset).toBe("fast");
    expect(plan?.runs[0].command).toContain("--preset fast");
  });
});
