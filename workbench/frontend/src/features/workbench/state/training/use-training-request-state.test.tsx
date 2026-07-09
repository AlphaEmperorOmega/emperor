import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { type ConfigField, type SearchAxis } from "@/lib/api";
import { type ConfigSection } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { type TrainingSearchState } from "@/lib/training-search";
import { useTrainingRequestState } from "@/features/workbench/state/training/use-training-request-state";

const fields: ConfigField[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    flag: "--hidden-dim",
    label: "Hidden Dim",
    section: "Model",
    sectionPath: ["Model"],
    type: "int",
    default: 64,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "num_epochs",
    configKey: "NUM_EPOCHS",
    flag: "--num-epochs",
    label: "Epochs",
    section: "Training",
    sectionPath: ["Training"],
    type: "int",
    default: 10,
    nullable: false,
    choices: [],
    locked: false,
  },
];

const configSections: ConfigSection[] = [{ title: "Model", fields }];

const searchAxes: SearchAxis[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    searchKey: "SEARCH_SPACE_HIDDEN_DIM",
    label: "Hidden Dim",
    section: "Model",
    type: "int",
    values: [128, 256],
    locked: false,
  },
];

const gridSearch: TrainingSearchState = {
  mode: "grid",
  selectedValues: { hidden_dim: [128, 256] },
  randomSamples: 10,
};

const snapshots: ConfigSnapshot[] = [
  {
    id: "wide",
    name: "Wide",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    overrides: { hidden_dim: "128", num_epochs: "4" },
    createdAt: "2026-06-01T00:00:00.000Z",
  },
];

function renderRequestState({
  selectedTrainingSnapshots = snapshots,
}: {
  selectedTrainingSnapshots?: ConfigSnapshot[];
} = {}) {
  return renderHook(() =>
    useTrainingRequestState({
      configSections,
      overrides: { hidden_dim: "192", dropout: "0.2" },
      configSnapshotCount: snapshots.length,
      selectedTrainingSnapshots,
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: ["baseline"],
      selectedDatasets: ["Mnist"],
      trainingSearch: gridSearch,
      searchAxes,
      searchLoading: false,
      trainingEnabled: true,
      logFolder: "runs",
    }),
  );
}

describe("useTrainingRequestState", () => {
  it("uses a mixed submitted run plan while at least one snapshot is checked", () => {
    const { result } = renderRequestState();

    expect(result.current.hasConfigSnapshots).toBe(true);
    expect(result.current.activeConfigSnapshotCount).toBe(1);
    expect(result.current.effectiveTrainingSearch.mode).toBe("off");
    expect(result.current.effectiveOverrides).toEqual({});
    expect(result.current.searchPayload).toBeUndefined();
    expect(result.current.snapshotRunPlan?.summary.totalRuns).toBe(2);
    expect(result.current.snapshotRunPlan?.summary.remainingEpochs).toBe(14);
    expect(result.current.snapshotRunPlan?.runs[0]).not.toHaveProperty(
      "snapshotId",
    );
    expect(result.current.snapshotRunPlan?.runs[0].overrides).toEqual({
      hidden_dim: "192",
      dropout: "0.2",
    });
    expect(result.current.snapshotRunPlan?.runs[1]).toMatchObject({
      snapshotId: "wide",
      snapshotName: "Wide",
      overrides: {
        hidden_dim: "192",
        num_epochs: "4",
        dropout: "0.2",
      },
    });
    expect(result.current.plannedRunCount).toBe(2);
    expect(result.current.canPlan).toBe(true);
  });

  it("falls back to normal overrides and search when all snapshots are unchecked", () => {
    const { result } = renderRequestState({
      selectedTrainingSnapshots: [],
    });

    expect(result.current.hasConfigSnapshots).toBe(true);
    expect(result.current.activeConfigSnapshotCount).toBe(0);
    expect(result.current.effectiveTrainingSearch).toEqual(gridSearch);
    expect(result.current.effectiveOverrides).toEqual({ dropout: "0.2" });
    expect(result.current.searchPayload).toEqual({
      mode: "grid",
      values: { hidden_dim: [128, 256] },
    });
    expect(result.current.snapshotRunPlan).toBeUndefined();
    expect(result.current.plannedRunCount).toBe(2);
    expect(result.current.canPlan).toBe(true);
  });
});
