import { describe, expect, it } from "vitest";
import { type SearchAxis } from "@/lib/api";
import {
  deriveTrainingSearchLockSummary,
  effectiveUnlockedTrainingSearch,
  estimatePlannedRuns,
  formatTrainingSearchList,
  unlockedSearchAxes,
  validateTrainingSearch,
  type TrainingSearchState,
} from "@/lib/training-search";

const axes: SearchAxis[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    searchKey: "SEARCH_SPACE_HIDDEN_DIM",
    label: "hidden dim",
    section: "Layer Stack Options",
    type: "int",
    values: [64, 128],
    locked: false,
  },
  {
    key: "stack_layer_norm_position",
    configKey: "STACK_LAYER_NORM_POSITION",
    searchKey: "SEARCH_SPACE_STACK_LAYER_NORM_POSITION",
    label: "stack layer norm position",
    section: "Layer Stack Options",
    type: "enum",
    values: ["BEFORE", "AFTER"],
    locked: true,
    lockedValue: "AFTER",
    lockedReason:
      "Locked by the POST_NORM preset because this preset locks `layer_norm_position`.",
    lockedByPresets: ["POST_NORM"],
    lockReasons: [
      "Locked by the POST_NORM preset because this preset locks `layer_norm_position`.",
    ],
  },
];

describe("training search lock handling", () => {
  it("formats domain lists with stable conjunction grammar", () => {
    expect(formatTrainingSearchList([])).toBe("");
    expect(formatTrainingSearchList(["A"])).toBe("A");
    expect(formatTrainingSearchList(["A", "B"])).toBe("A and B");
    expect(formatTrainingSearchList(["A", "B", "C"])).toBe("A, B, and C");
  });

  it("treats all axes as all unlocked axes", () => {
    expect(unlockedSearchAxes(axes).map((axis) => axis.key)).toEqual([
      "hidden_dim",
    ]);
  });

  it("derives preset-owned and selected-skipped warnings", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {
        hidden_dim: [64],
        stack_layer_norm_position: ["BEFORE", "AFTER"],
      },
      randomSamples: 10,
    };

    const summary = deriveTrainingSearchLockSummary(search, axes);

    expect(summary.lockedAxisCount).toBe(1);
    expect(summary.lockedAxisLabels).toEqual(["stack layer norm position"]);
    expect(summary.lockedPresetLabels).toEqual(["POST_NORM"]);
    expect(summary.lockedAxesMessage).toContain("POST_NORM");
    expect(summary.lockedAxesMessage).toContain("stack layer norm position");
    expect(summary.skippedSelectedAxisCount).toBe(1);
    expect(summary.skippedSelectedAxisMessage).toBe(
      "1 selected axis was skipped because a selected preset owns it.",
    );
  });

  it("excludes locked selected axes from the effective search state", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {
        hidden_dim: [64],
        stack_layer_norm_position: ["BEFORE", "AFTER"],
      },
      randomSamples: 10,
    };

    expect(effectiveUnlockedTrainingSearch(search, axes)).toEqual({
      mode: "grid",
      selectedValues: {
        hidden_dim: [64],
      },
      randomSamples: 10,
    });
  });

  it("can estimate base runs when every selected search axis was skipped", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {},
      randomSamples: 10,
    };

    expect(estimatePlannedRuns(search, 2, 3)).toBe(0);
    expect(
      estimatePlannedRuns(search, 2, 3, { emptySearchRunsAsBase: true }),
    ).toBe(6);
  });

  it("rejects a stale locked selected axis defensively", () => {
    const search: TrainingSearchState = {
      mode: "grid",
      selectedValues: {
        stack_layer_norm_position: ["BEFORE"],
      },
      randomSamples: 10,
    };

    expect(validateTrainingSearch(search, axes)).toEqual({
      ready: false,
      message: "stack layer norm position is locked by this preset.",
    });
  });
});
