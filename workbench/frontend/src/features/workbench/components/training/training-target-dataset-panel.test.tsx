import { type ComponentProps } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TrainingTargetDatasetPanel } from "@/features/workbench/components/training/training-target-dataset-panel";
import { type ConfigSnapshotMutationOutcome } from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";

type TrainingSetup = ComponentProps<
  typeof TrainingTargetDatasetPanel
>["setup"];

function setup(): TrainingSetup {
  return {
    model: {
      selectedType: "linears",
      selected: "linear",
      typeOptions: [{ value: "linears", label: "Linears" }],
      options: [{ value: "linear", label: "Linear" }],
      selectType: vi.fn(),
      select: vi.fn(),
    },
    variants: {
      primaryPreset: "baseline",
      selectedPresets: ["baseline"],
      selectedSnapshotIds: [],
      presetOptions: [{ value: "baseline", label: "baseline" }],
      snapshots: [
        {
          id: "snapshot-1",
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          name: "Wide",
          overrides: { hidden_dim: "128" },
          createdAt: "2026-06-01T00:00:00.000Z",
        },
      ],
      snapshotMutation: {
        phase: "idle",
        kind: null,
        snapshotId: null,
        error: "",
        canRetry: false,
      },
      selectPrimaryPreset: vi.fn(),
      selectPresets: vi.fn(),
      togglePreset: vi.fn(),
      excludePreset: vi.fn(),
      makePresetPrimary: vi.fn(),
      selectAllPresets: vi.fn(),
      selectOnlyPrimaryPreset: vi.fn(),
      selectSnapshots: vi.fn(),
      includeSnapshot: vi.fn(),
      excludeSnapshot: vi.fn(),
      removeSnapshot: vi.fn(
        async (): Promise<ConfigSnapshotMutationOutcome> => ({
          ok: true,
          kind: "remove",
          snapshotId: "snapshot-1",
          record: null,
        }),
      ),
      retrySnapshotMutation: vi.fn(async () => null),
      dismissSnapshotMutation: vi.fn(),
      createPresetSnapshot: vi.fn(),
      editSnapshot: vi.fn(),
      duplicateSnapshot: vi.fn(),
    },
    experimentTask: {
      selected: "image-classification",
      options: [
        { value: "image-classification", label: "Image Classification" },
      ],
      select: vi.fn(),
    },
    datasets: {
      selected: ["Mnist"],
      options: [
        { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
      ],
      select: vi.fn(),
      toggle: vi.fn(),
      selectAll: vi.fn(),
      selectFirst: vi.fn(),
    },
    monitors: {
      selected: [],
      options: [
        {
          name: "linear",
          label: "Linear",
          description: "Linear activations",
          kinds: ["activation"],
          defaultEnabled: false,
        },
      ],
      isLoading: false,
      select: vi.fn(),
      selectAll: vi.fn(),
      clear: vi.fn(),
    },
  };
}

describe("Training setup Adapter", () => {
  it("renders the grouped setup ownership and switches variant projections", async () => {
    const value = setup();
    render(<TrainingTargetDatasetPanel setup={value} />);
    const user = userEvent.setup();

    expect(
      screen.getByRole("combobox", { name: "training model type" }),
    ).toHaveTextContent("Linears");
    expect(
      screen.getByRole("combobox", { name: "Experiment task" }),
    ).toHaveTextContent("Image Classification");
    expect(
      screen.getByRole("combobox", { name: /^Presets 1 \/ 1 selected$/ }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("combobox", { name: "Config snapshots" }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Snapshots" }));
    expect(
      screen.getByRole("combobox", {
        name: /^Config snapshots 0 \/ 1 selected$/,
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("combobox", { name: /^Presets / }),
    ).not.toBeInTheDocument();
  });

  it("surfaces the grouped mutation failure and delegates retry", async () => {
    const value = setup();
    value.variants.snapshotMutation = {
      phase: "failed",
      kind: "remove",
      snapshotId: "snapshot-1",
      error: "Removal rejected.",
      canRetry: true,
    };
    render(<TrainingTargetDatasetPanel setup={value} />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: "Snapshots" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Removal rejected.");
    await user.click(screen.getByRole("button", { name: "Retry change" }));
    expect(value.variants.retrySnapshotMutation).toHaveBeenCalledTimes(1);
  });
});
