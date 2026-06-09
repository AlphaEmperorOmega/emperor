import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  buildTrainingModelOptions,
  buildTrainingMonitorOptions,
  buildTrainingPresetOptions,
} from "@/features/viewer/state/training/training-panel-options";

describe("training panel options", () => {
  it("maps models and presets to select options", () => {
    expect(buildTrainingModelOptions(["linear", "neuron"])).toEqual([
      { value: "linear", label: "linear" },
      { value: "neuron", label: "neuron" },
    ]);
    expect(
      buildTrainingPresetOptions([
        { name: "baseline", label: "Baseline", description: "Base run" },
      ]),
    ).toEqual([{ value: "baseline", label: "baseline" }]);
  });

  it("maps monitor metadata for the training multi-select", () => {
    const options = buildTrainingMonitorOptions([
      {
        name: "linear",
        label: "Linear monitor",
        description: "Layer activations",
        kinds: ["scalar", "histogram"],
        defaultEnabled: true,
      },
      {
        name: "empty",
        label: "Empty monitor",
        description: "No channels",
        kinds: [],
        defaultEnabled: false,
      },
    ]);

    expect(options[0]).toMatchObject({
      value: "linear",
      label: "Linear monitor",
      description: "Layer activations",
    });
    render(<>{options[0].meta}</>);
    expect(screen.getByText("scalar / histogram")).toBeInTheDocument();
    expect(options[1].meta).toBeUndefined();
  });
});
